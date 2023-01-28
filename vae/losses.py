from keras import backend as K
import tensorflow as tf
import math


def kl_divergence(means, log_vars):
    """
    Computes batch KL divergence.
    Args:
    mean: Encoder mean, tensor of size [batch_size, num_latent].
    log_var: Encoder log_var, tensor of size [batch_size, num_latent].
    Returns:
    batch_dkl: tensor of size ().
    """
    
    kl = K.square(means) + K.exp(log_vars) - log_vars - 1  # shape: (?,L)
    sample_dkl = 0.5 * K.sum(kl, axis=-1)  # shape: (?,)
    batch_dkl = K.mean(sample_dkl, axis=0)  # shape: ()

    return batch_dkl


def bernoulli_loss(original, reconstruction, isbinaryinput):
    """
    Computes batch cross-entropy loss for multi-bernoulli decoder.
    Args:
    original: VAE input, tensor of size [batch_size, initial_dim].
    reconstruction: VAE output, tensor of size [batch_size, initial_dim].
    isbinaryinput: Boolean, incicates whether pixels are binary.
    Returns:
    batch_xent: tensor of size ().
    """

    # if input is not binary, lower bound in cross entropy is not zero
    if isbinaryinput is True:
        xent_lb = 0
    else:
        dist = K.clip(original, 1e-7, 1 - 1e-7)
        xent_lb = -K.sum(dist * K.log(dist), axis=-1)  # shape:(?,)

    # cross-entropy
    # Clip reconstructions to [0+e,1-e] for numerical stability
    reconstruction_clipped = K.clip(reconstruction, 1e-7, 1 - 1e-7)
    a = original * K.log(reconstruction_clipped)
    b = (1 - original) * K.log(1 - reconstruction_clipped)
    xent = - K.sum(a + b, axis=-1)
    sample_xent = xent - xent_lb  # shape:(?,)
    batch_xent = K.mean(sample_xent, axis=0)  # shape:()

    return batch_xent


def gaussian_loss(original, reconstruction):
    """
    Computes batch L2 loss for Gaussian decoder.
    Args:
    original: VAE input, tensor of size [batch_size, initial_dim].
    reconstruction: VAE output, tensor of size [batch_size, initial_dim].
    Returns:
    batch_l2: tensor of size ().
    """

    # L2 loss
    sample_l2 = K.sum(K.square(original - reconstruction), axis=-1)  # (?,)
    batch_l2 = K.mean(sample_l2, axis=0)  # shape: ()

    return batch_l2


def compute_covariance_mean(means):
    """
    Computes the batch covariance matrix of means for DIP-VAE I.
    Uses cov(means) = E[means^T * means] - E[means]^T * E[means].
    Args:
    means: Encoder mean, tensor of size [batch_size, num_latent].
    Returns:
    covariance_means: tensor of size [num_latent, num_latent].
    """

    batch_size = K.cast(K.shape(means)[0], dtype='float32')

    # Compute a = E[means^T*means], shape: (L,L)
    a = K.dot(K.transpose(means), means) / batch_size

    # Compute b = E[means]^T*E[means], shape: (L,L)
    expected_means = K.mean(means, axis=0, keepdims=True)  # shape:(1,L)
    b = K.dot(K.transpose(expected_means), expected_means)  # shape:(L,L)

    covariance_means = a - b
    return covariance_means


def compute_covariance_z(means, log_vars):
    """
    Computes the batch covariance matrix of z for DIP-VAE II.
    Uses Cov_q(z)(z) = E[Cov] + Cov(means).
    Args:
    means: Encoder mean, tensor of size [batch_size, num_latent].
    log_vars: Encoder log_vars, tensor of size [batch_size, num_latent].
    Returns:
    cov_z: tensor of size [num_latent, num_latent].
    """

    # Cov(means)
    covariance_means = compute_covariance_mean(means)
    # E[Cov]
    mean_sigma_squared = K.mean(K.exp(log_vars), axis=0)  # shape:(L,)
    expected_covariance = tf.matrix_diag(mean_sigma_squared)  # shape:(L,L)

    covariance_z = expected_covariance + covariance_means
    return covariance_z


def dip_vae_regularizer(cov_matrix, lambda_od, lambda_d):
    """
    Compute regularizers for DIP-VAE I & II.
    Args:
    cov_matrix: Tensor of size [num_latent, num_latent] to regularize.
    lambda_od: Weight of penalty for off diagonal elements.
    lambda_d: Weight of penalty for diagonal elements.
    Returns:
    dip_regularizer: tensor of size ().
    """

    cov_matrix_diagonal = tf.diag_part(cov_matrix)  # (L,)
    cov_matrix_off_diagonal = cov_matrix - tf.diag(cov_matrix_diagonal)

    off_diag_penalty = lambda_od * K.sum(cov_matrix_off_diagonal ** 2)
    diag_penalty = lambda_d * K.sum((cov_matrix_diagonal - 1) ** 2)

    dip_regularizer = off_diag_penalty + diag_penalty
    return dip_regularizer


def gaussian_log_density(z, mean, log_var):
    """
    Computes element-wise Gaussian log density.
    Uses log(q(z)) = -0.5 * (log(2pi) + log(sigma^2) + (z-mu)^2 * sigma^(-2))
    Args:
    means: Encoder mean, tensor of size [batch_size, num_latent].
    log_vars: Encoder log_vars, tensor of size [batch_size, num_latent].
    z: Gaussian samples, tensor of size [batch_size, num_latent].
    Returns:
    log density: tensor of size [batch_size, num_latent].
    """

    pi = tf.constant(math.pi)
    log_2pi = K.log(2. * pi)
    inv_var = K.exp(-log_var)
    log_density = -0.5 * (log_2pi + log_var + inv_var * (z-mean) * (z-mean))
    return log_density


def total_correlation(z, mean, logvar):
    """Estimate of total correlation on a batch.
    We need to compute the expectation over a batch of: E_j [log(q(z(x_j))) -
    log(prod_l q(z(x_j)_l))]. We ignore the constants as they do not matter
    for the minimization. The constant should be equal to (num_latents - 1) *
    log(batch_size * dataset_size)
    Args:
    z: [batch_size, num_latents]-tensor with sampled representation.
    z_mean: [batch_size, num_latents]-tensor with mean of the encoder.
    z_logvar: [batch_size, num_latents]-tensor with log variance of the encoder
    .
    Returns:
    Total correlation estimated on a batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size [batch_size, batch_size, num_latents]. In the following
    # comments, [batch_size, batch_size, num_latents] are indexed by [j, i, l].
    log_qz_prob = gaussian_log_density(
      tf.expand_dims(z, 1), tf.expand_dims(mean, 0),
      tf.expand_dims(logvar, 0))
    # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
    # + constant) for each sample in the batch, which is a vector of size
    # [batch_size,].
    log_qz_product = tf.reduce_sum(
      tf.reduce_logsumexp(log_qz_prob, axis=1, keepdims=False),
      axis=1,
      keepdims=False)
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = tf.reduce_logsumexp(
      tf.reduce_sum(log_qz_prob, axis=2, keepdims=False),
      axis=1,
      keepdims=False)
    return tf.reduce_mean(log_qz - log_qz_product)
