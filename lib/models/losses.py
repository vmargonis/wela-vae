import math

from tensorflow import Tensor, constant
from tensorflow.keras import backend as K
from tensorflow.linalg import diag, diag_part
from tensorflow.math import reduce_logsumexp


def kl_divergence(mean: Tensor, log_var: Tensor) -> Tensor:
    """
    Computes batch KL divergence.

    :param mean: Encoder mean, tensor of size (batch_size, latent_dim).
    :param log_var: Encoder log variance, tensor of size (batch_size, latent_dim).
    :return: batch KL divergence, tensor of size ().
    """
    kl = K.square(mean) + K.exp(log_var) - log_var - 1  # shape:(?,L)
    sample_kl = 0.5 * K.sum(kl, axis=-1)  # shape:(?,)

    return K.mean(sample_kl, axis=0)


def bernoulli_loss(
    original: Tensor,
    reconstruction: Tensor,
    is_binary_input: bool,
) -> Tensor:
    """
    Computes batch cross-entropy loss for multi-bernoulli decoder.

    :param original: VAE input, tensor of size (batch_size, initial_dim).
    :param reconstruction: VAE output, tensor of size (batch_size, initial_dim).
    :param is_binary_input: Incicates whether pixels are binary.
    :return: Batch cross entropy loss, tensor of size ().
    """
    # if input is not binary, lower bound in cross entropy is not zero
    if is_binary_input:
        xent_lower_bound = 0

    else:
        original_clipped = K.clip(original, 1e-7, 1 - 1e-7)
        xent_lower_bound = -K.sum(
            original_clipped * K.log(original_clipped),
            axis=-1,
        )  # shape:(?,)

    reconstruction_clipped = K.clip(reconstruction, 1e-7, 1 - 1e-7)
    # cross-entropy
    a = original * K.log(reconstruction_clipped)
    b = (1 - original) * K.log(1 - reconstruction_clipped)  # noqa
    xent = -K.sum(a + b, axis=-1)
    sample_xent = xent - xent_lower_bound  # shape:(?,)

    return K.mean(sample_xent, axis=0)


def gaussian_loss(original: Tensor, reconstruction: Tensor) -> Tensor:
    """
    Computes batch L2 loss for Gaussian decoder.

    :param original: VAE input, tensor of size (batch_size, initial_dim).
    :param reconstruction: VAE output, tensor of size (batch_size, initial_dim).
    :return: Batch L2 loss, tensor of size ().
    """
    loss = K.sum(K.square(original - reconstruction), axis=-1)  # shape:(?,)
    return K.mean(loss, axis=0)


def compute_covariance_mean(mean: Tensor) -> Tensor:
    """
    Computes the batch covariance matrix of means for DIP-VAE I.
    Uses cov(means) = E[means^T * means] - E[means]^T * E[means].

    :param mean: Encoder mean, tensor of size (batch_size, latent_dim).
    :return: Covariance of encoder mean, tensor of size (latent_dim, latent_dim).
    """
    batch_size = K.cast(K.shape(mean)[0], dtype="float32")

    # Compute a = E[means^T * means], shape:(L,L)
    a = K.dot(K.transpose(mean), mean) / batch_size

    # Compute b = E[means]^T * E[means], shape:(L,L)
    expected_means = K.mean(mean, axis=0, keepdims=True)  # shape:(1,L)
    b = K.dot(K.transpose(expected_means), expected_means)  # shape:(L,L)

    return a - b


def compute_covariance_z(mean: Tensor, log_var: Tensor) -> Tensor:
    """
    Computes the batch covariance matrix of Gaussian samples z
    for DIP-VAE II. Uses Cov_q(z)(z) = E[Cov] + Cov(means).

    :param mean: Encoder mean, tensor of size (batch_size, latent_dim).
    :param log_var: Encoder log variance, tensor of size (batch_size, latent_dim).
    :return: batch covariance matrix of z, tensor of size (latent_dim, latent_dim).
    """
    covariance_means = compute_covariance_mean(mean)  # shape:(L,L)
    mean_sigma_squared = K.mean(K.exp(log_var), axis=0)  # shape:(L,)
    expected_covariance = diag(mean_sigma_squared)  # shape:(L,L)

    return expected_covariance + covariance_means


def dip_vae_regularizer(
    cov_matrix: Tensor,
    lambda_off_diag: float,
    lambda_diag: float,
) -> Tensor:
    """
    Compute regularizers for DIP-VAE I & II.

    :param cov_matrix: Covariance Matrix to regularize, size (latent_dim, latent_dim)
    :param lambda_off_diag: Weight of penalty for off-diagonal elements.
    :param lambda_diag: Weight of penalty for diagonal elements.
    :return: DIP regularizer, tensor of size ().
    """
    cov_matrix_diagonal = diag_part(cov_matrix)  # shape:(L,)
    cov_matrix_off_diagonal = cov_matrix - diag(cov_matrix_diagonal)

    off_diag_penalty = lambda_off_diag * K.sum(cov_matrix_off_diagonal**2)
    diag_penalty = lambda_diag * K.sum((cov_matrix_diagonal - 1) ** 2)

    return off_diag_penalty + diag_penalty


def gaussian_log_density(z: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
    """
    Computes element-wise Gaussian log density.
    Uses log(q(z)) = -0.5 * (log(2pi) + log(sigma^2) + (z-mu)^2 * sigma^(-2))

    :param z: gaussian samples, tensor of size (batch_size, latent_dim).
    :param mean: Encoder mean, tensor of size (batch_size, latent_dim).
    :param log_var: Encoder log variance, tensor of size (batch_size, latent_dim).
    :return: log density, tensor of size (batch_size, latent_dim).
    """
    log_2pi = K.log(2.0 * constant(math.pi))
    inverse_vars = K.exp(-log_var)
    return -0.5 * (log_2pi + log_var + inverse_vars * (z - mean) * (z - mean))


def total_correlation(z: Tensor, mean: Tensor, log_var: Tensor) -> Tensor:
    """
    Estimation of total correlation on a batch.
    We need to compute the expectation over a batch of:
    E_j [log(q(z(x_j))) - log(prod_l q(z(x_j)_l))].
    Constants are ignored as they do not matter for the minimization.

    :param z: Gaussian samples, tensor of shape (batch_size, latent_dim).
    :param mean: Encoder mean, tensor of shape (batch_size, latent_dim).
    :param log_var: Encoder log variance, tensor of shape (batch_size, latent_dim).
    :return: Estimated total correlation for the batch.
    """
    # Compute log(q(z(x_j)|x_i)) for every sample in the batch, which is a
    # tensor of size (batch_size, batch_size, latent_dim). In the following
    # comments, (batch_size, batch_size, latent_dim) are indexed by (j, i, l).
    log_qz_prob = gaussian_log_density(
        K.expand_dims(z, axis=1),
        K.expand_dims(mean, axis=0),
        K.expand_dims(log_var, axis=0),
    )
    # Compute log prod_l p(z(x_j)_l) =
    # sum_l(log(sum_i(q(z(z_j)_l|x_i))) + constant)
    # for each sample in the batch, which is a vector of size (batch_size,).
    log_qz_product = K.sum(
        reduce_logsumexp(log_qz_prob, axis=1, keepdims=False), axis=1, keepdims=False
    )
    # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + constant =
    # log(sum_i(prod_l q(z(x_j)_l|x_i))) + constant.
    log_qz = reduce_logsumexp(K.sum(log_qz_prob, axis=2), axis=1, keepdims=False)

    return K.mean(log_qz - log_qz_product)
