from keras import backend as K
from keras.optimizers import Adagrad, RMSprop, Adam, SGD
from vae.losses import gaussian_loss, bernoulli_loss


def reparameterization_trick(args):
    """Reparameterization trick:  z = mu + eps*sigma, eps~N(0,1)

    Parameters
    ----------
    args : tf.Tensor, tf.Tensor
        [mean, log_var], tensors of shape (batch_size, latent_dim)

    Returns
    -------
    tf.Tensor
        Gaussian samples, z = mu + eps*sigma, eps~N(0,1),
        tensor of shape (batch_size, latent_dim).
    """

    mean, log_var = args

    white_noise = K.random_normal(
        shape=(K.shape(mean)[0], K.shape(mean)[1])
    )

    return mean + K.exp(0.5 * log_var) * white_noise


def get_reconstruction_loss(input_vec, output_vec, config):
    """Get recostruction loss based on I/O and config."""

    if config["output_dist"] == "bernoulli":
        reconstruction_loss = bernoulli_loss(
            input_vec,
            output_vec,
            config["is_binary_input"],
        )

    elif config["output_dist"] == "gaussian":
        reconstruction_loss = gaussian_loss(input_vec, output_vec)

    else:
        raise NotImplementedError("Loss not supported.")

    return reconstruction_loss


def add_optimizer(config):
    """Optimizer selector."""

    learning_rate = config['optimizer']['learning_rate']

    if config['optimizer']['type'] == 'Adagrad':
        optimizer = Adagrad(learning_rate=learning_rate)
    elif config['optimizer']['type'] == 'Adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif config['optimizer']['type'] == 'RMSprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    elif config['optimizer']['type'] == 'SGD':
        optimizer = SGD(learning_rate=learning_rate)
    else:
        raise NotImplementedError("Optimizer not supported.")

    return optimizer
