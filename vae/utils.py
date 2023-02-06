from keras import backend as K
from keras.optimizers import Adagrad, RMSprop, Adam, SGD


def reparameterization_trick(args):
    """Reparameterization trick:  z = mu + eps*sigma, eps~N(0,1)

    Parameters
    ----------
    args : tf.Tensor, tf.Tensor
        [means, log_vars], tensors of shape (batch_size, latent_dim)

    Returns
    -------
    tf.Tensor
        Gaussian samples, z = mu + eps*sigma, eps~N(0,1),
        tensor of shape (batch_size, latent_dim).
    """

    means, log_vars = args

    white_noise = K.random_normal(
        shape=(K.shape(means)[0], K.shape(means)[1])
    )

    return means + K.exp(0.5 * log_vars) * white_noise


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
