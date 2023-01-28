from keras import backend as K
from keras.optimizers import Adagrad, RMSprop, Adam, SGD


# Reparameterazation trick
def reparam_trick(args):
    """
    Reparameterization trick
    :param args: means, log_vars. tensors of shape (batch_size, latent_dim)
    :return: z = mu + eps*sigma, eps~N(0,1), shape (batch_size, latent_dim)
    """
    mean, log_var = args
    n_samples = K.shape(mean)[0]  # num. of samples in batch
    n_features = K.shape(mean)[1]  # dimension of dataset

    epsilon = K.random_normal(shape=(n_samples, n_features))
    z = mean + K.exp(0.5 * log_var) * epsilon
    return z


def add_optimizer(config):
    lr = config['optimizer']['learning_rate']  # learning rate

    if config['optimizer']['type'] == 'Adagrad':
        optimizer = Adagrad(lr=lr)
    elif config['optimizer']['type'] == 'Adam':
        optimizer = Adam(lr=lr)
    elif config['optimizer']['type'] == 'RMSprop':
        optimizer = RMSprop(lr=lr)
    elif config['optimizer']['type'] == 'SGD':
        optimizer = SGD(lr=lr)
    else:
        raise NotImplementedError("Optimizer not supported.")
    return optimizer
