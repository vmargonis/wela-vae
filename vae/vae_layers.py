import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.initializers.initializers_v2 import GlorotUniform
from typing import Dict, Tuple


def _reparameterization_trick(args: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """Reparameterization trick:  z = mu + eps*sigma, eps~N(0,1)

    Parameters
    ----------
    args : Tuple[tf.Tensor, tf.Tensor]
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


def reparameterize(config: Dict) -> Model:

    mean = Input(shape=(config["latent_dim"],))
    log_var = Input(shape=(config["latent_dim"],))

    z_sample = Lambda(
        _reparameterization_trick,
        output_shape=(config["latent_dim"],),
        name="z_sample",
    )((mean, log_var))

    reparameterizer = Model(
        inputs=[mean, log_var],
        outputs=z_sample,
        name="reparameterizer"
    )

    return reparameterizer


def stacked_encoder(
        config: Dict,
) -> Tuple[Input, Dense, Dense, Model]:

    # glorot weight initializer
    w_init = GlorotUniform(seed=config["weight_seed"])

    # input placeholder
    input_vector = Input(shape=(config["initial_dim"],))

    # hidden layer(s)
    num_layers = len(config["encoder"]["units"])

    # First hidden is connected to input
    hidden = Dense(
        units=config["encoder"]["units"][0],
        activation=config["encoder"]["activation"][0],
        kernel_initializer=w_init,
    )(input_vector)

    # rest of layers connected to each other
    for i in range(1, num_layers):
        hidden = Dense(
            units=config["encoder"]["units"][i],
            activation=config["encoder"]["activation"][i],
            kernel_initializer=w_init,
        )(hidden)

    # Mean Layer (Latent vector - vae representation)
    mean = Dense(
        units=config["latent_dim"],
        activation=config["encoder"]["output_activation"],
        kernel_initializer=w_init,
        name="encoder_mean",
    )(hidden)

    # Log var layer
    log_var = Dense(
        units=config["latent_dim"],
        activation=config["encoder"]["output_activation"],
        kernel_initializer=w_init,
        name="encoder_log_sigma_squared",
    )(hidden)

    encoder = Model(
        inputs=input_vector,
        outputs=[mean, log_var],
        name="encoder",
    )

    return input_vector, mean, log_var, encoder


def stacked_decoder(config: Dict) -> Model:

    # glorot weight initializer
    w_init = GlorotUniform(seed=config["weight_seed"])

    z_sample = Input(shape=(config["latent_dim"],))

    # First hidden is connected to input
    hidden = Dense(
        units=config["decoder"]["units"][0],
        activation=config["decoder"]["activation"][0],
        kernel_initializer=w_init,
    )(z_sample)

    # hidden layer(s) / rest of layers connected to each other
    num_layers = len(config["decoder"]["units"])

    for i in range(1, num_layers):
        hidden = Dense(
            units=config["decoder"]["units"][i],
            activation=config["decoder"]["activation"][i],
            kernel_initializer=w_init,
        )(hidden)

    output_vector = Dense(
        units=config["initial_dim"],
        activation=config["decoder"]["output_activation"],
        kernel_initializer=w_init,
        name="output_vector",
    )(hidden)

    decoder = Model(
        inputs=z_sample,
        outputs=output_vector,
        name="decoder",
    )

    return decoder
