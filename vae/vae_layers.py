import keras
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.initializers.initializers_v2 import GlorotUniform
from vae.utils import reparameterization_trick
from typing import Dict, Tuple


def reparameterize(config: Dict) -> keras.Model:

    mean = Input(shape=(config["latent_dim"],))
    log_var = Input(shape=(config["latent_dim"],))

    z_sample = Lambda(
        reparameterization_trick,
        output_shape=(config["latent_dim"],),
        name="z_sample",
    )([mean, log_var])

    reparameterizer = Model(
        inputs=[mean, log_var],
        outputs=z_sample,
        name="reparameterizer"
    )

    return reparameterizer


def stacked_encoder(
        config: Dict,
) -> Tuple[keras.Input, keras.layers.Dense, keras.layers.Dense, keras.Model]:

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

    # Log vars layer
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


def stacked_decoder(config: Dict) -> keras.Model:

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
