import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate
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
    white_noise = K.random_normal(shape=(K.shape(mean)[0], K.shape(mean)[1]))

    return mean + K.exp(0.5 * log_var) * white_noise


def reparameterize(config: Dict) -> Model:
    """Reparameterization layer."""

    mean = Input(shape=(config["latent_dim"],))
    log_var = Input(shape=(config["latent_dim"],))

    z_sample = Lambda(
        _reparameterization_trick,
        output_shape=(config["latent_dim"],),
        name="z_sample",
    )((mean, log_var))

    reparameterizer = Model(
        inputs=[mean, log_var], outputs=z_sample, name="reparameterizer"
    )

    return reparameterizer


def dense_encoder(
    config: Dict,
) -> Tuple[Input, Dense, Dense, Model]:
    """Dense encoder layers."""

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


def dense_encoder_with_labels(
    config: Dict,
) -> Tuple[Input, Input, Input, Dense, Dense, Model]:
    """Dense encoder layers with two extra inputs-labels. Used by WeLa-VAE models."""

    # glorot weight initializer
    w_init = GlorotUniform(seed=config["weight_seed"])

    # input placeholders
    input_vector = Input(shape=(config["initial_dim"],))
    label_1 = Input(shape=(config["label_dim"],))
    label_2 = Input(shape=(config["label_dim"],))
    # concatenate
    combined_input = concatenate([input_vector, label_1, label_2])

    # hidden layer(s)
    num_layers = len(config["encoder"]["units"])

    # First hidden is connected to input
    hidden = Dense(
        units=config["encoder"]["units"][0],
        activation=config["encoder"]["activation"][0],
        kernel_initializer=w_init,
    )(combined_input)

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
        inputs=[input_vector, label_1, label_2],
        outputs=[mean, log_var],
        name="encoder",
    )

    return input_vector, label_1, label_2, mean, log_var, encoder


def dense_decoder(config: Dict) -> Model:
    """Dense decoder layers. If config["label_dim"] is not None, which indicates
    a WeLa-VAE architecture, the decoder also outputs two extra layers, one for each
    label vector.
    """

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

    if config["label_dim"]:
        label_out_1 = Dense(
            units=config["label_dim"],
            activation="softmax",
            kernel_initializer=w_init,
            name="label_1_output",
        )(hidden)

        label_out_2 = Dense(
            units=config["label_dim"],
            activation="softmax",
            kernel_initializer=w_init,
            name="label_2_output",
        )(hidden)

        outputs = [output_vector, label_out_1, label_out_2]

    else:
        outputs = output_vector

    decoder = Model(
        inputs=z_sample,
        outputs=outputs,
        name="decoder",
    )

    return decoder
