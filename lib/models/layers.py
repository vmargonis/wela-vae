from typing import Dict, Tuple

from tensorflow import Tensor
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import Dense, Input, Lambda, concatenate
from tensorflow.keras.models import Model


def _reparameterization_trick(args: Tuple[Tensor, Tensor]) -> Tensor:
    """
    Reparameterization trick:  z = mu + eps*sigma, eps~N(0,1).

    :param args: Tuple (mean, log_var), tensors of shape (batch_size, latent_dim).
    :return: Gaussian samples, z = mu + eps*sigma, eps~N(0,1),
        tensor of shape (batch_size, latent_dim).
    """
    mean, log_var = args
    white_noise = K.random_normal(shape=(K.shape(mean)[0], K.shape(mean)[1]))

    return mean + K.exp(0.5 * log_var) * white_noise


def reparameterize(config: Dict) -> Model:
    """
    Reparameterization layer.
    """
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
    """
    Dense encoder layers.
    """
    w_seed = config["weight_seed"]
    input_vector = Input(shape=(config["initial_dim"],))
    num_layers = len(config["encoder"]["units"])

    hidden = Dense(
        units=config["encoder"]["units"][0],
        activation=config["encoder"]["activation"][0],
        kernel_initializer=GlorotUniform(seed=w_seed),
    )(input_vector)

    for i in range(1, num_layers):
        hidden = Dense(
            units=config["encoder"]["units"][i],
            activation=config["encoder"]["activation"][i],
            kernel_initializer=GlorotUniform(seed=w_seed),
        )(hidden)

    # Latent vector - representations
    mean = Dense(
        units=config["latent_dim"],
        activation=config["encoder"]["output_activation"],
        kernel_initializer=GlorotUniform(seed=w_seed),
        name="encoder_mean",
    )(hidden)

    log_var = Dense(
        units=config["latent_dim"],
        activation=config["encoder"]["output_activation"],
        kernel_initializer=GlorotUniform(seed=w_seed),
        name="encoder_log_sigma_squared",
    )(hidden)

    encoder = Model(inputs=input_vector, outputs=[mean, log_var], name="encoder")

    return input_vector, mean, log_var, encoder


def dense_encoder_with_labels(
    config: Dict,
) -> Tuple[Input, Input, Input, Dense, Dense, Model]:
    """
    Dense encoder layers with two extra inputs-labels. Used by WeLa-VAE models.
    """
    w_seed = config["weight_seed"]

    input_vector = Input(shape=(config["initial_dim"],))
    label_1 = Input(shape=(config["label_dim"],))
    label_2 = Input(shape=(config["label_dim"],))
    combined_input = concatenate([input_vector, label_1, label_2])

    num_layers = len(config["encoder"]["units"])

    hidden = Dense(
        units=config["encoder"]["units"][0],
        activation=config["encoder"]["activation"][0],
        kernel_initializer=GlorotUniform(seed=w_seed),
    )(combined_input)

    for i in range(1, num_layers):
        hidden = Dense(
            units=config["encoder"]["units"][i],
            activation=config["encoder"]["activation"][i],
            kernel_initializer=GlorotUniform(seed=w_seed),
        )(hidden)

    # Latent vector - lib representations
    mean = Dense(
        units=config["latent_dim"],
        activation=config["encoder"]["output_activation"],
        kernel_initializer=GlorotUniform(seed=w_seed),
        name="encoder_mean",
    )(hidden)

    log_var = Dense(
        units=config["latent_dim"],
        activation=config["encoder"]["output_activation"],
        kernel_initializer=GlorotUniform(seed=w_seed),
        name="encoder_log_sigma_squared",
    )(hidden)

    encoder = Model(
        inputs=[input_vector, label_1, label_2],
        outputs=[mean, log_var],
        name="encoder",
    )

    return input_vector, label_1, label_2, mean, log_var, encoder


def dense_decoder(config: Dict) -> Model:
    """
    Dense decoder layers.
    """
    w_seed = config["weight_seed"]
    z_sample = Input(shape=(config["latent_dim"],))
    num_layers = len(config["decoder"]["units"])

    hidden = Dense(
        units=config["decoder"]["units"][0],
        activation=config["decoder"]["activation"][0],
        kernel_initializer=GlorotUniform(seed=w_seed),
    )(z_sample)

    for i in range(1, num_layers):
        hidden = Dense(
            units=config["decoder"]["units"][i],
            activation=config["decoder"]["activation"][i],
            kernel_initializer=GlorotUniform(seed=w_seed),
        )(hidden)

    output = Dense(
        units=config["initial_dim"],
        activation=config["decoder"]["output_activation"],
        kernel_initializer=GlorotUniform(seed=config["weight_seed"]),
        name="output_vector",
    )(hidden)

    decoder = Model(inputs=z_sample, outputs=output, name="decoder")
    return decoder


def dense_decoder_with_labels(config: Dict) -> Model:
    """
    Dense decoder layers with two extra outputs-labels. Used by WeLa-VAE models.
    """
    w_seed = config["weight_seed"]
    z_sample = Input(shape=(config["latent_dim"],))
    num_layers = len(config["decoder"]["units"])

    hidden = Dense(
        units=config["decoder"]["units"][0],
        activation=config["decoder"]["activation"][0],
        kernel_initializer=GlorotUniform(seed=w_seed),
    )(z_sample)

    for i in range(1, num_layers):
        hidden = Dense(
            units=config["decoder"]["units"][i],
            activation=config["decoder"]["activation"][i],
            kernel_initializer=GlorotUniform(seed=w_seed),
        )(hidden)

    output_vector = Dense(
        units=config["initial_dim"],
        activation=config["decoder"]["output_activation"],
        kernel_initializer=GlorotUniform(seed=w_seed),
        name="output_vector",
    )(hidden)

    label_1_out = Dense(
        units=config["label_dim"],
        activation="softmax",
        kernel_initializer=GlorotUniform(seed=w_seed),
        name="label_1_output",
    )(hidden)

    label_2_out = Dense(
        units=config["label_dim"],
        activation="softmax",
        kernel_initializer=GlorotUniform(seed=w_seed),
        name="label_2_output",
    )(hidden)

    outputs = [output_vector, label_1_out, label_2_out]

    decoder = Model(inputs=z_sample, outputs=outputs, name="decoder")
    return decoder
