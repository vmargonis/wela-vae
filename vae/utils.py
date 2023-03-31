import tensorflow as tf
from keras.optimizers import Adagrad, RMSprop, Adam, SGD
from vae.losses import gaussian_loss, bernoulli_loss
from typing import Dict


def get_reconstruction_loss(
    input_vec: tf.Tensor,
    output_vec: tf.Tensor,
    config: Dict,
) -> tf.Tensor:
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
        raise ValueError(
            f"Unkown output distribution {config['output_dist']}."
            "Choose either 'gaussian' or 'bernoulli'."
        )

    return reconstruction_loss


def add_optimizer(config: Dict):
    """Optimizer selector."""

    learning_rate = config["optimizer"]["learning_rate"]

    if config["optimizer"]["type"] == "Adagrad":
        optimizer = Adagrad(learning_rate=learning_rate)

    elif config["optimizer"]["type"] == "Adam":
        optimizer = Adam(learning_rate=learning_rate)

    elif config["optimizer"]["type"] == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)

    elif config["optimizer"]["type"] == "SGD":
        optimizer = SGD(learning_rate=learning_rate)

    else:
        raise ValueError(
            f"Unkown optimizer {config['optimizer']}."
            "Choose from: 'Adagrad', 'Adam', 'RMSprop', 'SGD'."
        )

    return optimizer
