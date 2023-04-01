import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adagrad, RMSprop, Adam, SGD
from lib.models.losses import (
    gaussian_loss,
    bernoulli_loss,
    compute_covariance_mean,
    compute_covariance_z,
    dip_vae_regularizer,
)
from typing import Dict


def get_reconstruction_loss(
    input_vec: tf.Tensor,
    output_vec: tf.Tensor,
    config: Dict,
) -> tf.Tensor:
    """Recostruction loss getter based on I/O and config."""

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


def get_dip_vae_regularizer(
    mean: tf.Tensor,
    log_var: tf.Tensor,
    params: Dict,
) -> tf.Tensor:
    """Computes DIPVAE regularizer based on DIPVAE type."""

    if params["dip_vae_type"] == "i":
        cov_matrix = compute_covariance_mean(mean)
    elif params["dip_vae_type"] == "ii":
        cov_matrix = compute_covariance_z(mean, log_var)
    else:
        raise ValueError(
            f"Unknown DIPVAE type {params['dip_vae_type']}."
            "Choose either 'i' or 'ii'."
        )

    dip_regularizer = dip_vae_regularizer(
        cov_matrix,
        params["lambda_off_diag"],
        params["lambda_diag"],
    )

    return dip_regularizer


def get_label_loss(
    input_label_1: tf.Tensor,
    output_label_1: tf.Tensor,
    input_label_2: tf.Tensor,
    output_label_2: tf.Tensor,
) -> tf.Tensor:
    """Compute reconstruction loss for the labels."""

    label_1_loss = K.categorical_crossentropy(input_label_1, output_label_1)
    label_2_loss = K.categorical_crossentropy(input_label_2, output_label_2)

    return K.mean(label_1_loss + label_2_loss, axis=-1)


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
