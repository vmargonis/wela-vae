from typing import Dict

from tensorflow import Tensor
from tensorflow.keras import backend as K

from lib.models.losses import (
    bernoulli_loss,
    compute_covariance_mean,
    compute_covariance_z,
    dip_vae_regularizer,
    gaussian_loss,
)


def get_reconstruction_loss(
    input_vec: Tensor,
    output_vec: Tensor,
    config: Dict,
) -> Tensor:
    """
    Recostruction loss getter based on I/O and config.
    """
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
    mean: Tensor,
    log_var: Tensor,
    config: Dict,
) -> Tensor:
    """
    Computes DIPVAE regularizer based on DIPVAE type.
    """
    if config["dip_vae_type"] == "i":
        cov_matrix = compute_covariance_mean(mean)
    elif config["dip_vae_type"] == "ii":
        cov_matrix = compute_covariance_z(mean, log_var)
    else:
        raise ValueError(
            f"Unknown DIPVAE type {config['dip_vae_type']}."
            "Choose either 'i' or 'ii'."
        )

    dip_regularizer = dip_vae_regularizer(
        cov_matrix,
        config["lambda_off_diag"],
        config["lambda_diag"],
    )

    return dip_regularizer


def get_label_loss(
    input_label_1: Tensor,
    output_label_1: Tensor,
    input_label_2: Tensor,
    output_label_2: Tensor,
) -> Tensor:
    """
    Computes reconstruction loss for the labels.
    """
    label_1_loss = K.categorical_crossentropy(input_label_1, output_label_1)
    label_2_loss = K.categorical_crossentropy(input_label_2, output_label_2)

    return K.mean(label_1_loss + label_2_loss, axis=-1)
