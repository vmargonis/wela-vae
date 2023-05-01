import os
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import SGD, Adagrad, Adam, RMSprop

from lib.config import Config
from lib.evaluation.metrics import compute_metric
from lib.evaluation.qualitative import make_qualitative_evaluation_figure
from lib.models.vae import DIPVAE, TCVAE, BetaVAE
from lib.models.welavae import WeLaBetaVAE, WeLaDIPVAE, WeLaTCVAE


def _reset_random_seeds() -> None:
    """Fixed random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(12)
    random.seed(12)
    np.random.seed(12)
    tf.random.set_seed(12)
    return None


def _load_configs() -> Tuple[Dict, Dict]:
    """Load train and vae configs from yaml file."""

    with open("lib/train_config.yaml", "r") as file:
        conf_dict = yaml.safe_load(file)

    return conf_dict["train"], conf_dict["vae"]


def _make_results_directory(
    vae_type: str,
    is_wela: bool,
    sub_path: str,
) -> str:
    """Makes qa or hist directory for the model."""
    assert sub_path in ["eval", "history"]

    par_dir = f"{Config.results_path}/{sub_path}"
    os.mkdir(par_dir) if not os.path.exists(par_dir) else None

    prefix = "wela_" if is_wela else ""
    sub_dir = f"{Config.results_path}/{sub_path}/{prefix}{vae_type}"
    os.mkdir(sub_dir) if not os.path.exists(sub_dir) else None

    return sub_dir


def _load_data(config: Dict) -> Tuple[np.array, np.array, np.array, np.array]:
    """Load Blobs dataset and angle/distance labels, with one-hot dimension
    specified in the yaml config.
    """

    dataset_zip = np.load(f"{Config.blobs_path}/data/blobs64.npz")
    blobs = dataset_zip["arr_0"]
    blobs = blobs.reshape((len(blobs), np.prod(blobs.shape[1:])))  # flatten

    ground_truth_zip = np.load(f"{Config.blobs_path}/data/blobs64_ground_truth.npz")
    true_factors = ground_truth_zip["arr_0"]

    # label one-hot dimension
    ldim = config["label_dim"]

    # load angle labels
    label_zip = np.load(f"{Config.blobs_path}/data/blobs64_anglelabels_res{ldim}.npz")
    angle_labels = label_zip["arr_0"]

    # load distance labels
    label_zip = np.load(f"{Config.blobs_path}/data/blobs64_distlabels_res{ldim}.npz")
    dist_labels = label_zip["arr_0"]

    return blobs, true_factors, angle_labels, dist_labels


def _add_optimizer(config: Dict):
    """Optimizer selector."""

    learning_rate = config["optimizer"]["learning_rate"]
    opt_type = config["optimizer"]["type"]

    if opt_type not in ["Adagrad", "Adam", "RMSprop", "SGD"]:
        raise ValueError(
            f"Unkown optimizer {config['optimizer']}."
            "Choose from: 'Adagrad', 'Adam', 'RMSprop', 'SGD'."
        )

    optimizer_dict = {"Adagrad": Adagrad, "Adam": Adam, "RMSprop": RMSprop, "SGD": SGD}
    optimizer_class = optimizer_dict.get(opt_type, SGD)
    return optimizer_class(learning_rate=learning_rate)


def _select_init_vae(vae_type: str, is_wela: bool, config: Dict):
    """VAE type selection and initialization."""

    if vae_type not in ["betavae", "tcvae", "dipvae"]:
        raise ValueError(
            f"Unkown vae type {vae_type}. Choose from: 'betavae', 'tcvae', 'dipvae'"
        )

    if is_wela:
        vae = (
            WeLaBetaVAE(config)
            if vae_type == "betavae"
            else WeLaTCVAE(config)
            if vae_type == "tcvae"
            else WeLaDIPVAE(config)
        )
    else:
        vae = (
            BetaVAE(config)
            if vae_type == "betavae"
            else TCVAE(config)
            if vae_type == "tcvae"
            else DIPVAE(config)
        )

    return vae


def training(
    vae_type: str,
    weight_seed: int,
    is_wela: bool = False,
    no_verbose: bool = False,
    loss_history: bool = False,
) -> None:
    """Training function."""

    _reset_random_seeds()
    qa_dir = _make_results_directory(vae_type, is_wela, "eval")
    train_config, vae_config = _load_configs()

    blobs, true_factors, angle_labels, dist_labels = _load_data(vae_config)
    data = [blobs, angle_labels, dist_labels] if is_wela else blobs

    vae_config["weight_seed"] = weight_seed
    model = _select_init_vae(vae_type, is_wela, vae_config)
    model.vae.compile(optimizer=_add_optimizer(train_config))

    early_stop = EarlyStopping(
        monitor="loss",
        patience=10,
        mode="min",
        restore_best_weights=True,
        start_from_epoch=20,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="loss",
        factor=0.1,
        patience=5,
        mode="min",
        cooldown=0,
        min_lr=train_config["optimizer"]["learning_rate"] * 0.1,
    )

    history = model.vae.fit(
        x=data,
        y=None,
        epochs=train_config["epochs"],
        batch_size=train_config["batch_size"],
        shuffle=True,
        callbacks=[early_stop, reduce_lr],
        verbose=(0 if no_verbose else "auto"),
    )

    if loss_history:
        hist_dir = _make_results_directory(vae_type, is_wela, "history")

        sns.set_style("white")
        plt.figure(figsize=(10, 7))
        plt.plot(history.history["loss"][1:])
        plt.title(model.str_repr)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(f"{hist_dir}/hist_{model.str_repr}.png")
        plt.close()

    means, log_vars = model.encoder.predict(data, verbose=0)

    make_qualitative_evaluation_figure(
        dataset=(data[0] if is_wela else data),
        mean_vec=means,
        log_var_vec=log_vars,
        decoder=model.decoder,
        is_wela=is_wela,
        figure_name=model.str_repr,
        output_directory=qa_dir,
    )
    print("\n Saved qualitative evaluation figure.")

    cartesian_mse = compute_metric(true_factors, means, metric="cartesian")
    polar_mse = compute_metric(true_factors, means, metric="polar")
    print(f"\n MSE: Cartesian={cartesian_mse:.2f}, Polar={polar_mse:.2f}")

    return None
