import argparse
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
from lib.evaluation.qualitative import make_qualitative_evaluation_figure
from lib.models.vae import DIPVAE, TCVAE, BetaVAE
from lib.models.welavae import WeLaBetaVAE, WeLaDIPVAE, WeLaTCVAE


def _load_configs() -> Tuple[Dict, Dict]:
    """Load train and vae configs from yaml file."""

    with open("lib/train_config.yaml", "r") as file:
        conf_dict = yaml.safe_load(file)

    return conf_dict["train"], conf_dict["vae"]


def _make_directories(vae_type: str, is_wela: bool) -> Tuple[str, str]:
    """Makes results directories for the model."""

    for sub_dir in ["eval", "history"]:
        par_dir = f"{Config.results_path}/{sub_dir}"
        os.mkdir(par_dir) if not os.path.exists(par_dir) else None

    prefix = "wela_" if is_wela else ""
    qa_dir = f"{Config.results_path}/eval/{prefix}{vae_type}"
    hist_dir = f"{Config.results_path}/history/{prefix}{vae_type}"
    for directory in [qa_dir, hist_dir]:
        os.mkdir(directory) if not os.path.exists(directory) else None

    return qa_dir, hist_dir


def _load_data(config: Dict) -> Tuple[np.array, np.array, np.array]:
    """Load Blobs dataset and angle/distance labels, with one-hot dimension
    specified in the yaml config.
    """

    dataset_zip = np.load(f"{Config.blobs_path}/data/blobs64.npz")
    blobs = dataset_zip["arr_0"]
    blobs = blobs.reshape((len(blobs), np.prod(blobs.shape[1:])))

    # label one-hot dimension
    ldim = config["label_dim"]

    # load angle labels
    label_zip = np.load(f"{Config.blobs_path}/data/blobs64_anglelabels_res{ldim}.npz")
    angle_labels = label_zip["arr_0"]

    # load distance labels
    label_zip = np.load(f"{Config.blobs_path}/data/blobs64_distlabels_res{ldim}.npz")
    dist_labels = label_zip["arr_0"]

    # Flatten for training
    return blobs, angle_labels, dist_labels


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


def _train(
    vae_type: str,
    is_wela: bool = False,
) -> None:
    """Training function."""

    qa_dir, hist_dir = _make_directories(vae_type, is_wela)
    train_config, vae_config = _load_configs()

    blobs, angle_labels, dist_labels = _load_data(vae_config)
    data = [blobs, angle_labels, dist_labels] if is_wela else blobs

    model = _select_init_vae(vae_type, is_wela, vae_config)
    model.vae.compile(optimizer=_add_optimizer(train_config))

    # TRAINING
    try:
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
            patience=3,
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
        )

        sns.set_style("white")
        plt.figure(figsize=(10, 7))
        plt.plot(history.history["loss"][1:])
        plt.title(model.str_repr)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(f"{hist_dir}/hist_{model.str_repr}.png")

    except KeyboardInterrupt:
        print("Training interrupted. Proceeding to evaluation.")

    finally:
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
        print("Saved qualitative evaluation figure.")

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        help="VAE type selection.",
        choices=["betavae", "tcvae", "dipvae"],
        required=True,
    )
    parser.add_argument(
        "-w",
        "--wela",
        help="If provided, WeLa variant will be used. Defaults to False.",
        action="store_true",
    )
    args = parser.parse_args()

    # fix random seeds for reproducibility
    seed = 12
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    _train(vae_type=args.type, is_wela=args.wela)

    return None


if __name__ == "__main__":
    main()
