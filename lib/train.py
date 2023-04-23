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


def _make_directories(vae_type: str, is_wela: bool) -> Tuple[str, str]:
    """Makes results directories for the model."""

    # ToDo: can qa_dir and hist_dir be make directly?
    for sub_dir in ["eval", "history"]:
        par_dir = f"{Config.results_path}/{sub_dir}"
        os.mkdir(par_dir) if not os.path.exists(par_dir) else None

    prefix = "wela_" if is_wela else ""
    qa_dir = f"{Config.results_path}/eval/{prefix}{vae_type}"
    hist_dir = f"{Config.results_path}/history/{prefix}{vae_type}"
    for directory in [qa_dir, hist_dir]:
        os.mkdir(directory) if not os.path.exists(directory) else None

    return qa_dir, hist_dir


def _load_data() -> np.array:
    """Load Blobs dataset."""

    # ToDo: load labels as well
    dataset_zip = np.load(f"{Config.blobs_path}/data/blobs64.npz")
    blobs = dataset_zip["arr_0"]

    # Flatten for training
    return blobs.reshape((len(blobs), np.prod(blobs.shape[1:])))


def _load_configs() -> Tuple[Dict, Dict]:
    """Load train and vae configs from yaml file."""

    with open("lib/train_config.yaml", "r") as file:
        conf_dict = yaml.safe_load(file)

    return conf_dict["train"], conf_dict["vae"]


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


def _train(
    data: np.array,
    vae_type: str,
    is_wela: bool = False,
) -> None:
    """Training function."""

    if vae_type not in ["betavae", "tcvae", "dipvae"]:
        raise ValueError(
            f"Unkown vae type {vae_type}. Choose from: 'betavae', 'tcvae', 'dipvae'"
        )

    qa_dir, hist_dir = _make_directories(vae_type, is_wela)
    train_config, vae_config = _load_configs()

    if is_wela:
        model = (
            WeLaBetaVAE(vae_config)
            if vae_type == "betavae"
            else WeLaTCVAE(vae_config)
            if vae_type == "tcvae"
            else WeLaDIPVAE(vae_config)
        )
    else:
        model = (
            BetaVAE(vae_config)
            if vae_type == "betavae"
            else TCVAE(vae_config)
            if vae_type == "tcvae"
            else DIPVAE(vae_config)
        )

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
        # ToDo: when is_wela=True data must contain three vectors in fit
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
        print("Training interrupted.")

    finally:
        # ToDo: when is_wela=True data must contain three vectors in predict
        means, log_vars = model.encoder.predict(data, verbose=0)

        make_qualitative_evaluation_figure(
            dataset=data,
            mean_vec=means,
            log_var_vec=log_vars,
            decoder=model.decoder,
            is_wela=is_wela,
            figure_name=model.str_repr,
            output_directory=qa_dir,
        )

        print("Saved qualitative evaluation figure.")

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        help="VAE type selection: betavae, tcvae or dipvae.",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--wela",
        type=bool,
        help="Boolean flag, whether to use the WeLa variant. Default is False.",
        default=False,
    )
    args = parser.parse_args()

    # fix random seeds for reproducibility
    seed = 12
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    x_train = _load_data()
    _train(data=x_train, vae_type=args.type, is_wela=args.wela)

    return None


if __name__ == "__main__":
    main()
