import argparse
import os
import random
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml
from keras.optimizers import SGD, Adagrad, Adam, RMSprop

from lib.config import Config
from lib.evaluation.qualitative import make_qualitative_evaluation_figure
from lib.models.vae import DIPVAE, TCVAE, BetaVAE
from lib.models.welavae import WeLaBetaVAE, WeLaDIPVAE, WeLaTCVAE


def _load_data() -> np.array:
    # load blob dataset
    dataset_zip = np.load(f"{Config.blobs_path}/data/blobs64.npz")
    blobs = dataset_zip["arr_0"]
    print("blobs original shape:", blobs.shape)

    # Flatten blobs: training set
    data = blobs.reshape((len(blobs), np.prod(blobs.shape[1:])))
    print("blobs flatten shape:", data.shape)

    return data


def _load_configs() -> Tuple[Dict, Dict]:
    with open("lib/train_config.yaml", "r") as file:
        conf_dict = yaml.safe_load(file)

    train_config = conf_dict["train"]
    vae_config = conf_dict["vae"]

    return train_config, vae_config


def _add_optimizer(config: Dict):
    """Optimizer selector."""

    learning_rate = config["optimizer"]["learning_rate"]
    opt_type = config["optimizer"]["type"]

    if opt_type not in ["Adagrad", "Adam", "RMSprop", "SGD"]:
        raise ValueError(
            f"Unkown optimizer {config['optimizer']}."
            "Choose from: 'Adagrad', 'Adam', 'RMSprop', 'SGD'."
        )

    if opt_type == "Adagrad":
        optimizer = Adagrad(learning_rate=learning_rate)
    elif opt_type == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif opt_type == "RMSprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate)

    return optimizer


def train(
    data: np.array,
    vae_type: str,
    is_wela: bool = False,
) -> None:
    if vae_type not in ["betavae", "tcvae", "dipvae"]:
        raise ValueError(
            f"Unkown vae type {vae_type}. Choose from: 'betavae', 'tcvae', 'dipvae'"
        )

    train_config, vae_config = _load_configs()

    if is_wela:
        if vae_type == "betavae":
            model = WeLaBetaVAE(vae_config)
        elif vae_type == "tcvae":
            model = WeLaTCVAE(vae_config)
        else:
            model = WeLaDIPVAE(vae_config)

    else:
        if vae_type == "betavae":
            model = BetaVAE(vae_config)
        elif vae_type == "tcvae":
            model = TCVAE(vae_config)
        else:
            model = DIPVAE(vae_config)

    model.vae.compile(optimizer=_add_optimizer(train_config))

    # TRAINING
    try:
        history = model.vae.fit(
            data,
            None,
            epochs=train_config["epochs"],
            batch_size=train_config["batch_size"],
            shuffle=True,
        )

        sns.set_style("white")
        plt.figure(figsize=(10, 7))
        plt.plot(history.history["loss"][1:])
        plt.title(model.str_repr)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.savefig(f"{Config.results_path}/history/hist_{model.str_repr}")

    except KeyboardInterrupt:
        print("Training interrupted.")

    finally:
        means, log_vars = model.encoder.predict(data, verbose=0)
        print("means shape:", means.shape)
        print("log_vars shape:", log_vars.shape)

        prefix = "wela_" if is_wela else ""
        fig_dir = f"{Config.results_path}/eval/{prefix}{vae_type}"
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        make_qualitative_evaluation_figure(
            dataset=data,
            mean_vec=means,
            log_var_vec=log_vars,
            decoder=model.decoder,
            is_wela=is_wela,
            figure_name=model.str_repr,
            output_directory=fig_dir,
        )

        print("Saved qualitative evaluation figure.")

        # SAVE REPRESENTATIONS AND LOG VARS
        # means_path = f"{Config.results_path}/vectors/means_{model.str_repr}.npz"
        # log_var_path = f"{Config.results_path}/vectors/log_vars_{model.str_repr}.npz"
        # np.savez_compressed(means_path, means)
        # np.savez_compressed(log_var_path, log_vars)
        # print("Saved representations and log_vars.")

        # SAVE MODELS
        # enc_path = f"{Config.results_path}/models/encoder_{model.str_repr}.h5"
        # dec_path = f"{Config.results_path}/models/decoder_{model.str_repr}.h5"
        # model.encoder.save(enc_path)
        # model.decoder.save(dec_path)
        # print("Saved models to disk")

    return None


def main():
    # fix random seed for reproducibility
    seed = 12
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # make necessary directories
    for sub_dir in ["eval", "models", "vectors", "history"]:
        if not os.path.exists(f"{Config.results_path}/{sub_dir}"):
            os.mkdir(f"{Config.results_path}/{sub_dir}")

    # load data
    x_train = _load_data()

    # train model
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        help="The VAE type to use: betavae, tcvae or dipvae",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--wela",
        type=bool,
        help="A boolean flag indicating whether to use the WeLa variant",
        default=False,
    )
    args = parser.parse_args()

    train(data=x_train, vae_type=args.type, is_wela=args.wela)


if __name__ == "__main__":
    main()
