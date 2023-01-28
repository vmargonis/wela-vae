import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from lib.evaluation.qualitative import make_qualitative_evaluation_figure
from lib.models.compile import compile_optimizer, compile_vae
from lib.utils import load_data, make_result_directories, parse_train_config


def train(vae_type: str, is_wela: bool = False) -> None:
    """
    Training function.
    """
    qa_dir, hist_dir = make_result_directories(vae_type, is_wela)
    train_config, vae_config = parse_train_config()

    blobs, angle_labels, dist_labels = load_data(vae_config)
    data = [blobs, angle_labels, dist_labels] if is_wela else blobs

    optimizer = compile_optimizer(train_config)
    model = compile_vae(vae_type, is_wela, vae_config)
    model.vae.compile(optimizer=optimizer)

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

    try:
        history = model.vae.fit(
            x=data,
            y=None,
            epochs=train_config["epochs"],
            batch_size=train_config["batch_size"],
            shuffle=True,
            callbacks=[early_stop, reduce_lr],
        )

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
    return


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

    train(vae_type=args.type, is_wela=args.wela)
    return


if __name__ == "__main__":
    # fix random seeds for reproducibility
    SEED = 12
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    main()
