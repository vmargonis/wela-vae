import argparse
import os

os.environ["PYTHONHASHSEED"] = str(12)  # recommended to be set before keras/tf import

from lib.train import training


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
        "-s",
        "--seed",
        help="Random seed for reproducibility.",
        required=True,
        type=int,
    )
    parser.add_argument(
        "-w",
        "--wela",
        help="If provided, WeLa variant will be used. Defaults to False.",
        action="store_true",
    )
    parser.add_argument(
        "--no-verbose",
        help="If provided, keras.fit() verbosity is set to silent. Default is False.",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--loss",
        help="If provided, Loss history figure will be produced. Defaults to False.",
        action="store_true",
    )
    args = parser.parse_args()

    training(
        vae_type=args.type,
        weight_seed=args.seed,
        is_wela=args.wela,
        no_verbose=args.no_verbose,
        loss_history=args.loss,
    )

    return None


if __name__ == "__main__":
    main()
