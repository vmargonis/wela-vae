from typing import Dict

from tensorflow.keras.optimizers import SGD, Adagrad, Adam, RMSprop

from lib.models.vae import DIPVAE, TCVAE, BetaVAE
from lib.models.welavae import WeLaBetaVAE, WeLaDIPVAE, WeLaTCVAE


def compile_optimizer(config: Dict):
    """
    Selects and compiles optimizer with learning rate.
    """
    optimizers = {"Adagrad": Adagrad, "Adam": Adam, "RMSprop": RMSprop, "SGD": SGD}
    learning_rate = config["optimizer"]["learning_rate"]
    optimizer = config["optimizer"]["type"]

    if optimizer not in optimizers.keys():
        raise ValueError(f"Unkown optimizer {optimizer}!")

    optimizer_class = optimizers[optimizer]
    return optimizer_class(learning_rate=learning_rate)


def compile_vae(vae_type: str, is_wela: bool, config: Dict):
    """
    VAE type selection and initialization.
    """
    if vae_type not in ("betavae", "tcvae", "dipvae"):
        raise ValueError(f"Unkown vae type {vae_type}!")

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
