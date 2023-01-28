import os
from typing import Dict, Tuple

import numpy as np
import yaml

from lib.config import Config


def parse_train_config() -> Tuple[Dict, Dict]:
    """
    Load train and vae configs from yaml file.
    """
    with open("lib/train_config.yaml", "r") as file:
        conf_dict = yaml.safe_load(file)

    return conf_dict["train"], conf_dict["vae"]


def make_result_directories(vae_type: str, is_wela: bool) -> Tuple[str, str]:
    """
    Makes results directories for the model.
    """
    for sub_dir in ["eval", "history"]:
        par_dir = f"{Config.results_path}/{sub_dir}"
        os.mkdir(par_dir) if not os.path.exists(par_dir) else None

    prefix = "wela_" if is_wela else ""
    qa_dir = f"{Config.results_path}/eval/{prefix}{vae_type}"
    hist_dir = f"{Config.results_path}/history/{prefix}{vae_type}"
    for directory in [qa_dir, hist_dir]:
        os.mkdir(directory) if not os.path.exists(directory) else None

    return qa_dir, hist_dir


def load_data(config: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load Blobs dataset and angle/distance labels, with one-hot dimension
    specified in the yaml config.
    """
    blobs = np.load(f"{Config.blobs_path}/data/blobs64.npz")["arr_0"]
    blobs = blobs.reshape((len(blobs), np.prod(blobs.shape[1:])))

    # label one-hot dimension
    ldim_ = config["label_dim"]

    angle_labels = np.load(
        f"{Config.blobs_path}/data/blobs64_anglelabels_res{ldim_}.npz"
    )["arr_0"]

    distance_labels = np.load(
        f"{Config.blobs_path}/data/blobs64_distlabels_res{ldim_}.npz"
    )["arr_0"]

    return blobs, angle_labels, distance_labels
