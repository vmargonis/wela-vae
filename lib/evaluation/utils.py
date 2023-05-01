from typing import Tuple

import numpy as np


def polar_to_cartesian(
    angle: np.array,
    distance: np.array,
) -> Tuple[np.array, np.array]:
    """Conversion from polar to cartesian coordinates."""

    x_pos = distance * np.cos(angle)
    y_pos = distance * np.sin(angle)
    return x_pos, y_pos


def normalize_polar(
    angle: np.array,
    distance: np.array,
) -> Tuple[np.array, np.array]:
    """Normalizes the learned polar representations. Maps angle to [0, pi/2] and
    distance to [0, sqrt(64^2 + 64^2)], linearly.

    Parameters
    ----------
    angle : np.array, 1d
        Latent channel vector corresponding to angle.
    distance : np.array, 1d
        Latent channel vector corresponding to distance.

    Returns
    -------
    Tuple of 1d numpy arrays,
        normalized angle and distance.
    """

    angle_max, angle_min = np.max(angle), np.min(angle)
    angle_norm = (angle - angle_min) * ((np.pi / 2) / (angle_max - angle_min))

    max_distance = np.sqrt(64.0**2 + 64.0**2)
    dist_max, dist_min = np.max(distance), np.min(distance)
    dist_norm = (distance - dist_min) * (max_distance / (dist_max - dist_min))

    return angle_norm, dist_norm


def normalize_cartesian(x_pos: np.array, y_pos: np.array) -> Tuple[np.array, np.array]:
    """Normalizes the learned cartesian representations. Maps both to [0, 64] linearly.

    Parameters
    ----------
    x_pos : np.array, 1d
        Latent channel vector corresponding to x-position.
    y_pos : np.array, 1d
        Latent channel vector corresponding to y-position.

    Returns
    -------
    Tuple of 1d numpy arrays,
        normalized x and y positions.
    """

    x_max, x_min = np.max(x_pos), np.min(x_pos)
    x_norm = (x_pos - x_min) * (64.0 / (x_max - x_min))

    y_max, y_min = np.max(y_pos), np.min(y_pos)
    y_norm = (y_pos - y_min) * (64.0 / (y_max - y_min))

    return x_norm, y_norm
