import numpy as np

from lib.evaluation.utils import (
    normalize_cartesian,
    normalize_polar,
    polar_to_cartesian,
)


def mse(true_factors: np.array, apx_factors: np.array) -> np.array:
    """Computes the MSE between the true factors of variation and the learned factors
    of variation, across all samples.

    Parameters
    ----------
    true_factors : np.array, shape=(n, 2)
        True factors of variation [x, y].
    apx_factors : np.array, shape=(n, 2)
        Learned factors of variation [x', y'].

    Returns
    -------
    np.array
        MSE, array of shape=()
    """

    return np.mean(np.sum((true_factors - apx_factors) ** 2, axis=1), axis=0)


def _evaluate_inversions(
    true_factors: np.array,
    representations: np.array,
    x_angle_idx: int,
    y_dist_idx: int,
    metric: str,
) -> np.array:
    """Given two specific channels of the latent space, one corresponding to X position
    (or angle) and another to Y position (or distance), the function returns the
    lowest MSE of the true (x, y) positions to the approximate ones, out of all possible
    inversions of the channels: left-to-right and top-to-bottom. If metric=`polar`, the
    representations are assumed to aproximate (angle, distance), and are first
    normalized and then converted to cartesian approximations. If metric=`cartesian`,
    only the normalization takes place.

    Parameters
    ----------
    true_factors : np.array, shape=(n, 2)
        True factors of variation, either [x, y] or [angle, distance].
    representations : np.array, shape=(n, latent_dim)
        Latent channel representations.
    x_angle_idx: int,
        Latent dimension's index to be treated as X position or angle.
    y_dist_idx: int,
        Latent dimension's index to be treated as X position or angle.
    metric: str,
        Which metric to use, polar or cartesian.


    Returns
    -------
    np.array, shape=()
        Best MSE out of all 4 possible inversions.
    """

    x_angle_apx = representations[:, x_angle_idx]
    y_dist_apx = representations[:, y_dist_idx]

    scores = []
    for multipliers in [[1, 1], [-1, 1], [1, -1], [-1, -1]]:
        if metric == "cartesian":
            x_apx, y_apx = normalize_cartesian(
                multipliers[0] * x_angle_apx,
                multipliers[1] * y_dist_apx,
            )
        else:
            angle_apx, dist_apx = normalize_polar(
                multipliers[0] * x_angle_apx,
                multipliers[1] * y_dist_apx,
            )
            x_apx, y_apx = polar_to_cartesian(angle_apx, dist_apx)

        apx_factors = np.array([x_apx, y_apx]).T
        scores.append(mse(true_factors, apx_factors))

    return min(scores)


def compute_metric(
    true_factors: np.array,
    representations: np.array,
    metric: str,
) -> np.array:
    """Computes the evaluation metric, trying all possible assignments of the latent
    channels

    Parameters
    ----------
    true_factors : np.array, shape=(n, 2)
        True factors of variation, either [x, y] or [angle, distance].
    representations : np.array, shape=(n, latent_dim)
        Latent channel representations.
    metric: str,
        Which metric to use, polar or cartesian.

    Returns
    -------
    np.array, shape=()
        Best MSE out of all possible assignments and inversions.
    """

    if metric not in ["cartesian", "polar"]:
        raise ValueError(f"{metric} metric unknown. Choose 'cartesian' or 'polar'")

    late_dim = representations.shape[1]

    scores = []
    for i in range(late_dim):
        for j in range(late_dim):
            if i != j:
                score = _evaluate_inversions(
                    true_factors=true_factors,
                    representations=representations,
                    x_angle_idx=i,
                    y_dist_idx=j,
                    metric=metric,
                )
                scores.append(score)

    return min(scores)
