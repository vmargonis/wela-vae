import numpy as np
from evaluation.shared import mse, polar_to_cartesian, normalize_polar


def evaluate_polar(true_factors, repres_clipped, angle_idx, dist_idx):
    
    """
    true factors: N x 2 true_factors[i,:] = [x,y]
    repres_clipped: N x late_dim, representations clipped to [-3, +3]
    angle_idx: latent dimension index to be treated as angle
    dist_idx: latent dimension index to be treated as distance

    Returns: evaluation score after trying 4 possible mappings
    """

    gaussian_angle = repres_clipped[:, angle_idx]
    gaussian_distance = repres_clipped[:, dist_idx]

    scores = np.zeros(4, dtype=float)
    # no inversion:
    angle_vec, dist_vec = normalize_polar(gaussian_angle, gaussian_distance)
    x_apx, y_apx = polar_to_cartesian(angle_vec, dist_vec)
    apx_factors = np.array([x_apx, y_apx]).T
    scores[0] = mse(true_factors, apx_factors)

    # invert angle
    angle_vec, dist_vec = normalize_polar(-1 * gaussian_angle,
                                          gaussian_distance)
    x_apx, y_apx = polar_to_cartesian(angle_vec, dist_vec)
    apx_factors = np.array([x_apx, y_apx]).T
    scores[1] = mse(true_factors, apx_factors)

    # invert distance
    angle_vec, dist_vec = normalize_polar(gaussian_angle,
                                          -1 * gaussian_distance)
    x_apx, y_apx = polar_to_cartesian(angle_vec, dist_vec)
    apx_factors = np.array([x_apx, y_apx]).T
    scores[2] = mse(true_factors, apx_factors)

    # invert both angle and distance
    angle_vec, dist_vec = normalize_polar(-1 * gaussian_angle,
                                          -1 * gaussian_distance)
    x_apx, y_apx = polar_to_cartesian(angle_vec, dist_vec)
    apx_factors = np.array([x_apx, y_apx]).T
    scores[3] = mse(true_factors, apx_factors)

    return np.min(scores)


def compute_metric_polar(true_factors, representations):
    
    """
    true factors: N x 2 true_factors[i,:] = [x,y]
    representations: N x late_dim

    Returns: Best MSE out of all assignments and inversions
    """

    late_dim = representations.shape[1]

    scores = []
    for i in range(late_dim):
        for j in range(late_dim):
            if i != j:
                angle_idx = i
                dist_idx = j
                score = evaluate_polar(true_factors, 
                                       representations,
                                       angle_idx, 
                                       dist_idx)
                scores.append(score)

    return min(scores)
