import numpy as np
from evaluation.shared import mse, normalize_cartesian


def evaluate_cartesian(true_factors, representations, x_idx, y_idx):
    
    """
    true factors: N x 2 true_factors[i,:] = [x,y]
    representations: N x late_dim
    x_idx: latent dimension index to be treated as x
    y_idx: latent dimension index to be treated as y

    Returns: evaluation score after trying 4 possible mappings
    """

    gaussian_x = representations[:, x_idx]
    gaussian_y = representations[:, y_idx]

    scores = np.zeros(4, dtype=float)

    # no inversion:
    x_apx, y_apx = normalize_cartesian(gaussian_x, gaussian_y)
    apx_factors = np.array([x_apx, y_apx]).T
    scores[0] = mse(true_factors, apx_factors)

    # invert x:
    x_apx, y_apx = normalize_cartesian(-1*gaussian_x, gaussian_y)
    apx_factors = np.array([x_apx, y_apx]).T
    scores[1] = mse(true_factors, apx_factors)

    # invert y:
    x_apx, y_apx = normalize_cartesian(gaussian_x, -1*gaussian_y)
    apx_factors = np.array([x_apx, y_apx]).T
    scores[2] = mse(true_factors, apx_factors)

    # invert x and y:
    x_apx, y_apx = normalize_cartesian(-1*gaussian_x, -1*gaussian_y)
    apx_factors = np.array([x_apx, y_apx]).T
    scores[3] = mse(true_factors, apx_factors)

    # print(scores, np.min(scores))

    return np.min(scores)


def compute_metric_cartesian(true_factors, representations):
    
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
                x_idx = i
                y_idx = j
                score = evaluate_cartesian(true_factors, 
                                           representations,
                                           x_idx,
                                           y_idx)
                scores.append(score)

    return min(scores)
