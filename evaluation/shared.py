import numpy as np


def mse(true_factors, apx_factors):
    
    """
    true factors: N x 2 true_factors[i,:] = [x, y]
    apx factors: N x 2 apx_factors[i,:] = [x_tilde, y_tilde]

    Returns: mean l2 distance accross N
    """
    
    diff = true_factors - apx_factors
    return np.mean(np.sum(diff ** 2, axis=1), axis=0)


def polar_to_cartesian(angle_vec, distance_vec):
    
    x_tilde = distance_vec * np.cos(angle_vec)
    y_tilde = distance_vec * np.sin(angle_vec)
    
    return x_tilde, y_tilde


def normalize_polar(gaussian_angle, gaussian_distance):
    
    """
    gaussian_angle: Nx1 vec to be treated as angle, ranges in [ga_min, ga_max]
    gaussian_distance: Nx1 vec to be treated as distance, in [gd_min, gd_max]
    """
    
    # map angles from [min, max] to [0, pi/2] linearly
    ga_max = np.max(gaussian_angle)
    ga_min = np.min(gaussian_angle)
    angle_vec = (gaussian_angle - ga_min) * ((np.pi / 2) / (ga_max-ga_min))

    # map distances from [min, max] to [0, max_distance] linearly
    max_distance = np.sqrt(64.0**2 + 64.0**2)
    gd_max = np.max(gaussian_distance)
    gd_min = np.min(gaussian_distance)
    dist_vec = (gaussian_distance - gd_min) * (max_distance / (gd_max-gd_min))

    return angle_vec, dist_vec


def normalize_cartesian(gaussian_x, gaussian_y):
    
    """
    gaussian_x: Nx1 vec to be treated as x, ranges in [gx_min, gx_max]
    gaussian_y: Nx1 vec to be treated as y, ranges in [gy_min, gy_max]
    """
    
    # map x and y from [min, max] to [0, 64] linearly
    gx_max = np.max(gaussian_x)
    gx_min = np.min(gaussian_x)
    x_vec = (gaussian_x - gx_min) * (64.0 / (gx_max - gx_min))

    gy_max = np.max(gaussian_y)
    gy_min = np.min(gaussian_y)
    y_vec = (gaussian_y - gy_min) * (64.0 / (gy_max - gy_min))

    return x_vec, y_vec
