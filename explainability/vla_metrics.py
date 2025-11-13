'''
Metrics useful for evaluating the performance for VLA outputs.
These metrics will be used to compare unperturbed and perturbed VLA outputs
to identify which perturbed inputs lead to signficant changes in the VLA outputs.
'''

import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from fastdtw import fastdtw


def calculate_vla_metric(
        unperturbed_episode_results,
        perturbed_episode_results,
        unperturbed_episode_lengths,
        perturbed_episode_lengths,
        unperturbed_trajectories,
        perturbed_trajectories,
        w_result=1,
        w_time=1,
        w_trajectory=1,
        W=None):
    """
    Computes the VLA metric given the results, lengths, and trajectories of the unperturbed and perturbed episodes.
    This metric is a weighted sum of the success metric, time metric, and trajectory difference metric.

    Args:
        unperturbed_episode_results (torch.Tensor): A binary tensor of shape (n, 1) for the results (1 - success, 0 - fail) of each trial of the unperturbed episode.
        perturbed_episode_results (torch.Tensor): A binary tensor of shape (n, 1) for the results (1 - success, 0 - fail) of each trial of the perturbed episode.
        unperturbed_episode_lengths (torch.Tensor): A tensor of shape (n, 1) for the average episode length of each trial of the unperturbed episode.
        perturbed_episode_lengths (torch.Tensor): A tensor of shape (n, 1) for the average episode length of each trial of the perturbed episode.
        unperturbed_trajectories (list): A list of n numpy arrays of shape (*, 8) that contains the trajectories of the unperturbed episodes.
        perturbed_trajectories (list): A list of n numpy arrays of shape (*, 8) that contains the trajectories of the perturbed episodes.
        w_result (float): weight for the success metric component.
        w_time (float): weight for the time metric component.
        w_trajectory (float): weight for the trajectory difference metric component.
        W (np.array): A diagonal matrix or 1d array that represents the weights of the different trajectory dimensions (8D)
    
    Returns:
        float: the VLA metric (-inf, inf) between the unperturbed and perturbed episodes.
    """
    assert min(w_result, w_time, w_trajectory) >= 0, "Each metric weight must be non-negative."
    assert sum([w_result, w_time, w_trajectory]) != 0, "Metric weights must not sum to 0."
    
    metric_weights = np.array([w_result, w_time, w_trajectory])
    metric_weights = metric_weights / np.linalg.norm(metric_weights)
    success_metric = calculate_success_metric(unperturbed_episode_results, perturbed_episode_results)
    time_metric = calculate_time_metric(unperturbed_episode_lengths, perturbed_episode_lengths)
    trajectory_metric = calculate_trajectory_difference_metric(unperturbed_trajectories, perturbed_trajectories, W)
    vla_metric = metric_weights[0] * success_metric + metric_weights[1] * time_metric + metric_weights[2] * trajectory_metric

    return vla_metric


def calculate_success_metric(unperturbed_episode_results, perturbed_episode_results):
    """
    Computes the absolute difference in success rates between an unperturbed and perturbed episode.

    Args:
        unperturbed_episode_results (torch.Tensor): A binary tensor of shape (n, 1) for the results (1 - success, 0 - fail) of each trial of the unperturbed episode.
        perturbed_episode_results (torch.Tensor): A binary tensor of shape (n, 1) for the results (1 - success, 0 - fail) of each trial of the perturbed episode.
    
    Returns:
        float: the difference in success rate between the unperturbed and perturbed episodes. 
    """
    assert torch.all((unperturbed_episode_results == 0) | (unperturbed_episode_results == 1)), "The results of the unperturbed episode trials must be either 0 (unsuccessful) or 1 (successful)."
    assert torch.all((perturbed_episode_results == 0) | (perturbed_episode_results == 1)), "The results of the perturbed episode trials must be either 0 (unsuccessful) or 1 (successful)."
    average_unperturbed_success_rate = torch.mean(unperturbed_episode_results.float())
    average_perturbed_success_rate = torch.mean(perturbed_episode_results.float())
    success_rate_difference = torch.abs(average_unperturbed_success_rate - average_perturbed_success_rate)
    return success_rate_difference

def calculate_time_metric(unperturbed_episode_lengths, perturbed_episode_lengths):
    """
    Computes the geometric increase in average episode length between an unperturbed and perturbed episode.

    Args:
        unperturbed_episode_results (torch.Tensor): A tensor of shape (n, 1) for the average episode length of each trial of the unperturbed episode.
        perturbed_episode_results (torch.Tensor): A tensor of shape (n, 1) for the average episode length of each trial of the perturbed episode.
    
    Returns:
        float: the geometric increase in average episode length between the unperturbed and perturbed episodes. 
    """
    assert torch.all(unperturbed_episode_lengths > 0), "The length of the unperturbed episode trials must be greater than 0."
    assert torch.all(perturbed_episode_lengths > 0), "The length of the perturbed episode trials must be greater than 0."
    average_unperturbed_episode_length = torch.mean(unperturbed_episode_lengths.float())
    average_perturbed_episode_length = torch.mean(perturbed_episode_lengths.float())
    episode_length_increase = average_unperturbed_episode_length / average_perturbed_episode_length
    return episode_length_increase

def calculate_trajectory_difference_metric(unperturbed_trajectories, perturbed_trajectories, W=None):
    """
    Computes the log Wasserstein distance between the set of n trials of the unperturbed trajectory and
    the set of n trials of the perturbed trajectory.

    Args:
        unperturbed_trajectories (list): A list of n numpy arrays of shape (*, 8) that contains the trajectories of the unperturbed episodes.
        perturbed_trajectories (list): A list of n numpy arrays of shape (*, 8) that contains the trajectories of the perturbed episodes.
        W (np.array): A diagonal matrix or 1d array that represents the weights of the different trajectory dimensions (8D)

    Returns:
        float: the trajectory difference (-inf, inf) between a set of trials for unperturbed and perturbed episodes.
    """
    # Calculate the cost matrix for the distance between the ith trial of the unperturbed trajectories
    # to the jth trial of the perturbed trajectories.
    assert len(unperturbed_trajectories) == len(perturbed_trajectories), (
        f"To simplify the Wasserstein distance metric calculation into a linear_sum_assignment optimization problem, "
        f"the number of trials for both distribuions must be the same, got num unperturbed {len(unperturbed_trajectories)} "
        f"and num perturbed {len(perturbed_trajectories)}."
    )
    assert len(unperturbed_trajectories) > 0, "There must be at least 1 trajectory in the list of perturbed and unperturbed trajectories."
    if (W is None):
        W = W = np.eye(8)  # 8 weights for the 8 DoF for the trajectories
    W = W / np.linalg.norm(W)
    if (W.ndim == 1):
        assert W.shape == (8,), f"The W matrix should have the same number of elements as the DoF for the trajectories, ie shape: (8,), got {W.shape}."
        W = np.diag(W)
    elif (W.ndim == 2):
        assert W.shape == (8, 8), f"The W matrix should have the same number of elements as the DoF for the trajectories, ie shape: (8, 8), got {W.shape}."
    else:
        assert False, f"W should have ndims as 1 or 2, got {W.ndim}."
    n = len(unperturbed_trajectories)

    trajectory_distance_matrix = get_dtw_trajectory_distance_matrix(unperturbed_trajectories, perturbed_trajectories, W)
    trajectory_wasserstein_distance = calculate_wasserstein_1_dist(trajectory_distance_matrix, n)
    trajectory_difference = np.log(trajectory_wasserstein_distance + 1e-16)

    return trajectory_difference

def get_dtw_trajectory_distance_matrix(unperturbed_trajectories, perturbed_trajectories, W=None):
    """
    Computes the dtw trajectory distance  matrix for the distance between the ith trial of the unperturbed trajectories
    to the jth trial of the perturbed trajectories.

    Args:
        unperturbed_trajectories (list): A list of n numpy arrays of shape (*, 8) that contains the trajectories of the unperturbed episodes.
        perturbed_trajectories (list): A list of n numpy arrays of shape (*, 8) that contains the trajectories of the perturbed episodes.
        W (np.array): A diagonal matrix that represents the weights of the different trajectory dimensions (8D)
    
    Return:
        numpy.array: An m x  matrix that gives the distance between the ith trial of the unperturbed trajectories to the
                     jth trial of the perturbed trajectories.
    """
    trajectory_distance_matrix = np.zeros((len(unperturbed_trajectories), len(perturbed_trajectories)))
    for i in range(0, len(unperturbed_trajectories)):
        for j in range(0, len(perturbed_trajectories)):
            trajectory_distance_matrix[i][j] = calculate_dtw_trajectory_difference(
                                                    unperturbed_trajectories[i],
                                                    perturbed_trajectories[j],
                                                    W)[0]
    return trajectory_distance_matrix

def calculate_wasserstein_1_dist(trajectory_distance_matrix, n):
    """
    Computes the wasserstein distance between two trajectories given the trajectory_distance matrix.
    We assume the trajectories have the same number of trials in the distributions so that we can
    simplify the problem to a Wasserstein 1D distance using linear_sum_assignment.

    Args:
        trajectory_distance_matrix (numpy.array): An n x n matrix that gives the distance between the 
                                                  ith trial of the unperturbed trajectories to the jth 
                                                  trial of the perturbed trajectories.
        n (int): number of trials for both trajectories
    
    Return:
        float: the trajectory difference between a set of trials for unperturbed and perturbed episodes.
    """
    unperturbed_idxs, perturbed_idxs = linear_sum_assignment(trajectory_distance_matrix)
    wasserstein_1_dist = trajectory_distance_matrix[unperturbed_idxs, perturbed_idxs].sum() / n
    return wasserstein_1_dist

def calculate_dtw_trajectory_difference(trajectory_1, trajectory_2, W=None):
    """
    Calculates the difference between 2 trajectories using Dynamic Time Warping (DTW).
    DTW aligns the trajectories. We then form quadrilaterals and triangles between the
    trajectory timestamps and sum up the measured areas of these shapes. We also
    normalize with respect to the maximum area and take the negative log of 1 - value.

    Args:
        trajectory_1 (numpy.array): A 2d array of size (T_1, 8) that has the values of each DoF of the VLA for each time step in the episode.
        trajectory_2 (numpy.array): A 2d array of size (T_2, 8) that has the values of each DoF of the VLA for each time step in the episode.
        W (np.array): A diagonal matrix that represents the weights of the different trajectory dimensions (8D)
    
    Returns:
        float: the area-based DTW difference between trajectory_1 and trajectory_2.
    """
    if (W is None):
        W = W = np.eye(8)  # 8 weights for the 8 DoF for the trajectories
    W = W / np.linalg.norm(W)
    if (W.ndim == 1):
        assert W.shape == (8,), f"The W matrix should have the same number of elements as the DoF for the trajectories, ie shape: (8,), got {W.shape}."
        W = np.diag(W)
    elif (W.ndim == 2):
        assert W.shape == (8, 8), f"The W matrix should have the same number of elements as the DoF for the trajectories, ie shape: (8, 8), got {W.shape}."
    else:
        assert False, f"W should have ndims as 1 or 2, got {W.ndim}."
    min_point_pos = get_min_point_pos(np.vstack((trajectory_1, trajectory_2)))
    max_point_pos = get_max_point_pos(np.vstack((trajectory_1, trajectory_2)))
    def dtw_distance_metric(point_1, point_2):
        return trajectory_8dof_normalized_point_distance(point_1, point_2, min_point_pos, max_point_pos)
    _, warp_path = fastdtw(trajectory_1, trajectory_2, dist=dtw_distance_metric)
    dtw_area, triangles = calculate_dtw_area(trajectory_1, trajectory_2, warp_path, W)
    return dtw_area, warp_path, triangles
 

def calculate_dtw_area(trajectory_1, trajectory_2, warp_path, W=None):
    """
    Calculate the area between 2 trajectories based on the DTW warp path.

    Args:
        trajectory_1 (numpy.array): A 2d array of size (T_1, 8) that has the values of each DoF of the VLA for each time step in the episode.
        trajectory_2 (numpy.array): A 2d array of size (T_2, 8) that has the values of each DoF of the VLA for each time step in the episode.
        warp_path (list): A list of tuples that indicate which points in trajectory_1 and trajectory_2 are aligned together.
        W (np.array): A diagonal matrix that represents the weights of the different trajectory dimensions (8D)
    
    Returns:
        float: the area between trajectory_1 and trajectory_2 based on the DTW warp path.
    """
    area = 0
    triangles = list()
    for i in range(1, len(warp_path)):
        curr_traj_1_idx, curr_traj_2_idx = warp_path[i]
        prev_curr_traj_1_idx, prev_curr_traj_2_idx = warp_path[i-1]
        sub_area_1 = (triangle_area(point_1=trajectory_1[curr_traj_1_idx, :], point_2=trajectory_2[curr_traj_2_idx, :], point_3=trajectory_1[prev_curr_traj_1_idx, :], W=W)
                    + triangle_area(point_1=trajectory_2[curr_traj_2_idx, :], point_2=trajectory_2[prev_curr_traj_2_idx, :], point_3=trajectory_1[prev_curr_traj_1_idx, :], W=W))
        sub_area_2 = (triangle_area(point_1=trajectory_1[curr_traj_1_idx, :], point_2=trajectory_2[prev_curr_traj_2_idx, :], point_3=trajectory_1[prev_curr_traj_1_idx, :], W=W)
                    + triangle_area(point_1=trajectory_2[curr_traj_2_idx, :], point_2=trajectory_2[prev_curr_traj_2_idx, :], point_3=trajectory_1[curr_traj_1_idx, :], W=W))

        if (sub_area_1 <= sub_area_2):
            area += sub_area_1
            if (triangle_area(point_1=trajectory_1[curr_traj_1_idx, :], point_2=trajectory_2[curr_traj_2_idx, :], point_3=trajectory_1[prev_curr_traj_1_idx, :], W=W) > 0):
                triangles.append([trajectory_1[curr_traj_1_idx, :], trajectory_2[curr_traj_2_idx, :], trajectory_1[prev_curr_traj_1_idx, :]])
            if (triangle_area(point_1=trajectory_2[curr_traj_2_idx, :], point_2=trajectory_2[prev_curr_traj_2_idx, :], point_3=trajectory_1[prev_curr_traj_1_idx, :], W=W) > 0):
                triangles.append([trajectory_2[curr_traj_2_idx, :], trajectory_2[prev_curr_traj_2_idx, :], trajectory_1[prev_curr_traj_1_idx, :]])
        else:
            area += sub_area_2
            if (triangle_area(point_1=trajectory_1[curr_traj_1_idx, :], point_2=trajectory_2[prev_curr_traj_2_idx, :], point_3=trajectory_1[prev_curr_traj_1_idx, :], W=W) > 0):
                triangles.append([trajectory_1[curr_traj_1_idx, :], trajectory_2[prev_curr_traj_2_idx, :], trajectory_1[prev_curr_traj_1_idx, :]])
            if (triangle_area(point_1=trajectory_2[curr_traj_2_idx, :], point_2=trajectory_2[prev_curr_traj_2_idx, :], point_3=trajectory_1[curr_traj_1_idx, :], W=W) > 0):
                triangles.append([trajectory_2[curr_traj_2_idx, :], trajectory_2[prev_curr_traj_2_idx, :], trajectory_1[curr_traj_1_idx, :]])
    
    return area, np.array(triangles)

def triangle_area(point_1, point_2, point_3, W=None):
    """
    Calculate the area of a triangle in an n-dimensional space.

    Args:
        point_1 (np.array): A n-dimension array that represents the point in a trajectory
        point_2 (np.array): A n-dimension array that represents the point in a trajectory
        point_3 (np.array): A n-dimension array that represents the point in a trajectory
        W (np.array): A diagonal matrix that represents the weights of the different trajectory dimensions (8D)
    
    Returns:
        float: the area of the triangle
    """
    u = point_2 - point_1
    v = point_3 - point_1
    if (W is None):
        W = np.eye(len(u)) / np.linalg.norm(np.eye(len(u)))
    G = np.array([[u @ W @ u, u @ W @ v],
                  [v @ W @ u, v @ W @ v]], dtype=float)

    triangle_area = float(0.5 * np.sqrt(np.linalg.det(G) + 1e-26))
    return triangle_area

def trajectory_8dof_normalized_point_distance(point_1, point_2, min_point_pos, max_point_pos, w_pos=1.0, w_rot=1.0, w_g=1.0):
    """
    Calculate the difference/distance between 2 points in the 8D space of the VLA outputs.
    8D: 3D for xyz position, 4D for quaternion orientation, 1D for gripper value. We normalize 
    only for the euclidean distance in 3D space because the other values are bounded between 0 
    and 1.

    Args:
        point_1 (numpy.array): A numpy array of shape (8) that denotes a singular point in the 8D VLA output space.
        point_2 (numpy.array): A numpy array of shape (8) that denotes a singular point in the 8D VLA output space.
        min_point_pos (numpy.array): A numpy array of shape (3) that denotes a point with the lowest XYZ coords that
                                    the episode has in 3D space.
        max_point_pos (numpy.array): A numpy array of shape (3) that denotes a point with the largest XYZ coords that
                                    the episode has in 3D space.
        w_pos (float): The weight/importance of the xyz difference between 2 points.
        w_rot (float): The weight/importance of the quaternion difference between 2 points.
        w_g (float): The weight/importance of the gripper value difference between 2 points.
    
    Return:
        float: aggregated and weighed difference between 2 points in the 8D VLA output shape.
    """
    assert point_1.squeeze().shape == (8,), f"Point 1 must have 8 DoF, got point shape as {point_1.squeeze().shape}"
    assert point_2.squeeze().shape == (8,), f"Point 2 must have 8 DoF, got point shape as {point_1.squeeze().shape}"

    point_1_pos = point_1[0:3]
    point_2_pos = point_2[0:3]
    point_1_quat = point_1[3:7]
    point_2_quat = point_2[3:7]
    point_1_g = point_1[7]
    point_2_g = point_2[7]

    pos_diff = np.linalg.norm(point_1_pos - point_2_pos) / (np.linalg.norm(max_point_pos - min_point_pos) + 1e-15)
    quat_diff = calculate_angle(point_1_quat, point_2_quat) / np.pi
    g_diff = np.abs(point_1_g - point_2_g)

    assert 0 <= pos_diff and pos_diff <= 1, "The normalized value of the position difference between points must be between 0 and 1."
    assert 0 <= quat_diff and quat_diff <= 1, "The normalized value of the quaternion/angle difference between points must be between 0 and 1."
    assert 0 <= g_diff and g_diff <= 1, "The normalized value of the gripper position difference between points must be between 0 and 1."

    point_distance = w_pos * pos_diff + w_rot * quat_diff + w_g * g_diff
    return point_distance

def calculate_angle(quaternion_1, quaternion_2):
    """
    Calculates the angle between two quaternions.

    Args:
        quaternion_1 (numpy.array): A numpy array of shape (4) that denotes a quaternion
        quaternion_2 (numpy.array): A numpy array of shape (4) that denotes a quaternion

    Return:
        float: angle (in radians) between the two quaternions
    """
    quaternion_1 = quaternion_1 / np.linalg.norm(quaternion_1)
    quaternion_2 = quaternion_2 / np.linalg.norm(quaternion_2)

    assert math.isclose(quaternion_1[0]**2 + quaternion_1[1]**2 + quaternion_1[2]**2 + quaternion_1[3]**2, 1, rel_tol=1e-8), f"Quaternion 1 is invalid: {quaternion_1}"
    assert math.isclose(quaternion_2[0]**2 + quaternion_2[1]**2 + quaternion_2[2]**2 + quaternion_2[3]**2, 1, rel_tol=1e-8), f"Quaternion 2 is invalid: {quaternion_2}"

    dot_product = np.clip(np.abs(np.dot(quaternion_1, quaternion_2)), -1.0, 1.0)
    angle_rad = 2 * np.arccos(dot_product)
    assert 0 <= angle_rad and angle_rad <= np.pi, "The magnitude of the angle must be between 0 and pi."

    return angle_rad

def get_min_point_pos(points):
    """
    Get the minimum xyz position from a set of 8D points.

    Args:
        points (numpy.array): A 2D numpy array of shape (T, 8) that has T points in the 8D VLA output space.
    
    Returns:
        numpy.array: A numpy array of shape (3) that denotes a point with the lowest XYZ coords that
                     the episode has in 3D space.
    """
    assert points.shape[0] > 0, "The number of points must be greater than 0."
    min_point_pos = np.array([np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])])
    return min_point_pos

def get_max_point_pos(points):
    """
    Get the maximum xyz position from a set of 8D points.

    Args:
        points (numpy.array): A 2D numpy array of shape (T, 8) that has T points in the 8D VLA output space.
    
    Returns:
        numpy.array: A numpy array of shape (3) that denotes a point with the largest XYZ coords that
                     the episode has in 3D space.
    """
    assert points.shape[0] > 0, "The number of points must be greater than 0."
    max_point_pos = np.array([np.max(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 2])])
    return max_point_pos
