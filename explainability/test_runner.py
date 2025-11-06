import numpy as np

import vla_metrics
from metric_utils import extract_episode_trials
import data_visualization


# og_traj = np.load("./test/states_orig.npy")
# no_change_traj = np.load("./test/states_no_change.npy")
# big_change_traj = np.load("./test/states_big_change.npy")
# small_change_traj = np.load("./test/states_small_prompt_change.npy")

episode_data_path = "./test/vla_output_1.json"
UNPERTURBED = 'unperturbed'
# print(og_traj.shape, no_change_traj.shape, big_change_traj.shape, small_change_traj.shape)

W = np.array([1, 1, 1, 0.2, 0.2, 0.2, 0.2, 0.5])


# print("OG vs. OG", vla_metrics.calculate_dtw_trajectory_difference(og_traj, og_traj, W)[0])
# print("OG vs. No Change", vla_metrics.calculate_dtw_trajectory_difference(og_traj, no_change_traj, W)[0])
# print("OG vs. Small Change", vla_metrics.calculate_dtw_trajectory_difference(og_traj, small_change_traj, W)[0])
# print("OG vs. Big Change", vla_metrics.calculate_dtw_trajectory_difference(og_traj, big_change_traj, W)[0])

# sample_1 = np.array([[0, 0, 0, 1, 1, 1, 1, 0], [2, 2, 2, 1, 1, 1, 1, 0], [2.2, 4, 2.5, 1, 1, 1, 1, 0], [2.2, 5, 3, 1, 1, 1, 1, 0], [2, 6, 4, 1, 1, 1, 1, 0]])
# sample_2 = np.array([[0, 0, 0, 1, 1, 1, 1, 0], [1, 1, 3, 1, 1, 1, 1, 0], [2, 3.8, 3, 1, 1, 1, 1, 0], [2, 4, 3, 1, 1, 1, 1, 0], [2, 5, 3, 1, 1, 1, 1, 0], [2, 6, 4, 1, 1, 1, 1, 0]])

episode_data = extract_episode_trials(episode_data_path)

unperturbed_episode = episode_data[UNPERTURBED]
perturbed_v2_episode = episode_data['perturbed_language_change_bowl']
# perturbed_v1_episode = episode_data['perturbed_language_remove_black_replace_place']

print("Trajectory Difference Metric:", vla_metrics.calculate_trajectory_difference_metric(unperturbed_episode, perturbed_v2_episode, W))
print("Same Trajectory Difference Metric:", vla_metrics.calculate_trajectory_difference_metric(unperturbed_episode[:2], unperturbed_episode[2:4], W))

for i, t in enumerate(perturbed_v2_episode):
    print(f"Perturbed trajectory metric {i} output:", vla_metrics.calculate_dtw_trajectory_difference(unperturbed_episode[0], t, W)[0])

sample_1 = unperturbed_episode[2]
sample_2 = perturbed_v2_episode[2]
# sample_1 = og_traj
# sample_2 = small_change_traj
sample_dtw_trajectory_diff, sample_dtw_warp_path, sample_triangles = vla_metrics.calculate_dtw_trajectory_difference(sample_1, sample_2, W)
print("Single Sample Trajectory Difference Metric:", np.log(sample_dtw_trajectory_diff))
warp_path = np.array([[sample_1[i,:3], sample_2[j,:3]] for i, j in sample_dtw_warp_path])
# for i in range(0, len(sample_triangles)):
#     print(sample_triangles[i])
# print(sample_dtw_warp_path)
data_visualization.visualize_trajectory_difference(sample_1, sample_2, triangles=sample_triangles, lines=warp_path, output_file="./test/sample_visualization.html")
data_visualization.visualize_trajectories(perturbed_v2_episode, output_file="./test/perturbed_v2_trajectories.html")


# print("2-trial trajectory difference:", vla_metrics.calculate_trajectory_difference_metric([og_traj, no_change_traj], [big_change_traj, small_change_traj]))
# print("OG vs. OG", vla_metrics.calculate_trajectory_difference_metric([og_traj], [og_traj], W))
# print("OG vs. No Change", vla_metrics.calculate_trajectory_difference_metric([og_traj], [no_change_traj], W))
# print("OG vs. Small Change", vla_metrics.calculate_trajectory_difference_metric([og_traj], [small_change_traj], W))
# print("OG vs. Big Change", vla_metrics.calculate_trajectory_difference_metric([og_traj], [big_change_traj], W))