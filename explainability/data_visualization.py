import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_trajectories(trajectories, output_file="visualization.html"):
    """
    Visualizes n trajectories in 3D space.

    Args:
        trajectory_1 (np.ndarray): A list of 2D array of shape (T1, 8) representing a list of trajectory.
        output_file (str): The file path to save the interactive 3D visualization.
    """
    fig = go.Figure()
    for i, traj in enumerate(trajectories):
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines+markers',
            # line=dict(color='red'),
            name=f'Trajectory {i}'
        ))
    fig.write_html(output_file)


def visualize_trajectory_difference(trajectory_1, trajectory_2, triangles=None, lines=None, output_file="visualization.html"):
    """
    Visualizes two trajectories and a set of triangles in 3D space.

    Args:
        trajectory_1 (np.ndarray): A 2D array of shape (T1, 8) representing the first trajectory.
        trajectory_2 (np.ndarray): A 2D array of shape (T2, 8) representing the second trajectory.
        triangles (np.ndarray): A 2D array of shape (N, 3, 3) where each row represents a triangle defined by 3 points in 3D space.
        lines (np.ndarray): A 2D array of shape (M, 2) where each row represents a line segment outputed by DTW.
        output_file (str): The file path to save the interactive 3D visualization.
    """
    fig = go.Figure()
    # ax = fig.add_subplot(111, projection='3d')

    # Plot the first trajectory in red
    # ax.plot(trajectory_1[:, 0], trajectory_1[:, 1], trajectory_1[:, 2], color='red', label='Trajectory 1')
    fig.add_trace(go.Scatter3d(
        x=trajectory_1[:, 0],
        y=trajectory_1[:, 1],
        z=trajectory_1[:, 2],
        mode='lines+markers',
        line=dict(color='red'),
        name='Trajectory 1'
    ))

    # Plot the second trajectory in blue
    # ax.plot(trajectory_2[:, 0], trajectory_2[:, 1], trajectory_2[:, 2], color='blue', label='Trajectory 2')
    fig.add_trace(go.Scatter3d(
        x=trajectory_2[:, 0],
        y=trajectory_2[:, 1],
        z=trajectory_2[:, 2],
        mode='lines+markers',
        line=dict(color='blue'),
        name='Trajectory 2'
    ))

    for point in trajectory_1:
        position = point[:3]  # XYZ position
        quaternion = point[3:7]  # Quaternion (x, y, z, w)
        direction = quaternion_to_direction_vector(quaternion)
        end_point = position + direction / 100  # Compute the end point of the vector
        fig.add_trace(go.Scatter3d(
            x=[position[0], end_point[0]],
            y=[position[1], end_point[1]],
            z=[position[2], end_point[2]],
            mode='lines',
            line=dict(color='orange')
        ))
    
    for point in trajectory_2:
        position = point[:3]  # XYZ position
        quaternion = point[3:7]  # Quaternion (x, y, z, w)
        direction = quaternion_to_direction_vector(quaternion)
        end_point = position + direction / 100 # Compute the end point of the vector
        fig.add_trace(go.Scatter3d(
            x=[position[0], end_point[0]],
            y=[position[1], end_point[1]],
            z=[position[2], end_point[2]],
            mode='lines',
            line=dict(color='purple')
        ))

    # Plot the triangles in yellow
    if triangles is not None:
        for triangle in triangles:
            fig.add_trace(go.Mesh3d(
                x=triangle[:, 0],
                y=triangle[:, 1],
                z=triangle[:, 2],
                color='yellow',
                opacity=0.5
            ))
            fig.add_trace(go.Scatter3d(
                x=triangle[:, 0],
                y=triangle[:, 1],
                z=triangle[:, 2],
                mode='lines',
                line=dict(color='gray'),
            ))

    if lines is not None:
        for line in lines:
            fig.add_trace(go.Scatter3d(
                x=line[:, 0],
                y=line[:, 1],
                z=line[:, 2],
                mode='lines',
                line=dict(color='green')
            ))

    # Set labels and legend
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()

    # Save the visualization to file
    # plt.savefig(output_file)
    # mpld3.save_html(fig, output_file)
    # plt.close()
    fig.write_html(output_file)

def quaternion_to_direction_vector(quaternion):
    """
    Converts a quaternion to a directional vector.

    Args:
        quaternion (np.ndarray): A 4-element array representing the quaternion (x, y, z, w).

    Returns:
        np.ndarray: A 3-element array representing the directional vector.
    """
    x, y, z, w = quaternion
    # Compute the direction vector (this assumes the quaternion is normalized)
    direction = np.array([
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x**2 + y**2)
    ])
    return direction

def add_orientation_vectors(trajectory):
    """
    Adds orientation vectors to a trajectory based on its quaternion values.

    Args:
        trajectory (np.ndarray): A 2D array of shape (T, 7), where the first 3 columns are XYZ positions
                                 and the next 4 columns are quaternions (x, y, z, w).

    Returns:
        list: A list of tuples, where each tuple contains the start point (XYZ) and the end point (XYZ + direction).
    """
    vectors = []
    for point in trajectory:
        position = point[:3]  # XYZ position
        quaternion = point[3:7]  # Quaternion (x, y, z, w)
        direction = quaternion_to_direction_vector(quaternion)
        end_point = position + direction  # Compute the end point of the vector
        vectors.append((position, end_point))
    return vectors

# Example usage
if __name__ == "__main__":
    # Example trajectories and triangles
    trajectory_1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    trajectory_2 = np.array([[0, 0, 1], [1, 1, 2], [2, 2, 3]])
    triangles = np.array([
        [[0, 0, 0], [1, 1, 1], [0, 0, 1]],
        [[1, 1, 1], [2, 2, 2], [1, 1, 2]]
    ])

    visualize_trajectories(trajectory_1, trajectory_2, triangles, "example_visualization.png")