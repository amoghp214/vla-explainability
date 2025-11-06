import json
import numpy as np

def extract_episode_trials(episode_json_path):
    """
    Extract a list of trajectories (n trials) from the episode JSON.
    The json should contain a dict of all the episode sets (n trials
    of a specific task with fixed perturbations). Each key is the 
    name of the episode type, e.g. 'unperturbed' or 
    'language_perturbed_remove_the'. The value of each key should be
    a list of all the trajectories resulting from running this 
    episode n times. Each trajectory trial is a 2d array of size
    (num_steps, num_dof). Therefore each value should be a 3d nested
    list.

    Args:
        episode_json_path (str): The path of the episode data in JSON format.
    
    Returns:
        list: A dict of lists of np.ndarray, each representing a trial's trajectory.
    """
    with open(episode_json_path, 'r') as f:
        episode_data = json.load(f)
    
    num_trials = -1
    parsed_data = dict()
    for name, episode_set in episode_data.items():
        curr_episode_set = []
        for episode in episode_set:
            np_episode = np.array(episode)
            assert np_episode.shape[1] == 8, f"Trajectories must be represented with shape (*, 8), got {np_episode.shape}."  # We expect the trajectories to have an 8 DoF output
            curr_episode_set.append(np_episode)
        if (num_trials == -1):
            num_trials = len(curr_episode_set)
        assert num_trials == len(curr_episode_set), ("Each episode set must have the same number of trials;",
                                                    f"previous set contained {num_trials} trials, and current ",
                                                    f"set contains {len(curr_episode_set)} trials.")
        parsed_data[name] = curr_episode_set
    
    return parsed_data
        

