

def generate_perturbation_dataset():
    """
    Create a dataset of perturbed episodes (x: perturbation is on/off, y: metric result)

    Args:

    Returns:
        torch.dataset: a dataset to fit the surrogate model on
    """
    # Run episodic inference on the unperturbed episode.

    # Given a set of perturbations, select perturbation combination and represent this combination as 1s and 0s for our surrogate model's input features.

    # Run episodic inference on selected perturbation combination.

    # Calculate the weighted metric score of the perturbed vs. unperturbed inferences.

    # Concatenate perturbation combination representation (x) with weighted metric scores (y)

    # Convert to dataset and return