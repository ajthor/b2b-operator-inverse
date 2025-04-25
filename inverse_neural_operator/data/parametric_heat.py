import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class ParametricHeatDataset(Dataset):
    """
    Custom dataset for parametric heat equation data where X and Y are assumed to be on a grid.

    This dataset handles the case where the HuggingFace dataset only has 'u' and 's' values,
    and we need to create the X and Y grid coordinates.
    """

    def __init__(self, huggingface_dataset, grid_size=51, device="cpu"):
        """
        Initialize the dataset by extracting 'u' and 's' values and creating coordinate grid.

        Args:
            huggingface_dataset: HuggingFace dataset with 'u' and 's' fields
            grid_size: Size of the square grid (default: 51)
            device: The device to put tensors on
        """
        self.device = device

        # Extract 'u' and 's' from the dataset and convert to tensors directly
        # Get all 'u' and 's' values at once as numpy arrays
        u_array = huggingface_dataset["u"]
        s_array = huggingface_dataset["s"]

        # Convert to tensors
        self.u_tensor = torch.tensor(u_array, device=device)
        self.s_tensor = torch.tensor(s_array, device=device)

        # Ensure correct dimensions
        if self.u_tensor.dim() == 2:  # [batch, values]
            self.u_tensor = self.u_tensor.unsqueeze(-1)  # [batch, values, 1]
        if self.s_tensor.dim() == 2:
            self.s_tensor = self.s_tensor.unsqueeze(-1)

        # Create a meshgrid for X and Y coordinates (normalized to [0,1])
        x = torch.linspace(0, 1, grid_size, device=device)
        y = torch.linspace(0, 1, grid_size, device=device)
        X, Y = torch.meshgrid(x, y, indexing="ij")

        # Flatten the grid coordinates
        self.grid_coords = torch.stack([X.flatten(), Y.flatten()], dim=1)

    def __len__(self):
        return len(self.u_tensor)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (X, u, Y, s) where:
            - X is the grid coordinates
            - u is the input function values
            - Y is the same grid coordinates (for this dataset)
            - s is the output function values
        """
        return (
            self.grid_coords,  # X
            self.u_tensor[idx],  # u
            self.grid_coords,  # Y (same as X for this dataset)
            self.s_tensor[idx],  # s
        )


class InputFunctionEncoderDataset(Dataset):
    """Dataset for input function encoder that splits the input domain points."""

    def __init__(self, base_dataset, device="cpu"):
        """
        Initialize from a base ParametricHeatDataset.

        Args:
            base_dataset: ParametricHeatDataset instance
            device: The device to put tensors on
        """
        self.device = device

        # Get the number of samples and grid points
        n_samples = len(base_dataset)
        n_points = len(base_dataset.grid_coords)
        split_idx = n_points // 2

        # Pre-allocate tensors for all samples
        self.example_xs = []
        self.example_ys = []
        self.xs = []
        self.ys = []

        X = base_dataset.grid_coords

        for i in range(n_samples):
            u = base_dataset.u_tensor[i]

            # Split points into example and test sets
            indices = torch.randperm(n_points, device=device)
            example_indices = indices[:split_idx]
            remaining_indices = indices[split_idx:]

            self.example_xs.append(X[example_indices])
            self.example_ys.append(u[example_indices])
            self.xs.append(X[remaining_indices])
            self.ys.append(u[remaining_indices])

        # Convert lists to stacked tensors
        self.example_xs = torch.stack(self.example_xs)
        self.example_ys = torch.stack(self.example_ys)
        self.xs = torch.stack(self.xs)
        self.ys = torch.stack(self.ys)

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (example_xs, example_ys, xs, ys)
        """
        return (self.example_xs[idx], self.example_ys[idx], self.xs[idx], self.ys[idx])


class OutputFunctionEncoderDataset(Dataset):
    """Dataset for output function encoder that splits the output domain points."""

    def __init__(self, base_dataset, device="cpu"):
        """
        Initialize from a base ParametricHeatDataset.

        Args:
            base_dataset: ParametricHeatDataset instance
            device: The device to put tensors on
        """
        self.device = device

        # Get the number of samples and grid points
        n_samples = len(base_dataset)
        n_points = len(base_dataset.grid_coords)
        split_idx = n_points // 2

        # Pre-allocate tensors for all samples
        self.example_xs = []
        self.example_ys = []
        self.xs = []
        self.ys = []

        Y = base_dataset.grid_coords

        for i in range(n_samples):
            s = base_dataset.s_tensor[i]

            # Split points into example and test sets
            indices = torch.randperm(n_points, device=device)
            example_indices = indices[:split_idx]
            remaining_indices = indices[split_idx:]

            self.example_xs.append(Y[example_indices])
            self.example_ys.append(s[example_indices])
            self.xs.append(Y[remaining_indices])
            self.ys.append(s[remaining_indices])

        # Convert lists to stacked tensors
        self.example_xs = torch.stack(self.example_xs)
        self.example_ys = torch.stack(self.example_ys)
        self.xs = torch.stack(self.xs)
        self.ys = torch.stack(self.ys)

    def __len__(self):
        return self.xs.shape[0]

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (example_xs, example_ys, xs, ys)
        """
        return (self.example_xs[idx], self.example_ys[idx], self.xs[idx], self.ys[idx])


def get_info(dataset):
    """
    Extract info from dataset.

    Args:
        dataset: Dataset to extract info from

    Returns:
        Info dictionary
    """
    # Take the first sample to determine dimensions
    sample = dataset[0]

    # Get input and output sizes from first and second components
    input_size = sample[0].shape[-1]
    output_size = sample[1].shape[-1]

    return {
        "input_size": input_size,
        "output_size": output_size,
    }


def load_data(params, device, split="train"):
    """
    Load a dataset from a specific split.

    Args:
        params: Parameters for processing
        device: The device to use
        split: The dataset split to load (default: "train")

    Returns:
        A tuple containing the datasets and info for the specified split
    """
    # Load the dataset from HuggingFace
    hf_dataset = load_dataset("ajthor/parametric_heat", split=split)

    # Create grid size from params
    grid_size = 51  # Default grid size of 51x51

    # Create our dataset
    model_dataset = ParametricHeatDataset(
        hf_dataset, grid_size=grid_size, device=device
    )
    model_info = get_info(model_dataset)

    # Create input function encoder dataset
    input_fe_dataset = InputFunctionEncoderDataset(model_dataset, device=device)
    input_info = get_info(input_fe_dataset)

    # Create output function encoder dataset
    output_fe_dataset = OutputFunctionEncoderDataset(model_dataset, device=device)
    output_info = get_info(output_fe_dataset)

    return (
        model_dataset,
        input_fe_dataset,
        output_fe_dataset,
        input_info,
        output_info,
        model_info,
    )
