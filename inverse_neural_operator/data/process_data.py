import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import math


class ModelDataset(Dataset):
    """Dataset for the main model that processes input and output functions."""

    def __init__(self, dataset, device="cpu"):
        """
        Initialize from any dataset format.

        Args:
            dataset: HuggingFace dataset with 'X', 'u', 'Y', and 's' fields
            device: The device to put tensors on
        """
        self.device = device
        self.n_samples = len(dataset)

        self.X = torch.tensor(dataset["X"], device=device)
        self.u = torch.tensor(dataset["u"], device=device)
        self.Y = torch.tensor(dataset["Y"], device=device)
        self.s = torch.tensor(dataset["s"], device=device)

        # Ensure correct dimensions
        if self.X.dim() == 2:
            self.X = self.X.unsqueeze(-1)
        if self.u.dim() == 2:
            self.u = self.u.unsqueeze(-1)
        if self.Y.dim() == 2:
            self.Y = self.Y.unsqueeze(-1)
        if self.s.dim() == 2:
            self.s = self.s.unsqueeze(-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (X, u, Y, s)
        """
        return (self.X[idx], self.u[idx], self.Y[idx], self.s[idx])

    def get_info(self):
        """Extract info from model dataset including spatial dimensions for iFNO."""
        # Basic info (existing)
        info = {
            "X_size": self.X.shape[-1],
            "u_size": self.u.shape[-1],
            "Y_size": self.Y.shape[-1],
            "s_size": self.s.shape[-1],
            "X_len": self.X.shape[0],
            "u_len": self.u.shape[0],
            "Y_len": self.Y.shape[0],
            "s_len": self.s.shape[0],
        }
        
        # Note: For iFNO support, datasets should implement custom Dataset classes
        # with hardcoded spatial dimensions in their get_info() methods
        # This generic ModelDataset is kept for backward compatibility
        
        return info


class InputFunctionEncoderDataset(Dataset):
    """Dataset for input function encoder that splits the input domain points."""

    def __init__(self, dataset, device="cpu"):
        """
        Initialize from any base dataset that provides X, u, Y, s tensors.

        Args:
            dataset: Dataset with __getitem__ returning (X, u, Y, s)
            device: The device to put tensors on
        """
        self.device = device

        self.n_samples = len(dataset)

        # self.X = dataset.X
        # self.u = dataset.u
        self.dataset = dataset

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (example_xs, example_ys, xs, ys)
        """
        # X = self.X[idx]
        # u = self.u[idx]
        X, u, Y, s = self.dataset[idx]

        # Do a randperm split
        B = X.shape[0]
        indices = torch.randperm(B, device=self.device)
        cut = B // 2
        example_indices = indices[:cut]
        remaining_indices = indices[cut:]

        example_xs = X[example_indices]
        example_ys = u[example_indices]
        xs = X[remaining_indices]
        ys = u[remaining_indices]

        return (example_xs, example_ys, xs, ys)


class OutputFunctionEncoderDataset(Dataset):
    """Dataset for output function encoder that splits the output domain points."""

    def __init__(self, dataset, device="cpu"):
        """
        Initialize from any base dataset that provides X, u, Y, s tensors.

        Args:
            dataset: Dataset with __getitem__ returning (X, u, Y, s)
            device: The device to put tensors on
        """
        self.device = device

        self.n_samples = len(dataset)

        # self.Y = dataset.Y
        # self.s = dataset.s
        self.dataset = dataset

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (example_xs, example_ys, xs, ys)
        """
        # Y = self.Y[idx]
        # s = self.s[idx]
        X, u, Y, s = self.dataset[idx]

        # Do a randperm split
        B = Y.shape[0]
        indices = torch.randperm(B, device=self.device)
        cut = B // 2
        example_indices = indices[:cut]
        remaining_indices = indices[cut:]

        example_xs = Y[example_indices]
        example_ys = s[example_indices]
        xs = Y[remaining_indices]
        ys = s[remaining_indices]

        return (example_xs, example_ys, xs, ys)


class IterableInputFunctionEncoderDataset(torch.utils.data.IterableDataset):
    """Iterable dataset for input function encoder that works with streaming datasets."""

    def __init__(self, dataset, device="cpu"):
        """
        Initialize from any base dataset that provides X, u, Y, s tensors.

        Args:
            dataset: Iterable dataset with iteration returning (X, u, Y, s)
            device: The device to put tensors on
        """
        self.device = device
        self.dataset = dataset

    def __iter__(self):
        """
        Iterate over the dataset and yield input function encoder samples.

        Yields:
            A tuple of (example_xs, example_ys, xs, ys) where:
            - example_xs are the example input coordinates
            - example_ys are the example input function values
            - xs are the remaining input coordinates
            - ys are the remaining input function values
        """
        for X, u, Y, s in self.dataset:
            # X: input coordinates, u: input function values
            # Split the input domain into examples and remaining points
            n_examples = 5
            total_points = X.shape[0]

            if total_points <= n_examples:
                # If we don't have enough points, use all as examples
                example_indices = torch.arange(total_points, device=self.device)
                remaining_indices = torch.arange(total_points, device=self.device)
            else:
                # Randomly select example indices
                example_indices = torch.randperm(total_points, device=self.device)[:n_examples]
                remaining_indices = torch.arange(total_points, device=self.device)

            example_xs = X[example_indices]
            example_ys = u[example_indices]
            xs = X[remaining_indices]
            ys = u[remaining_indices]

            yield (example_xs, example_ys, xs, ys)


class IterableOutputFunctionEncoderDataset(torch.utils.data.IterableDataset):
    """Iterable dataset for output function encoder that works with streaming datasets."""

    def __init__(self, dataset, device="cpu"):
        """
        Initialize from any base dataset that provides X, u, Y, s tensors.

        Args:
            dataset: Iterable dataset with iteration returning (X, u, Y, s)
            device: The device to put tensors on
        """
        self.device = device
        self.dataset = dataset

    def __iter__(self):
        """
        Iterate over the dataset and yield output function encoder samples.

        Yields:
            A tuple of (example_xs, example_ys, xs, ys) where:
            - example_xs are the example output coordinates
            - example_ys are the example output function values
            - xs are the remaining output coordinates
            - ys are the remaining output function values
        """
        for X, u, Y, s in self.dataset:
            # Y: output coordinates, s: output function values
            # Split the output domain into examples and remaining points
            n_examples = 5
            total_points = Y.shape[0]

            if total_points <= n_examples:
                # If we don't have enough points, use all as examples
                example_indices = torch.arange(total_points, device=self.device)
                remaining_indices = torch.arange(total_points, device=self.device)
            else:
                # Randomly select example indices
                example_indices = torch.randperm(total_points, device=self.device)[:n_examples]
                remaining_indices = torch.arange(total_points, device=self.device)

            example_xs = Y[example_indices]
            example_ys = s[example_indices]
            xs = Y[remaining_indices]
            ys = s[remaining_indices]

            yield (example_xs, example_ys, xs, ys)
