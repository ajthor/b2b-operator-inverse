import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


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
        """Extract info from model dataset."""
        return {
            "X_size": self.X.shape[-1],
            "u_size": self.u.shape[-1],
            "Y_size": self.Y.shape[-1],
            "s_size": self.s.shape[-1],
            "X_len": self.X.shape[0],
            "u_len": self.u.shape[0],
            "Y_len": self.Y.shape[0],
            "s_len": self.s.shape[0],
        }


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

        self.X = dataset.X
        self.u = dataset.u

        # # Do a randperm split
        # B = X.shape[0]
        # N = X.shape[1]
        # idx = torch.multinomial(torch.ones(B, N, device=device), N, replacement=False)

        # X = X.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, X.shape[2]))
        # u = u.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, u.shape[2]))

        # cut = N // 2
        # n1, n2 = cut, N - cut

        # self.example_xs, self.xs = X.split([n1, n2], dim=1)
        # self.example_ys, self.ys = u.split([n1, n2], dim=1)

        # # Ensure correct dimensions
        # if self.example_xs.dim() == 2:
        #     self.example_xs = self.example_xs.unsqueeze(-1)
        # if self.example_ys.dim() == 2:
        #     self.example_ys = self.example_ys.unsqueeze(-1)
        # if self.xs.dim() == 2:
        #     self.xs = self.xs.unsqueeze(-1)
        # if self.ys.dim() == 2:
        #     self.ys = self.ys.unsqueeze(-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (example_xs, example_ys, xs, ys)
        """
        X = self.X[idx]
        u = self.u[idx]

        # Do a randperm split
        B = X.shape[0]
        idx = torch.randperm(B, device=self.device)
        cut = B // 2
        example_indices = idx[:cut]
        remaining_indices = idx[cut:]

        example_xs = X[example_indices]
        example_ys = u[example_indices]
        xs = X[remaining_indices]
        ys = u[remaining_indices]

        # # Ensure correct dimensions
        # if example_xs.dim() == 2:
        #     example_xs = example_xs.unsqueeze(-1)
        # if example_ys.dim() == 2:
        #     example_ys = example_ys.unsqueeze(-1)
        # if xs.dim() == 2:
        #     xs = xs.unsqueeze(-1)
        # if ys.dim() == 2:
        #     ys = ys.unsqueeze(-1)

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

        self.Y = dataset.Y
        self.s = dataset.s

        # # Do a randperm split
        # B = Y.shape[0]
        # N = Y.shape[1]
        # idx = torch.multinomial(torch.ones(B, N, device=device), N, replacement=False)

        # Y = Y.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, Y.shape[2]))
        # s = s.gather(dim=1, index=idx.unsqueeze(-1).expand(-1, -1, s.shape[2]))

        # cut = N // 2
        # n1, n2 = cut, N - cut

        # self.example_xs, self.xs = Y.split([n1, n2], dim=1)
        # self.example_ys, self.ys = s.split([n1, n2], dim=1)

        # # Ensure correct dimensions
        # if self.example_xs.dim() == 2:
        #     self.example_xs = self.example_xs.unsqueeze(-1)
        # if self.example_ys.dim() == 2:
        #     self.example_ys = self.example_ys.unsqueeze(-1)
        # if self.xs.dim() == 2:
        #     self.xs = self.xs.unsqueeze(-1)
        # if self.ys.dim() == 2:
        #     self.ys = self.ys.unsqueeze(-1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns:
            A tuple of (example_xs, example_ys, xs, ys)
        """
        Y = self.Y[idx]
        s = self.s[idx]

        # Do a randperm split
        B = Y.shape[0]
        idx = torch.randperm(B, device=self.device)
        cut = B // 2
        example_indices = idx[:cut]
        remaining_indices = idx[cut:]

        example_xs = Y[example_indices]
        example_ys = s[example_indices]
        xs = Y[remaining_indices]
        ys = s[remaining_indices]

        # # Ensure correct dimensions
        # if example_xs.dim() == 2:
        #     example_xs = example_xs.unsqueeze(-1)
        # if example_ys.dim() == 2:
        #     example_ys = example_ys.unsqueeze(-1)
        # if xs.dim() == 2:
        #     xs = xs.unsqueeze(-1)
        # if ys.dim() == 2:
        #     ys = ys.unsqueeze(-1)

        return (example_xs, example_ys, xs, ys)
