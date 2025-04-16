import torch

import tqdm


class LinearB2BOperator(torch.nn.Module):
    """
    Linear operator for the inverse problem of parameter estimation.
    """

    def __init__(self, input_dim, output_dim):
        super(LinearB2BOperator, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear.weight.requires_grad = False

    def forward(self, alpha):
        return self.linear(alpha)

    def inverse(self, beta):
        """Compute the inverse of the linear operator."""
        return torch.linalg.solve(self.linear.weight, beta)


def train(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    input_function_encoder,
    output_function_encoder,
    n_epochs,
    summary_writer,
    model_name,
    params,
    device,
):

    n = params.input_fe_n_basis
    m = params.output_fe_n_basis

    SXX = torch.zeros((n, n), device=device)
    SXY = torch.zeros((n, m), device=device)

    with torch.no_grad():
        tqdm_bar = tqdm.tqdm(len(train_dataloader))
        for batch in train_dataloader:

            # Compute the alpha and beta coefficients
            X = batch["X"]
            u = batch["u"]
            Y = batch["Y"]
            s = batch["s"]
            alpha = input_function_encoder.compute_coefficients(X, u)
            beta = output_function_encoder.compute_coefficients(Y, s)

            # Compute the normal equations in chunks
            SXX += torch.einsum("ij,ik->jk", alpha, alpha)
            SXY += torch.einsum("ij,ik->jk", alpha, beta)

            tqdm_bar.update(1)

        # Add small regularization term to SXX
        SXX += 1e-6 * torch.eye(n, device=device)

        # Compute the linear operator
        W = torch.linalg.solve(SXX, SXY)

        model.linear.weight.copy_(W.T)
