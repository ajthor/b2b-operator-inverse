import torch

from datasets import load_dataset
from torch.utils.data import DataLoader

from function_encoder.model.mlp import MLP, MultiHeadedMLP
from function_encoder.function_encoder import FunctionEncoder, BasisFunctions
from function_encoder.losses import basis_normalization_loss
from function_encoder.utils.training import fit

from invertible_network import InvertibleNetwork

import tqdm

import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


torch.manual_seed(42)

# Load dataset

ds = load_dataset("ajthor/derivative_polynomial", split="train")
ds = ds.with_format("torch", device=device)


dataloader = DataLoader(ds, batch_size=50, shuffle=True)


# Define model

n_basis = 8

input_basis_functions = MultiHeadedMLP(layer_sizes=[1, 64, 1], num_heads=n_basis)
input_function_encoder = FunctionEncoder(input_basis_functions)

output_basis_functions = MultiHeadedMLP(layer_sizes=[1, 64, 1], num_heads=n_basis)
output_function_encoder = FunctionEncoder(output_basis_functions)

operator = InvertibleNetwork(input_size=n_basis)


# Train model


epochs = 1000
learning_rate = 1e-3


# Train the input function encoder
def input_loss_function(model, batch):
    X, f = batch["X"], batch["f"]
    X = X.unsqueeze(-1)
    f = f.unsqueeze(-1)

    example_xs = X
    example_ys = f
    xs = X
    ys = f

    coefficients = model.compute_coefficients(example_xs, example_ys)
    y_pred = model(xs, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, ys)
    norm_loss = basis_normalization_loss(model.basis_functions(xs))

    return pred_loss + norm_loss


input_function_encoder = fit(
    model=input_function_encoder,
    ds=dataloader,
    loss_function=input_loss_function,
    epochs=epochs,
    learning_rate=learning_rate,
)


# Train the output function encoder
def output_loss_function(model, batch):
    Y, Tf = batch["Y"], batch["Tf"]
    Y = Y.unsqueeze(-1)
    Tf = Tf.unsqueeze(-1)

    example_xs = Y
    example_ys = Tf
    xs = Y
    ys = Tf

    coefficients = model.compute_coefficients(example_xs, example_ys)
    y_pred = model(xs, coefficients)

    pred_loss = torch.nn.functional.mse_loss(y_pred, ys)
    norm_loss = basis_normalization_loss(model.basis_functions(xs))

    return pred_loss + norm_loss


output_function_encoder = fit(
    model=output_function_encoder,
    ds=dataloader,
    loss_function=output_loss_function,
    epochs=epochs,
    learning_rate=learning_rate,
)


# Train the operator
def operator_loss_function(model, batch):
    X, f, Y, Tf = batch["X"], batch["f"], batch["Y"], batch["Tf"]
    X = X.unsqueeze(-1)
    f = f.unsqueeze(-1)
    Y = Y.unsqueeze(-1)
    Tf = Tf.unsqueeze(-1)

    alpha = input_function_encoder.compute_coefficients(X, f)
    beta = output_function_encoder.compute_coefficients(Y, Tf)

    beta_pred = model(alpha)
    alpha_pred = model.inverse(beta_pred)

    pred_loss = torch.nn.functional.mse_loss(beta_pred, beta)
    reconstruction_loss = torch.nn.functional.mse_loss(alpha_pred, alpha)

    return pred_loss + reconstruction_loss


def train_operator(model, dataloader, epochs, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    with tqdm.tqdm(range(epochs)) as tqdm_bar:
        for epoch in tqdm_bar:

            for batch in dataloader:
                optimizer.zero_grad()
                loss = operator_loss_function(model, batch)
                loss.backward()
                optimizer.step()
                break

            if epoch % 10 == 0:
                tqdm_bar.set_postfix_str(f"Loss {loss.item():.3e}")


train_operator(operator, dataloader, epochs=10000, learning_rate=learning_rate)


# Plot

input_function_encoder.eval()
output_function_encoder.eval()
operator.eval()

point = ds.take(1)[0]

X = point["X"]
f = point["f"]
Y = point["Y"]
Tf = point["Tf"]

idx = torch.argsort(X, dim=0).squeeze()
X = X[idx]
f = f[idx]

idx = torch.argsort(Y, dim=0).squeeze()
Y = Y[idx]
Tf = Tf[idx]

X = X.unsqueeze(-1).unsqueeze(0)
f = f.unsqueeze(-1).unsqueeze(0)
Y = Y.unsqueeze(-1).unsqueeze(0)
Tf = Tf.unsqueeze(-1).unsqueeze(0)

alpha = input_function_encoder.compute_coefficients(X, f)
beta = output_function_encoder.compute_coefficients(Y, Tf)

f_est = input_function_encoder(X, alpha)
Tf_est = output_function_encoder(Y, beta)

beta_pred = operator(alpha)
alpha_pred = operator.inverse(beta)

f_pred = input_function_encoder(X, alpha_pred)
Tf_pred = output_function_encoder(Y, beta_pred)

X = X.squeeze().cpu().detach().numpy()
f = f.squeeze().cpu().detach().numpy()
f_est = f_est.squeeze().cpu().detach().numpy()
f_pred = f_pred.squeeze().cpu().detach().numpy()

Y = Y.squeeze().cpu().detach().numpy()
Tf = Tf.squeeze().cpu().detach().numpy()
Tf_est = Tf_est.squeeze().cpu().detach().numpy()
Tf_pred = Tf_pred.squeeze().cpu().detach().numpy()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(X, f, label="f")
ax[0].plot(X, f_est, label="f_est")
ax[0].plot(X, f_pred, label="f_pred")

ax[1].plot(Y, Tf, label="Tf")
ax[1].plot(Y, Tf_est, label="Tf_est")
ax[1].plot(Y, Tf_pred, label="Tf_pred")

plt.legend()
plt.show()
