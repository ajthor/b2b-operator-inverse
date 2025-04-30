import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader


def evaluate_random(
    model,
    dataset,
    input_function_encoder=None,
    output_function_encoder=None,
    n_samples=1,
    model_type="standard",
):
    """
    Evaluate a random case from the dataset.

    Args:
        model: The model to evaluate
        dataset: The dataset to sample from
        input_function_encoder: The encoder for input functions
        output_function_encoder: The encoder for output functions
        n_samples: Number of samples to generate (for variational models)
        model_type: Type of model ("standard", "variational", "deeponet")

    Returns:
        point: The sampled data point
        pred: The predicted function
        alpha_pred: The predicted coefficients
        loss: The loss value
    """
    model.eval()
    with torch.no_grad():
        index = np.random.choice(len(dataset), 1, replace=False)[0]
        subset = Subset(dataset, [index])

        dataloader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
        )

        for point in dataloader:
            pred, alpha_pred = evaluate(
                model,
                point,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                n_samples=n_samples,
                model_type=model_type,
            )

            loss = loss_function(
                model=model,
                batch=point,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                model_type=model_type,
            )

            return point, pred, alpha_pred, loss


def find_best_case(
    model,
    dataset,
    input_function_encoder=None,
    output_function_encoder=None,
    n_samples=1,
    model_type="standard",
):
    """
    Find the best case (lowest loss) from the dataset.

    Args:
        model: The model to evaluate
        dataset: The dataset to search through
        input_function_encoder: The encoder for input functions
        output_function_encoder: The encoder for output functions
        n_samples: Number of samples to generate (for variational models)
        model_type: Type of model ("standard", "variational", "deeponet")

    Returns:
        best_case: The data point with the lowest loss
        best_pred: The corresponding prediction
        best_alpha_pred: The corresponding predicted coefficients
        best_case_loss: The lowest loss value
    """
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
        best_case = None
        best_case_loss = float("inf")
        best_case_index = -1
        best_pred = None
        best_alpha_pred = None

        for i, point in enumerate(dataloader):
            loss = loss_function(
                model=model,
                batch=point,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                model_type=model_type,
            )
            if loss < best_case_loss:
                best_case_loss = loss
                best_case = point
                best_case_index = i

                # Calculate prediction for the best case
                best_pred, best_alpha_pred = evaluate(
                    model,
                    point,
                    input_function_encoder=input_function_encoder,
                    output_function_encoder=output_function_encoder,
                    n_samples=n_samples,
                    model_type=model_type,
                )

        return best_case, best_pred, best_alpha_pred, best_case_loss


def find_worst_case(
    model,
    dataset,
    input_function_encoder=None,
    output_function_encoder=None,
    n_samples=1,
    model_type="standard",
):
    """
    Find the worst case (highest loss) from the dataset.

    Args:
        model: The model to evaluate
        dataset: The dataset to search through
        input_function_encoder: The encoder for input functions
        output_function_encoder: The encoder for output functions
        n_samples: Number of samples to generate (for variational models)
        model_type: Type of model ("standard", "variational", "deeponet")

    Returns:
        worst_case: The data point with the highest loss
        worst_pred: The corresponding prediction
        worst_alpha_pred: The corresponding predicted coefficients
        worst_case_loss: The highest loss value
    """
    model.eval()
    with torch.no_grad():
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
        )
        worst_case = None
        worst_case_loss = float("-inf")
        worst_case_index = -1
        worst_pred = None
        worst_alpha_pred = None

        for i, point in enumerate(dataloader):
            loss = loss_function(
                model=model,
                batch=point,
                input_function_encoder=input_function_encoder,
                output_function_encoder=output_function_encoder,
                model_type=model_type,
            )
            if loss > worst_case_loss:
                worst_case_loss = loss
                worst_case = point
                worst_case_index = i

                # Calculate prediction for the worst case
                worst_pred, worst_alpha_pred = evaluate(
                    model,
                    point,
                    input_function_encoder=input_function_encoder,
                    output_function_encoder=output_function_encoder,
                    n_samples=n_samples,
                    model_type=model_type,
                )

        return worst_case, worst_pred, worst_alpha_pred, worst_case_loss
