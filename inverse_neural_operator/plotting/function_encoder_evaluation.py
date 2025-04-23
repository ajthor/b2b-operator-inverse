import argparse

import numpy as np

import torch
from torch.utils.data import Subset, DataLoader

import matplotlib.pyplot as plt


def plot_function_encoder_evaluations(
    model,
    dataset,
    evaluate_instance,
    file_name="results/function_encoder_evaluation.png",
):
    # Get 9 random points from the dataset
    indices = np.random.choice(len(dataset), 9, replace=False)
    subset = Subset(dataset, indices)

    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
    )

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    for i, point in enumerate(dataloader):
        pred = evaluate_instance(model, point)
        pred = pred.squeeze(0).cpu().numpy()

        example_xs, example_ys, xs, ys = point
        example_xs = example_xs.squeeze(0).cpu().numpy()
        example_ys = example_ys.squeeze(0).cpu().numpy()
        xs = xs.squeeze(0).cpu().numpy()
        ys = ys.squeeze(0).cpu().numpy()

        # Sort by the xs
        sort_indices = np.argsort(xs, axis=0)
        xs = xs[sort_indices.squeeze()]
        ys = ys[sort_indices.squeeze()]
        pred = pred[sort_indices.squeeze()]

        # Plot the input data
        axs[i // 3, i % 3].scatter(
            example_xs, example_ys, label="Example Data", color="gray", alpha=0.5
        )
        axs[i // 3, i % 3].plot(xs, ys, label="True")
        axs[i // 3, i % 3].plot(xs, pred, label="Prediction")

    plt.tight_layout()

    plt.savefig(file_name)
    plt.close()
