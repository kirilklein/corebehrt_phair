from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_and_save_hist(
    tensor_data: torch.Tensor,
    name: str,
    split: str,
    folder: str,
    positive_indices: list = None,
    density=True,
) -> None:
    """Plot and save histogram of tensor_data to folder with name split_name.png.
    name: str: Name of the tensor_data
    split: str: Name of the split (train, val, test)
    folder: str: Folder to save the histogram to
    positive_indices: list: Indices of positive samples in tensor_data
    density: bool: If True, plot density histogram
    """
    fig, ax = plt.subplots()
    if positive_indices:
        bins = np.histogram_bin_edges(tensor_data, bins=50)
        negative_indices = [
            i for i in range(len(tensor_data)) if i not in positive_indices
        ]
        ax.hist(
            tensor_data[negative_indices].cpu().numpy(),
            bins=bins,
            color="b",
            alpha=0.5,
            label="negative",
            density=density,
        )
        ax.hist(
            tensor_data[positive_indices].cpu().numpy(),
            bins=bins,
            color="r",
            alpha=0.5,
            label="positive",
            density=density,
        )
        ax.legend()
    else:
        ax.hist(tensor_data, bins=50)
    ax.set_xlabel(name)
    ax.set_title(f"{split} {name}")
    fig.savefig(join(folder, f"{split}_{name}.png"), dpi=150, bbox_inches="tight")
