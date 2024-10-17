from os.path import join

import numpy as np
import pandas as pd
import torch


def save_combined_predictions_evaluation(
    n_splits: int, evaluation_folder: str, mode="val"
) -> None:
    """Combine predictions from all folds and save to finetune folder."""
    predictions = []
    pids = []
    for fold in range(1, n_splits + 1):
        fold_folder = join(evaluation_folder, f"fold_{fold}")

        fold_pids = torch.load(join(fold_folder, f"{mode}_pids.pt"))
        fold_predictions = np.load(
            join(fold_folder, f"probas_{mode}.npz"), allow_pickle=True
        )["probas"]

        predictions.append(fold_predictions)
        pids.extend(fold_pids)

    predictions = np.concatenate(predictions).flatten()

    df = pd.DataFrame({"PID": pids, "proba": predictions})
    df.to_csv(join(evaluation_folder, "counterfactual_predictions.csv"), index=False)
