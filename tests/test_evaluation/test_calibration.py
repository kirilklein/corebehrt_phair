import unittest
import pandas as pd
import torch
import os
import tempfile
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from ehr2vec.evaluation.calibration import train_calibrator, calibrate_data


class TestCalibrationPipeline(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.finetune_folder = self.test_dir.name

        # Create sample data
        self.predictions_data = {
            "pid": [1, 2, 3, 4, 5, 6, 7, 8],
            "proba": [0.1, 0.4, 0.8, 0.7, 0.3, 0.5, 0.9, 0.6],
            "target": [0, 1, 1, 0, 0, 1, 1, 0],
        }
        self.predictions_df = pd.DataFrame(self.predictions_data)
        self.predictions_df.to_csv(
            os.path.join(self.finetune_folder, "predictions_and_targets.csv"),
            index=False,
        )

        # Create fold folders and PID files
        self.n_splits = 2
        for fold in range(1, self.n_splits + 1):
            fold_folder = os.path.join(self.finetune_folder, f"fold_{fold}")
            os.makedirs(fold_folder, exist_ok=True)

            train_pids = (
                torch.tensor([1, 2, 3, 4]) if fold == 1 else torch.tensor([5, 6, 7, 8])
            )
            val_pids = (
                torch.tensor([5, 6, 7, 8]) if fold == 1 else torch.tensor([1, 2, 3, 4])
            )

            torch.save(train_pids, os.path.join(fold_folder, "train_pids.pt"))
            torch.save(val_pids, os.path.join(fold_folder, "val_pids.pt"))

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    def test_train_calibrator(self):
        # Test isotonic calibration
        if not self.predictions_df.empty:
            calibrator = train_calibrator(self.predictions_df, method="isotonic")
            self.assertIsInstance(calibrator, IsotonicRegression)

        # Test sigmoid calibration
        if not self.predictions_df.empty:
            calibrator = train_calibrator(self.predictions_df, method="sigmoid")
            self.assertIsInstance(calibrator, LogisticRegression)

        # Test invalid method
        with self.assertRaises(ValueError):
            train_calibrator(self.predictions_df, method="invalid")

    def test_calibrate_data(self):
        # Train a calibrator
        if not self.predictions_df.empty:
            calibrator = train_calibrator(self.predictions_df, method="isotonic")

            # Calibrate validation data
            val_data = self.predictions_df.iloc[:4]
            calibrated_val_data = calibrate_data(calibrator, val_data)

            # Check if the calibrated data has the same number of rows
            self.assertEqual(len(calibrated_val_data), len(val_data))
            # Check if the 'proba' column has been updated
            self.assertFalse((calibrated_val_data["proba"] == val_data["proba"]).all())


if __name__ == "__main__":
    unittest.main()
