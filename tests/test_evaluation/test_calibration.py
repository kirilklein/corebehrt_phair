import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ehr2vec.common.default_args import ORG_PID_COL, PROBA_COL, TARGET_COL
from ehr2vec.evaluation.calibration import calibrate_data, train_calibrator


class TestCalibrationPipeline(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory
        self.test_dir = tempfile.TemporaryDirectory()
        self.finetune_folder = self.test_dir.name

        # Create sample data
        self.predictions_data = {
            ORG_PID_COL: [1, 2, 3, 4, 5, 6, 7, 8],
            PROBA_COL: [0.1, 0.4, 0.8, 0.7, 0.3, 0.5, 0.9, 0.6],
            TARGET_COL: [0, 1, 1, 0, 0, 1, 1, 0],
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

    def test_train_calibrator_isotonic(self):
        """Test training isotonic calibrator"""
        if not self.predictions_df.empty:
            calibrator = train_calibrator(self.predictions_df, method="isotonic")
            self.assertIsInstance(calibrator, IsotonicRegression)

            # Test predictions are monotonic
            test_inputs = [0.1, 0.3, 0.5, 0.7, 0.9]
            predictions = calibrator.predict(test_inputs)
            self.assertTrue(all(x <= y for x, y in zip(predictions, predictions[1:])))

    def test_train_calibrator_sigmoid(self):
        """Test training sigmoid calibrator"""
        if not self.predictions_df.empty:
            calibrator = train_calibrator(self.predictions_df, method="sigmoid")
            self.assertIsInstance(calibrator, LogisticRegression)

            # Test predictions are in valid range
            test_inputs = np.array([0.1, 0.3, 0.5, 0.7, 0.9]).reshape(-1, 1)
            predictions = calibrator.predict_proba(test_inputs)[:, 1]
            self.assertTrue(all(0 <= p <= 1 for p in predictions))

    def test_train_calibrator_invalid(self):
        """Test invalid calibration method raises error"""
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
            self.assertFalse(
                (calibrated_val_data[PROBA_COL] == val_data[PROBA_COL]).all()
            )


if __name__ == "__main__":
    unittest.main()
