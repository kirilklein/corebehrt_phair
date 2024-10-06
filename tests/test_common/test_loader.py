import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
from ehr2vec.common.config import Config
from ehr2vec.common.loader import (
    load_checkpoint_and_epoch,
    load_model_cfg_from_checkpoint,
    FeaturesLoader,
    load_propensities,
    load_outcomes,
)


class TestLoaders(unittest.TestCase):
    @patch("ehr2vec.common.loader.load_predictions_from_finetune_dir")
    def test_load_propensities(self, mock_load_predictions):
        # Set up mock return value
        mock_data = pd.DataFrame(
            {"pid": [1, 2, 3], "target": [0, 1, 0], "proba": [0.2, 0.8, 0.5]}
        )
        mock_load_predictions.return_value = mock_data

        # Run function
        result = load_propensities("test_folder")

        # Check expected outcome
        expected_result = mock_data.rename(
            columns={"pid": "PID", "target": "treatment", "proba": "ps"}
        ).set_index("PID")
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("pandas.read_csv")
    def test_load_outcomes(self, mock_read_csv):
        # Set up mock return value
        mock_data = pd.DataFrame(
            {"PID": [1, 2, 3], "TIMESTAMP": ["2023-01-01", "2023-01-02", "2023-01-03"]}
        )
        mock_read_csv.return_value = mock_data

        # Run function
        result = load_outcomes("test_outcome_path")

        # Check expected outcome
        expected_result = mock_data.set_index("PID")
        expected_result["outcome"] = 1
        expected_result = expected_result.drop(columns=["TIMESTAMP"])
        pd.testing.assert_frame_equal(result, expected_result)

    @patch("ehr2vec.common.loader.ModelLoader.load_checkpoint")
    def test_load_checkpoint_and_epoch(self, mock_load_checkpoint):
        # Set up mock config and return values
        mock_cfg = MagicMock(spec=Config)
        mock_cfg.paths = MagicMock()
        mock_cfg.paths.model_path = "test_model_path"
        mock_checkpoint = {"epoch": 5}
        mock_load_checkpoint.return_value = mock_checkpoint

        # Run function
        checkpoint, epoch = load_checkpoint_and_epoch(mock_cfg)

        # Check expected outcome
        self.assertEqual(checkpoint, mock_checkpoint)
        self.assertEqual(epoch, 5)

    @patch("ehr2vec.common.loader.load_config")
    def test_load_model_cfg_from_checkpoint(self, mock_load_config):
        # Set up mock config
        mock_cfg = MagicMock(spec=Config)
        mock_cfg.paths = MagicMock()
        mock_cfg.paths.model_path = "test_model_path"
        mock_old_cfg = MagicMock(spec=Config)
        mock_old_cfg.model = "old_model_config"
        mock_load_config.return_value = mock_old_cfg

        # Run function
        result = load_model_cfg_from_checkpoint(mock_cfg, "config_name")

        # Check expected outcome
        self.assertTrue(result)
        self.assertEqual(mock_cfg.model, "old_model_config")

    @patch("torch.load")
    @patch("os.path.exists")
    def test_features_loader_load_vocabulary(self, mock_exists, mock_torch_load):
        # Set up mock config
        mock_cfg = MagicMock()
        mock_cfg.paths = {"data_path": "test_data_path"}
        mock_exists.return_value = True
        mock_torch_load.return_value = "vocabulary"

        # Create instance of FeaturesLoader
        loader = FeaturesLoader(mock_cfg)

        # Run function
        result = loader.load_vocabulary("tokenized_data_path")

        # Construct the expected path dynamically
        expected_path = os.path.join("tokenized_data_path", "vocabulary.pt")

        # Assert mocks were called with expected arguments
        mock_exists.assert_called_once_with(expected_path)
        mock_torch_load.assert_called_once_with(expected_path)

        # Check expected outcome
        self.assertEqual(result, "vocabulary")


if __name__ == "__main__":
    unittest.main()
