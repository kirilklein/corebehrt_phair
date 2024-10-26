import os
import shutil
import unittest
from unittest.mock import MagicMock, patch

from ehr2vec.common.config import Config
from ehr2vec.common.utils import Data
from ehr2vec.data.prepare_data import DatasetPreparer


class TestDatasetPreparerIndexDates(unittest.TestCase):
    def setUp(self):
        # Create a mock configuration
        self.mock_config = MagicMock(spec=Config)
        self.mock_config.paths = MagicMock()
        self.mock_config.paths.predefined_splits = "mock/path/splits"
        self.mock_config.outcome = MagicMock()
        self.mock_config.outcome.n_hours_censoring = -12  # New censoring time

        # Initialize DatasetPreparer with the mock config
        self.dataset_preparer = DatasetPreparer(self.mock_config)

    def tearDown(self):
        # Remove the temporary test directory after each test
        if os.path.exists("MagicMock"):
            shutil.rmtree("MagicMock")

    @patch("ehr2vec.data.prepare_data.load_config")
    def test_load_predefined_split_config(self, mock_load_config):
        # Arrange
        expected_config = MagicMock(spec=Config)
        mock_load_config.return_value = expected_config

        # Act
        result = self.dataset_preparer._load_predefined_split_config()

        # Assert
        mock_load_config.assert_called_once_with(
            os.path.join("mock/path/splits", "finetune_config.yaml")
        )
        self.assertEqual(result, expected_config)

    def test_calculate_censoring_delta(self):
        # Arrange
        predefined_config = MagicMock(spec=Config)
        predefined_config.outcome = MagicMock()
        predefined_config.outcome.n_hours_censoring = -24  # Original censoring time

        # Act
        delta = self.dataset_preparer._calculate_censoring_delta(predefined_config)

        # Assert
        # New censoring (-12) minus predefined censoring (-24) should be 12
        self.assertEqual(delta, 12)

    def test_validate_censoring_delta_valid(self):
        # Arrange
        valid_delta = 12  # Positive delta is valid

        # Act & Assert
        try:
            self.dataset_preparer._validate_censoring_delta(valid_delta)
        except ValueError:
            self.fail("_validate_censoring_delta raised ValueError unexpectedly!")

    def test_validate_censoring_delta_invalid(self):
        # Arrange
        invalid_delta = -12  # Negative delta should raise error

        # Act & Assert
        with self.assertRaises(ValueError) as context:
            self.dataset_preparer._validate_censoring_delta(invalid_delta)

        self.assertEqual(
            str(context.exception),
            "New censoring time must be later than the predefined one.",
        )

    @patch("ehr2vec.data.prepare_data.DatasetPreparer._load_predefined_split_config")
    def test_adjust_predefined_index_dates(self, mock_load_config):
        # Arrange
        mock_data = MagicMock(spec=Data)
        mock_data.index_dates = [100, 200, 300]  # Original index dates

        predefined_config = MagicMock(spec=Config)
        predefined_config.outcome = MagicMock()
        predefined_config.outcome.n_hours_censoring = -24  # Original censoring time
        mock_load_config.return_value = predefined_config

        # Act
        result = self.dataset_preparer._adjust_predefined_index_dates(mock_data)

        # Assert
        # Delta should be 12 (-12 - (-24) = 12)
        expected_dates = [112, 212, 312]  # Original dates + 12
        self.assertEqual(result.index_dates, expected_dates)

    @patch("ehr2vec.data.prepare_data.DatasetPreparer._load_predefined_split_config")
    def test_adjust_predefined_index_dates_invalid_censoring(self, mock_load_config):
        # Arrange
        mock_data = MagicMock(spec=Data)
        mock_data.index_dates = [100, 200, 300]

        predefined_config = MagicMock(spec=Config)
        predefined_config.outcome = MagicMock()
        predefined_config.outcome.n_hours_censoring = -12  # Original censoring time
        self.mock_config.outcome.n_hours_censoring = -24
        mock_load_config.return_value = predefined_config
        # Act & Assert
        # New censoring (-24) is earlier than original (-12), should raise error
        with self.assertRaises(ValueError) as context:
            self.dataset_preparer._adjust_predefined_index_dates(mock_data)

        self.assertEqual(
            str(context.exception),
            "New censoring time must be later than the predefined one.",
        )


if __name__ == "__main__":
    unittest.main()
