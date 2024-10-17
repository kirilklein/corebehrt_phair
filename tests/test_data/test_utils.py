import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from ehr2vec.data.utils import Utilities, remove_duplicate_indices


class TestUtilities(unittest.TestCase):
    def test_process_datasets(self):
        data = MagicMock()
        data.pids = [1, 2, 3]
        func = Mock(side_effect=lambda x: x)
        func.__name__ = "func"
        Utilities.process_data(data, func)
        func.assert_called_once_with(data)

    def test_log_patient_nums(self):
        pass

    def test_get_abspos_from_origin_point(self):
        timestamps = [datetime(1, 1, i) for i in range(1, 11)]
        origin_point = {"year": 1, "month": 1, "day": 1}
        abspos = Utilities.get_abspos_from_origin_point(timestamps, origin_point)

        self.assertEqual(abspos, [24 * i for i in range(10)])

    def test_get_relative_timestamps_in_hours(self):
        timestamps = pd.Series(pd.to_datetime([f"2020-01-0{i}" for i in range(1, 10)]))
        origin_point = datetime(**{"year": 2020, "month": 1, "day": 1})
        rel_timestamps = Utilities.get_relative_timestamps_in_hours(
            timestamps, origin_point
        )
        expected = [24 * i for i in range(9)]
        np.testing.assert_allclose(rel_timestamps, expected, rtol=1e-5, atol=1e-8)

    def test_check_and_adjust_max_segment(self):
        data = Mock(features={"segment": [[1, 2, 3], [4, 5, 6]]})
        model_config = Mock(type_vocab_size=5)
        Utilities.check_and_adjust_max_segment(data, model_config)

        self.assertEqual(model_config.type_vocab_size, 7)

    def test_get_token_to_index_map(self):
        vocab = {
            "[PAD]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[MASK]": 3,
            "[UNK]": 4,
            "A": 5,
            "B": 6,
        }
        token2index, new_vocab = Utilities.get_token_to_index_map(vocab)
        self.assertEqual(token2index, {5: 0, 6: 1})
        self.assertEqual(new_vocab, {"A": 0, "B": 1})

    def test_get_gender_token(self):
        BG_GENDER_KEYS = {
            "male": ["M", "Mand", "male", "Male", "man", "MAN", "1"],
            "female": ["W", "Kvinde", "F", "female", "Female", "woman", "WOMAN", "0"],
        }
        vocabulary = {"BG_GENDER_Male": 0, "BG_GENDER_Female": 1}
        for gender, values in BG_GENDER_KEYS.items():
            for value in values:
                result = Utilities.get_gender_token(vocabulary, value)
                if gender == "male":
                    self.assertEqual(result, 0)
                elif gender == "female":
                    self.assertEqual(result, 1)

    def test_get_background_indices(self):
        data = Mock(
            vocabulary={
                "[SEP]": -1,
                "BG_Gender": 0,
                "BG_Age": 1,
                "BG_Country": 2,
                "Foo": 3,
            },
            features={"concept": [[0, 1, 3], [0, 1, 3]]},
        )
        data_none = Mock(
            vocabulary={"[SEP]": -1, "Foo": 3}, features={"concept": [[3], [3]]}
        )

        background_indices = Utilities.get_background_indices(data)
        self.assertEqual(background_indices, [0, 1])

        background_indices = Utilities.get_background_indices(data_none)
        self.assertEqual(background_indices, [])

    def test_code_starts_with(self):
        self.assertTrue(Utilities.code_starts_with("123", ("1", "2")))
        self.assertFalse(Utilities.code_starts_with("345", ("1", "2")))

    def test_log_pos_patients_num(self):
        pass

    @patch(
        "os.listdir", return_value=[f"checkpoint_epoch{i}_end.pt" for i in range(10)]
    )
    def test_get_last_checkpoint_epoch(self, mock_listdir):
        last_epoch = Utilities.get_last_checkpoint_epoch("dir")

        self.assertEqual(last_epoch, 9)

    def test_split_train_val(self):
        features = {"concept": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}
        pids = [1, 2, 3]
        val_ratio = 0.35

        train_features, train_pids, val_features, val_pids = Utilities.split_train_val(
            features, pids, val_ratio
        )

        self.assertEqual(train_features, {"concept": [[1, 2, 3], [4, 5, 6]]})
        self.assertEqual(val_features, {"concept": [[7, 8, 9]]})
        self.assertEqual(train_pids, [1, 2])
        self.assertEqual(val_pids, [3])

    def test_iter_patients(self):
        pass

    def test_censor(self):
        patient0 = {"concept": [1, 2, 3], "abspos": [1, 2, 3]}
        patient1 = {"concept": [4, 5, 6], "abspos": [4, 5, 6]}

        result = Utilities.censor(patient0, 2)
        result1 = Utilities.censor(patient1, 10)

        self.assertEqual(result, {"concept": [1, 2], "abspos": [1, 2]})
        self.assertEqual(result1, {"concept": [4, 5, 6], "abspos": [4, 5, 6]})

    def test__generate_censor_flags(self):
        abspos = [1, 2, 3, 4, 5]
        event_timestamp = 3

        result = Utilities._generate_censor_flags(abspos, event_timestamp)

        self.assertEqual(result, [True, True, True, False, False])

    def test_regex_function(self):
        mock_vocabulary = {"A01": 0, "B02": 1, "C03": 2, "A11": 3, "D04": 4, "001": 5}

        regex = r"^A"
        result = Utilities.get_codes_from_regex(mock_vocabulary, regex)
        expected_result = {0, 3}  # Values for "A01" and "A11"
        self.assertEqual(result, expected_result)

        regex = r"\d"
        result = Utilities.get_codes_from_regex(mock_vocabulary, regex)
        expected_result = {5}  # No values expected
        self.assertEqual(result, expected_result)

        regex = r"^X"
        result = Utilities.get_codes_from_regex(mock_vocabulary, regex)
        expected_result = set()  # No matches expected
        self.assertEqual(result, expected_result)

    # Define the unittest class
    class TestRemoveDuplicateIndices(unittest.TestCase):

        def test_remove_duplicate_indices_with_duplicates(self):
            """
            Test that the function removes duplicate indices and logs a warning.
            """
            # Create a DataFrame with duplicate indices
            df = pd.DataFrame(
                {"A": [10, 20, 30, 40], "B": ["a", "b", "c", "d"]}, index=[1, 2, 2, 3]
            )  # Duplicate index at 2

            # Expected DataFrame after removing duplicates
            expected_df = pd.DataFrame(
                {"A": [10, 20, 40], "B": ["a", "b", "d"]}, index=[1, 2, 3]
            )

            with self.assertLogs("effect_estimation", level="WARNING") as log:
                result_df = remove_duplicate_indices(df)

            # Check that the warning was logged
            self.assertTrue(
                any(
                    "Found 1 duplicate indices after merging. Keeping first occurrence."
                    in message
                    for message in log.output
                ),
                "Expected warning message not found in logs.",
            )

            # Check that the duplicates were removed correctly
            pd.testing.assert_frame_equal(result_df, expected_df)

        def test_remove_duplicate_indices_no_duplicates(self):
            """
            Test that the function returns the same DataFrame when there are no duplicates and does not log a warning.
            """
            # Create a DataFrame without duplicate indices
            df = pd.DataFrame(
                {"A": [10, 20, 30], "B": ["a", "b", "c"]}, index=[1, 2, 3]
            )

            expected_df = df.copy()

            with self.assertLogs("effect_estimation", level="WARNING") as log:
                result_df = remove_duplicate_indices(df)

            # Check that no warnings were logged
            self.assertEqual(
                len(log.output),
                0,
                "No warnings should be logged when there are no duplicate indices.",
            )

            # Check that the DataFrame remains unchanged
            pd.testing.assert_frame_equal(result_df, expected_df)

        def test_remove_duplicate_indices_all_duplicates(self):
            """
            Test that the function handles DataFrames where all indices are duplicates.
            """
            # Create a DataFrame where all indices are the same
            df = pd.DataFrame(
                {"A": [10, 20, 30], "B": ["a", "b", "c"]}, index=[1, 1, 1]
            )  # All indices are 1

            # Expected DataFrame after removing duplicates
            expected_df = pd.DataFrame({"A": [10], "B": ["a"]}, index=[1])

            with self.assertLogs("effect_estimation", level="WARNING") as log:
                result_df = remove_duplicate_indices(df)

            # Check that the warning was logged
            self.assertTrue(
                any(
                    "Found 2 duplicate indices after merging. Keeping first occurrence."
                    in message
                    for message in log.output
                ),
                "Expected warning message not found in logs.",
            )

            # Check that only the first occurrence is kept
            pd.testing.assert_frame_equal(result_df, expected_df)

        def test_remove_duplicate_indices_empty_dataframe(self):
            """
            Test that the function handles an empty DataFrame without errors.
            """
            # Create an empty DataFrame
            df = pd.DataFrame(columns=["A", "B"])

            expected_df = df.copy()

            with self.assertLogs("effect_estimation", level="WARNING") as log:
                result_df = remove_duplicate_indices(df)

            # Check that no warnings were logged
            self.assertEqual(
                len(log.output),
                0,
                "No warnings should be logged for an empty DataFrame.",
            )

            # Check that the result is an empty DataFrame
            pd.testing.assert_frame_equal(result_df, expected_df)


if __name__ == "__main__":
    unittest.main()
