import unittest
from typing import Set

from ehr2vec.double_robust.counterfactual import (
    get_frequency_of_codes,
    get_probability_of_codes,
    remove_codes,
    insert_random_code_to_end,
    create_counterfactual_data,
)
from ehr2vec.common.utils import Data


class TestCounterfactuals(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_features = {
            "concept": [
                [1, 2, 3],
                [2, 3, 4],
                [1, 3, 5],
            ],
            "age": [
                [20, 21, 22],
                [30, 31, 32],
                [40, 41, 42],
            ],
        }
        self.exposure_codes: Set[int] = {1, 2, 6}  # Note: 6 doesn't appear in features

    def test_get_frequency_of_codes(self):
        """Test the get_frequency_of_codes function with various scenarios."""
        # Test normal case
        result = get_frequency_of_codes(self.sample_features, self.exposure_codes)
        expected = {1: 2, 2: 2, 6: 0}
        self.assertEqual(result, expected)

        # Test empty features
        empty_features = {"concept": []}
        result = get_frequency_of_codes(empty_features, self.exposure_codes)
        expected = {1: 0, 2: 0, 6: 0}
        self.assertEqual(result, expected)

        # Test empty exposure codes
        result = get_frequency_of_codes(self.sample_features, set())
        self.assertEqual(result, {})

    def test_get_probability_of_codes(self):
        """Test the get_probability_of_codes function."""
        code_frequencies = {1: 2, 2: 2, 6: 0}
        result = get_probability_of_codes(code_frequencies)

        total = sum(code_frequencies.values()) + 1  # +1 for smoothing
        expected = {1: 2 / total, 2: 2 / total, 6: 0 / total}

        self.assertEqual(result, expected)

    def test_remove_codes(self):
        """Test the remove_codes function."""
        patient = {
            "concept": [1, 2, 3, 4],
            "age": [20, 21, 22, 23],
        }
        codes_to_remove = [1, 2]

        result = remove_codes(patient, codes_to_remove)
        expected = {
            "concept": [3, 4],
            "age": [22, 23],
        }

        self.assertEqual(result, expected)

    def test_insert_random_code_to_end(self):
        """Test the insert_random_code_to_end function."""
        patient = {
            "concept": [3, 4],
            "age": [20, 21],
        }
        exposure_codes = {1, 2}
        code_probabilities = {1: 0.6, 2: 0.4}

        result = insert_random_code_to_end(patient, exposure_codes, code_probabilities)

        # Check structure
        self.assertEqual(len(result["concept"]), len(patient["concept"]) + 1)
        self.assertEqual(len(result["age"]), len(patient["age"]) + 1)

        # Check last age value is repeated
        self.assertEqual(result["age"][-1], patient["age"][-1])

        # Check inserted code is from exposure_codes
        self.assertIn(result["concept"][-1], exposure_codes)

    def test_create_counterfactual_data(self):
        """Test the create_counterfactual_data function."""
        original_data = Data(
            features=self.sample_features,
            pids=["p1", "p2", "p3"],
            outcomes=[0, 1, 0],
            vocabulary={"code1": 1, "code2": 2, "code3": 3, "code4": 4, "code5": 5},
        )
        exposure_regex_list = [
            "code1",
            "code[35]",
        ]  # This should match codes 1, 3 and 5
        exposure_codes = {1, 3, 5}
        result = create_counterfactual_data(original_data, exposure_regex_list)

        # Check that the structure is preserved
        self.assertEqual(result.pids, original_data.pids)
        self.assertEqual(result.outcomes, original_data.outcomes)
        self.assertEqual(result.vocabulary, original_data.vocabulary)

        for orig_seq, new_seq in zip(
            original_data.features["concept"], result.features["concept"]
        ):
            orig_has_exposure = any(code in exposure_codes for code in orig_seq)
            new_has_exposure = any(code in exposure_codes for code in new_seq)

            if orig_has_exposure:
                # If original had exposure:
                # 1. New sequence should not have any exposure codes
                self.assertFalse(new_has_exposure)
                # 2. New sequence should be shorter by the number of exposure codes
                orig_exposure_count = sum(
                    1 for code in orig_seq if code in exposure_codes
                )
                self.assertEqual(len(new_seq), len(orig_seq) - orig_exposure_count)
            else:
                # If original had no exposure:
                # 1. New sequence should have exactly one exposure code
                self.assertTrue(new_has_exposure)
                # 2. New sequence should be longer by 1
                self.assertEqual(len(new_seq), len(orig_seq) + 1)
                # 3. The added code should be one of the exposure codes
                self.assertIn(new_seq[-1], exposure_codes)
                # 4. All codes except the last one should match the original
                self.assertEqual(new_seq[:-1], orig_seq)


if __name__ == "__main__":
    unittest.main()
