import unittest
from typing import Set

from ehr2vec.double_robust.counterfactual import (
    get_frequency_of_codes,
    get_probability_of_codes,
    create_counterfactual_data,
    swap_codes,
)
from ehr2vec.common.utils import Data


class TestCounterfactuals(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sample_features = {
            "concept": [
                [1, 2, 3, 5],  # ends with exposure code
                [2, 3, 4, 9],  # ends with control code
                [1, 3, 4, 4],  # ends with neither
                [1, 5, 3],
            ],
            "age": [[20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43], [50, 51, 52]],
        }
        self.exposure_codes: Set[int] = {5, 6}  # 6 doesn't appear in features
        self.control_code = 9

    def test_create_counterfactual_data(self):
        """Test the create_counterfactual_data function."""
        original_data = Data(
            features=self.sample_features,
            pids=["p1", "p2", "p3"],
            outcomes=[0, 1, 0],
            vocabulary={"code4": 4, "code5": 5, "code6": 6, "code9": 9},
        )
        exposure_regex_list = ["code[56]"]  # This should match codes 5 and 6
        result = create_counterfactual_data(
            original_data, exposure_regex_list, control_regex="code9"
        )

        # Check that the structure is preserved
        self.assertEqual(result.pids, original_data.pids)
        self.assertEqual(result.outcomes, original_data.outcomes)
        self.assertEqual(result.vocabulary, original_data.vocabulary)

        # Check sequence transformations
        orig_seqs = original_data.features["concept"]
        new_seqs = result.features["concept"]
        # First sequence: ends with exposure code (5) -> should end with control code (9)
        self.assertEqual(new_seqs[0][-1], self.control_code)
        self.assertEqual(new_seqs[0][:-1], orig_seqs[0][:-1])

        # Second sequence: ends with control code (9) -> should end with exposure code (5 or 6)
        self.assertIn(new_seqs[1][-1], self.exposure_codes)
        self.assertEqual(new_seqs[1][:-1], orig_seqs[1][:-1])

        # Third sequence: ends with neither -> should remain unchanged
        self.assertEqual(new_seqs[2], orig_seqs[2])

        # Check that age features remain unchanged
        self.assertEqual(result.features["age"], original_data.features["age"])

        # last sequence has exposure code in the middle
        self.assertEqual(new_seqs[3][0], orig_seqs[3][0])
        self.assertEqual(new_seqs[3][1], self.control_code)
        self.assertEqual(new_seqs[3][2], orig_seqs[3][2])

    def test_get_frequency_of_codes(self):
        """Test the get_frequency_of_codes function with various scenarios."""
        # Test normal case
        result = get_frequency_of_codes(self.sample_features, self.exposure_codes)
        expected = {6: 0, 5: 2}
        self.assertEqual(result, expected)

        # Test empty features
        empty_features = {"concept": []}
        result = get_frequency_of_codes(empty_features, self.exposure_codes)
        expected = {5: 0, 6: 0}
        self.assertEqual(result, expected)

        # Test empty exposure codes
        result = get_frequency_of_codes(self.sample_features, set())
        self.assertEqual(result, {})

    def test_get_probability_of_codes(self):
        """Test the get_probability_of_codes function."""
        code_frequencies = {1: 2, 2: 2, 6: 0, 5: 2}
        result = get_probability_of_codes(code_frequencies)

        total = sum(code_frequencies.values()) + 1  # +1 for smoothing
        expected = {1: 2 / total, 2: 2 / total, 6: 0 / total, 5: 2 / total}

        self.assertEqual(result, expected)

    def test_swap_codes(self):
        # Test setup
        concepts = [1, 2, 3, 4, 5]
        exposure_code_probabilities = {5: 0.7, 6: 0.3}
        control_code = 9

        # Test case 1: Exposure code at the end
        result = swap_codes(concepts, exposure_code_probabilities, control_code)
        self.assertEqual(result[-1], control_code)
        self.assertEqual(result[:-1], concepts[:-1])

        # Test case 2: Control code at the end
        concepts_with_control = [1, 2, 3, 4, 9]
        result = swap_codes(
            concepts_with_control, exposure_code_probabilities, control_code
        )
        self.assertIn(result[-1], {5, 6})  # Should be one of the exposure codes
        self.assertEqual(result[:-1], concepts_with_control[:-1])

        # Test case 3: No relevant codes
        concepts_no_codes = [1, 2, 3, 4]
        result = swap_codes(
            concepts_no_codes, exposure_code_probabilities, control_code
        )
        self.assertEqual(result, concepts_no_codes)

        # Test case 4: Relevant code in middle (should still only swap last occurrence)
        concepts_middle_code = [1, 5, 3, 4, 5]
        result = swap_codes(
            concepts_middle_code, exposure_code_probabilities, control_code
        )
        self.assertEqual(result[-1], control_code)
        self.assertEqual(result[1], 5)  # Earlier 5 should remain unchanged


if __name__ == "__main__":
    unittest.main()
