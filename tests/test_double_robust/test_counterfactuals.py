import unittest
from ehr2vec.common.utils import Data
from ehr2vec.double_robust.counterfactual import (
    CounterfactualGenerator,
    swap_codes,
    get_probability_of_codes,
)


class TestCounterfactuals(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.vocabulary = {
            "drug_A": 5,
            "drug_B": 6,
            "control": 9,
            "other": 1,
        }
        self.sample_features = {
            "concept": [
                [1, 2, 3, 5],  # ends with exposure code
                [2, 3, 4, 9],  # ends with control code
                [1, 3, 4, 4],  # ends with neither
                [1, 5, 3],  # has exposure code
            ],
            "age": [[20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43], [50, 51, 52]],
        }
        self.data = Data(
            features=self.sample_features,
            pids=["p1", "p2", "p3", "p4"],
            outcomes=[0, 1, 0, 0],
            vocabulary=self.vocabulary,
            exposures=[1, 0, 0, 1],
        )

    def test_simple_exposure_flip(self):
        """Test counterfactual generation with no regex (simple exposure flip)."""
        generator = CounterfactualGenerator(data=self.data)
        result = generator.generate()

        # Check that structure is preserved
        self.assertEqual(result.pids, self.data.pids)
        self.assertEqual(result.outcomes, self.data.outcomes)
        self.assertEqual(result.vocabulary, self.data.vocabulary)

        # Check that exposures are flipped
        self.assertEqual(result.exposures, [0, 1, 1, 0])

        # Check that features remain unchanged
        self.assertEqual(result.features, self.data.features)

    def test_regex_counterfactual(self):
        """Test counterfactual generation with regex patterns."""
        generator = CounterfactualGenerator(
            data=self.data, exposure_regex=["drug_[AB]"], control_regex="control"
        )
        result = generator.generate()

        # Check basic structure
        self.assertEqual(result.pids, self.data.pids)
        self.assertEqual(result.outcomes, self.data.outcomes)
        self.assertEqual(result.vocabulary, self.data.vocabulary)

        # Check code swapping in sequences
        orig_seqs = self.data.features["concept"]
        new_seqs = result.features["concept"]

        # First sequence: ends with exposure code (5) -> should end with control code (9)
        self.assertEqual(new_seqs[0][-1], 9)
        self.assertEqual(new_seqs[0][:-1], orig_seqs[0][:-1])

        # Second sequence: ends with control code (9) -> should end with exposure code (5 or 6)
        self.assertIn(new_seqs[1][-1], {5, 6})
        self.assertEqual(new_seqs[1][:-1], orig_seqs[1][:-1])

        # Fourth sequence: has exposure code (5) -> should be replaced with control code (9)
        self.assertEqual(new_seqs[3][1], 9)
        self.assertEqual(new_seqs[3][0], orig_seqs[3][0])
        self.assertEqual(new_seqs[3][2], orig_seqs[3][2])

    def test_count_first_exposure_code_occurrence(self):
        """Test counting first occurrences of exposure codes."""
        generator = CounterfactualGenerator(
            data=self.data, exposure_regex=["drug_[AB]"], control_regex="control"
        )
        generator._initialize_codes()

        result = generator._count_first_exposure_code_occurrence()
        # Code 5 appears in sequences [0] and [3], code 6 doesn't appear
        expected = {5: 2, 6: 0}
        self.assertEqual(result, expected)

    def test_get_probability_of_codes(self):
        """Test probability calculation from code frequencies."""
        code_frequencies = {5: 2, 6: 0}
        result = get_probability_of_codes(code_frequencies)

        expected = {5: 1.0, 6: 0.0}  # 2/2 for code 5, 0/2 for code 6
        self.assertEqual(result, expected)

    def test_swap_codes(self):
        """Test code swapping logic."""
        exposure_probs = {5: 1.0, 6: 0.0}  # Deterministic for testing
        control_code = 9

        # Test exposure code replacement
        concepts = [1, 2, 3, 5]
        result = swap_codes(concepts, exposure_probs, control_code)
        self.assertEqual(result[-1], control_code)
        self.assertEqual(result[:-1], concepts[:-1])

        # Test control code replacement
        concepts = [1, 2, 3, 9]
        result = swap_codes(concepts, exposure_probs, control_code)
        self.assertEqual(result[-1], 5)  # Should always pick 5 with our probabilities
        self.assertEqual(result[:-1], concepts[:-1])

        # Test no relevant codes
        concepts = [1, 2, 3, 4]
        result = swap_codes(concepts, exposure_probs, control_code)
        self.assertEqual(result, concepts)


if __name__ == "__main__":
    unittest.main()
