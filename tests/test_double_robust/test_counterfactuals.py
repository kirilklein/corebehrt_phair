import unittest

from ehr2vec.common.utils import Data
from ehr2vec.double_robust.counterfactual import (
    create_counterfactual_data,
    insert_random_code_to_end,
    remove_codes,
)


class TestCounterfactuals(unittest.TestCase):
    def setUp(self):
        self.vocabulary = {
            "[CLS]": 0,
            "[SEP]": 1,
            "BG_1": 2,
            "BG_2": 3,
            "A1": 4,
            "B1": 5,
            "C1": 6,
            "D1": 7,
        }
        self.exposure_regex_list = ["A1", "B1"]
        self.exposure_codes = {4, 5}

    def test_remove_codes(self):
        patient = {
            "concept": [0, 2, 3, 1, 4, 5, 6],
            "abspos": [0, 1, 2, 3, 10, 15, 20],
            "segment": [1, 1, 1, 1, 2, 2, 2],
        }
        result = remove_codes(patient, self.exposure_codes)
        expected = {
            "concept": [0, 2, 3, 1, 6],
            "abspos": [0, 1, 2, 3, 20],
            "segment": [1, 1, 1, 1, 2],
        }
        self.assertEqual(result, expected)

    def test_insert_random_code_to_end(self):
        patient = {
            "concept": [0, 2, 3, 1, 4, 5],
            "abspos": [0, 1, 2, 3, 10, 15],
            "segment": [1, 1, 1, 1, 2, 2],
        }
        exposure_codes = {4, 5}  # A1 and B1
        result = insert_random_code_to_end(patient, exposure_codes)

        self.assertEqual(len(result["concept"]), len(patient["concept"]) + 1)
        self.assertEqual(len(result["abspos"]), len(patient["abspos"]) + 1)
        self.assertEqual(len(result["segment"]), len(patient["segment"]) + 1)
        self.assertIn(result["concept"][-1], exposure_codes)

    def test_create_counterfactual_data_with_exposure(self):
        features = {
            "concept": [
                [0, 2, 3, 1, 4, 5],
                [0, 2, 3, 1, 6, 7],
            ],
            "abspos": [
                [0, 1, 2, 3, 10, 15],
                [0, 1, 2, 3, 10, 15],
            ],
            "segment": [
                [1, 1, 1, 1, 2, 2],
                [1, 1, 1, 1, 2, 2],
            ],
        }
        data = Data(
            features=features,
            outcomes=None,
            pids=None,
            mode=None,
            vocabulary=self.vocabulary,
        )

        result = create_counterfactual_data(data, self.exposure_regex_list)

        # Check that the exposure code was removed from the first patient
        self.assertEqual(len(result.features["concept"][0]), 4)
        self.assertNotIn(5, result.features["concept"][0])

        # Check that an exposure code was added to the second patient
        self.assertEqual(len(result.features["concept"][1]), 7)
        self.assertIn(result.features["concept"][1][-1], {4, 5})

    def test_create_counterfactual_data_without_exposure(self):
        features = {
            "concept": [
                [0, 2, 3, 1, 6, 7],
                [0, 2, 3, 1, 6, 7],
            ],
            "abspos": [
                [0, 1, 2, 3, 10, 15],
                [0, 1, 2, 3, 10, 15],
            ],
            "segment": [
                [1, 1, 1, 1, 2, 2],
                [1, 1, 1, 1, 2, 2],
            ],
        }
        data = Data(
            features=features,
            outcomes=None,
            pids=None,
            mode=None,
            vocabulary=self.vocabulary,
        )

        result = create_counterfactual_data(data, self.exposure_regex_list)

        # Check that an exposure code was added to both patients
        self.assertEqual(len(result.features["concept"][0]), 7)
        self.assertEqual(len(result.features["concept"][1]), 7)
        self.assertIn(result.features["concept"][0][-1], {4, 5})
        self.assertIn(result.features["concept"][1][-1], {4, 5})

    def test_create_counterfactual_data_mixed(self):
        features = {
            "concept": [
                [0, 2, 3, 1, 4, 6],
                [0, 2, 3, 1, 6, 7],
                [0, 2, 3, 1, 5, 7],
            ],
            "abspos": [
                [0, 1, 2, 3, 10, 15],
                [0, 1, 2, 3, 10, 15],
                [0, 1, 2, 3, 10, 15],
            ],
            "segment": [
                [1, 1, 1, 1, 2, 2],
                [1, 1, 1, 1, 2, 2],
                [1, 1, 1, 1, 2, 2],
            ],
        }
        data = Data(
            features=features,
            outcomes=None,
            pids=None,
            mode=None,
            vocabulary=self.vocabulary,
        )

        result = create_counterfactual_data(data, self.exposure_regex_list)

        # Check that the exposure code was removed from the first and third patients
        self.assertEqual(len(result.features["concept"][0]), 5)
        self.assertEqual(len(result.features["concept"][2]), 5)
        self.assertNotIn(4, result.features["concept"][0])
        self.assertNotIn(5, result.features["concept"][2])

        # Check that an exposure code was added to the second patient
        self.assertEqual(len(result.features["concept"][1]), 7)
        self.assertIn(result.features["concept"][1][-1], {4, 5})


if __name__ == "__main__":
    unittest.main()
