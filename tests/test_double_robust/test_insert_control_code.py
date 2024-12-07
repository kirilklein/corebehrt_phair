import unittest
from ehr2vec.common.utils import Data
from ehr2vec.double_robust.counterfactual import insert_control_codes


class TestControlCodeHandler(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_data = Data(
            features={
                "concept": [
                    [0, 1],  # Control patient
                    [2, 3],  # Case patient
                    [4, 5],  # Control patient
                    [6, 7, 8],  # Control patient
                ],
                "abspos": [
                    [10.0, 30.0],
                    [20.0, 40.0],
                    [15.0, 35.0],
                    [10.0, 15.0, 15.0],
                ],
                "age": [[50, 51], [60, 61], [70, 71], [50, 81, 81]],
                "segment": [[1, 1], [1, 1], [1, 1], [1, 1, 1]],
            },
            pids=["P1", "P2", "P3", "P4"],
            outcomes=[None, 100, None, None],
            index_dates=[25.0, 30.0, 20.0, 15.0],
            vocabulary={"CONTROL": 1000},
        )

        self.control_patients = {"P1", "P3", "P4"}
        self.control_code = "CONTROL"
        self.setup_expected_results()

    def setup_expected_results(self):
        self.expected_concepts = [
            [0, 1000, 1],  # Control code inserted at index 1
            [2, 3],  # No change
            [1000, 4, 5],  # Control code inserted at index 1
            [6, 1000, 7, 8],  # Control code inserted at index 1
        ]
        self.expected_abspos = [
            [10.0, 25.0, 30.0],  # Index date inserted
            [20.0, 40.0],  # No change
            [20.0, 15.0, 35.0],  # Index date inserted
            [10.0, 15.0, 15.0, 15.0],  # Index date inserted
        ]
        self.expected_age = [
            [50, 51, 51],  # Copied from closest event
            [60, 61],  # No change
            [70, 70, 71],  # Copied from closest event
            [50, 81, 81, 81],  # Copied from closest event
        ]
        self.expected_segments = [
            [1, 1, 1],  # Copied from closest event
            [1, 1],  # No change
            [1, 1, 1],  # Copied from closest event
            [1, 1, 1, 1],  # Copied from closest event
        ]

    def test_insert_control_codes(self):
        # Run the method
        result = insert_control_codes(
            self.test_data, self.control_patients, self.control_code
        )
        # Verify results
        self.assertEqual(result.features["concept"], self.expected_concepts)
        self.assertEqual(result.features["abspos"], self.expected_abspos)
        self.assertEqual(result.features["age"], self.expected_age)
        self.assertEqual(result.features["segment"], self.expected_segments)


if __name__ == "__main__":
    unittest.main()
