import unittest
from ehr2vec.common.utils import Data
from ehr2vec.double_robust.counterfactual import (
    insert_control_codes,
    insert_control_code_for_patient,
)


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
                    [9],  # Single event
                ],
                "abspos": [
                    [10.0, 30.0],
                    [20.0, 40.0],
                    [15.0, 35.0],
                    [10.0, 15.0, 15.0],
                    [5.0],
                ],
                "age": [[50, 51], [60, 61], [70, 71], [50, 81, 81], [45]],
                "segment": [[1, 1], [1, 1], [1, 1], [1, 1, 1], [1]],
            },
            pids=["P1", "P2", "P3", "P4", "P5", "P6"],
            outcomes=[None, 100, None, None, None, None],
            index_dates=[25.0, 30.0, 20.0, 15.0, 10.0, 5.0],
            vocabulary={"CONTROL": 1000},
        )

        self.control_patients = {"P1", "P3", "P4", "P5", "P6"}
        self.control_code = "CONTROL"
        self.setup_expected_results()

    def setup_expected_results(self):
        self.expected_concepts = [
            [0, 1000, 1],  # Control code still at index 1 (25.0 < 30.0)
            [2, 3],  # No change
            [4, 1000, 5],  # Control code at index 1 (20.0 > 15.0)
            [6, 1000, 7, 8],  # Control code at index 1 (15.0 = 15.0)
            [9, 1000],  # Control code after 9 (10.0 > 5.0)
        ]
        self.expected_abspos = [
            [10.0, 25.0, 30.0],  # Index date inserted
            [20.0, 40.0],  # No change
            [15.0, 20.0, 35.0],  # Index date inserted
            [10.0, 15.0, 15.0, 15.0],  # Index date inserted
            [5.0, 10.0],  # Index date inserted
        ]
        self.expected_age = [
            [50, 51, 51],  # Copied from closest event
            [60, 61],  # No change
            [70, 70, 71],  # Copied from closest event
            [50, 81, 81, 81],  # Copied from closest event
            [45, 45],  # Age copied
        ]
        self.expected_segments = [
            [1, 1, 1],  # Copied from closest event
            [1, 1],  # No change
            [1, 1, 1],  # Copied from closest event
            [1, 1, 1, 1],  # Copied from closest event
            [1, 1],  # Segment copied
        ]

    def test_insert_control_codes(self):
        """Test bulk insertion of control codes."""

        insert_control_codes(self.test_data, self.control_patients, self.control_code)

        # Test each feature type separately with descriptive messages
        self.assertEqual(
            self.test_data.features["concept"],
            self.expected_concepts,
            "Control code insertion failed for concepts",
        )
        self.assertEqual(
            self.test_data.features["abspos"],
            self.expected_abspos,
            "Time position insertion failed",
        )
        self.assertEqual(
            self.test_data.features["age"],
            self.expected_age,
            "Age value copying failed",
        )
        self.assertEqual(
            self.test_data.features["segment"],
            self.expected_segments,
            "Segment value copying failed",
        )

    def test_single_patient_transformation(self):
        """Test control code insertion for a single patient."""
        # Test for a control patient
        patient_data = {
            "concept": self.test_data.features["concept"][0],
            "abspos": self.test_data.features["abspos"][0],
            "age": self.test_data.features["age"][0],
            "segment": self.test_data.features["segment"][0],
        }

        insert_control_code_for_patient(
            patient_data,  # Pass the single patient's data
            25.0,  # index_date
            self.test_data.vocabulary[self.control_code],
        )

        self.assertEqual(
            patient_data["concept"],
            self.expected_concepts[0],
            "Single control patient transformation failed",
        )

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        with self.assertRaises(KeyError):
            insert_control_codes(
                self.test_data,
                self.control_patients,
                "INVALID_CODE",  # Non-existent control code
            )


if __name__ == "__main__":
    unittest.main()
