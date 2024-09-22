import unittest
from unittest.mock import patch
from ehr2vec.data_fixes.censor import Censorer  # Replace with the correct import path


class TestCensorer(unittest.TestCase):

    def setUp(self):
        # Adjust according to the new class parameters
        self.censorer = Censorer(n_hours=1, n_hours_diag_censoring=1, vocabulary={'[CLS]': 0, '[SEP]': 1, 'BG_GENDER_Male': 2, 'Diagnosis1': 3, 'Diagnosis2': 4})
        self.censorer.background_length = 3

    def test_censor(self):
        features = {
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2', 'Medication1', '[SEP]']],
            'abspos': [[0, 0, 0, 1, 1, 2, 3, 3]],
            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1]]   
        }
        index_dates = [1]
        expected_result = {
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2']],
            'abspos': [[0, 0, 0, 1, 1, 2]],
            'attention_mask': [[1, 1, 1, 1, 1, 1]]
        }
        result = self.censorer.censor(features, index_dates)
        self.assertEqual(result, expected_result)

    def test_if_tokenized(self):
        self.assertFalse(self.censorer._identify_if_tokenized(['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2']))
        self.assertTrue(self.censorer._identify_if_tokenized([0, 6, 1, 7, 1, 8]))

    def test_identify_background(self):
        background_flags = self.censorer._identify_background(['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2'], tokenized_flag=False)
        self.assertEqual(background_flags, [True, True, True, False, False, False])

        background_flags_tokenized = self.censorer._identify_background([0, 2, 1, 3, 1, 4], tokenized_flag=True)
        self.assertEqual(background_flags_tokenized, [True, True, True, False, False, False])

    def test_generate_censor_flags(self):
        abspos = [0, 0, 0, 1, 1, 2, 3, 3]
        patient = {'abspos': abspos}
        event_timestamp = 1
        censor_flags = self.censorer._generate_censor_flags(patient, event_timestamp)

        self.assertEqual(censor_flags, [True, True, True, True, True, True, False, False])

    def test_censor_patient(self):
        patient = {
            'concept': ['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2', 'Medication1', '[SEP]'],
            'abspos': [0, 0, 0, 1, 1, 2, 3, 3],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]
        }
        index_timestamp = 1
        expected_patient = {
            'concept': ['[CLS]', 'BG_GENDER_Male', '[SEP]', 'Diagnosis1', '[SEP]', 'Diagnosis2'],
            'abspos': [0, 0, 0, 1, 1, 2],
            'attention_mask': [1, 1, 1, 1, 1, 1]
        }
        result = self.censorer._censor_patient(patient, index_timestamp)
        self.assertEqual(result, expected_patient)


class TestCensorerDiagnosesLater(unittest.TestCase):

    def setUp(self):
        self.vocabulary = {'[CLS]': 0, '[SEP]': 1, 'BG_GENDER_Male': 2, 
                           'M1': 3, 'M2': 4, 'M3': 5,
                           'D1': 6, 'D2': 7, 'D3': 8}
        self.index_dates = [2.1]
        self.features = {
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 
                         'D1', 'M1', 'D1', '[SEP]', 
                         'M2', 'D2', '[SEP]',
                         'M3', '[SEP]',
                         'D3', '[SEP]']],
            'abspos': [[0, 0, 0, 
                        1, 2, 2.5, 2.5,
                        3, 4, 4, 
                        5, 5,
                        6, 6]],
            'attention_mask': [[1] * 14],
            'segment': [[0, 0, 0, 
                         1, 1, 1, 1, 
                         2, 2, 2, 
                         3, 3,
                         4, 4]]
        }
        self.features = self.convert_concept_to_ids(self.features)
        self.patient = {k: v[0] for k, v in self.features.items()}

        # Initialize censorers
        self.censorer = self.create_censorer(censor_diag_end_of_visit=False)
        self.censorer_end_of_visit = self.create_censorer(censor_diag_end_of_visit=True)
        
        # Expected results
        self.expected_result = self.convert_concept_to_ids({
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'D1', 'M1', 'D1', '[SEP]', 'D2', '[SEP]']],
            'abspos': [[0, 0, 0, 1, 2, 2.5, 2.5, 4, 4]],
            'attention_mask': [[1] * 9],
            'segment': [[0, 0, 0, 1, 1, 1, 1, 2, 2]]
        })
        self.expected_result_end_of_visit = self.convert_concept_to_ids({
            'concept': [['[CLS]', 'BG_GENDER_Male', '[SEP]', 'D1', 'M1', 'D1', '[SEP]']],
            'abspos': [[0, 0, 0, 1, 2, 2.5, 2.5]],
            'attention_mask': [[1] * 7],
            'segment': [[0, 0, 0, 1, 1, 1, 1]]
        })
        self.expected_censor_flags = [True, True, True, True, True, True, True, False, True, True, False, False, False, False]

    def create_censorer(self, censor_diag_end_of_visit):
        censorer = Censorer(n_hours=0, n_hours_diag_censoring=3.5, vocabulary=self.vocabulary, censor_diag_end_of_visit=censor_diag_end_of_visit)
        censorer.background_length = 3
        return censorer

    def convert_concept_to_ids(self, feature_dict):
        feature_dict['concept'][0] = [self.vocabulary[c] for c in feature_dict['concept'][0]]
        return feature_dict

    def test_censor(self):
        self.run_censor_test(self.censorer, self.expected_result)
        self.run_censor_test(self.censorer_end_of_visit, self.expected_result_end_of_visit)

    def run_censor_test(self, censorer, expected_result):
        result = censorer.censor(self.features, self.index_dates)
        self.assertEqual(result, expected_result)

    def test_generate_censor_flags(self):
        self.run_generate_censor_flags_test(
            self.censorer._generate_censor_flags, 
            self.expected_censor_flags
        )

    def run_generate_censor_flags_test(self, generate_flags_func, expected_flags):
        flags = generate_flags_func(self.patient, self.index_dates[0])
        self.assertEqual(flags, expected_flags)

    def test_generate_sep_diag_censor_flags(self):
        self.run_generate_censor_flags_test(
            self.censorer._generate_sep_diag_censor_flags, 
            [False, False, False, True, False, True, True, False, True, True, False, False, False, False]
        )

    def test_generate_sep_diag_censor_flags_end_of_visit(self):
        self.run_generate_censor_flags_test(
            self.censorer_end_of_visit._generate_sep_diag_censor_flags_end_of_visit, 
            [False, False, False, True, False, True, True, False, False, False, False, False, False, False]
        )

    def test_get_diagnoses_flags(self):
        concept = ['D1', 'M1', 'M2', '[SEP]', 'D2']
        expected_flags = [True, False, False, False, True]
        diagnoses_flags = self.censorer._get_diagnoses_flags([self.vocabulary[c] for c in concept])
        self.assertEqual(diagnoses_flags, expected_flags)

    def test_get_last_segment_before_timestamp(self):
        last_segment = self.censorer._get_last_segment_before_timestamp([0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 0, 1.9, 4, 5, 5.5, 6, 7], 2)
        self.assertEqual(last_segment, 1)

    def test_return_last_index_for_element(self):
        last_index = self.censorer._return_last_index_for_element([0, 0, 1, 1, 1, 2], 1)
        self.assertEqual(last_index, 4)

    def test_combine_lists_with_and(self):
        self.run_combine_lists_test([True, False, True], [True, True, False], [True, False, False], self.censorer._combine_lists_with_and)

    def test_combine_lists_with_or(self):
        self.run_combine_lists_test([True, False, True], [True, True, False], [True, True, True], self.censorer._combine_lists_with_or)

    def run_combine_lists_test(self, list1, list2, expected_result, combine_func):
        result = combine_func(list1, list2)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
