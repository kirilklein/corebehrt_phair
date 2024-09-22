import unittest
from datetime import datetime

import pandas as pd

from ehr2vec.data.creators import (AbsposCreator, AgeCreator,
                                   BackgroundCreator, DeathCreator,
                                   SegmentCreator)
from ehr2vec.data.utils import Utilities
from ehr2vec.tests.helpers import ConfigMock


class TestBaseCreator(unittest.TestCase):
    def setUp(self):
        self.cfg = ConfigMock()
        self.cfg.age = {'round': 2}
        self.cfg.abspos = {'year': 2020, 'month': 1, 'day': 26}
        self.cfg.segment = True
        self.cfg.background = ['GENDER']

        self.concepts = pd.DataFrame({
            'PID': ['1', '2', '3', '1'],
            'CONCEPT': ['DA1', 'DA2', 'MA1', 'DA2'],
            'TIMESTAMP': pd.to_datetime(['2020-01-02', '2021-03-20', '2022-05-08', '2023-01-02']),
            'ADMISSION_ID': ['A', 'B', 'C', 'D']
        })
        self.patients_info = pd.DataFrame({
            'PID': ['1', '2', '3'],
            'BIRTHDATE': pd.to_datetime(['2000-01-02', '2000-03-20', '2000-05-08']),
            'DATE_OF_DEATH': pd.to_datetime(['2020-01-02', '2021-03-20', '2022-05-08']),
            'GENDER': ['Male', 'Female', 'Male']
        })

    def test_age_creator(self):
        creator = AgeCreator(self.cfg)
        result = creator.create(self.concepts, self.patients_info)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('AGE', result.columns)
        self.assertEqual(result.AGE.tolist(), [20, 21, 22, 23])

    def test_abspos_creator(self):
        creator = AbsposCreator(self.cfg)
        result = creator.create(self.concepts, self.patients_info)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('ABSPOS', result.columns)
        origin_point = datetime(**self.cfg.abspos)
        self.assertTrue(result.ABSPOS.iloc[0] < 0)
        self.assertTrue(all(result.ABSPOS.iloc[1:] > 0))
        self.assertEqual(result.ABSPOS.tolist(), ((self.concepts.TIMESTAMP - origin_point).dt.total_seconds() / 60 / 60).tolist())

    def test_segment_creator(self):
        creator = SegmentCreator(self.cfg)
        result = creator.create(self.concepts, self.patients_info)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('SEGMENT', result.columns)
        self.assertEqual(result.SEGMENT.tolist(), [1, 1, 1, 2])

    def test_background_creator(self):
        creator = BackgroundCreator(self.cfg)
        result = creator.create(self.concepts, self.patients_info)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('SEGMENT', result.columns)
        self.assertIn('AGE', result.columns)
        self.assertIn('ABSPOS', result.columns)
        for _, patient in result.groupby("PID"):
            self.assertEqual(patient.CONCEPT.str.startswith('BG_').sum(), len(self.cfg.background))

    def test_death_date_creator(self):
        # Ensure SEGMENT is added to the concepts DataFrame
        segment_creator = SegmentCreator(self.cfg)
        concepts_with_segment = segment_creator.create(self.concepts, self.patients_info)
        
        # Now test DeathDateCreator
        creator = DeathCreator(self.cfg)
        result = creator.create(concepts_with_segment, self.patients_info)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('CONCEPT', result.columns)
        self.assertIn('AGE', result.columns)
        self.assertIn('ABSPOS', result.columns)
        self.assertIn('SEGMENT', result.columns)
        
        death_entries = result[result['CONCEPT'] == 'Death']
        self.assertEqual(len(death_entries), len(self.patients_info))
        
        expected_ages = ((self.patients_info['DATE_OF_DEATH'] - self.patients_info['BIRTHDATE']).dt.days / 365.25).round(2).tolist()
        self.assertEqual(death_entries['AGE'].tolist(), expected_ages)
        
        abspos_at_death = Utilities.get_abspos_from_origin_point(self.patients_info['DATE_OF_DEATH'], self.cfg.abspos)
        self.assertEqual(death_entries['ABSPOS'].tolist(), abspos_at_death.tolist())
        
        for pid in self.patients_info['PID']:
            self.assertEqual(death_entries[death_entries['PID'] == pid]['SEGMENT'].iloc[0], concepts_with_segment[concepts_with_segment['PID'] == pid]['SEGMENT'].iloc[-1])

if __name__ == '__main__':
    unittest.main()
