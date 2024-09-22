import unittest
from datetime import datetime
from typing import Dict, List, Union

import numpy as np
import pandas as pd

from ehr2vec.common.utils import Data
from ehr2vec.downstream_tasks.outcomes import OutcomeHandler


# Mock Utilities class as it seems to be used for time computations
class MockUtilities:
    @staticmethod
    def get_abspos_from_origin_point(timestamps: Union[pd.Series, List[datetime]], 
                                     origin_point: Dict[str, int])->Union[pd.Series, List[float]]:
        """Get the absolute position in hours from the origin point"""
        origin_point_dt = datetime(**origin_point)
        
        if isinstance(timestamps, pd.Series):
            # Convert timestamps from Series of datetime to Series of float hours
            return (timestamps - origin_point_dt).dt.total_seconds() / 60 / 60
        elif isinstance(timestamps, list):
            # Convert list of datetime to list of float hours
            return [(timestamp - origin_point_dt).total_seconds() / 60 / 60 for timestamp in timestamps]

# Unit tests

class TestOutcomeHandler(unittest.TestCase):
    
    def setUp(self):
        # Mock data for Data class
        self.data = Data(
            features={
                'concept': [['A', 'B', 'Death'], ['C', 'Death', 'D'], ['E', 'F'], ['G', 'H']],
                'abspos': [[10., 20., 30.], [40., 51., 60.], [70., 80.], [90., 100.]]
            },
            pids=['P1', 'P2', 'P3', 'P4'],
            outcomes=[None, 100, 200, 110],
            index_dates=[0, 50, 150, 50],
            vocabulary={'Death': 'Death'}
        )
        
        # Mock DataFrames for outcomes and exposures
        self.outcomes_df = pd.DataFrame({
            'PID': ['P1', 'P2', 'P3', 'P4'],
            'TIMESTAMP': [None, 100, 200, None]
        })
        
        self.exposures_df = pd.DataFrame({
            'PID': ['P1', 'P2', 'P3', 'P4'],
            'TIMESTAMP': [0, 50, 150, 50]
        })

        # Initialize the OutcomeHandler instance
        self.handler = OutcomeHandler(
            survival=True,
            end_of_time={'year': 2021, 'month': 12, 'day': 31, 'hour': 23, 'minute': 59, 'second': 59}
        )
        self.handler_wdeath = OutcomeHandler(
            survival=False,
            end_of_time={'year': 2021, 'month': 12, 'day': 31, 'hour': 23, 'minute': 59, 'second': 59},
            death_is_event=True
        )

    def test_assign_time2event(self):
        # Test method for assign_time2event
        result_data = self.handler.assign_time2event(self.data)
        self.assertEqual(result_data.times2event, [30.0, 50.0, 50.0, 60.0], "Time to event calculation is incorrect.")

    def test_assign_time2event_wdeath(self):
        # Test method for assign_time2event
        result_data = self.handler_wdeath.assign_time2event(self.data)
        self.assertEqual(result_data.times2event, [30.0, 1.0, 50.0, 60.0], "Time to event calculation is incorrect.")

    def test_get_death_abspos(self):
        # Test method for get_death_abspos
        expected_death_abspos = [30.0, 51.0, np.nan, np.nan]
        result = self.handler.get_death_abspos(self.data).to_list()

        np.testing.assert_array_equal(result, expected_death_abspos, "Death positions are incorrectly computed.")
    
    def test_compute_end_of_time_abspos(self):
        # Test method for compute_end_of_time_abspos
        end_of_time_dt = datetime(year=2021, month=12, day=31, hour=23, minute=59, second=59)
        expected_end_time = MockUtilities.get_abspos_from_origin_point(
            [end_of_time_dt],
            OutcomeHandler.ORIGIN_POINT
        )[0]
        result = self.handler.compute_end_of_time_abspos()
        self.assertEqual(result[0], expected_end_time, "End of time computation is incorrect.")

