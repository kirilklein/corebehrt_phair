import pandas as pd
from datetime import datetime
import itertools
import numpy as np

class BaseCreator():
    def __init__(self, config: dict):
        self.config = config

    def __call__(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
         # Create PID -> BIRTHDATE dict
        if 'BIRTHDATE' not in patients_info.columns:
            if 'DATE_OF_BIRTH' in patients_info.columns:
                patients_info = patients_info.rename(columns={'DATE_OF_BIRTH': 'BIRTHDATE'})
            else:
                raise KeyError('BIRTHDATE column not found in patients_info')
        return self.create(concepts, patients_info)

class AgeCreator(BaseCreator):
    feature = id = 'age'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        birthdates = pd.Series(patients_info['BIRTHDATE'].values, index=patients_info['PID']).to_dict()
        # Calculate approximate age
        ages = (((concepts['TIMESTAMP'] - concepts['PID'].map(birthdates)).dt.days / 365.25) + 0.5).round()

        concepts['AGE'] = ages
        return concepts

class AbsposCreator(BaseCreator):
    
    feature = id = 'abspos'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        origin_point = datetime(**self.config.abspos)
        # Calculate hours since origin point
        abspos = (concepts['TIMESTAMP'] - origin_point).dt.total_seconds() / 60 / 60

        concepts['ABSPOS'] = abspos
        return concepts

class SegmentCreator(BaseCreator):
    feature = id = 'segment'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        if 'ADMISSION_ID' in concepts.columns:
            seg_col = 'ADMISSION_ID'
        elif 'SEGMENT' in concepts.columns:
            seg_col = 'SEGMENT'
        else:
            raise KeyError('No segment column found in concepts')
    
        segments = concepts.groupby('PID')[seg_col].transform(lambda x: pd.factorize(x)[0]+1) # change back
        
        concepts['SEGMENT'] = segments
        return concepts

class ValueCreator(BaseCreator):
    feature = id = 'lab_value'

    def assign_value(self, result):
        cancel_names = [
            'Afbestilt', 'Aflyst', 'Annuleret', 'Annulleret','Cancelled',
            'afbestilt', 'aflyst', 'annuleret', 'annulleret', 'cancelled'
        ]
        if result in [f'Q{i}' for i in range(1, 11)]:
            result
        elif result in cancel_names:
            return "Cancelled"
        else:
            return "Other"

    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        concepts['LAB_VALUE'] = concepts['RESULT'].apply(self.assign_value)
        return concepts

class BackgroundCreator(BaseCreator):
    id = 'background'
    prepend_token = "BG_"
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame):
        # Create background concepts
        background = {
            'PID': patients_info['PID'].tolist() * len(self.config.background),
            'CONCEPT': itertools.chain.from_iterable(
                [(self.prepend_token + col + '_' +patients_info[col].astype(str)).tolist() for col in self.config.background])
        }

        if 'segment' in self.config:
            background['SEGMENT'] = 0

        if 'lab_value' in self.config:
            background['LAB_VALUE'] = "[N/A]"

        if 'age' in self.config:
            background['AGE'] = -1

        if 'abspos' in self.config:
            origin_point = datetime(**self.config.abspos)
            start = (patients_info['BIRTHDATE'] - origin_point).dt.total_seconds() / 60 / 60
            background['ABSPOS'] = start.tolist() * len(self.config.background)

        # Prepend background to concepts
        background = pd.DataFrame(background)
        return pd.concat([background, concepts])

class BinnedValueCreator(BaseCreator):
    feature = id = 'binned_value'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        print(concepts.columns)
        # Create value concepts
        concepts['RESULT'] = pd.to_numeric(concepts['RESULT'], errors='coerce')
        concepts['index'] = concepts.index

        values = concepts[concepts['RESULT'].notna()].copy()
        #values['RESULT'] = values.groupby('CONCEPT')['RESULT'].transform(lambda x: (x - x.min()) / (x.max() - x.min())) # Normalize values to [0, 1] within concept
        values['RESULT'] = (values['RESULT'] * 100).astype(int)
        values['CONCEPT'] = 'VAL_' + values['RESULT'].astype(str)
        
        concepts['order'] = 0
        values['order'] = 1
        concatted_df = pd.concat((concepts, values)).sort_values(['index', 'order']).drop(["index", "order"], axis=1)
        return concatted_df

class QuartileValueCreator(BaseCreator):
    feature = id = 'quartile_value'
    def create(self, concepts: pd.DataFrame, patients_info: pd.DataFrame)-> pd.DataFrame:
        print(concepts.columns)
        # Create value concepts
        concepts = concepts[concepts['RESULT'].astype(str).str.startswith('Q')].copy()
        concepts['index'] = concepts.index

        values = concepts[concepts['RESULT'].notna()].copy()
        values['CONCEPT'] = 'VAL_' + values['RESULT'].astype(str)
        
        concepts['order'] = 0
        values['order'] = 1
        concatted_df = pd.concat((concepts, values)).sort_values(['index', 'order']).drop(["index", "order"], axis=1)
        return concatted_df

