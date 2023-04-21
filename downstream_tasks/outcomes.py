import pandas as pd

from data.concept_loader import ConceptLoader

class OutcomeMaker():
    def __init__(self, config: dict):
        # We reload concepts to load in custom events
        self.concepts_with_custom, _ = ConceptLoader()(
            concepts=config.loader.concepts + ['custom_events'], 
            data_dir=config.loader.data_dir,
            patients_info=config.loader.patients_info,
        )
        self.outcomes = config.outcomes

    def __call__(self, patients_info: pd.DataFrame):
        # Initalize patient_outcomes dict
        patient_outcomes = {pid: {k: None for k in self.outcomes} for pid in self.concepts_with_custom['PID'].unique()}

        # Create {PID: DATE_OF_DEATH} dict
        if self.outcomes.DEATH:
            death = pd.Series(patients_info['DATE_OF_DEATH'].values, index=patients_info['PID']).to_dict()

        for pid, patient in self.concepts_with_custom.groupby('PID'):    # For each patient
            for key in self.outcomes:                   # For each outcome
                # Hospital admission
                if key == 'HOSPITAL_ADMISSION':
                    patient_outcomes[pid]['HOSPITAL_ADMISSION'] = self.hospital_admission(patient)
                
                # Death
                if key == 'DEATH':
                    patient_outcomes[pid]['DEATH'] = death.get(pid)

                # ICU admission
                if key == 'ICU_ADMISSION':
                    patient_outcomes[pid]['ICU_ADMISSION'] = self.icu_admission(patient)

                # Mechanical ventilation
                if key == 'MECHANICAL_VENTILATION':
                    patient_outcomes[pid]['MECHANICAL_VENTILATION'] = self.respirator(patient)

        # Add patient_outcomes to patient info (converted to dict)
        info = patients_info.set_index('PID').to_dict('index')
        for pid, values in patient_outcomes.items():
            for key, outcome in values.items():
                info.setdefault(pid, {})[f'OUTCOME_{key}'] = outcome
        
        # Convert back to dataframe
        return pd.DataFrame.from_dict(info, orient='index').reset_index().rename(columns={'index': 'PID'})

    @staticmethod
    def hospital_admission(patient: pd.DataFrame):
        admission = patient['ADMISSION_ID'].values
        for idx, adm in enumerate(admission):
            if not adm.startswith('unq_'):
                return patient['TIMESTAMP'].iloc[idx]
        return None

    @staticmethod
    def icu_admission(patient: pd.DataFrame):
        concepts = patient['CONCEPT'].values
        for idx, concept in enumerate(concepts):
            if concept == 'ICU':
                return patient['TIMESTAMP'].iloc[idx]
        return None

    @staticmethod
    def respirator(patient: pd.DataFrame):
        concepts = patient['CONCEPT'].values
        for idx, concept in enumerate(concepts):
            if concept == 'RESPIRATOR':
                return patient['TIMESTAMP'].iloc[idx]
        return None


