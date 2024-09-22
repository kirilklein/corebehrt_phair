"""Create tokenized features from formatted data. config template: data.yaml"""
from collections import defaultdict
from os.path import abspath, dirname, join

import pandas as pd
from tqdm import tqdm

from ehr2vec.common.azure import AzurePathContext, save_to_blobstore
from ehr2vec.common.config import load_config
from ehr2vec.common.logger import TqdmToLogger
from ehr2vec.common.setup import DirectoryPreparer, get_args
from ehr2vec.common.utils import check_patient_counts
from ehr2vec.data.concept_loader import ConceptLoaderLarge
from ehr2vec.downstream_tasks.outcomes import OutcomeMaker

BLOBSTORE = 'CINF'
CONFIG_NAME = 'outcome/outcomes_simvastatin.yaml'

args = get_args(CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)


            
def process_data(loader, cfg, features_cfg, logger)->dict:
    all_outcomes = defaultdict(list)
    for (concept_batch, patient_batch) in tqdm(loader(), desc='Batch Process Data', file=TqdmToLogger(logger)):
        check_patient_counts(concept_batch, patient_batch, logger)
        pids = concept_batch.PID.unique()
        outcome_tables = OutcomeMaker(cfg, features_cfg)(concept_batch, patient_batch, pids)
        # Concatenate the tables for each key
        for key, df in outcome_tables.items():
            if key in all_outcomes:
                all_outcomes[key] = pd.concat([all_outcomes[key], df])
            else:
                all_outcomes[key] = df
    return all_outcomes

def main_data(config_path):
    cfg = load_config(config_path)
    cfg.paths.outcome_dir = join(cfg.features_dir, 'outcomes', cfg.outcomes_name)
    
    cfg, _, mount_context = AzurePathContext(cfg, dataset_name=BLOBSTORE).azure_outcomes_setup()

    logger = DirectoryPreparer(config_path).prepare_directory_outcomes(cfg.paths.outcome_dir, cfg.outcomes_name)
    logger.info('Mount Dataset')
    logger.info('Starting outcomes creation')
    features_cfg = load_config(join(cfg.features_dir, 'data_config.yaml'))
    outcome_tables = process_data(ConceptLoaderLarge(**cfg.loader), cfg, features_cfg, logger)
    
    for key, df in outcome_tables.items():
        df.to_csv(join(cfg.paths.outcome_dir, f'{key}.csv'), index=False)
    
    logger.info('Finish outcomes creation')

    if cfg.env=='azure':
        save_to_blobstore(local_path='outcomes', 
                          remote_path=join(BLOBSTORE, 'outcomes', cfg.paths.run_name))
        mount_context.stop()
    logger.info('Done') 

if __name__ == '__main__':
    main_data(config_path)

