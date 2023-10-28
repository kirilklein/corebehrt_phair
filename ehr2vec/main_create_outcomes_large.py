"""Create tokenized features from formatted data. config template: data.yaml"""
import os
from collections import defaultdict
from os.path import join

import torch
from common.azure import save_to_blobstore
from common.config import load_config
from common.logger import TqdmToLogger
from common.setup import AzurePathContext, DirectoryPreparer, get_args
from common.utils import check_patient_counts
from data.concept_loader import ConceptLoaderLarge
from downstream_tasks.outcomes import OutcomeMaker
from tqdm import tqdm

BLOBSTORE = 'PHAIR'
CONFIG_NAME = 'outcomes_test.yaml'

args = get_args(CONFIG_NAME)
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)

            
def process_data(loader, cfg, features_cfg, logger):
    all_outcomes = defaultdict(list)
    for (concept_batch, patient_batch) in tqdm(loader(), desc='Batch Process Data', file=TqdmToLogger(logger)):
        check_patient_counts(concept_batch, patient_batch, logger)
        pids = concept_batch.PID.unique()
        outcomes = OutcomeMaker(cfg, features_cfg)(concept_batch, patient_batch, pids)
        for key, value in outcomes.items():
            all_outcomes[key].extend(value)
    return all_outcomes

def main_data(config_path):
    cfg = load_config(config_path)
    outcome_dir = join(cfg.features_dir, 'outcomes', cfg.outcomes_name)
    
    cfg, _, mount_context = AzurePathContext(cfg).azure_outcomes_setup()

    logger = DirectoryPreparer(config_path).prepare_directory_outcomes(outcome_dir, cfg.outcomes_name)
    logger.info('Mount Dataset')
    logger.info('Starting outcomes creation')
    features_cfg = load_config(join(cfg.features_dir, 'data_config.yaml'))
    outcomes = process_data(ConceptLoaderLarge(**cfg.loader), cfg, features_cfg, logger)
    
    torch.save(outcomes, join(outcome_dir, f'{cfg.outcomes_name}.pt'))
    
    logger.info('Finish outcomes creation')

    if cfg.env=='azure':
        save_to_blobstore(local_path=join('outcomes',cfg.run_name), 
                          remote_path=join(BLOBSTORE, 'outcomes', cfg.run_name))
        mount_context.stop()
    logger.info('Done') 

if __name__ == '__main__':
    main_data(config_path)

