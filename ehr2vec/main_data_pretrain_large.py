"""
Input: Formatted Data
- Load concepts
- Handle wrong data
- Exclude patients with <k concepts
- Split data
- Tokenize
- truncate train and val
"""
import os
import shutil
from os.path import join

import torch
from typing import List, Dict, Union
from common.azure import AzurePathContext, save_to_blobstore
from common.config import load_config
from common.logger import TqdmToLogger
from common.setup import DirectoryPreparer, get_args
from common.utils import check_directory_for_features
from data.batch import Batches, BatchTokenize
from data.concept_loader import ConceptLoaderLarge
from data.featuremaker import FeatureMaker
from data.tokenizer import EHRTokenizer
from data_fixes.exclude import Excluder
from data_fixes.handle import Handler
from tqdm import tqdm

CONFIG_NAME = 'data_pretrain.yaml'
BLOBSTORE = 'PHAIR'

args = get_args(CONFIG_NAME, 'data_pretrain')
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_path)


def main_data(config_path):
    """
        Loads data
        Finds outcomes
        Creates features
        Handles wrong data
        Excludes patients with <k concepts
        Splits data
        Tokenizes
        Saves
    """
    cfg = load_config(config_path)
    cfg, _, mount_context = AzurePathContext(cfg, dataset_name=BLOBSTORE).azure_data_pretrain_setup()

    logger = DirectoryPreparer(config_path).prepare_directory(cfg)  
    logger.info('Mount Dataset')
    
    logger.info('Initialize Processors')
    logger.info('Starting feature creation and processing')
    if not check_directory_for_features(cfg.loader.data_dir):
        pids = create_and_save_features(ConceptLoaderLarge(**cfg.loader), 
                                        Handler(**cfg.handler), 
                                        Excluder(**cfg.excluder), 
                                        cfg, logger)
        torch.save(pids, join(cfg.output_dir, 'features', 'pids_features.pt'))
    else:
        pids = torch.load(join(cfg.loader.data_dir, 'features', 'pids_features.pt'))
    logger.info('Finished feature creation and processing')
    
    exclude_pids = load_exclude_pids(cfg)
    assigned_pids = load_assigned_pids(cfg)
    for split, split_assigned_pids in assigned_pids.items():
        logger.info(f"Number of assigned pids in split {split}: {len(split_assigned_pids)}")
    logger.info(f"Number of pids to exclude: {len(exclude_pids)}")
    logger.info('Splitting batches')
    batches = Batches(cfg, pids, 
                      exclude_pids=exclude_pids, 
                      assigned_pids=assigned_pids)
    logger.info("Check for existing splits")
    batches_split = batches.split_batches()
    tokenized_dir_name = cfg.get('tokenized_dir_name','tokenized')
    check_and_clear_directory(cfg, logger, tokenized_dir_name=tokenized_dir_name)
    logger.info('Tokenizing')
    tokenizer = EHRTokenizer(config=cfg.tokenizer)
    batch_tokenize = BatchTokenize(pids, tokenizer, cfg, tokenized_dir_name=tokenized_dir_name)
    shutil.copy(config_path, join(cfg.output_dir, tokenized_dir_name,  'data_cfg.yaml'))
    
    batch_tokenize.tokenize(batches_split)
    logger.info('Finished tokenizing')
    
    
    if cfg.env=='azure':
        features_dir_name  = cfg.paths.get('features_dir_name', cfg.paths.run_name)
        save_to_blobstore(local_path='data/', 
                          remote_path=join(BLOBSTORE, 'features', features_dir_name))
        mount_context.stop()
    logger.info('Finished')

def check_and_clear_directory(cfg, logger, tokenized_dir_name='tokenized'):
    tokenized_dir = join(cfg.output_dir, tokenized_dir_name)
    tokenized_files = os.listdir(tokenized_dir) 
    if len(tokenized_files)>0:
        logger.warning(f"The directory {tokenized_dir} is not empty.")
        logger.warning(f"Deleting tokenized files.")
        for file in tokenized_files:
            file_path = join(tokenized_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)

def create_and_save_features(conceptloader, handler, excluder, cfg, logger, )-> list:
    """
    Creates features and saves them to disk.
    Returns a list of lists of pids for each batch
    """
    pids = []
    for i, (concept_batch, patient_batch) in enumerate(tqdm(conceptloader(), desc='Batch Process Data', file=TqdmToLogger(logger))):
        feature_maker = FeatureMaker(cfg.features) # Otherwise appended to old features
        features_batch, pids_batch = feature_maker(concept_batch, patient_batch)
        features_batch = handler(features_batch)
        features_batch, _, kept_indices  = excluder(features_batch)
        kept_pids = [pids_batch[idx] for idx in kept_indices]
        torch.save(features_batch, join(cfg.output_dir, 'features', f'features_{i}.pt'))
        torch.save(kept_pids, join(cfg.output_dir, 'features', f'pids_features_{i}.pt'))
        pids.append(kept_pids)
    return pids

def load_exclude_pids(cfg)->List:
    """
    Loads pids from file
    Excluded pids
    """
    if cfg.get('exclude_pids', None) is None:
        return []
    return load_pids(cfg.exclude_pids)

def load_assigned_pids(cfg)->Dict:
    """ Loads pids which should be assigned to certain splits."""
    if cfg.get('assigned_pids', None) is None:
        return {}
    assigned_pids = {}
    for split, files in cfg.assigned_pids.items():
        assigned_pids[split] = load_pids(files)
    return assigned_pids

def load_pids(files: Union[List, str])->List:
    if isinstance(files, str):
        return torch.load(files)    
    pids = []
    for file in files:
        pids.extend(torch.load(file))
    return pids



if __name__ == '__main__':
    main_data(config_path)


