import os
from datetime import datetime
from os.path import abspath, dirname, join, split

import pandas as pd
import torch

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.initialize import ModelManager
from ehr2vec.common.logger import log_config
from ehr2vec.common.setup import (fix_tmp_prefixes_for_azure_paths, get_args,
                                  initialize_configuration_finetune,
                                  setup_logger,
                                  update_test_cfg_with_pt_ft_cfgs)
from ehr2vec.common.utils import Data
from ehr2vec.data.dataset import BinaryOutcomeDataset
from ehr2vec.data.prepare_data import DatasetPreparer
from ehr2vec.evaluation.encodings import EHRTester
from ehr2vec.evaluation.utils import save_data

CONFIG_NAME = 'finetune/finetune_evaluate.yaml'
BLOBSTORE='CINF'

args = get_args(CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def test_fold(cfg, finetune_folder: str, test_folder: str, fold:int, test_data: Data=None, run=None, logger=None)->None:
    """Test model on one fold. Save test results in test folder."""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    logger.info("Saving test pids")
    torch.save(test_data.pids, join(test_folder, 'test_pids.pt'))
    test_dataset = BinaryOutcomeDataset(test_data.features, test_data.outcomes)
    modelmanager = ModelManager(cfg, model_path=fold_folder)
    checkpoint = modelmanager.load_checkpoint()
    modelmanager.load_model_config() 
    logger.info('Load best finetuned model to compute test scores')
    model = modelmanager.initialize_finetune_model(checkpoint, test_dataset)

    tester = EHRTester( 
        model=model, 
        test_dataset=None, # test only after training
        args=cfg.tester_args,
        metrics=cfg.metrics,
        cfg=cfg,
        run=run,
        logger=logger,
        accumulate_logits=True,
        test_folder=test_folder,
        mode=f'test_fold{fold}'
    )
    
    tester.model = model
    tester.test_dataset = test_dataset
    tester.evaluate(modelmanager.get_epoch(), mode='test')

def cv_test_loop(test_data: Data, finetune_folder: str, test_folder: str, 
                 n_splits:int, cfg=None, logger=None, run=None)->None:
    """Loop over cross validation folds. Save test results in test folder."""
    for fold in range(n_splits):
        fold += 1
        logger.info(f"Testing fold {fold}/{n_splits}")
        test_fold(cfg, finetune_folder, test_folder, fold, test_data, run, logger)

def compute_and_save_scores_mean_std(n_splits:int, test_folder: str, mode='test', logger=None)->None:
    """Compute mean and std of test/val scores. And save to finetune folder."""
    logger.info(f"Compute mean and std of {mode} scores")
    scores = []
    for fold in range(1, n_splits+1):
        table_path = join(test_folder, f'{mode}_fold{fold}_scores.csv')
        fold_scores = pd.read_csv(table_path)
        scores.append(fold_scores)
    scores = pd.concat(scores)
    scores_mean_std = scores.groupby('metric')['value'].agg(['mean', 'std'])
    scores_mean_std.to_csv(join(test_folder, f'{mode}_scores_mean_std.csv'))


def main():
    cfg, run, mount_context, azure_context = initialize_configuration_finetune(config_path, dataset_name=BLOBSTORE)
    
    # create test folder
    date = datetime.now().strftime("%Y%m%d-%H%M")
    test_folder = join(cfg.paths.output_path, f'test_{date}')
    os.makedirs(test_folder, exist_ok=True)

    finetune_folder = cfg.paths.get("model_path")
    logger = setup_logger(test_folder, 'test_info.log')
    logger.info(f"Config Paths: {cfg.paths}")
    logger.info(f"Update config with pretrain and ft information.")
    cfg = update_test_cfg_with_pt_ft_cfgs(cfg, finetune_folder)
    cfg = fix_tmp_prefixes_for_azure_paths(cfg, azure_context)
    cfg.save_to_yaml(join(test_folder, 'evaluate_config.yaml'))
    logger.info(f"Config Paths after fix: {cfg.paths}")

    fold_dirs = [fold_folder for fold_folder in os.listdir(finetune_folder) if fold_folder.startswith('fold_')]
    n_splits = len(fold_dirs)
    log_config(cfg, logger)
    cfg.paths.run_name = split(test_folder)[-1]

    if not cfg.data.get('preprocess', False):
        logger.info(f"Load processed test data from {cfg.paths.data_dir}")
        test_data = Data.load_from_directory(cfg.paths.data_dir, mode='test')
    else:
        logger.info(f"Prepare test data from {cfg.paths.data_dir}")
        dataset_preparer = DatasetPreparer(cfg)
        data = dataset_preparer.prepare_finetune_data()    
        if 'predefined_pids' in cfg.paths:
            logger.info(f"Load test pids from {cfg.paths.predefined_pids}")
            test_pids = torch.load(join(cfg.paths.predefined_pids, 'test_pids.pt')) 
            if len(test_pids)!=len(set(test_pids)):
                logger.warn(f'Test pids contain duplicates. Test pids len {len(test_pids)}, unique pids {len(set(test_pids))}.')
                logger.info('Removing duplicates')
                test_pids = list(set(test_pids))
            test_data = data.select_data_subset_by_pids(test_pids, mode='test')
        else:
            logger.info(f"Use all data for testing.")
            test_data = data
    save_data(test_data, test_folder)
    cv_test_loop(test_data, finetune_folder, test_folder, n_splits, cfg, logger, run)
    compute_and_save_scores_mean_std(n_splits, test_folder, mode='test', logger=logger)    
    
    if cfg.env=='azure':
        save_to_blobstore(local_path='', # uses everything in 'outputs' 
                          remote_path=join(BLOBSTORE, fix_tmp_prefixes_for_azure_paths(cfg.paths.model_path)))
        mount_context.stop()
    logger.info('Done')


if __name__ == '__main__':
    main()
