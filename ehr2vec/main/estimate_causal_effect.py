"""This script is used to measure the effect of treatment on the outcome using propensity scores."""
import os
from datetime import datetime
from os.path import abspath, dirname, join, split

import numpy as np
import pandas as pd
from pycaysal.api import estimator

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.calibration import calibrate_cv
from ehr2vec.common.loader import load_predictions_from_finetune_dir
from ehr2vec.common.logger import log_config
from ehr2vec.common.setup import (fix_tmp_prefixes_for_azure_paths, get_args,
                                  setup_logger)

CONFIG_NAME = 'causal_inference/measure_effect.yaml'
BLOBSTORE='CINF'

args = get_args(CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    cfg, run, mount_context, azure_context = initialize_configuration_measure_effect(config_path, dataset_name=BLOBSTORE)
    
    # create test folder
    date = datetime.now().strftime("%Y%m%d-%H%M")
    exp_folder = join(cfg.paths.output_path, f'experiment_{date}')
    os.makedirs(exp_folder, exist_ok=True)

    ps_folder = cfg.paths.get("ps_model_path")
    # later we will add outcome folder
    logger = setup_logger(exp_folder, 'info.log')
    logger.info(f"Config Paths: {cfg.paths}")
    logger.info(f"Update config with pretrain and ft information.")
    
    pids, probas, targets = load_predictions_from_finetune_dir(ps_folder)
    pids_outcome, probas_outcome, _ = load_predictions_from_finetune_dir(outcome_folder)
    pids_outcome_treated, probas_outcome_treated, _ = load_predictions_from_finetune_dir(outcome_treated_folder)
    pids_outcome_untreated, probas_outcome_untreated, _ = load_predictions_from_finetune_dir(outcome_untreated_folder)
    
    
    if cfg.get('calibration', False):
        probas = calibrate_cv(probas, targets)
    # for testing set some of the targets to 0
    # select random targets to set to 0
    print(targets.sum())
    print(len(targets))
    indices = np.random.choice(range(len(targets)), 60, replace=False)
    targets[indices] = 0

    # do estimation here
    #measure_effect(cfg.method, probas, targets, pids, exp_folder, logger)
    log_config(cfg, logger)
    cfg.paths.run_name = split(exp_folder)[-1]
    
    if cfg.env=='azure':
        save_to_blobstore(local_path='', # uses everything in 'outputs' 
                          remote_path=join(BLOBSTORE, fix_tmp_prefixes_for_azure_paths(cfg.paths.model_path)))
        mount_context.stop()
    logger.info('Done')


if __name__ == '__main__':
    main()
