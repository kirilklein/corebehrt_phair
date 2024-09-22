"""
Compute feature importance for 'concept' features using perturbation method.
Using the five folds from the cross-validation.
"""
import os
from datetime import datetime
from os.path import abspath, dirname, join, split

import numpy as np
import shap
import torch
from torch.utils.data import DataLoader

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.initialize import ModelManager
from ehr2vec.common.logger import log_config
from ehr2vec.common.setup import (fix_tmp_prefixes_for_azure_paths, get_args,
                                  initialize_configuration_finetune,
                                  setup_logger,
                                  update_test_cfg_with_pt_ft_cfgs)
from ehr2vec.common.utils import Data
from ehr2vec.data.dataset import BinaryOutcomeDataset
from ehr2vec.feature_importance.shap import BEHRTWrapper, EHRMasker
from ehr2vec.feature_importance.shap_utils import insert_concept_shap_values
from ehr2vec.feature_importance.utils import log_most_important_features
from ehr2vec.trainer.utils import get_tqdm

CONFIG_NAME = 'feature_importance/shap_feature_importance.yaml'
BLOBSTORE='CINF'
DEAFAULT_VAL_SPLIT = 0.2

args = get_args(CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_fold(
        all_shap_values: np.ndarray, 
        cfg, data:Data, fold:int, finetune_folder:str, fi_folder:str,
        logger)->np.ndarray:
    """Compute feature importance on one fold"""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    save_fold_folder = join(fi_folder, f'fold_{fold}')

    os.makedirs(save_fold_folder, exist_ok=True)

    # initialize datasets
    logger.info('Initializing datasets')
    dataset = BinaryOutcomeDataset(data.features, data.outcomes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, ) # SHAP will create n_permutations copies of the input
    
    # load BEHRT model
    modelmanager = ModelManager(cfg, model_path=fold_folder)
    checkpoint = modelmanager.load_checkpoint()
    modelmanager.load_model_config() 
    logger.info('Load best finetuned model to compute feature importance')
    finetuned_model = modelmanager.initialize_finetune_model(checkpoint, data)
    masker = EHRMasker(data.vocabulary)
    finetuned_model.eval()
    # to be able to handle the input, we pass 1 sample at a time
    # the explainer creates n_permutation copies of it, 
    # which we then pass as a batch to the modelwrapper
    with torch.no_grad():
        for i, batch in enumerate(get_tqdm(dataloader)):
            shap_batch_size = cfg.shap.get('batch_size', 16)
            wrapped_model = BEHRTWrapper(finetuned_model, batch)
            concepts = batch['concept'].numpy().reshape(-1, 1, batch['concept'].shape[1])
            explainer = shap.PermutationExplainer(wrapped_model, masker=masker, 
                                                  batch_size=shap_batch_size)
            
            # resize bs, seq_len to bs, 1, seq_len
            # batch_size, 1, seq_len. expected by SHAP
            n_permutations = concepts.shape[-1] * 2 + 1 # required by SHAP
            shap_values = explainer.shap_values(concepts, npermutations=n_permutations)
            all_shap_values = insert_concept_shap_values(all_shap_values, concepts, shap_values)
            log_most_important_features(all_shap_values, data.vocabulary, num_features=20) #for testing, can be removed later
            # if i > 100: # for testing
             #   break
    return all_shap_values


def cv_loop_predefined_splits(
        data: Data, 
        predefined_splits_dir: str, 
        finetune_folder:str,
        fi_folder: str,
        logger
        )->int:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    fold_dirs = [join(predefined_splits_dir, d) for d in os.listdir(predefined_splits_dir) if os.path.isdir(os.path.join(predefined_splits_dir, d)) and 'fold_' in d]
    all_shap_values = np.zeros(len(data.vocabulary))
    for fold_dir in fold_dirs:
        fold = int(split(fold_dir)[1].split('_')[1])
        logger.info(f"Training fold {fold}/{len(fold_dirs)}")
        all_shap_values = compute_fold(
            all_shap_values=all_shap_values,
            cfg=cfg, 
            data=data,
            fold=fold, 
            finetune_folder=finetune_folder, 
            fi_folder=fi_folder,
            logger=logger)
        
    return all_shap_values


def prepare_and_load_data():
    cfg, run, mount_context, azure_context = initialize_configuration_finetune(config_path, dataset_name=BLOBSTORE)
    date = datetime.now().strftime("%Y%m%d-%H%M")
    
    fi_folder = join(cfg.paths.output_path, f'feature_importance_shap_{date}')
    os.makedirs(fi_folder, exist_ok=True)

    finetune_folder = cfg.paths.get("model_path")
    logger = setup_logger(fi_folder, 'info.log')
    logger.info(f"Config Paths: {cfg.paths}")
    logger.info(f"Update config with pretrain and ft information.")
    cfg = update_test_cfg_with_pt_ft_cfgs(cfg, finetune_folder)
    cfg = fix_tmp_prefixes_for_azure_paths(cfg, azure_context)
    
    cfg.save_to_yaml(join(fi_folder, 'feature_importance_config.yaml'))
   
    log_config(cfg, logger)
    cfg.paths.run_name = split(fi_folder)[-1]

    if not cfg.data.get('preprocess', False):
        logger.info(f"Load processed test data from {cfg.paths.model_path}")
        data = Data.load_from_directory(cfg.paths.model_path, mode='')
    else:
        raise ValueError("Not implemented yet. Just use preprocessed data.")
        #dataset_preparer = DatasetPreparer(cfg)
        #data = dataset_preparer.prepare_finetune_data() 
    return data, mount_context, cfg, run, logger, fi_folder, azure_context


if __name__ == '__main__':
    data, mount_context, cfg, run, logger, fi_folder, azure_context = prepare_and_load_data()
    if 'test_features.pt' in os.listdir(cfg.paths.model_path):
        test_data = Data.load_from_directory(cfg.paths.model_path, mode='test')
    else:
        test_data = Data()
        
    shap_values = cv_loop_predefined_splits(data, 
                            predefined_splits_dir=cfg.paths.model_path, 
                            finetune_folder=cfg.paths.model_path,
                            fi_folder=fi_folder,
                            logger=logger)
    # print('all shap_values', shap_values)
    torch.save(shap_values, join(fi_folder, 'shap_values.pt'))
    log_most_important_features(shap_values, data.vocabulary, num_features=20)

    if cfg.env=='azure':
        save_to_blobstore(local_path='', # uses everything in 'outputs' 
                          remote_path=join(BLOBSTORE, cfg.paths.model_path))
        mount_context.stop()

    logger.info('Done')
