"""
Compute feature importance for 'concept' features using perturbation method.
Using the five folds from the cross-validation.
"""
import os
from datetime import datetime
from os.path import abspath, dirname, join, split
from typing import Dict, List

import numpy as np
import shap
import torch
from torch.utils.data import DataLoader

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.initialize import ModelManager
from ehr2vec.common.loader import load_and_select_splits
from ehr2vec.common.logger import log_config
from ehr2vec.common.setup import (fix_tmp_prefixes_for_azure_paths, get_args,
                                  initialize_configuration_finetune,
                                  setup_logger,
                                  update_test_cfg_with_pt_ft_cfgs)
from ehr2vec.common.utils import Data
from ehr2vec.data.dataset import BinaryOutcomeDataset
from ehr2vec.dataloader.collate_fn import dynamic_padding
from ehr2vec.feature_importance.shap import DeepSHAP_BEHRTWrapper
from ehr2vec.feature_importance.shap_utils import (insert_shap_values,
                                                   split_batch_into_bg_and_fg)
from ehr2vec.feature_importance.utils import log_most_important_features_deep
from ehr2vec.trainer.utils import get_tqdm

CONFIG_NAME = 'feature_importance/shap_deep_feature_importance.yaml'
BLOBSTORE='CINF'

args = get_args(CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_fold(
        all_shap_values: Dict[str, np.ndarray],
        val_data: Data, 
        cfg, fold:int, 
        finetune_folder:str, fi_folder:str,
        logger)->np.ndarray:
    """Compute feature importance on one fold"""
    fold_folder = join(finetune_folder, f'fold_{fold}')
    save_fold_folder = join(fi_folder, f'fold_{fold}')

    os.makedirs(save_fold_folder, exist_ok=True)

    # initialize datasets
    logger.info('Initializing datasets')
    dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)

    batch_size = cfg.dataloader.get('batch_size',512)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dynamic_padding) # SHAP will create n_permutations copies of the input
    
    # load BEHRT model
    modelmanager = ModelManager(cfg, model_path=fold_folder)
    checkpoint = modelmanager.load_checkpoint()
    modelmanager.load_model_config() 
    logger.info('Load best finetuned model to compute feature importance')
    finetuned_model = modelmanager.initialize_finetune_model(checkpoint, data)
    finetuned_model.eval()
    finetuned_model.to(device)
    
    wrapped_model = DeepSHAP_BEHRTWrapper(finetuned_model)
    
    for i, batch in enumerate(get_tqdm(dataloader)):
        batch.pop('target')

        n_bg_samples = cfg.shap.get('n_background_samples', 100)
        concepts = batch['concept'][n_bg_samples:]
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_emb = wrapped_model.get_embeddings(batch)
        bg_batch_emb, fg_batch_emb = split_batch_into_bg_and_fg(
            batch_emb, n_bg_samples=n_bg_samples) 
        bg_batch_emb_ls = list(bg_batch_emb.values())
        fg_batch_emb_ls = list(fg_batch_emb.values())
        explainer = shap.DeepExplainer(wrapped_model, data=bg_batch_emb_ls)
        
        shap_values = explainer.shap_values(X=fg_batch_emb_ls)
        shap_values_dict = {feature: shap_values[i] for i, feature in enumerate(fg_batch_emb.keys())}
                
        attention_mask = batch['attention_mask'][n_bg_samples:]
        all_shap_values = insert_shap_values(
            all_shap_values, concepts, shap_values_dict, attention_mask)
        log_most_important_features_deep(all_shap_values, data.vocabulary, num_features=20) #for testing, can be removed later
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
    all_shap_values = {'concept':np.zeros(len(data.vocabulary)),
                       'age':np.zeros(1),
                       'abspos':np.zeros(1),
                       'segment':np.zeros(1),
    }
    for fold_dir in fold_dirs:
        fold = int(split(fold_dir)[1].split('_')[1])
        logger.info(f"Training fold {fold}/{len(fold_dirs)}")
        _, val_data = load_and_select_splits(fold_dir, data)
        all_shap_values = compute_fold(
            all_shap_values=all_shap_values,
            val_data=val_data,
            cfg=cfg, 
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
    torch.save(shap_values, join(fi_folder, 'shap_values.pt'))
    log_most_important_features_deep(shap_values, data.vocabulary, num_features=20)

    if cfg.env=='azure':
        save_to_blobstore(local_path='', # uses everything in 'outputs' 
                          remote_path=join(BLOBSTORE, cfg.paths.model_path))
        mount_context.stop()

    logger.info('Done')
