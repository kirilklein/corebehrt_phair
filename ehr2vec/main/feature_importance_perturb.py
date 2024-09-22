"""
Compute feature importance for 'concept' features using perturbation method.
Using the five folds from the cross-validation.
"""
import os
from datetime import datetime
from os.path import abspath, dirname, join, split

import torch

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.initialize import ModelManager
from ehr2vec.common.loader import load_and_select_splits
from ehr2vec.common.logger import log_config
from ehr2vec.common.saver import Saver
from ehr2vec.common.setup import (fix_tmp_prefixes_for_azure_paths, get_args,
                                  initialize_configuration_finetune,
                                  setup_logger,
                                  update_test_cfg_with_pt_ft_cfgs)
from ehr2vec.common.utils import Data, compute_number_of_warmup_steps
from ehr2vec.data.dataset import BinaryOutcomeDataset
from ehr2vec.data.split import split_data_into_train_val
from ehr2vec.evaluation.visualization import plot_most_important_features
from ehr2vec.feature_importance.perturb import PerturbationModel
from ehr2vec.feature_importance.perturb_utils import average_sigmas, log_most_important_features_for_perturbation_model, compute_concept_frequency
from ehr2vec.trainer.trainer import EHRTrainer


CONFIG_NAME = 'feature_importance/finetune_feature_importance.yaml'
BLOBSTORE='CINF'
DEAFAULT_VAL_SPLIT = 0.2

args = get_args(CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def finetune_fold(
        cfg, data:Data, 
        fold:int, finetune_folder:str, fi_folder:str, 
        run, test_data: Data=None, )->None:
    """Finetune model on one fold"""

    train_data, val_data = split_data_into_train_val(data, cfg.data.get('val_split', DEAFAULT_VAL_SPLIT))
    
    fold_folder = join(finetune_folder, f'fold_{fold}')
    save_fold_folder = join(fi_folder, f'fold_{fold}')
    if 'scheduler' in cfg:
        logger.info('Computing number of warmup steps')
        compute_number_of_warmup_steps(cfg, len(train_data))
    os.makedirs(save_fold_folder, exist_ok=True)
    os.makedirs(join(save_fold_folder, "checkpoints"), exist_ok=True)

    logger.info("Saving patient numbers")
    logger.info("Saving pids")
    torch.save(train_data.pids, join(save_fold_folder, 'train_pids.pt'))
    torch.save(val_data.pids, join(save_fold_folder, 'val_pids.pt'))
    if len(test_data) > 0:
        torch.save(test_data.pids, join(fold_folder, 'test_pids.pt'))
    Saver(fi_folder).save_patient_nums(train_data, val_data)

    # initialize datasets
    logger.info('Initializing datasets')
    train_dataset = BinaryOutcomeDataset(train_data.features, train_data.outcomes)
    val_dataset = BinaryOutcomeDataset(val_data.features, val_data.outcomes)
    test_dataset = BinaryOutcomeDataset(test_data.features, test_data.outcomes) if len(test_data) > 0 else None

    # load BEHRT model
    modelmanager = ModelManager(cfg, model_path=fold_folder)
    checkpoint = modelmanager.load_checkpoint()
    modelmanager.load_model_config() 
    logger.info('Load best finetuned model to compute test scores')
    finetuned_model = modelmanager.initialize_finetune_model(checkpoint, train_dataset)
    
    if cfg.model.get('scale_with_frequency', False):
        concept_frequency = compute_concept_frequency(train_data.features, train_data.vocabulary) 
        torch.save(concept_frequency, join(fi_folder, 'concept_frequency.pt'))
    else:
        concept_frequency = None
    # initialize perturbation model
    perturbation_model = PerturbationModel(finetuned_model, cfg.model, concept_frequency)
    logger.info("Trainable parameters in the perturbation model")
    for name, param in perturbation_model.named_parameters():
        if param.requires_grad:
            logger.info(f"{name}: {param.size()}")
    assert len(train_data.vocabulary)==len(perturbation_model.get_sigmas_weights()), f"Vocabulary size {len(train_data.vocabulary)} does not match sigmas size {len(perturbation_model.get_sigmas_weights())}"
    modelmanager.model_path = None # to initialize training components form scratch
    
    optimizer, sampler, scheduler, cfg = modelmanager.initialize_training_components(
        perturbation_model, train_dataset)

    logger.info("Optimizer parameters")
    for group in optimizer.param_groups:
        logger.info("Optimizer group:")
        for param in group['params']:
            logger.info(f"size {param.size()}")

    trainer = EHRTrainer( 
        model=perturbation_model, 
        optimizer=optimizer,
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        test_dataset=None, # test only after training
        args=cfg.trainer_args,
        metrics=cfg.metrics,
        sampler=sampler,
        scheduler=scheduler,
        cfg=cfg,
        run=run,
        logger=logger,
        accumulate_logits=True,
        run_folder=save_fold_folder,
    )
    trainer.train()
    if test_dataset is not None:
        trainer.test_dataset = test_dataset
        trainer._evaluate(checkpoint['epoch'], mode='test')
    # save sigmas from the model
    perturbation_model.save_sigmas(join(fi_folder, f'sigmas_fold_{fold}.pt'))
    log_most_important_features_for_perturbation_model(perturbation_model, train_data.vocabulary)


def _limit_patients(indices_or_pids: list, split: str)->list:
    if f'number_of_{split}_patients' in cfg.data:
        number_of_patients = cfg.data.get(f'number_of_{split}_patients')
        if len(indices_or_pids) >= number_of_patients:
            indices_or_pids = indices_or_pids[:number_of_patients]
            logger.info(f"Number of {split} patients is limited to {number_of_patients}")
        else:
            raise ValueError(f"Number of train patients is {len(indices_or_pids)}, but should be at least {number_of_patients}")
    return indices_or_pids

def cv_loop_predefined_splits(
        data: Data, 
        predefined_splits_dir: str, 
        finetune_folder:str,
        fi_folder: str,
        run,
        test_data: Data,
        )->int:
    """Loop over predefined splits"""
    # find fold_1, fold_2, ... folders in predefined_splits_dir
    fold_dirs = [join(predefined_splits_dir, d) for d in os.listdir(predefined_splits_dir) if os.path.isdir(os.path.join(predefined_splits_dir, d)) and 'fold_' in d]
    N_SPLITS = len(fold_dirs)
    for fold_dir in fold_dirs:
        fold = int(split(fold_dir)[1].split('_')[1])
        logger.info(f"Training fold {fold}/{len(fold_dirs)}")
        _, val_data = load_and_select_splits(fold_dir, data)
        val_pids = _limit_patients(val_data.pids, 'val')
        if len(val_pids)<len(val_data.pids):
            val_data = data.select_data_subset_by_pids(val_pids, mode='val')
        # use only validation data to train the perturbatino model    
        finetune_fold(cfg=cfg, 
                      val_data=val_data, 
                      fold=fold, 
                      finetune_folder=finetune_folder, 
                      fi_folder=fi_folder, run=run, 
                      test_data=test_data)
        
    return N_SPLITS


def prepare_and_load_data():
    cfg, run, mount_context, azure_context = initialize_configuration_finetune(config_path, dataset_name=BLOBSTORE)
    date = datetime.now().strftime("%Y%m%d-%H%M")
    
    fi_folder = join(cfg.paths.output_path, f'feature_importance_perturb_{date}')
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
        
    n_splits = cv_loop_predefined_splits(data, 
                            predefined_splits_dir=cfg.paths.model_path, 
                            finetune_folder=cfg.paths.model_path,
                            fi_folder=fi_folder,
                            run=run,
                            test_data=test_data)
    sigmas = average_sigmas(fi_folder, n_splits)
    torch.save(sigmas, join(fi_folder, 'sigmas_average.pt')) # save averaged sigmas
    plot_most_important_features(data.vocabulary, sigmas, fi_folder)
    if cfg.env=='azure':
        save_to_blobstore(local_path='', # uses everything in 'outputs' 
                          remote_path=join(BLOBSTORE, cfg.paths.model_path))
        mount_context.stop()

    logger.info('Done')
