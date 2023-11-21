import logging
import os
from os.path import join
from typing import Tuple

import torch
from common.config import Config, load_config
from common.utils import Data
from data.utils import Utilities
from transformers import BertConfig

logger = logging.getLogger(__name__)  # Get the logger for this module

VOCABULARY_FILE = 'vocabulary.pt'
TREE_FILE = 'tree.pt'
TREE_MATRIX_FILE = 'tree_matrix.pt'
CHECKPOINT_FOLDER = 'checkpoints'
VAL_RATIO = 0.2
# TODO: Add option to load test set only!
        
def load_checkpoint_and_epoch(cfg: Config)->Tuple:
    model_path = cfg.paths.get('model_path', None)
    checkpoint = ModelLoader(cfg).load_checkpoint() if model_path is not None else None
    epoch = Utilities.get_last_checkpoint_epoch(join(model_path, CHECKPOINT_FOLDER)) if model_path is not None else None
    return checkpoint, epoch

def load_model_cfg_from_checkpoint(cfg: Config, config_name: str)->None:
    """If training from checkpoint, we need to get the old config"""
    model_path = cfg.paths.get('model_path', None)
    if model_path is not None: # if we are training from checkpoint, we need to load the old config
        old_cfg = load_config(join(cfg.paths.model_path, config_name))
        cfg.model = old_cfg.model

class FeaturesLoader():
    def __init__(self, cfg):
        self.path_cfg = cfg.paths
        self.cfg = cfg
    def load_tokenized_data(self)->Tuple[dict, list, dict, list, dict]:
        tokenized_dir = self.path_cfg.get('tokenized_dir', 'tokenized')
        logger.info('Loading tokenized data from %s', tokenized_dir)
        tokenized_data_path = join(self.path_cfg.data_path, tokenized_dir)
        if os.path.exists(join(tokenized_data_path, 'tokenized_train.pt')): # this is needed for backwards compatibility. New splits are pretrain and finetune_test.
            split = "train"
        else:
            split = "pretrain"
        logger.info("Loading tokenized data train")
        train_features  = torch.load(join(tokenized_data_path, f'tokenized_{split}.pt'))
        train_pids = torch.load(join(tokenized_data_path,  f'pids_{split}.pt'))
        
        if os.path.exists(join(tokenized_data_path, 'tokenized_val.pt')): 
            logger.info("Loading tokenized data val")
            val_features = torch.load(join(tokenized_data_path, 'tokenized_val.pt'))
            val_pids = torch.load(join(tokenized_data_path, 'pids_val.pt'))
        else:
            logger.info("No validation set found. Split train into train and val.")
            train_features, train_pids, val_features, val_pids = Utilities.split_train_val(train_features, train_pids, val_ratio=self.cfg.data.get('val_ratio', VAL_RATIO))
        logger.info(f"Train size: {len(train_pids)}, Val size: {len(val_pids)}")
        logger.info("Loading vocabulary")
        try:
            vocabulary = torch.load(join(tokenized_data_path, VOCABULARY_FILE))
        except:
            vocabulary = torch.load(join(self.path_cfg.data_path, VOCABULARY_FILE))
        return train_features, train_pids, val_features, val_pids, vocabulary

    def load_tokenized_finetune_data(self, mode: str)->Data:
        """Load features for finetuning"""
        tokenized_dir = self.path_cfg.get('tokenized_dir', 'tokenized')
        tokenized_file = self.path_cfg.get('tokenized_file', 'tokenized_val.pt')
        tokenized_pids = self.path_cfg.get('tokenized_pids', 'pids_val.pt')
        
        tokenized_data_path = join(self.path_cfg.data_path, tokenized_dir)
        
        logger.info(f"Loading tokenized data from {tokenized_data_path}")
        features  = torch.load(join(tokenized_data_path, tokenized_file))
        pids = torch.load(join(tokenized_data_path,  tokenized_pids))
        
        logger.info("Loading vocabulary")
        try:
            vocabulary = torch.load(join(tokenized_data_path, VOCABULARY_FILE))
        except:
            vocabulary = torch.load(join(self.path_cfg.data_path, VOCABULARY_FILE))
        return Data(features, pids, vocabulary=vocabulary, mode=mode)
    
    def load_outcomes(self)->Tuple[dict, dict]:
        logger.info(f'Load outcomes from {self.path_cfg.outcome}')
        censoring_timestamps_path = self.path_cfg.censor if self.path_cfg.get("censor", False) else self.path_cfg.outcome
        logger.info(f'Load censoring_timestamps from {censoring_timestamps_path}')
        outcomes = torch.load(self.path_cfg.outcome)
        censor_outcomes = torch.load(self.path_cfg.censor) if self.path_cfg.get('censor', False) else outcomes   
        return outcomes, censor_outcomes

    def load_tree(self)->Tuple[dict, torch.Tensor, dict]:
        hierarchical_path = join(self.path_cfg.data_path, 
                                 self.path_cfg.hierarchical_dir)
        tree = torch.load(join(hierarchical_path, TREE_FILE))
        tree_matrix = torch.load(join(hierarchical_path, TREE_MATRIX_FILE))
        h_vocabulary = torch.load(join(hierarchical_path, VOCABULARY_FILE))
        return tree, tree_matrix, h_vocabulary 
    
    def load_finetune_data(self, path: str=None, mode: str='val')->Data:
        """Load features for finetuning"""
        path = self.path_cfg.finetune_features_path if path is None else path
        features = torch.load(join(path, f'features.pt'))
        outcomes = torch.load(join(path, f'outcomes.pt'))
        pids = torch.load(join(path, f'pids.pt'))
        vocabulary = torch.load(join(path, 'vocabulary.pt'))
        return Data(features, pids, outcomes, vocabulary=vocabulary, mode=mode)

class ModelLoader():
    def __init__(self, cfg: Config, model_path: str=None):
        """Load model from config and checkpoint."""
        self.cfg = cfg
        if model_path is not None:
            self.model_path = model_path
        elif self.cfg.paths.get('model_path', None) is not None:
            self.model_path = self.cfg.paths.model_path
        else:
            self.model_path = None
    
    def load_model(self, model_class, add_config:dict={}, checkpoint: dict=None, kwargs={}):
        """Load model from config and checkpoint. model_class is the class of the model to be loaded."""
        checkpoint = self.load_checkpoint() if checkpoint is None else checkpoint
        # Load the config from file
        config = BertConfig.from_pretrained(self.model_path) 
        config.update(add_config)
        model = model_class(config, **kwargs)
        
        return self.load_state_dict_into_model(model, checkpoint)
    
    def load_state_dict_into_model(self, model: torch.nn.Module, checkpoint: dict)->torch.nn.Module:
        load_result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        missing_keys = load_result.missing_keys

        if len([k for k in missing_keys if k.startswith('embeddings')])>0:
            pretrained_model_embeddings = model.embeddings.__class__.__name__
            raise ValueError(f"Embeddings not loaded. Ensure that model.behrt_embeddings is compatible with pretrained model embeddings {pretrained_model_embeddings}.")
        logger.warning("missing state dict keys: %s", missing_keys)
        return model

    def load_checkpoint(self)->dict:
        """Load checkpoint, if checkpoint epoch provided. Else load last checkpoint."""
        checkpoints_dir = join(self.model_path, CHECKPOINT_FOLDER)
        checkpoint_epoch = self.get_checkpoint_epoch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint_path = join(checkpoints_dir,f'checkpoint_epoch{checkpoint_epoch}_end.pt')
        logger.info("Loading checkpoint from %s", checkpoint_path)
        return torch.load(checkpoint_path, map_location=device)
    
    def get_checkpoint_epoch(self)->int:
        """Get checkpoint epoch from config or return the last checkpoint_epoch for this model."""
        checkpoint_epoch = self.cfg.paths.get('checkpoint_epoch', None)
        if checkpoint_epoch is None:
            logger.info("No checkpoint provided. Loading last checkpoint.")
            checkpoint_epoch = Utilities.get_last_checkpoint_epoch(join(
                self.model_path, CHECKPOINT_FOLDER))
        return checkpoint_epoch




