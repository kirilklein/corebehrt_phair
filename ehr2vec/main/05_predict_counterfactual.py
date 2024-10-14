import os
from datetime import datetime
from os.path import abspath, dirname, join, split

import torch

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.initialize import ModelManager
from ehr2vec.common.loader import load_and_select_splits
from ehr2vec.common.logger import log_config
from ehr2vec.common.setup import (
    fix_tmp_prefixes_for_azure_paths,
    get_args,
    initialize_configuration_finetune,
    setup_logger,
    update_test_cfg_with_pt_ft_cfgs,
)
from ehr2vec.common.utils import Data
from ehr2vec.data.dataset import BinaryOutcomeDataset
from ehr2vec.data_fixes.truncate import Truncator
from ehr2vec.double_robust.counterfactual import create_counterfactual_data
from ehr2vec.double_robust.save import save_combined_predictions_evaluation
from ehr2vec.evaluation.encodings import EHRTester
from ehr2vec.evaluation.utils import save_data
from ehr2vec.common.default_args import DEFAULT_BLOBSTORE
from ehr2vec.common.loader import load_config

DEFAULT_CONFIG_NAME = "example_configs/05_predict_counterfactual.yaml"


args = get_args(DEFAULT_CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)


def predict_fold(
    cfg,
    finetune_folder: str,
    counterfactual_folder: str,
    fold: int,
    counterfactual_val_data: Data,
    run=None,
    logger=None,
) -> None:
    """Predict on validation data for one fold."""
    fold_folder = join(finetune_folder, f"fold_{fold}")
    save_folder = join(counterfactual_folder, f"fold_{fold}")

    os.makedirs(save_folder, exist_ok=True)
    torch.save(counterfactual_val_data.pids, join(save_folder, f"val_pids.pt"))
    logger.info(f"Predicting for fold {fold}")

    counterfactual_val_dataset = BinaryOutcomeDataset(
        counterfactual_val_data.features, counterfactual_val_data.outcomes
    )

    modelmanager = ModelManager(cfg, model_path=fold_folder)
    checkpoint = modelmanager.load_checkpoint()
    modelmanager.load_model_config()
    logger.info("Load best finetuned model to compute predictions")
    model = modelmanager.initialize_finetune_model(
        checkpoint, counterfactual_val_dataset
    )

    tester = EHRTester(
        model=model,
        test_dataset=counterfactual_val_dataset,
        args=cfg.tester_args,
        metrics=cfg.get("metrics", None),
        cfg=cfg,
        run=run,
        logger=logger,
        accumulate_logits=True,
        test_folder=save_folder,
        mode=f"val",
    )

    tester.evaluate(modelmanager.get_epoch(), mode="test")


def cv_predict_loop(
    counterfactual_data: Data,
    finetune_folder: str,
    counterfactual_folder: str,
    n_splits: int,
    cfg=None,
    logger=None,
    run=None,
) -> None:
    """Loop over cross validation folds. Predict on validation data for each fold."""
    for fold in range(1, n_splits + 1):
        logger.info(f"Processing fold {fold}/{n_splits}")
        fold_dir = join(finetune_folder, f"fold_{fold}")
        _, counterfactual_val_data = load_and_select_splits(
            fold_dir, counterfactual_data
        )
        predict_fold(
            cfg,
            finetune_folder,
            counterfactual_folder,
            fold,
            counterfactual_val_data,
            run,
            logger,
        )


def main(config_path: str):
    cfg = load_config(config_path)
    cfg, run, mount_context, azure_context = initialize_configuration_finetune(
        config_path, dataset_name=cfg.get("project", DEFAULT_BLOBSTORE)
    )

    date = datetime.now().strftime("%Y%m%d-%H%M")
    counterfactual_folder = join(
        cfg.paths.output_path, f"counterfactual_predictions_{date}"
    )
    os.makedirs(counterfactual_folder, exist_ok=True)

    finetune_folder = cfg.paths.get("model_path")
    logger = setup_logger(counterfactual_folder, "counterfactual_info.log")
    logger.info(f"Config Paths: {cfg.paths}")
    logger.info(f"Update config with pretrain and ft information.")
    cfg = update_test_cfg_with_pt_ft_cfgs(cfg, finetune_folder)
    cfg = fix_tmp_prefixes_for_azure_paths(cfg, azure_context)
    cfg.save_to_yaml(join(counterfactual_folder, "counterfactual_config.yaml"))

    log_config(cfg, logger)
    cfg.paths.run_name = split(counterfactual_folder)[-1]

    logger.info(f"Load processed data from {cfg.paths.model_path}")
    data = Data.load_from_directory(cfg.paths.model_path, mode="")
    counterfactual_data = create_counterfactual_data(
        data, cfg.data.counterfactual.exposure_regex
    )
    counterfactual_data.features = Truncator(cfg.data.truncation_len, data.vocabulary)(
        counterfactual_data.features
    )
    counterfactual_data.mode = "val"  # important so we load the correct pids
    save_data(counterfactual_data, counterfactual_folder)
    n_splits = len([d for d in os.listdir(finetune_folder) if d.startswith("fold_")])
    cv_predict_loop(
        counterfactual_data,
        finetune_folder,
        counterfactual_folder,
        n_splits,
        cfg,
        logger,
        run,
    )
    save_combined_predictions_evaluation(n_splits, counterfactual_folder, mode="val")
    if cfg.env == "azure":
        save_to_blobstore(
            local_path="",  # uses everything in 'outputs'
            remote_path=join(
                cfg.get("project", DEFAULT_BLOBSTORE), fix_tmp_prefixes_for_azure_paths(cfg.paths.model_path)
            ),
        )
        mount_context.stop()
    logger.info("Done")


if __name__ == "__main__":
    main(config_path)
