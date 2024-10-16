import argparse
import logging
import os
import sys
import uuid
from os.path import join, split
from pathlib import Path
from shutil import copyfile
from typing import Tuple

from ehr2vec.common.azure import AzurePathContext
from ehr2vec.common.config import Config, load_config

logger = logging.getLogger(__name__)  # Get the logger for this module

CHECKPOINTS_DIR = "checkpoints"


def get_args(default_config_name, default_run_name=None):
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=join("configs", default_config_name),
        help="Configuration file, path relative to configs/",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=(
            default_run_name if default_run_name else default_config_name.split(".")[0]
        ),
    )
    args, _ = parser.parse_known_args()
    if not args.config_path.startswith("configs"):
        args.config_path = join("configs", args.config_path)
    if not args.config_path.endswith(".yaml"):
        args.config_path += ".yaml"

    return args


def setup_logger(dir: str, log_file: str = "info.log"):
    """Sets up the logger."""
    os.makedirs(dir, exist_ok=True)  # Ensure the directory exists
    log_path = join(dir, log_file)

    # Configure file handler
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    # Configure stream handler (for stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # Log a test message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_path}")

    return logger


def copy_data_config(cfg: Config, run_folder: str) -> None:
    """
    Copy data_config.yaml to run folder.
    By default copy from tokenized folder, if not available, copy from data folder.
    """
    tokenized_dir_name = cfg.paths.get("tokenized_dir", "tokenized")

    try:
        copyfile(
            join(cfg.paths.data_path, tokenized_dir_name, "data_cfg.yaml"),
            join(run_folder, "data_config.yaml"),
        )
    except:
        copyfile(
            join(cfg.paths.data_path, "data_config.yaml"),
            join(run_folder, "data_config.yaml"),
        )


def copy_pretrain_config(cfg: Config, run_folder: str) -> None:
    """
    Copy pretrain_config.yaml to run folder.
    """
    pretrain_model_path = cfg.paths.get("pretrain_model_path")
    model_path = cfg.paths.get("model_path")
    pt_cfg_name = "pretrain_config.yaml"

    pretrain_cfg_path = (
        pretrain_model_path if pretrain_model_path is not None else model_path
    )
    if pretrain_cfg_path is None:
        raise ValueError(
            "Either pretrain_model_path or model_path must be specified in the configuration."
        )

    if os.path.exists(join(pretrain_cfg_path, pt_cfg_name)):
        pretrain_cfg_path = join(pretrain_cfg_path, pt_cfg_name)
    elif os.path.exists(join(pretrain_cfg_path, "fold_1", pt_cfg_name)):
        pretrain_cfg_path = join(pretrain_cfg_path, "fold_1", pt_cfg_name)
    else:
        raise FileNotFoundError(
            f"Could not find pretrain config in {pretrain_cfg_path}"
        )
    try:
        copyfile(pretrain_cfg_path, join(run_folder, pt_cfg_name))
    except:
        logger.warning(
            f"Could not copy pretrain config from {pretrain_cfg_path} to {run_folder}"
        )


def update_test_cfg_with_pt_ft_cfgs(cfg: Config, finetune_folder: str) -> Config:
    """
    Update config with pretrain and ft information.
    Used for testing/feature importance calculation after finetuning.
    """
    finetune_config = load_config(join(finetune_folder, "finetune_config.yaml"))
    pretrain_config = load_config(join(finetune_folder, "pretrain_config.yaml"))
    if cfg.data.get("preprocess", False):
        cfg.data.update(finetune_config.data)
        cfg.outcome = finetune_config.outcome
        cfg.data.update(pretrain_config.data)
    cfg.model = finetune_config.model
    cfg.paths.update(finetune_config.paths)
    cfg.model.update(pretrain_config.model)
    return cfg


def remove_tmp_prefixes(path: str) -> Path:
    """Remove 'tmp' prefixes from a path."""
    path_parts = Path(path).parts
    start_index = next(
        (i for i, part in enumerate(path_parts) if not part.startswith("tmp")), 1
    )
    return Path(*path_parts[start_index:])


def remove_tmp_prefixes_from_path_cfg(path_cfg: Config) -> Config:
    """Update all paths in a cfg by removing 'tmp' prefixes."""
    return Config(
        {
            key: str(remove_tmp_prefixes(value))
            for key, value in path_cfg.items()
            if "tmp" in value
        }
    )


def fix_tmp_prefixes_for_azure_paths(
    cfg: Config, azure_context: AzurePathContext
) -> Config:
    """
    Fix paths in config for azure.
    The saved finetune configs have /tmp/tmp.../actual/path
    after removing it, we need to prepend the new mounted path
    to every path in the config.
    """
    if cfg.env == "azure":
        cfg.paths = remove_tmp_prefixes_from_path_cfg(cfg.paths)
        azure_context.cfg = cfg
        cfg, _, _ = azure_context.azure_finetune_setup()
    return cfg


def initialize_configuration_finetune(cfg: Config, dataset_name: str):
    """
    Load and adjust the configuration. Used if finetune models are loaded.
    E.g. in test or feature importance scripts.
    """
    azure_context = AzurePathContext(cfg, dataset_name=dataset_name)
    cfg, run, mount_context = azure_context.azure_finetune_setup()
    if cfg.env == "azure":
        cfg.paths.output_path = "outputs"
    else:
        cfg.paths.output_path = cfg.paths.model_path
    return cfg, run, mount_context, azure_context


def initialize_configuration_effect_estimation(cfg: Config, dataset_name: str):
    """
    Load and adjust the configuration. Used if finetune models are loaded.
    E.g. in test or feature importance scripts.
    """
    azure_context = AzurePathContext(cfg, dataset_name=dataset_name)
    cfg, run, mount_context = azure_context.azure_estimate_setup()
    if cfg.env == "azure":
        cfg.paths.output_path = "outputs"
    else:
        if cfg.paths.output_path is None:
            raise ValueError("output_path must be provided in the configuration.")
    return cfg, run, mount_context, azure_context


class DirectoryPreparer:
    """Prepares directories for training and evaluation."""

    def __init__(self, config_path) -> None:
        self.config_path = config_path

    def create_directory_and_copy_config(
        self, output_dir: str, new_config_name: str
    ) -> logging.Logger:
        """Creates output directory and copies config file"""
        os.makedirs(output_dir, exist_ok=True)
        destination = join(output_dir, new_config_name)
        copyfile(self.config_path, destination)
        return setup_logger(output_dir)

    def prepare_directory(self, cfg: Config):
        """Creates output directory and copies config file"""
        logger = self.create_directory_and_copy_config(
            cfg.output_dir, "data_config.yaml"
        )
        os.makedirs(join(cfg.output_dir, "features"), exist_ok=True)
        os.makedirs(join(cfg.output_dir, cfg.tokenized_dir_name), exist_ok=True)
        copyfile(
            self.config_path,
            join(cfg.output_dir, cfg.tokenized_dir_name, "data_config.yaml"),
        )
        return logger

    def prepare_directory_outcomes(self, outcome_dir: str, outcomes_name: str):
        """Creates output directory for outcomes and copies config file"""
        return self.create_directory_and_copy_config(
            outcome_dir, f"outcome_{outcomes_name}_config.yaml"
        )

    def prepare_embedding_directory(self, cfg: Config):
        """Creates output directory and copies config file"""
        return self.create_directory_and_copy_config(cfg.output_dir, "emb_config.yaml")

    def prepare_encodings_directory(self, cfg: Config):
        """Creates output directory and copies config file"""
        return self.create_directory_and_copy_config(
            cfg.output_dir, "encodings_config.yaml"
        )

    @staticmethod
    def setup_run_folder(
        cfg: Config, run_folder: str = None
    ) -> Tuple[logging.Logger, str]:
        """Creates a run folder and checkpoints folder inside it. Returns logger and run folder path."""
        # Generate unique run_name if not provided
        run_name = (
            cfg.paths.run_name if hasattr(cfg.paths, "run_name") else uuid.uuid4().hex
        )
        if run_folder is None:
            run_folder = join(cfg.paths.output_path, run_name)

        os.makedirs(run_folder, exist_ok=True)
        os.makedirs(join(run_folder, CHECKPOINTS_DIR), exist_ok=True)
        logger = setup_logger(run_folder)
        logger.info(f"Run folder: {run_folder}")
        return logger, run_folder

    @staticmethod
    def adjust_paths_for_finetune(cfg: Config) -> Config:
        """
        Adjusts the following paths in the configuration for the finetune environment:
        - output_path: set to pretrain_model_path
        - run_name: constructed according to setting
        """
        pretrain_model_path = cfg.paths.get("pretrain_model_path")
        model_path = cfg.paths.get("model_path")
        if model_path is not None:
            model_path = split(model_path)[
                0
            ]  # Use directory of model path (the model path will be constructed in the finetune script)
        save_folder_path = cfg.paths.get("save_folder_path")

        # Determine the output path with a priority order
        output_path = pretrain_model_path or model_path or save_folder_path
        if output_path is None:
            raise ValueError(
                "Either pretrain_model_path, model_path, or save_folder_path must be provided."
            )
        cfg.paths.output_path = output_path
        cfg.paths.run_name = DirectoryPreparer.construct_finetune_model_dir_name(cfg)
        return cfg

    @staticmethod
    def get_event_name(path: str) -> str:
        return split(path)[-1].strip(".csv")

    @staticmethod
    def construct_finetune_model_dir_name(cfg: Config) -> str:
        """
        Constructs the name of the finetune model directory.
        Based on the outcome type, the censor type, and the number of hours pre- or post- outcome.
        """
        outcome_name = DirectoryPreparer.get_event_name(cfg.paths.outcome)
        censor_name = (
            DirectoryPreparer.get_event_name(cfg.paths.exposure)
            if cfg.paths.get("exposure", False)
            else outcome_name
        )
        finetune_folder_name = f"finetune_{outcome_name}_censored_"

        n_hours_censor = cfg.outcome.get("n_hours_censoring", None)
        n_hours_str = (
            DirectoryPreparer.handle_n_hours(n_hours_censor)
            if n_hours_censor is not None
            else "at"
        )

        if cfg.outcome.get("index_date", None) is not None:
            censor_name = DirectoryPreparer.handle_index_date(cfg.outcome.index_date)

        finetune_folder_name = f"{finetune_folder_name}{n_hours_str}_{censor_name}"

        n_hours_start_follow_up = cfg.outcome.get("n_hours_follow_up", None)
        n_hours_follow_up_str = (
            DirectoryPreparer.handle_n_hours(n_hours_start_follow_up)
            if n_hours_start_follow_up is not None
            else "at"
        )

        finetune_folder_name = f"{finetune_folder_name}_followup_start_{n_hours_follow_up_str}_index_date_{cfg.paths.run_name}"
        return finetune_folder_name

    @staticmethod
    def handle_n_hours(n_hours: int) -> str:
        days = True if abs(n_hours) > 48 else False
        window = int(abs(n_hours / 24)) if days else abs(n_hours)
        days_hours = "days" if days else "hours"
        pre_post = "pre" if n_hours < 0 else "post"
        return f"{window}_{days_hours}_{pre_post}"

    @staticmethod
    def handle_index_date(n_hours: dict) -> str:
        censor_event = [f"{k}{v}" for k, v in n_hours.items() if v is not None]
        return "_".join(censor_event)
