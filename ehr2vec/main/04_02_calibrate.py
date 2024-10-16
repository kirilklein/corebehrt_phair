from os.path import abspath, dirname, join

from ehr2vec.common.azure import save_to_blobstore
from ehr2vec.common.default_args import DEFAULT_BLOBSTORE
from ehr2vec.common.loader import load_config
from ehr2vec.common.setup import (
    get_args,
    initialize_configuration_finetune,
    setup_logger,
)
from ehr2vec.common.cli import override_config_from_cli
from ehr2vec.evaluation.calibration import compute_and_save_calibration

DEFAULT_CONFIG_NAME = "example_configs/04_02_calibrate.yaml"


args = get_args(DEFAULT_CONFIG_NAME)
config_path = join(dirname(dirname(abspath(__file__))), args.config_path)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    override_config_from_cli(cfg)
    cfg, run, mount_context, azure_context = initialize_configuration_finetune(
        cfg, dataset_name=cfg.get("project", DEFAULT_BLOBSTORE)
    )
    finetune_folder = cfg.paths.output_path
    logger = setup_logger(finetune_folder, f"calibration.log")
    logger.info("Starting calibration")
    compute_and_save_calibration(finetune_folder, cfg.calibration)
    logger.info("Done")
    if cfg.env == "azure":
        save_path = cfg.paths.model_path
        save_to_blobstore(
            local_path=cfg.paths.run_name,
            remote_path=join(
                cfg.get("project", DEFAULT_BLOBSTORE), save_path, cfg.paths.run_name
            ),
        )
        mount_context.stop()
    logger.info("Done")


if __name__ == "__main__":
    main(config_path)
