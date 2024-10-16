try:
    import wandb

    use_wandb = True
except ImportError:
    use_wandb = False
from ehr2vec.common.config import Config


def initialize_wandb(run, cfg: Config, wandb_kwargs: dict):
    """
    Initialize Wandb if available, else return run.
    Return the run object.
    """
    if not use_wandb:
        return run

    wandb_config = create_wandb_config(cfg)
    wandb.init(config=wandb_config, **wandb_kwargs)
    return wandb.run


def create_wandb_config(cfg: Config) -> dict:
    """
    Create a wandb config dictionary from a Config object.
    """
    config_keys = [
        "trainer_args",
        "model",
        "optimizer",
        "data",
        "test_args",
        "scheduler",
        "outcome",
        "estimator",
        "paths",
        "ps_file",
    ]
    wandb_config = {key: {} for key in config_keys}

    for key in wandb_config:
        if hasattr(cfg, key):
            wandb_config[key] = (
                cfg[key] if isinstance(cfg[key], dict) else {"value": cfg[key]}
            )

    return wandb_config


def finish_wandb() -> None:
    if use_wandb:
        wandb.finish()
