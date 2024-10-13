try:
    import wandb

    use_wandb = True
except ImportError:
    use_wandb = False
PROJECT_NAME = "PHAIR"


def initialize_wandb(run, cfg):
    """
    Initialize Wand if available, else return run.
    Return a tuple of (use_wandb, run)
    """
    if not use_wandb:
        return run
    else:
        wandb_config = {}
        if hasattr(cfg, "trainer_args"):
            wandb_config.update(cfg.trainer_args)
        if hasattr(cfg, "test_args"):
            wandb_config.update(cfg.test_args)
        if hasattr(cfg, "model"):
            wandb_config.update(cfg.model)
        wandb.init(project=PROJECT_NAME, config=wandb_config)
        run = wandb.run
        return run


def finish_wandb():
    if use_wandb:
        wandb.finish()
