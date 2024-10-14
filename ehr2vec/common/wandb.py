try:
    import wandb

    use_wandb = True
except ImportError:
    use_wandb = False


def initialize_wandb(run, cfg, wandb_kwargs):
    """
    Initialize Wand if available, else return run.
    Return a tuple of (use_wandb, run)
    """
    if not use_wandb:
        return run
    else:
        wandb_config = {}
        for key in [
            "trainer_args",
            "model",
            "optimizer",
            "data",
            "test_args",
            "scheduler",
        ]:
            if hasattr(cfg, key):
                wandb_config.update(cfg[key])
        wandb.init(config=wandb_config, **wandb_kwargs)
        run = wandb.run
        return run


def finish_wandb():
    if use_wandb:
        wandb.finish()
