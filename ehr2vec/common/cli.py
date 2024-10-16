import sys
from ehr2vec.common.config import Config


def override_config_from_cli(cfg):
    """
    Overrides the configuration object with values provided from the CLI.
    """
    overrides = parse_cli_args()
    for key, value, is_new in overrides:
        value = cfg.str_to_num(value)
        set_nested_attr(cfg, key, value, allow_new=is_new)


def parse_cli_args():
    """
    Parses command-line arguments in the form 'key=value' or '+key=value'.
    Returns a list of tuples: (key, value, is_new)
    """
    args = sys.argv[1:]
    overrides = []
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            is_new = False
            if key.startswith("+"):
                is_new = True
                key = key[1:]  # Remove '+' prefix
            overrides.append((key, value, is_new))
        elif arg.startswith("--"): # ignore argparse arguments, already parsed
            pass
        else:
            raise ValueError(f"Argument '{arg}' is not in 'key=value' format.")
    return overrides


def set_nested_attr(obj, attr_path, value, allow_new=False):
    """
    Sets a nested attribute in a Config object based on a dot-separated path.
    If allow_new is True, it will create new attributes as needed.
    """
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        if not hasattr(obj, attr):
            if allow_new:
                setattr(obj, attr, Config())
            else:
                raise AttributeError(
                    f"Attribute '{attr}' does not exist in the configuration."
                )
        obj = getattr(obj, attr)
    if hasattr(obj, attrs[-1]) or allow_new:
        setattr(obj, attrs[-1], value)
    else:
        raise AttributeError(
            f"Attribute '{attrs[-1]}' does not exist in the configuration."
        )
