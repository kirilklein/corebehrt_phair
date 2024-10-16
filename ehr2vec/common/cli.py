import sys
from typing import List, Tuple

from ehr2vec.common.config import Config


def override_config_from_cli(cfg: Config):
    """
    Overrides the configuration object with values provided from the CLI.
    """
    overrides: List[Tuple[str, str, bool]] = parse_cli_args()
    for key, value, is_new in overrides:
        value = cfg.str_to_num(value)
        set_nested_attr(cfg, key, value, allow_new=is_new)


def parse_cli_args() -> List[Tuple[str, str, bool]]:
    """
    Parses command-line arguments in the form 'key=value' or '+key=value'.
    Returns a list of tuples: (key, value, is_new)
    """
    args = sys.argv[1:]
    return process_args(args)


def process_args(args: List[str]) -> List[Tuple[str, str, bool]]:
    """
    Processes the list of command-line arguments.
    """
    overrides = []
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if "=" in arg:
            overrides.append(parse_key_value_arg(arg))
        elif arg.startswith("--"):
            skip_next = handle_argparse_arg(args, i)
        else:
            raise ValueError(f"Argument '{arg}' is not in 'key=value' format.")
    return overrides


def parse_key_value_arg(arg: str) -> Tuple[str, str, bool]:
    """
    Parses a key=value argument.
    """
    key, value = arg.split("=", 1)
    is_new = False
    if key.startswith("+"):
        is_new = True
        key = key[1:]  # Remove '+' prefix
    return (key, value, is_new)


def handle_argparse_arg(args: List[str], index: int) -> bool:
    """
    Handles argparse-style arguments.
    """
    # ignore argparse arguments, already parsed
    # skip next one as well
    if index + 1 < len(args):
        return True
    return False


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
