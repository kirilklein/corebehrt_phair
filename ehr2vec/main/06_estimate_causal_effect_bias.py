from os.path import abspath, dirname, join

from ehr2vec.common.setup import get_args
from ehr2vec.effect_estimation.main_estimator import EffectEstimator_with_bias


def main(config_path: str):
    estimator = EffectEstimator_with_bias.from_config(config_path)
    estimator.run()


if __name__ == "__main__":
    args = get_args("example_configs/06_estimate_effect_binary.yaml")
    config_path = join(dirname(dirname(abspath(__file__))), args.config_path)
    main(config_path)
