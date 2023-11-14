import datetime

import click
import numpy as np
import yaml
import sys

from modules.base import BaselineConfig
from modules.base import LoggerConfig
from modules.baseline.samplers import build_sampler
from modules.clf import build_classifier
from modules.data import _build_Xy
from modules.eval import evaluate
from modules.logger import build_logger

from modules.utils import enable_stochastic_process

np.random.seed(1)


@click.command()
@click.option("--config", "-C", type=str, help="Path to config file")
def main(config: str):
    _config = config

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    # Logger 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    logger_config = LoggerConfig(**config["logger"])
    logger = build_logger(name=logger_config.name,
                          log_path=logger_config.log_path,
                          telegram_handler=logger_config.telegram_handler)

    # logging config
    logger.debug("<START>")
    logger.debug(config)
    

    # Baseline method
    # ++++++++++++++++++++++++++++++++++++++++++++++++++
    baseline_config = BaselineConfig(**config['baseline'])

    # dataset
    # -------------------------------------------------- 
    (X, y), (X_test, y_test), le = _build_Xy(baseline_config.dataset)

    # Sampler
    # -------------------------------------------------- 
    sampler = build_sampler(baseline_config.sampler_name)
    try:
        X, y = sampler.fit_resample(X, y)

    except Exception as e:
        logger.critical(e)
        sys.exit(1)

    # CLF 
    # -------------------------------------------------- 
    clf = build_classifier(baseline_config.classifier)
    clf.fit(X, y)

    # Evaluation 
    # -------------------------------------------------- 
    y_pred = clf.predict(X_test)
    p, r, f = evaluate(y_test, y_pred, baseline_config.score_avg_method)

    # Logging result
    # -------------------------------------------------- 
    logger.debug(f"Precision: {p:.5f}, Recall: {r:.5f}, F1: {f:.5f}")
    logger.debug("<END>")
    logger.info("-" * 80)

    # Logging to result
    with open(baseline_config.result_pth, "a") as fn:
        t = datetime.datetime.now()
        fn.write(f"{t},{_config},{p:.5f},{r:.5f},{f:.5f}\n")

if __name__ == "__main__": main()
