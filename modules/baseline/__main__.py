import datetime
from collections import Counter

import click
import numpy as np

from modules.eval import evaluate
from modules.logger import build_logger

np.random.seed(1)


@click.command()
@click.option("--config", type=str, help="Path to config file")
def main(config: str):
    config_pth = config

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    # Logger
    logger_config = LoggerConfig(**config["logger"])
    logger = build_logger(logger_config.logger_name, logger_config.logger_dir)

    # logging config
    logger.info("<START>")
    logger.info(config)

    # dataset
    (X_train, y_train), (X_valid, y_valid), le = _build_Xy(os_config.dataset)
