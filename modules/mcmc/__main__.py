import datetime
import click
import yaml

from modules.logger import build_logger
from modules.base import (
        MCMCConfig,
        LoggerConfig
    )

from modules.data import _build_Xy
from modules.clf import build_classifier
from modules.mcmc.mcmc import MCMCOverSampling
from modules.eval import evaluate

@click.command()
@click.option("--config", "-C", type=str, help="Path to config file")
def main(config: str):
    _config = config

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    logger_config = LoggerConfig(**config['logger'])
    logger = build_logger(name=logger_config.logger_name,
                          log_path=logger_config.logger_dir)

    # logging config
    logger.info("<START>")
    logger.info(config)
    

    mcmc_config = MCMCConfig(**config["mcmc"])

    # dataset
    (X, y), (X_test, y_test), le = _build_Xy(mcmc_config.dataset)

    # Resample
    sampler_d = lambda: build_classifier(mcmc_config.sampler_d)
    sampler = MCMCOverSampling(d_factory=sampler_d,
                               max_iter=mcmc_config.max_iter,
                               step_size=mcmc_config.step_size)

    sampler.fit(X, y)
    X, y = sampler.resample(X,y)

    clf = build_classifier(mcmc_config.classifier)
    clf.fit(X, y)

    # Evaluation
    y_pred = clf.predict(X_test)

    p, r, f = evaluate(y_test, y_pred, mcmc_config.score_avg_method)

    # Logging result
    logger.info(f"Precision: {p:.5f}, Recall: {r:.5f}, F1: {f:.5f}")
    logger.info("<END>")
    logger.info("-" * 80)

    # Logging to result
    with open(mcmc_config.result_pth, "a") as fn:
        t = datetime.datetime.now()
        fn.write(f"{t},{_config},{p:.5f},{r:.5f},{f:.5f}\n")

if __name__ == "__main__": main()
