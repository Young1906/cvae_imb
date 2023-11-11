import datetime
import click
import yaml

from modules.logger import build_logger
from modules.base import (
        MCMCExpConfig,
        LoggerConfig
    )

from modules.data import build_synthetic_dataset 
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
    logger = build_logger(name=logger_config.name,
                          log_path=logger_config.log_path,
                          telegram_handler=logger_config.telegram_handler)

    logger.debug("<START>")
    logger.debug(config)

    exp_config = MCMCExpConfig(**config["mcmc"])
    
    # dataset
    (X, y), (X_test, y_test) =\
            build_synthetic_dataset(exp_config.n_samples, exp_config.ratio, .2)

    # Resample
    sampler_d = lambda: build_classifier(exp_config.sampler_d)
    sampler = MCMCOverSampling(d_factory=sampler_d,
                               max_iter=exp_config.max_iter,
                               step_size=exp_config.step_size)

    sampler.fit(X, y)
    X, y = sampler.resample(X,y)

    clf = build_classifier(exp_config.classifier)
    clf.fit(X, y)

    # Evaluation
    y_pred = clf.predict(X_test)

    p, r, f = evaluate(y_test, y_pred, exp_config.score_avg_method)

    # Logging result
    logger.debug(f"Precision: {p:.5f}, Recall: {r:.5f}, F1: {f:.5f}")
    logger.debug("<END>")
    logger.debug("-" * 80)

    # Logging to result
    with open(exp_config.result_pth, "a") as fn:
        t = datetime.datetime.now()
        fn.write(f"{t},{_config},{p:.5f},{r:.5f},{f:.5f}\n")


if __name__ == "__main__": main()
