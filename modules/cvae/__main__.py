import datetime
from collections import Counter

import click
import lightning as L
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from modules.base import LoggerConfig
from modules.base import OverSamplingConfig
from modules.clf import build_classifier
from modules.data import _build_Xy
from modules.cvae.generate import generate
from modules.cvae.generate import load_decoder
from modules.eval import evaluate
from modules.logger import build_logger

L.seed_everything(1, workers=True)

@click.command()
@click.option("--config", type=str, help="Path to config file")
def main(config: str):
    """
    """
    config_pth = config

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    os_config = OverSamplingConfig(**config['oversampling'])

    # Logger
    logger_config = LoggerConfig(**config['logger'])
    logger = build_logger(
            logger_config.logger_name,
            logger_config.logger_dir)

    # logging config
    logger.info("<START>")
    logger.info(config)

    # dataset
    (X_train, y_train), (X_valid, y_valid), le =\
            _build_Xy(os_config.dataset)

    (N, _), (N_val, _) = X_train.shape, X_valid.shape

    # generate new sample to train
    pth = f"{os_config.checkpoint_pth}/{os_config.checkpoint_fn}.ckpt"
    decoder = load_decoder(
            pth=pth,
            input_dim=os_config.input_dim,
            encoder=os_config.encoder,
            decoder=os_config.decoder,
            z_dim=os_config.z_dim,
            n_class=os_config.n_class)

    to_generate = []
    counter = Counter(y_train)
    _max = np.max(list(counter.values()))
    for (c, n) in counter.items():
        n_samples = _max - n

        if n_samples:
            to_generate.append((c, n_samples))

    X_syn, y_syn = [], []
    for (c, n) in to_generate:
        _y = torch.ones(n, dtype=torch.int64) * c
        samples = generate(
                decoder,
                n,
                os_config.z_dim, 
                _y, 
                os_config.n_class)

        _y = _y.detach().cpu().numpy()

        X_syn.append(samples)
        y_syn.append(_y)

    X_train_syn, y_train_syn = \
            np.concatenate([X_train, *X_syn], axis=0),\
            np.concatenate([y_train, *y_syn])

    # train classifier 
    clf = build_classifier(os_config.classifier)
    clf.fit(X_train_syn, y_train_syn)

    # Evaluation
    y_pred = clf.predict(X_valid)

    p, r, f= evaluate(y_valid, y_pred, os_config.score_avg_method)

    # Logging result
    logger.info(f"Precision: {p:.5f}, Recall: {r:.5f}, F1: {f:.5f}")
    logger.info("<END>")
    logger.info("-"*80)

    # Logging to result
    with open(os_config.result_pth, "a") as fn:
        t = datetime.datetime.now()
        fn.write(f"{t},{config_pth},{p:.5f},{r:.5f},{f:.5f}\n")


    
if __name__ == "__main__": main()
