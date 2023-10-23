from collections import Counter

import click
import lightning as L
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from modules.base import OverSamplingConfig
from modules.cvae.clf import build_classifier
from modules.cvae.data import _build_Xy
from modules.cvae.generate import generate
from modules.cvae.generate import load_decoder

L.seed_everything(1, workers=True)

@click.command()
@click.option("--config", type=str, help="Path to config file")
def main(config: str):
    """
    """

    with open(config, "r") as f:
        config = yaml.safe_load(f)

    oversampling_config = OverSamplingConfig(**config['oversampling'])
    print(oversampling_config)
    return

    # dataset
    (X_train, y_train), (X_valid, y_valid), le =\
            _build_Xy(ds_name)

    # baseline model
    clf = build_classifier(clf_name)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_valid)
    print("Baseline", f1_score(y_valid, y_pred, average='weighted'))

    # generate new sample to train
    decoder = load_decoder(pth, 32, encoder, decoder, z_dim, n_class)

    #counter: number of sample per classes
    counter = Counter(y_train)
    _max = np.max(list(counter.values()))

    to_generate = []
    for (c, n) in counter.items():
        n_samples = _max - n

        if n_samples:
            to_generate.append((c, n_samples))

    X_syn, y_syn = [], []
    for (c, n) in to_generate:
        _y = torch.ones(n, dtype=torch.int64) * c
        samples = generate(decoder, n, z_dim, _y, n_class)
        _y = _y.detach().cpu().numpy()

        X_syn.append(samples)
        y_syn.append(_y)

    X_train_syn, y_train_syn = \
            np.concatenate([X_train, *X_syn], axis=0),\
            np.concatenate([y_train, *y_syn])

    # baseline model
    clf = build_classifier(clf_name)
    clf.fit(X_train_syn, y_train_syn)

    y_pred = clf.predict(X_valid)
    print("CVAE    ", f1_score(y_valid, y_pred, average='weighted'))
    

    
if __name__ == "__main__": main()
