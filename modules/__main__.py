from collections import Counter

import click
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from modules.clf import build_classifier
from modules.data import _build_Xy
from modules.generate import generate
from modules.generate import load_decoder

@click.command()
@click.option(
        "--ds_name", "-N", type=str, help="Name of the encoder")
@click.option(
        "--clf_name", "-F", type=str, help="SKLearn classifier")
@click.option(
        "--pth", "-P", type=str, help="Path to checkpoint")
@click.option(
        "--encoder", "-E", type=str, help="Name of the encoder")
@click.option(
        "--decoder", "-D", type=str, help="Name of the decoder")
@click.option(
        "--z_dim", "-Z", type=int, help="Latent dim")
@click.option(
        "--n_class", "-C", type=int, help="N classes")

def main(
        ds_name: str,
        clf_name: str,
        pth:str,
        encoder: str,
        decoder: str,
        z_dim: int,
        n_class: int):
    """
    """
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
