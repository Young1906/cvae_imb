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

    os_config = OverSamplingConfig(**config['oversampling'])

    # dataset
    (X_train, y_train), (X_valid, y_valid), le =\
            _build_Xy(os_config.dataset)

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

    # baseline model
    clf = build_classifier(os_config.classifier)
    clf.fit(X_train_syn, y_train_syn)

    y_pred = clf.predict(X_valid)
    print("CVAE: ",
          f1_score

              y_valid,
              y_pred,
              average=os_config.f1_score_avg))
    

    
if __name__ == "__main__": main()
