# Conditional VAE for imbalance learning

Generating minority class samples using conditional VAE

## [IMPORTANT] Experiment procedure

### Validation method

- For `sklearn.model_selection.train_test_split`, set `random_state=1`, `stratify=y`.
- ...



## Commands

- Train a CVAE model to generate random samples from any class

```bash
    python -m modules.cvae.train --config <pth/to/config>
```

For sample of config file, see `config/dev.yaml`

