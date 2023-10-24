# Conditional VAE for imbalance learning
Minority class oversampling using conditional VAE

## Initial result

### F1-score

|               |   catboost |   decision_tree |     gbc |     knn |      lr |     mlp |     svm |
|---------------|------------|-----------------|---------|---------|---------|---------|---------|
| balance       |    0.85942 |         0.77089 | 0.84533 | 0.81475 | 0.9355  | 0.94947 | 0.89619 |
| breast-cancer |    0.95122 |         0.875   | 0.93827 | 0.95122 | 0.97619 | 0.96386 | 0.95238 |
| breast-tissue |    0.95253 |         0.67143 | 0.73608 | 0.68254 | 0.71955 | 0.69156 | 0.60072 |
| connectionist |    0.78947 |         0.71429 | 0.8     | 0.81818 | 0.68182 | 0.82051 | 0.8     |
| frogs         |    0.99379 |         0.9763  | 0.98196 | 0.99436 | 0.95598 | 0.99263 | 0.98871 |
| heart_2cl     |    0.91304 |         0.86047 | 0.87912 | 0.84211 | 0.91954 | 0.89655 | 0.92308 |
| ionosphere    |    0.94737 |         0.88172 | 0.96842 | 0.88235 | 0.88889 | 0.9375  | 0.94737 |
| parkinsons    |    0.94737 |         0.91228 | 0.96552 | 0.88136 | 0.82143 | 0.93333 | 0.92063 |

### Precision

|               |   catboost |   decision_tree |     gbc |     knn |      lr |     mlp |     svm |
|---------------|------------|-----------------|---------|---------|---------|---------|---------|
| balance       |    0.85285 |         0.80325 | 0.82973 | 0.82215 | 0.95435 | 0.94853 | 0.89222 |
| breast-cancer |    0.975   |         0.92105 | 0.97436 | 0.975   | 0.97619 | 0.97561 | 0.95238 |
| breast-tissue |    0.96364 |         0.67955 | 0.76591 | 0.69697 | 0.81591 | 0.66818 | 0.59578 |
| connectionist |    0.83333 |         0.68182 | 0.8     | 0.75    | 0.625   | 0.84211 | 0.8     |
| frogs         |    0.99211 |         0.9741  | 0.97865 | 0.99101 | 0.95383 | 0.99432 | 0.98649 |
| heart_2cl     |    0.85714 |         0.86047 | 0.83333 | 0.9697  | 0.90909 | 0.88636 | 0.875   |
| ionosphere    |    0.91837 |         0.87234 | 0.93878 | 0.80357 | 0.83019 | 0.9     | 0.91837 |
| parkinsons    |    0.96429 |         0.92857 | 0.96552 | 0.86667 | 0.85185 | 0.90323 | 0.85294 |

### Recall

|               |   catboost |   decision_tree |     gbc |     knn |      lr |     mlp |     svm |
|---------------|------------|-----------------|---------|---------|---------|---------|---------|
| balance       |    0.872   |         0.744   | 0.864   | 0.808   | 0.928   | 0.952   | 0.912   |
| breast-cancer |    0.92857 |         0.83333 | 0.90476 | 0.92857 | 0.97619 | 0.95238 | 0.95238 |
| breast-tissue |    0.95455 |         0.68182 | 0.72727 | 0.68182 | 0.72727 | 0.72727 | 0.63636 |
| connectionist |    0.75    |         0.75    | 0.8     | 0.9     | 0.75    | 0.8     | 0.8     |
| frogs         |    0.99548 |         0.97851 | 0.98529 | 0.99774 | 0.95814 | 0.99095 | 0.99095 |
| heart_2cl     |    0.97674 |         0.86047 | 0.93023 | 0.74419 | 0.93023 | 0.90698 | 0.97674 |
| ionosphere    |    0.97826 |         0.8913  | 1       | 0.97826 | 0.95652 | 0.97826 | 0.97826 |
| parkinsons    |    0.93103 |         0.89655 | 0.96552 | 0.89655 | 0.7931  | 0.96552 | 1       |


## Checklist

### Dataset

- [x] Breast Cancer
- [x] Frogs MFCCs
- [x] Breast Tissue
- [x] Connectionist BenchMark
- [x] Ionosphere
- [x] Parkinsons
- [ ] Heart Training Subset ( Heart2CL? )
- [x] Balance

Other (not included in Hoang's repo)

- [x] Ecoli

### Classifier

- [x] SMV
- [x] Logistic Regression
- [x] MLP

Other:

- [x] KNN
- [x] Decision Tree
- [x] Gradient Boosting Classifier
- [x] CatBoost


## [IMPORTANT] Experiment procedure

### Validation method

- For `sklearn.model_selection.train_test_split`, set `random_state=1`, `stratify=y`.
- ...



## Commands

- Train a CVAE model to generate random samples from any class

```bash
    python -m modules.cvae.train --config <pth/to/config>
```

- CVAE oversampling experiment
```bash
    python -m modules.cvae --config <pth/to/config>
```

For sample of config file, see `config/dev.yaml`

## Note/Questions
- **Breast tissues**: Is the baseline result trained on Scaled Dataset?
- **Heart 2CL dataset**: training on spectf.data as train and spectf.test as validation or concate and split later( as in Thu's [notebook](https://colab.research.google.com/drive/1zm-V7dIAE5F61NxAcNASD9WBR1YzJXcv?usp=sharing#scrollTo=8-kXWlmtl-OM)?)

- **MLP**: where is the code for implementations?

## Citations

```bibtex
{
}
```

## Resource

- [Notebook](https://colab.research.google.com/drive/1zm-V7dIAE5F61NxAcNASD9WBR1YzJXcv?usp=sharing#scrollTo=pvXSYmgVoP9D) handling dataset
- [Hoang's repo](https://github.com/Cavan1Ed1s0n/MissingData/)
