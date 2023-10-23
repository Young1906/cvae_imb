# Conditional VAE for imbalance learning
Minority class oversampling using conditional VAE

## Initial result (all on SVM)

| Dataset | CVAE F1 | Best Baseline |
| --- | --- | --- |
| Breast Tissue | **60.072%** | 59.6186 (KMean Smote) | 
| Frogs MFCCs | 98.871% | **99.0373%** (ADASYN) | 
| Heart 2CL | **92.308%** | 79.6326% (KMean Smote) | 
| Ionosphere | **94.737%** | 94.082% (ADASYN) | 
| Breast Cancer | 95.238%| | 
| Connectionist | 80.000%| | 
| Parkinsons |92.063% | | 
| Balance |89.619% | | 


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
- [ ] MLP

Other (not included in Hoang's repo)

- [x] KNN
- [x] Decision Tree


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

- MLP: where is the code for implementations?

## Citations

```bibtex
```

## Resource

- [Notebook](https://colab.research.google.com/drive/1zm-V7dIAE5F61NxAcNASD9WBR1YzJXcv?usp=sharing#scrollTo=pvXSYmgVoP9D) handling dataset
- [Hoang's repo](https://github.com/Cavan1Ed1s0n/MissingData/)
