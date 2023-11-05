# Conditional VAE & MCMC for imbalance learning
Minority class oversampling using conditional VAE and MCMC

## TLDR

Oversampling


### MCMC

To sample from from class $c_i$ with probability density function $p(X | y = c_i)$, we can utilize Markov Chain Monte Carlo, we need quantity:

$$
\begin{equation}
\begin{aligned}
H & = \frac{p(x | y = c_i)}{p(x_t | y = c_i)} \\ 
    & = \frac{p(x, y)/p(y)}{p(x_t, y)/p(y)} \\
    & = \frac{p(y|x)}{p(y|x_t)} \times \frac{p(x)}{p(x_t)}
\end{aligned}
\end{equation}
$$

The first term $p(y | x)$, we can train a classifier to approximate this quantity. The choice of $d_\theta(.)$ is very flexible, can be `scikit-learn` implementation of `LogisticRegression()` for binary classification problem or a simple fully connected network.

$$
p(y | x) = d_\theta(x)
$$


For the second term, we can just assume x to be  a parametrics distribution such as Multivariate Gaussian. 

$$
x \sim \mathcal{N}(.; \mu, \Sigma)
$$


## Conditional VAE

## MCMC

### F1-score

#### balance


|                             |   catboost |   decision_tree |      gbc |      knn |       lr |      mlp |      svm |
|-----------------------------|------------|-----------------|----------|----------|----------|----------|----------|
| baseline                    |   0.88484  |        0.7924   | 0.84133  | 0.8448   | 0.83759  | 0.9158   | 0.87434  |
| instance-hardness-threshold |   0.649083 |        0.63392  | 0.625469 | 0.694536 | 0.690433 | 0.781246 | 0.643765 |
| mcmc                        |   0.860509 |        0.783372 | 0.853808 | 0.751991 | 0.813427 | 0.851543 | 0.822859 |
| smotenn                     |   0.830028 |        0.76745  | 0.823676 | 0.72628  | 0.727295 | 0.856183 | 0.842741 |
| svm-smote                   |   0.866276 |        0.782571 | 0.838979 | 0.766065 | 0.937529 | 0.936136 | 0.879181 |




#### breast-cancer


|                             |   catboost |   decision_tree |      gbc |      knn |       lr |      mlp |      svm |
|-----------------------------|------------|-----------------|----------|----------|----------|----------|----------|
| adasyn                      |   0.963726 |        0.92482  | 0.964729 | 0.934469 | 0.966229 | 0.974775 | 0.963148 |
| baseline                    |   0.96472  |        0.92982  | 0.95578  | 0.96472  | 0.98246  | 0.96491  | 0.96491  |
| instance-hardness-threshold |   0.931053 |        0.905954 | 0.911638 | 0.967267 | 0.957589 | 0.962209 | 0.932929 |
| kmean-smote                 |   0.963528 |        0.946055 | 0.956376 | 0.964169 | 0.98246  | 0.97362  | 0.966071 |
| mcmc                        |   0.960615 |        0.920891 | 0.957596 | 0.964141 | 0.979513 | 0.973083 | 0.970138 |
| smotenn                     |   0.955334 |        0.930066 | 0.954677 | 0.962603 | 0.974896 | 0.969024 | 0.963726 |
| svm-smote                   |   0.965443 |        0.934669 | 0.963558 | 0.944245 | 0.969699 | 0.969607 | 0.966678 |




#### breast-tissue


|                             |   catboost |   decision_tree |      gbc |      knn |       lr |      mlp |      svm |
|-----------------------------|------------|-----------------|----------|----------|----------|----------|----------|
| baseline                    |   0.85584  |        0.68254  | 0.70899  | 0.68254  | 0.7197   | 0.6048   | 0.68773  |
| instance-hardness-threshold |   0.607123 |        0.695747 | 0.654201 | 0.642091 | 0.646367 | 0.656354 | 0.643466 |
| kmean-smote                 |   0.779612 |        0.656127 | 0.720984 | 0.647563 | 0.714568 | 0.625538 | 0.621993 |
| mcmc                        |   0.812517 |        0.636917 | 0.701231 | 0.672871 | 0.713274 | 0.614817 | 0.644064 |
| smotenn                     |   0.532918 |        0.401472 | 0.44234  | 0.440399 | 0.497419 | 0.536628 | 0.471377 |
| svm-smote                   |   0.793561 |        0.640507 | 0.676877 | 0.653841 | 0.703457 | 0.578586 | 0.584208 |




#### connectionist


|                             |   catboost |   decision_tree |      gbc |      knn |       lr |      mlp |      svm |
|-----------------------------|------------|-----------------|----------|----------|----------|----------|----------|
| baseline                    |   0.80865  |        0.78388  | 0.7619   | 0.80865  | 0.66515  | 0.85649  | 0.78584  |
| instance-hardness-threshold |   0.718629 |        0.691953 | 0.754288 | 0.749861 | 0.69969  | 0.775605 | 0.769993 |
| kmean-smote                 |   0.79466  |        0.705158 | 0.791593 | 0.824899 | 0.66515  | 0.84242  | 0.796625 |
| mcmc                        |   0.78999  |        0.727265 | 0.79188  | 0.815136 | 0.666803 | 0.832863 | 0.787317 |
| smotenn                     |   0.729115 |        0.692104 | 0.708845 | 0.746553 | 0.723268 | 0.768199 | 0.763791 |
| svm-smote                   |   0.793117 |        0.718943 | 0.786785 | 0.828207 | 0.666803 | 0.829727 | 0.801262 |




#### ecoli


|                             |   catboost |   decision_tree |      gbc |      knn |       lr |      mlp |      svm |
|-----------------------------|------------|-----------------|----------|----------|----------|----------|----------|
| adasyn                      |   0.928518 |        0.895019 | 0.895905 | 0.891433 | 0.875908 | 0.893759 | 0.92563  |
| baseline                    |   0.93977  |        0.87613  | 0.89589  | 0.92563  | 0.83368  | 0.86695  | 0.95422  |
| instance-hardness-threshold |   0.873125 |        0.846557 | 0.850082 | 0.93653  | 0.906889 | 0.932539 | 0.919902 |
| kmean-smote                 |   0.93977  |        0.918843 | 0.917057 | 0.92563  | 0.890109 | 0.906185 | 0.94266  |
| smotenn                     |   0.926451 |        0.899624 | 0.907805 | 0.938297 | 0.90536  | 0.909889 | 0.941007 |
| svm-smote                   |   0.914534 |        0.896039 | 0.896081 | 0.91078  | 0.863615 | 0.900843 | 0.914442 |




#### frogs


|                             |   catboost |   decision_tree |      gbc |      knn |       lr |      mlp |      svm |
|-----------------------------|------------|-----------------|----------|----------|----------|----------|----------|
| adasyn                      |   0.991992 |        0.969297 | 0.982935 | 0.990325 | 0.914038 | 0.988695 | 0.98821  |
| baseline                    |   0.99235  |        0.97217  | 0.97773  | 0.99304  | 0.945    | 0.98749  | 0.98539  |
| instance-hardness-threshold |   0.942741 |        0.924118 | 0.938025 | 0.973303 | 0.899496 | 0.955235 | 0.940739 |
| kmean-smote                 |   0.99124  |        0.969987 | 0.979499 | 0.993183 | 0.943959 | 0.988511 | 0.98429  |
| mcmc                        |   0.990824 |        0.968045 | 0.976623 | 0.99304  | 0.944587 | 0.986889 | 0.981733 |
| smotenn                     |   0.991057 |        0.96825  | 0.981045 | 0.99263  | 0.946693 | 0.990041 | 0.986099 |
| svm-smote                   |   0.991897 |        0.972866 | 0.981463 | 0.991109 | 0.913043 | 0.988752 | 0.986825 |




#### heart_2cl


|                             |   catboost |   decision_tree |        gbc |        knn |         lr |        mlp |      svm |
|-----------------------------|------------|-----------------|------------|------------|------------|------------|----------|
| adasyn                      |   0.870075 |        0.786499 |   0.852897 |   0.61319  |   0.828677 |   0.845241 | 0.843241 |
| baseline                    |   0.81583  |        0.6629   |   0.76978  |   0.689    |   0.87901  |   0.86232  | 0.70599  |
| instance-hardness-threshold |   0.67905  |        0.682048 |   0.70114  |   0.579159 |   0.706417 |   0.708984 | 0.653413 |
| kmean-smote                 |   0.80072  |      nan        | nan        | nan        | nan        | nan        | 0.90573  |
| mcmc                        |   0.825023 |        0.757118 |   0.814438 |   0.687449 |   0.862697 |   0.828984 | 0.705375 |
| smotenn                     |   0.716437 |        0.746014 |   0.713832 |   0.506955 |   0.761331 |   0.766891 | 0.770195 |
| svm-smote                   |   0.865111 |        0.761178 |   0.824415 |   0.652935 |   0.81106  |   0.819186 | 0.828017 |




#### ionosphere


|                             |   catboost |   decision_tree |      gbc |      knn |       lr |      mlp |      svm |
|-----------------------------|------------|-----------------|----------|----------|----------|----------|----------|
| adasyn                      |   0.938429 |        0.867163 | 0.945689 | 0.882835 | 0.859011 | 0.913173 | 0.95077  |
| baseline                    |   0.94239  |        0.87263  | 0.95707  | 0.81814  | 0.85137  | 0.89847  | 0.91358  |
| instance-hardness-threshold |   0.941819 |        0.876733 | 0.897152 | 0.865263 | 0.828159 | 0.828468 | 0.950189 |
| kmean-smote                 |   0.931323 |        0.873593 | 0.951293 | 0.853458 | 0.846479 | 0.898105 | 0.915561 |
| mcmc                        |   0.943563 |        0.868606 | 0.949509 | 0.819262 | 0.849251 | 0.90752  | 0.914571 |
| smotenn                     |   0.870816 |        0.843817 | 0.874487 | 0.862034 | 0.833399 | 0.86297  | 0.911532 |
| svm-smote                   |   0.93038  |        0.865027 | 0.95902  | 0.851247 | 0.840101 | 0.899386 | 0.930393 |




#### parkinsons


|                             |   catboost |   decision_tree |      gbc |      knn |       lr |      mlp |      svm |
|-----------------------------|------------|-----------------|----------|----------|----------|----------|----------|
| adasyn                      |   0.925863 |        0.857061 | 0.935659 | 0.867348 | 0.824004 | 0.922237 | 0.867809 |
| baseline                    |   0.91818  |        0.81731  | 0.91818  | 0.86364  | 0.76511  | 0.88792  | 0.85551  |
| instance-hardness-threshold |   0.76441  |        0.740177 | 0.730031 | 0.769219 | 0.744809 | 0.748039 | 0.736623 |
| kmean-smote                 |   0.91922  |        0.911572 | 0.914647 | 0.89345  | 0.77503  | 0.909967 | 0.857671 |
| mcmc                        |   0.92173  |        0.864725 | 0.901116 | 0.86364  | 0.783017 | 0.876905 | 0.854521 |
| smotenn                     |   0.893541 |        0.796221 | 0.882765 | 0.865124 | 0.787241 | 0.854859 | 0.839845 |
| svm-smote                   |   0.930761 |        0.867852 | 0.928469 | 0.846404 | 0.790834 | 0.929065 | 0.860945 |

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

- Reproduce all experiments

```bash
    make all
```

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
