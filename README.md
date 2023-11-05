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


|                             |   catboost |   decision_tree |    gbc |    knn |     lr |    mlp |    svm |
|-----------------------------|------------|-----------------|--------|--------|--------|--------|--------|
| baseline                    |     0.8848 |          0.7924 | 0.8413 | 0.8448 | 0.8376 | 0.9158 | 0.8743 |
| instance-hardness-threshold |     0.6491 |          0.6339 | 0.6255 | 0.6945 | 0.6904 | 0.7812 | 0.6438 |
| mcmc                        |     0.8605 |          0.7834 | 0.8538 | 0.752  | 0.8134 | 0.8515 | 0.8229 |
| smotenn                     |     0.83   |          0.7675 | 0.8237 | 0.7263 | 0.7273 | 0.8562 | 0.8427 |
| svm-smote                   |     0.8663 |          0.7826 | 0.839  | 0.7661 | 0.9375 | 0.9361 | 0.8792 |




#### breast-cancer


|                             |   catboost |   decision_tree |    gbc |    knn |     lr |    mlp |    svm |
|-----------------------------|------------|-----------------|--------|--------|--------|--------|--------|
| adasyn                      |     0.9637 |          0.9248 | 0.9647 | 0.9345 | 0.9662 | 0.9748 | 0.9631 |
| baseline                    |     0.9647 |          0.9298 | 0.9558 | 0.9647 | 0.9825 | 0.9649 | 0.9649 |
| instance-hardness-threshold |     0.9311 |          0.906  | 0.9116 | 0.9673 | 0.9576 | 0.9622 | 0.9329 |
| kmean-smote                 |     0.9635 |          0.9461 | 0.9564 | 0.9642 | 0.9825 | 0.9736 | 0.9661 |
| mcmc                        |     0.9606 |          0.9209 | 0.9576 | 0.9641 | 0.9795 | 0.9731 | 0.9701 |
| smotenn                     |     0.9553 |          0.9301 | 0.9547 | 0.9626 | 0.9749 | 0.969  | 0.9637 |
| svm-smote                   |     0.9654 |          0.9347 | 0.9636 | 0.9442 | 0.9697 | 0.9696 | 0.9667 |




#### breast-tissue


|                             |   catboost |   decision_tree |    gbc |    knn |     lr |    mlp |    svm |
|-----------------------------|------------|-----------------|--------|--------|--------|--------|--------|
| baseline                    |     0.8558 |          0.6825 | 0.709  | 0.6825 | 0.7197 | 0.6048 | 0.6877 |
| instance-hardness-threshold |     0.6071 |          0.6957 | 0.6542 | 0.6421 | 0.6464 | 0.6564 | 0.6435 |
| kmean-smote                 |     0.7796 |          0.6561 | 0.721  | 0.6476 | 0.7146 | 0.6255 | 0.622  |
| mcmc                        |     0.8125 |          0.6369 | 0.7012 | 0.6729 | 0.7133 | 0.6148 | 0.6441 |
| smotenn                     |     0.5329 |          0.4015 | 0.4423 | 0.4404 | 0.4974 | 0.5366 | 0.4714 |
| svm-smote                   |     0.7936 |          0.6405 | 0.6769 | 0.6538 | 0.7035 | 0.5786 | 0.5842 |




#### connectionist


|                             |   catboost |   decision_tree |    gbc |    knn |     lr |    mlp |    svm |
|-----------------------------|------------|-----------------|--------|--------|--------|--------|--------|
| baseline                    |     0.8086 |          0.7839 | 0.7619 | 0.8086 | 0.6652 | 0.8565 | 0.7858 |
| instance-hardness-threshold |     0.7186 |          0.692  | 0.7543 | 0.7499 | 0.6997 | 0.7756 | 0.77   |
| kmean-smote                 |     0.7947 |          0.7052 | 0.7916 | 0.8249 | 0.6652 | 0.8424 | 0.7966 |
| mcmc                        |     0.79   |          0.7273 | 0.7919 | 0.8151 | 0.6668 | 0.8329 | 0.7873 |
| smotenn                     |     0.7291 |          0.6921 | 0.7088 | 0.7466 | 0.7233 | 0.7682 | 0.7638 |
| svm-smote                   |     0.7931 |          0.7189 | 0.7868 | 0.8282 | 0.6668 | 0.8297 | 0.8013 |




#### ecoli


|                             |   catboost |   decision_tree |    gbc |    knn |     lr |    mlp |    svm |
|-----------------------------|------------|-----------------|--------|--------|--------|--------|--------|
| adasyn                      |     0.9285 |          0.895  | 0.8959 | 0.8914 | 0.8759 | 0.8938 | 0.9256 |
| baseline                    |     0.9398 |          0.8761 | 0.8959 | 0.9256 | 0.8337 | 0.8669 | 0.9542 |
| instance-hardness-threshold |     0.8731 |          0.8466 | 0.8501 | 0.9365 | 0.9069 | 0.9325 | 0.9199 |
| kmean-smote                 |     0.9398 |          0.9188 | 0.9171 | 0.9256 | 0.8901 | 0.9062 | 0.9427 |
| smotenn                     |     0.9265 |          0.8996 | 0.9078 | 0.9383 | 0.9054 | 0.9099 | 0.941  |
| svm-smote                   |     0.9145 |          0.896  | 0.8961 | 0.9108 | 0.8636 | 0.9008 | 0.9144 |




#### frogs


|                             |   catboost |   decision_tree |    gbc |    knn |     lr |    mlp |    svm |
|-----------------------------|------------|-----------------|--------|--------|--------|--------|--------|
| adasyn                      |     0.992  |          0.9693 | 0.9829 | 0.9903 | 0.914  | 0.9887 | 0.9882 |
| baseline                    |     0.9923 |          0.9722 | 0.9777 | 0.993  | 0.945  | 0.9875 | 0.9854 |
| instance-hardness-threshold |     0.9427 |          0.9241 | 0.938  | 0.9733 | 0.8995 | 0.9552 | 0.9407 |
| kmean-smote                 |     0.9912 |          0.97   | 0.9795 | 0.9932 | 0.944  | 0.9885 | 0.9843 |
| mcmc                        |     0.9908 |          0.968  | 0.9766 | 0.993  | 0.9446 | 0.9869 | 0.9817 |
| smotenn                     |     0.9911 |          0.9682 | 0.981  | 0.9926 | 0.9467 | 0.99   | 0.9861 |
| svm-smote                   |     0.9919 |          0.9729 | 0.9815 | 0.9911 | 0.913  | 0.9888 | 0.9868 |




#### heart_2cl


|                             |   catboost |   decision_tree |      gbc |      knn |       lr |      mlp |    svm |
|-----------------------------|------------|-----------------|----------|----------|----------|----------|--------|
| adasyn                      |     0.8701 |          0.7865 |   0.8529 |   0.6132 |   0.8287 |   0.8452 | 0.8432 |
| baseline                    |     0.8158 |          0.6629 |   0.7698 |   0.689  |   0.879  |   0.8623 | 0.706  |
| instance-hardness-threshold |     0.6791 |          0.682  |   0.7011 |   0.5792 |   0.7064 |   0.709  | 0.6534 |
| kmean-smote                 |     0.8007 |        nan      | nan      | nan      | nan      | nan      | 0.9057 |
| mcmc                        |     0.825  |          0.7571 |   0.8144 |   0.6874 |   0.8627 |   0.829  | 0.7054 |
| smotenn                     |     0.7164 |          0.746  |   0.7138 |   0.507  |   0.7613 |   0.7669 | 0.7702 |
| svm-smote                   |     0.8651 |          0.7612 |   0.8244 |   0.6529 |   0.8111 |   0.8192 | 0.828  |




#### ionosphere


|                             |   catboost |   decision_tree |    gbc |    knn |     lr |    mlp |    svm |
|-----------------------------|------------|-----------------|--------|--------|--------|--------|--------|
| adasyn                      |     0.9384 |          0.8672 | 0.9457 | 0.8828 | 0.859  | 0.9132 | 0.9508 |
| baseline                    |     0.9424 |          0.8726 | 0.9571 | 0.8181 | 0.8514 | 0.8985 | 0.9136 |
| instance-hardness-threshold |     0.9418 |          0.8767 | 0.8972 | 0.8653 | 0.8282 | 0.8285 | 0.9502 |
| kmean-smote                 |     0.9313 |          0.8736 | 0.9513 | 0.8535 | 0.8465 | 0.8981 | 0.9156 |
| mcmc                        |     0.9436 |          0.8686 | 0.9495 | 0.8193 | 0.8493 | 0.9075 | 0.9146 |
| smotenn                     |     0.8708 |          0.8438 | 0.8745 | 0.862  | 0.8334 | 0.863  | 0.9115 |
| svm-smote                   |     0.9304 |          0.865  | 0.959  | 0.8512 | 0.8401 | 0.8994 | 0.9304 |




#### parkinsons


|                             |   catboost |   decision_tree |    gbc |    knn |     lr |    mlp |    svm |
|-----------------------------|------------|-----------------|--------|--------|--------|--------|--------|
| adasyn                      |     0.9259 |          0.8571 | 0.9357 | 0.8673 | 0.824  | 0.9222 | 0.8678 |
| baseline                    |     0.9182 |          0.8173 | 0.9182 | 0.8636 | 0.7651 | 0.8879 | 0.8555 |
| instance-hardness-threshold |     0.7644 |          0.7402 | 0.73   | 0.7692 | 0.7448 | 0.748  | 0.7366 |
| kmean-smote                 |     0.9192 |          0.9116 | 0.9146 | 0.8934 | 0.775  | 0.91   | 0.8577 |
| mcmc                        |     0.9217 |          0.8647 | 0.9011 | 0.8636 | 0.783  | 0.8769 | 0.8545 |
| smotenn                     |     0.8935 |          0.7962 | 0.8828 | 0.8651 | 0.7872 | 0.8549 | 0.8398 |
| svm-smote                   |     0.9308 |          0.8679 | 0.9285 | 0.8464 | 0.7908 | 0.9291 | 0.8609 |

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
