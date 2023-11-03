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

### F1-score

|               |   catboost |   decision_tree |     gbc |     knn |      lr |     mlp |     svm |
|---------------|------------|-----------------|---------|---------|---------|---------|---------|
| balance       |    0.85942 |         0.77089 | 0.84533 | 0.81475 | **0.9355** | **0.94947** | **0.89619** |
| breast-cancer |    0.95122 |         0.875   | 0.93827 | 0.95122 | 0.97619 | 0.96386 | 0.95238 |
| breast-tissue |    0.95253 |         0.67143 | 0.73608 | 0.68254 | **0.71955** | **0.69156** | **0.60072** |
| connectionist |    0.78947 |         0.71429 | 0.8     | 0.81818 | 0.68182 | 0.82051 | 0.8     |
| frogs         |    0.99379 |         0.9763  | 0.98196 | 0.99436 | **0.95598** | **0.99263** | 0.98871 |
| heart_2cl     |    0.91304 |         0.86047 | 0.87912 | 0.84211 | **0.91954** | **0.89655** | **0.92308** |
| ionosphere    |    0.94737 |         0.88172 | 0.96842 | 0.88235 | **0.88889** | **0.9375**  | **0.94737** |
| parkinsons    |    0.94737 |         0.91228 | 0.96552 | 0.88136 | **0.82143** | **0.93333** | **0.92063** |

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

## MCMC

### F1 score

|               | catboost          | decision_tree     | gbc               | knn               | lr                | mlp               | svm               |
|---------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| balance       | 0.85534 ± 0.01010 | 0.77878 ± 0.01344 | 0.85522 ± 0.00869 | 0.76407 ± 0.02277 | 0.81874 ± 0.02713 | 0.86270 ± 0.02613 | 0.82459 ± 0.02124 |
| breast-cancer | 0.95912 ± 0.00776 | 0.92483 ± 0.01633 | 0.95609 ± 0.00711 | 0.96472 ± 0.00000 | 0.97981 ± 0.00405 | 0.97278 ± 0.00472 | 0.96897 ± 0.00435 |
| breast-tissue | 0.79040 ± 0.04859 | 0.68911 ± 0.07027 | 0.70142 ± 0.05085 | 0.66342 ± 0.02148 | **0.71234 ± 0.06512** | 0.59860 ± 0.06217 | **0.63469 ± 0.05318** |
| connectionist | 0.78764 ± 0.01644 | 0.72634 ± 0.05323 | 0.79243 ± 0.01853 | 0.81272 ± 0.00909 | 0.66846 ± 0.00843 | **0.83991 ± 0.02029** | 0.78333 ± 0.01880 |
| frogs         | 0.99064 ± 0.00091 | 0.96854 ± 0.00444 | 0.97717 ± 0.00200 | 0.99304 ± 0.00000 | **0.94524 ± 0.00232** | 0.98745 ± 0.00194 | 0.98169 ± 0.00087 |
| heart_2cl     | 0.82107 ± 0.02139 | 0.74410 ± 0.05252 | 0.81084 ± 0.04649 | 0.68900 ± 0.00000 | **0.89092 ± 0.03232** | **0.85742 ± 0.04248** | **0.79788 ± 0.08831** |
| ionosphere    | 0.94449 ± 0.01031 | 0.86825 ± 0.03093 | 0.95142 ± 0.00873 | 0.82372 ± 0.00996 | **0.85594 ± 0.01186** | **0.90219 ± 0.01152** | 0.91556 ± 0.00505 |
| parkinsons    | 0.91270 ± 0.02577 | 0.84864 ± 0.03577 | 0.89533 ± 0.03298 | 0.86208 ± 0.00585 | 0.78929 ± 0.02070 | 0.86293 ± 0.02470 | **0.85317 ± 0.00876** |

### Precision

|               | catboost          | decision_tree     | gbc               | knn               | lr                | mlp               | svm               |
|---------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| balance       | 0.85048 ± 0.01227 | 0.80887 ± 0.01252 | 0.83717 ± 0.01189 | 0.84035 ± 0.01774 | 0.89089 ± 0.03389 | 0.90199 ± 0.01847 | 0.86760 ± 0.01755 |
| breast-cancer | 0.95966 ± 0.00758 | 0.92566 ± 0.01621 | 0.95713 ± 0.00674 | 0.96518 ± 0.00000 | 0.97983 ± 0.00401 | 0.97285 ± 0.00473 | 0.96902 ± 0.00439 |
| breast-tissue | 0.81126 ± 0.05526 | 0.71117 ± 0.08801 | 0.72665 ± 0.06484 | 0.69018 ± 0.01253 | 0.74354 ± 0.08645 | 0.60262 ± 0.07566 | 0.66122 ± 0.08099 |
| connectionist | 0.78875 ± 0.01734 | 0.74320 ± 0.05750 | 0.79385 ± 0.01846 | 0.81521 ± 0.00827 | 0.67868 ± 0.00700 | 0.84281 ± 0.02219 | 0.78576 ± 0.01775 |
| frogs         | 0.99065 ± 0.00091 | 0.96859 ± 0.00443 | 0.97722 ± 0.00200 | 0.99307 ± 0.00000 | 0.94525 ± 0.00232 | 0.98746 ± 0.00193 | 0.98169 ± 0.00087 |
| heart_2cl     | 0.83945 ± 0.03059 | 0.74168 ± 0.05949 | 0.81908 ± 0.05670 | 0.66435 ± 0.00000 | 0.87540 ± 0.02209 | 0.83726 ± 0.02456 | 0.72622 ± 0.08195 |
| ionosphere    | 0.94881 ± 0.01035 | 0.86930 ± 0.03075 | 0.95417 ± 0.00902 | 0.85292 ± 0.00656 | 0.87340 ± 0.01084 | 0.91087 ± 0.01092 | 0.92013 ± 0.00431 |
| parkinsons    | 0.91533 ± 0.02663 | 0.85095 ± 0.03620 | 0.90074 ± 0.03723 | 0.86839 ± 0.00760 | 0.78860 ± 0.02119 | 0.87159 ± 0.02783 | 0.88944 ± 0.00452 |

### Recall

|               | catboost          | decision_tree     | gbc               | knn               | lr                | mlp               | svm               |
|---------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| balance       | 0.86427 ± 0.01213 | 0.75520 ± 0.01660 | 0.87787 ± 0.01011 | 0.71040 ± 0.03378 | 0.78213 ± 0.03354 | 0.84000 ± 0.03718 | 0.79493 ± 0.03243 |
| breast-cancer | 0.95936 ± 0.00767 | 0.92515 ± 0.01629 | 0.95643 ± 0.00697 | 0.96491 ± 0.00000 | 0.97983 ± 0.00402 | 0.97280 ± 0.00472 | 0.96900 ± 0.00438 |
| breast-tissue | 0.79546 ± 0.04507 | 0.70151 ± 0.06394 | 0.71515 ± 0.04970 | 0.66364 ± 0.02227 | 0.71667 ± 0.06175 | 0.63333 ± 0.04830 | 0.65455 ± 0.04636 |
| connectionist | 0.78809 ± 0.01667 | 0.72857 ± 0.05309 | 0.79285 ± 0.01860 | 0.81349 ± 0.00887 | 0.66984 ± 0.00809 | 0.84047 ± 0.02053 | 0.78333 ± 0.01880 |
| frogs         | 0.99064 ± 0.00091 | 0.96859 ± 0.00443 | 0.97721 ± 0.00200 | 0.99305 ± 0.00000 | 0.94533 ± 0.00232 | 0.98745 ± 0.00194 | 0.98170 ± 0.00087 |
| heart_2cl     | 0.84444 ± 0.01889 | 0.74938 ± 0.04823 | 0.83457 ± 0.03853 | 0.72222 ± 0.00000 | 0.91563 ± 0.04422 | 0.89144 ± 0.05743 | 0.89660 ± 0.10057 |
| ionosphere    | 0.94554 ± 0.01011 | 0.86948 ± 0.03019 | 0.95211 ± 0.00861 | 0.83568 ± 0.00840 | 0.86291 ± 0.01087 | 0.90516 ± 0.01087 | 0.91737 ± 0.00479 |
| parkinsons    | 0.91453 ± 0.02593 | 0.85096 ± 0.03651 | 0.89915 ± 0.03306 | 0.87008 ± 0.00640 | 0.79145 ± 0.02065 | 0.87008 ± 0.02381 | 0.87008 ± 0.00640 |


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
