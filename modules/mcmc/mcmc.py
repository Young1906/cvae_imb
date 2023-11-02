"""
.idea:
    using MCMC to oversampling minority class to addresse imbalance learning
    problem.

    quantity needed :
    H = p(x | y) / p(x_t | y) \
            = [p(x, y) / p(y)] / [p(x_t, y) / p(y)]\
            = [p(y | x) * p(x)]/ [p(y | x_t) * p(x_t)]
            = p(y | x) / p(y | x_t) * p(x) / p(x_t)

    to model :
    + p(y | x) -> train a classifier d_theta(x) = p(y | x)
    + p(x) -> simply assume x ~ N(mu, sigma)
"""

import numpy as np
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import f1_score

from modules.clf import build_classifier


class MCMCOverSampling:
    def __init__(self, d_factory: callable, max_iter: int = 1, step_size: float = .25):
        self.max_iter = max_iter
        self.d_factory = d_factory
        self.step_size = step_size

    @staticmethod
    def gaussian_loglikelihood(x, mu, scale) -> float:
        return (- np.log(scale) \
                - 0.5 * (x - mu) * (x - mu) / scale).sum()

    @staticmethod
    def __mcmc(d, f, q, X, y, target, n_samples: int, burn_in: int = None):
        """
        d: trained classifier
        f: prior of X
        q: proposal distribution
        X, y: observation
        target: class to sample from
        n_samples: number of samples
        """

        if burn_in is None:
            burn_in = n_samples // 4

        # Initial value: 
        Xt = X[y == target, :]
        N, _ = Xt.shape
        idx = np.random.choice(N)
        x0 = Xt[idx,:]

        # Decision function
        _d = lambda x: d.predict_proba(np.expand_dims(x, 0))[0, target] # p(y = target | x)

        # Sample
        samples = []
        counter = 0
        while True:
            # Proposal
            x = q(x0)

            # Compute H = p(x|y) / p(x_t | y)
            H = np.exp(f(x) - f(x0)) * _d(x) / _d(x0)
            H = 1 if H > 1 else H

            u = np.random.uniform()

            if u < H:
                samples.append(x)
                x0 = x
                counter +=1

            if counter >= (n_samples + burn_in) :
                samples = samples[burn_in:]
                return np.stack(samples)

    @staticmethod
    def __wrapMCMC(d, f, q, X, y):
        """
        Automatically compute number of obs need to samples
        """
        counter = Counter(y)
        _max = np.max(list(counter.values()))

        to_samples = []

        # Compute number of samples per class
        for c, n in counter.items():
            _n = _max - n
            if _n:
                to_samples.append((c, _n))

        X_syn, y_syn = [], []
        for (c, n) in to_samples:
            _X = MCMCOverSampling.__mcmc(d, f, q, X, y, target=c, n_samples=n)
            X_syn.append(_X)
            y_syn.append(np.ones(n) * c)

        return np.concatenate([X, *X_syn], 0), np.concatenate([y, *y_syn])



    def fit(self, X, y):
        # Modeling p(X) by Gaussian
        Mu = X.mean(axis=0)
        Sig = np.std(X, axis=0)

        # Prior pdf
        f = lambda x: self.gaussian_loglikelihood(x, Mu, Sig)


        X, X_valid, y, y_valid = train_test_split(
                X, y,
                test_size=.25,
                stratify=y)

        # initial classifier
        d = self.d_factory()
        d = self.__fit_d(d, X, y)

        # Proposal distribution
        q = lambda x: np.random.normal(x, self.step_size * Sig)

        self.best_f1 = 0
        self.best_d = None

        bar = tqdm(range(self.max_iter))

        for _ in bar: 
            X_syn, y_syn = self.__wrapMCMC(d, f, q, X, y)
            d = self.__fit_d(d, X_syn, y_syn)

            f1 = f1_score(y_valid, d.predict(X_valid), average="weighted")

            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_d = d
            bar.set_description(f"Best F1: {self.best_f1: .5f}")

        return self.__wrapMCMC(self.best_d, f, q, X, y)


    def __fit_d(self, d, X, y):
        """
        d: sklearn classifier
        X, y: data
        """
        d.fit(X, y)
        return d


    def resample(self, X, y):
        return self.fit(X, y)

if __name__ == "__main__":
    # d_factory = lambda: build_classifier("mlp")
    # d_factory = lambda: RandomForestClassifier()


    omc = MCMCOverSampling(lambda: build_classifier("lr"), max_iter= 100, step_size=.25)

    from modules.data import _build_Xy
    (X, y), (X_test, y_test), le = _build_Xy("breast-tissue")

    # Baseline
    clf = build_classifier("svm")
    clf.fit(X, y)
    print(f1_score(y_test, clf.predict(X_test), average="weighted"))


    X_syn, y_syn = omc.fit(X, y)
    clf = build_classifier("svm")
    clf.fit(X_syn, y_syn)
    print(f1_score(y_test, clf.predict(X_test), average="weighted"))


