"""
"""

import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split

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
        # Modeling p(X) by Multivariate Gaussian
        Mu = X.mean(axis=0)
        Sig = np.std(X, axis=0)

        # Prior pdf
        f = lambda x: self.gaussian_loglikelihood(x, Mu, Sig)


        X, X_valid, y, y_valid = train_test_split(
                X, y,
                test_size=.25,
                stratify=y)

        # initial classifier
        d0 = self.d_factory()
        d0, (m, s) = self.__fit_d(d0, X, y)

        # Proposal distribution
        q = lambda x: np.random.normal(x, self.step_size * Sig)

        X_syn, y_syn = self.__wrapMCMC(d0, f, q, X, y)



    def __fit_d(self, d, X, y):
        """
        d: sklearn classifier
        X, y: data
        """

        cv = cross_val_score(d, X, y, cv=5)
        d.fit(X, y)

        return d, (cv.mean(), cv.std())


    def resample(self):
        pass


if __name__ == "__main__":
    d_factory = lambda: LogisticRegression()
    omc = MCMCOverSampling(d_factory, max_iter= 5, step_size=.25)
    X, y = np.random.normal(0, 1, (128, 16)),\
            np.random.randint(low=0, high=2, size=(128))

    omc.fit(X, y)

