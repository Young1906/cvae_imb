from collections import Counter

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from modules.clf import build_classifier
from modules.data import _build_Xy


def enable_stochastic_process(f):
    def wrapper(*args, **kwargs):
        np.random.seed() # cancel effect of numpy seed
        result = f(*args, **kwargs)
        np.random.seed(1) # enable effect of numpy seed
        return result
    return wrapper


class MCMCOverSampling:
    def __init__(self, d_factory: callable, max_iter: int = 1, step_size: float = .25):
        self.max_iter = max_iter
        self.d_factory = d_factory
        self.step_size = step_size

        self.__fitted = False

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
        fail_counter = 0

        while True:
            # Proposal
            x = q(x0)

            # Compute H = p(x|y) / p(x_t | y)
            H = np.exp(f(x) - f(x0)) * _d(x) / _d(x0)
            H = 1 if H > 1 else H

            u = np.random.uniform()

            if u < H:
                if _d(x) == d.predict_proba(np.array([x],)).max():
                    samples.append(x)
                    x0 = x
                    counter +=1
                    fail_counter = 0

                else:
                    fail_counter += 1

            if counter >= (n_samples + burn_in) :
                samples = samples[burn_in:]
                return np.stack(samples)

            if fail_counter > 50:
                # restart the chain from a random in X
                idx = np.random.choice(N)
                x0 = Xt[idx,:]
                fail_counter = 0
                

    @staticmethod
    @enable_stochastic_process
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
        # Splitting
        X_train, X_valid, y_train, y_valid = train_test_split(
                X, y,
                test_size=.25,
                stratify=y)

        # Modeling p(X) by Gaussian
        Mu = X_train.mean(axis=0)
        Sig = np.std(X_train, axis=0)

        # Prior pdf
        f = lambda x: self.gaussian_loglikelihood(x, Mu, Sig)

        # initial classifier
        d = self.d_factory()
        d = self.__fit_d(d, X_train, y_train)

        # Proposal distribution
        q = lambda x: np.random.normal(x, self.step_size * Sig)

        self.best_f1 = 0
        self.best_d = d 

        bar = tqdm(range(self.max_iter))
        for i in bar: 
            X_syn, y_syn = \
                    self.__wrapMCMC(self.best_d, f, q, X_train, y_train)
            d = self.__fit_d(d, X_syn, y_syn)
            f1 = f1_score(y_valid, d.predict(X_valid), average="macro")

            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_d = d
            bar.set_description(f"Best F1: {self.best_f1: .5f}")

        self.__fitted=True


    def __fit_d(self, d, X, y):
        """
        d: sklearn classifier
        X, y: data
        """
        d.fit(X, y)
        return d


    def resample(self, X, y):

        if not self.__fitted:
            raise ValueError("The sampler is not fitted")

        # Modeling p(X) by Gaussian
        Mu = X.mean(axis=0)
        Sig = np.std(X, axis=0)

        # Prior pdf
        f = lambda x: self.gaussian_loglikelihood(x, Mu, Sig)
        # Proposal distribution
        q = lambda x: np.random.normal(x, self.step_size * Sig)
        
        return self.__wrapMCMC(self.best_d, f, q, X, y)


if __name__ == "__main__":

    omc = MCMCOverSampling(
            lambda: build_classifier("catboost"),
            max_iter= 5, step_size=.25)

    (X, y), (X_test, y_test), le = _build_Xy("breast-tissue")

    # Baseline
    clf = build_classifier("svm")
    clf.fit(X, y)
    print(f1_score(y_test, clf.predict(X_test), average="weighted"))


    omc.fit(X, y)
    X_syn, y_syn = omc.resample(X, y)

    clf = build_classifier("svm")
    clf.fit(X_syn, y_syn)
    print(f1_score(y_test, clf.predict(X_test), average="weighted"))
