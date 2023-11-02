"""
List of available classifier
- svm
- lr
- decision_tree
- mlp
- gradient boosting classifier
- catboost
"""

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier


def build_classifier(name: str):
    """
    Return corresponing sklearn classifier
    """
    if name == "svm":
        clf = SVC()

    elif name == "nb":
        clf = GaussianNB()

    elif name == "lr":
        clf = LogisticRegression()

    elif name == "knn":
        clf = KNeighborsClassifier()

    elif name == "decision_tree":
        clf = DecisionTreeClassifier()

    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(16, 8))

    elif name == "gbc":
        clf = GradientBoostingClassifier()

    elif name == "catboost":
        clf = CatBoostClassifier(verbose=0)

    else:
        raise NotImplementedError(name)

    return clf
