import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier


def build_classifier(name: str):
    if name == "svm":
        return SVC()

    elif name == "lr":
        return LogisticRegression()

    elif name == "knn":
        return KNeighborsClassifier() 

    elif name == "decision_tree":
        return DecisionTreeClassifier()

    elif name == "mlp":
        return MLPClassifier(hidden_layer_sizes=(32, 16))

    elif name == "gbc":
        return  GradientBoostingClassifier()

    elif name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(verbose=0)

    else:
        raise NotImplementedError(name)
