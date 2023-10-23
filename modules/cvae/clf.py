import sklearn
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def build_classifier(name: str):
    if name == "svm":
        return SVC()

    elif name == "lr":
        return LogisticRegression()

    elif name == "knn":
        return KNeighborsClassifier() 

    elif name == "decision_tree":
        return DecisionTreeClassifier()

    else:
        raise NotImplementedError(name)
