"""
Author: Tu T. Do
Email: tu.dothanh1906@gmail.com
"""
import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, avg: str) -> list[float]:
    """
    Evaluate precision, recall and f1 score
    """
    p = precision_score(y_true, y_pred, average=avg)
    r = recall_score(y_true, y_pred, average=avg)
    f = f1_score(y_true, y_pred, average=avg)

    return p, r, f
