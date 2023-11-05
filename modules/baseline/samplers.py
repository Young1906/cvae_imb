"""
Author: Tu T. Do
Email: tu.dothanh1906@gmail.com
"""

import numpy as np
from modules.utils import enable_stochastic_process
from collections import Counter

def build_sampler(name: str):
    """
    Baseline methods, including: 
    - ADASYN
    - KMeanSmote
    - SmoteNN
    - SVMSMote
    - InsHard
    """
    if name == "adasyn":
        from imblearn.over_sampling import ADASYN
        sampler = ADASYN()

    elif name == "kmean-smote":
        from imblearn.over_sampling import KMeansSMOTE
        sampler = KMeansSMOTE()

    elif name == "smotenn":
        from imblearn.combine import SMOTEENN
        sampler = SMOTEENN()

    elif name == "svm-smote":
        from imblearn.over_sampling import SVMSMOTE
        sampler = SVMSMOTE()

    elif name == "instance-hardness-threshold":
        from imblearn.under_sampling import InstanceHardnessThreshold
        sampler = InstanceHardnessThreshold()
    
    elif name == "baseline":
        sampler = Baseline()

    else: 
        raise NotImplementedError(name)

    return Wrapper(sampler)


class Wrapper:
    def __init__(self, sampler):
        self.sampler=sampler

    @enable_stochastic_process
    def fit_resample(self, X, y):
        return self.sampler.fit_resample(X, y)

class Baseline:
    def __init__(self): pass
    def fit_resample(self, X, y): return X, y



# if __name__ == '__main__':
#     X, y = np.random.normal(0, 1, size=(512, 16)),\
#             np.random.uniform(0, 1, size=(512))
#     y = (y > .3) * 1
# 
#     print(Counter(y))
# 
#     samplers = build_sampler("instance-hardness-threshold")
#     Xr, yr = samplers.fit_resample(X, y)
#     print(Counter(yr))



