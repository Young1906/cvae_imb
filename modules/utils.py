import numpy as np

def enable_stochastic_process(f):
    def wrapper(*args, **kwargs):
        np.random.seed() # cancel effect of numpy seed
        result = f(*args, **kwargs)
        np.random.seed(1) # enable effect of numpy seed
        return result
    return wrapper

