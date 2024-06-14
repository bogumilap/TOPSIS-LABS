import numpy as np

def softmax(array: list[float]):
    np_array = np.array(array)
    exp_array = np.exp(np_array)
    return list(exp_array / np.sum(exp_array))
