from math import exp


# TODO: use numpy to speed this up
def softmax(array: list[float]):
    to_e_power = list(map(exp, array))
    denominator = sum(to_e_power)
    return list(map(lambda x: x / denominator, to_e_power))
