import numpy as np
import math


def mean_squared_error(y_true, y_predict):
    """MSE"""
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to zhe size of y_predict"

    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """RMSE"""
    assert len(y_true) == len(y_predict), 'the size of y_true must be equal to the size of y_predict'

    return math.sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """MAE"""
    assert len(y_true) == len(y_predict), 'the size of y_true must bu equal of th e size of y_predict'

    return np.sum(np.absolute((y_true - y_predict))) / len(y_true)


def r2_score(y_true, y_predict):
    return 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
