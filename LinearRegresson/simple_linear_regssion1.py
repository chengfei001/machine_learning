import numpy as np


class SimpleLinearRegssion1:
    def __init__(self):
        """初始化simple linear resgression1 模型"""
        self.a_ = 0.0
        self.b_ = 0.0

    def fit(self, x_train, y_train):
        assert x_train.ndim == 1, '必须为一维数据'
        assert len(x_train) == len(y_train), 'x_train和y_train的数据量必须相等'
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.
        d = 0.

        for x_train_i, y_train_i in zip(x_train, y_train):
            num += (x_train_i - x_mean) * (y_train_i - y_mean)
            d += (x_train_i - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        assert x_predict.ndim == 1, '数据必须为一维的。'
        assert self.a_ is not None and self.b_ is not None, '必须是fit后的数据'
        return np.array([self._predict(x_predict) for x in x_predict])

    def _predict(self, x_single):
        return self.a_ * x_single + self.b_
