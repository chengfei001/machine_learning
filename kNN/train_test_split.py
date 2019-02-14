import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import logging
from sklearn import datasets
from kNN.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO)

iris = datasets.load_iris()
'''Desc dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])'''
X = iris.data
y = iris.target

'''练习测试'''
# shuffle_indexes = np.random.permutation(len(X))
# logging.info(shuffle_indexes)
#
# test_ratio = 0.2
# test_size = int(len(X) * test_ratio)
#
# test_indexes = shuffle_indexes[:test_size]
# train_indexes = shuffle_indexes[test_size:]
#
# X_train = X[train_indexes]
# y_train = y[train_indexes]
#
# X_test = X[test_indexes]
# y_train = y[train_indexes]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# logging.info(X_train.shape)
# logging.info(X_train.shape)
# logging.info("-------------")
# logging.info(X_test.shape)
# logging.info(y_test.shape)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
y_predict = neigh.predict(X_test)
logging.info(y_predict)
logging.info(y_test)
logging.info('预测准确率：%f', sum(y_predict == y_test) / len(y_test))
