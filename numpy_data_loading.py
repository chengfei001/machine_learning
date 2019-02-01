import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import logging
from sklearn import datasets

logging.basicConfig(level= logging.INFO)


iris  = datasets.load_iris()
logging.info(iris.keys())
logging.info(iris['DESCR'])
logging.info(iris.data)
logging.info(iris.data.shape)
logging.info(iris.feature_names)
logging.info(iris.target)
logging.info(iris.target.shape)
logging.info(iris.target_names)

X = iris.data[:,2:]
logging.info(X.shape)

# 所有数据，萼片的长度x和宽度y
# plt.scatter(X[:, 1], X[:, 0])
# plt.show()

# 用不同颜色标出不同类别的鸢尾花
y = iris.target

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='green', marker='+')
plt.scatter(X[y == 2, 0], X[y == 2, 1], color='blue', marker='.')
plt.show()

