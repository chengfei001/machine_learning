import numpy as np
import logging
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

'''最值归一化'''
'''一维数据'''
X = np.random.randint(0, 100, size=100)
X = (X - np.min(X)) / np.max(X) - np.min(X)
logging.info(X)

'''二维数据'''
Y = np.random.randint(0, 100, (50, 2))
Y = np.array(Y, dtype=float)
for i in range(0, 2):
    Y[:, i] = (Y[:, i] - np.min(Y[:, i])) / (np.max(Y[:, i]) - np.min(Y[:, i]))

logging.info(Y[:10, :])
plt.scatter(Y[:, 0],Y[:, 1])
plt.show()

# 均值
logging.info(np.mean(Y[:, 0]))
# 方差
logging.info(np.std(Y[:, 0]))

'''均值方差归一化'''
X2 = np.random.randint(0, 100, (50, 2))
X2 = np.array(X2, dtype=float)
for i in range(0, 2):
    X2[:, i] = (X2[:, i] - np.mean(X2[:, i])) / np.std(X2[:, i])

logging.info(X2)
plt.scatter(X2[:, 0], X2[:, 1])
plt.show()