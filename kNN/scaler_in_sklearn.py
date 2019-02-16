import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO)

data = datasets.load_iris()
X = data.data
y = data.target

'''将数据分为训练集和测试集'''
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

'''归一化处理---预处理'''
standardScaler = StandardScaler()
# 放入数据计算均值和方差
standardScaler.fit(X_train)
# logging.info(standardScaler.mean_)
# logging.info(standardScaler.scale_)

# 归一化处理
X_train_standard = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

# logging.info(X_train_standard)
# logging.info(X_test_standard)

# 开始KNN分类
knn_clf = KNeighborsClassifier(n_neighbors=3)
# 分类
knn_clf.fit(X_train_standard, y_train)
# 归一化数据准确度百分比
logging.info(knn_clf.score(X_test_standard, y_test))
# 错误方法，没有做归一化的数据分类准确度百分比
logging.info(knn_clf.score(X_test, y_test))
