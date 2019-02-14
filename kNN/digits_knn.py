import numpy as py
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

digits = datasets.load_digits()
logging.info(digits.keys())
logging.info(digits['DESCR'])

x = digits.data
y = digits.target

'''参数x和y分别是数据和结果标签'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(x_train, y_train)
y_predict = knn_clf.predict(x_test)

logging.info(accuracy_score(y_test, y_predict))

logging.info(knn_clf.score(x_test, y_test))

'''最合适的K值'''
best_score = 0.0
best_k = 0
for k in range(2, 15):
    knn_clf = KNeighborsClassifier(n_neighbors=k, weights='distance®')
    knn_clf.fit(x_train, y_train)
    score  = knn_clf.score(x, y)
    if score > best_score:
        best_k = k
        best_score = score

logging.info(best_k)
logging.info(best_score)
