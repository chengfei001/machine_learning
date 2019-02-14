import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import logging
from sklearn import datasets
import datetime

start_time = datetime.datetime.now()

data = datasets.load_digits()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)


logging.basicConfig(level=logging.INFO)


'''配置网格参数'''
param_grid = (
    {
        'n_neighbors': [k for k in range(1, 11)],
        'weights': ['uniform']},
    {
        'n_neighbors': [k for k in range(1, 11)],
        'weights': ['distance'],
        'p': [p for p in range(1, 6)]
    }
)
knn_clf = KNeighborsClassifier()
'''n_jobs参与运算的cpu核数， verbose在运行中输出参数，默认为0，不输出，数字越大输出的内容越多'''
grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1, verbose=2, cv=5)
grid_search.fit(X_train, y_train)

end_time = datetime.datetime.now()

spend_time = end_time - start_time

logging.info('spend time:%s', spend_time)

