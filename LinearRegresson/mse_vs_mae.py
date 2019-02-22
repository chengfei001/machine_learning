import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import logging
from  LinearRegresson.simple_linear_regssion1 import SimpleLinearRegssion2
from sklearn.model_selection import  train_test_split
from LinearRegresson.metrics import mean_squared_error
from LinearRegresson.metrics import root_mean_squared_error
from LinearRegresson.metrics import mean_absolute_error
import math
import sklearn.metrics

logging.basicConfig(level=logging.INFO)

boston = datasets.load_boston()
# logging.info(boston.DESCR)
# logging.info(boston.feature_names)
x = boston.data[:, 5]
y = boston.target

x = x[y < 50.0]
y = y[y < 50.0]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=666)

# logging.info(x_train.shape )

reg = SimpleLinearRegssion2()
reg.fit(x_train, y_train)

logging.info('a=%f, b=%f', reg.a_, reg.b_)

plt.scatter(x_train, y_train)
plt.plot(x_train, reg.predict(x_train), color='b')
plt.show()

y_predict = reg.predict(x_test)

# MSE
mse_test =mean_squared_error(y_test, y_predict)
logging.info(mse_test)
# RMSE
rmse_test = root_mean_squared_error(y_test, y_predict)
logging.info(rmse_test)

# MAE
mae_test = mean_absolute_error(y_test, y_predict)
logging.info(mae_test)


# sklearn MSE

sklearn_mse_test = sklearn.metrics.mean_squared_error(y_test, y_predict)
logging.info('sklearn_mse_test:%f', sklearn_mse_test)

# sklearn RMSE
sklearn_rmse_test = math.sqrt(sklearn_mse_test)
logging.info('sklearn_rmse_test:%f', sklearn_rmse_test)

# sklearn MAE
sklearn_mae_test = sklearn.metrics.mean_absolute_error(y_test, y_predict)
logging.info('sklearn_mae_test:%f', sklearn_mae_test)



