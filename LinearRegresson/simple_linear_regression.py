import numpy as np
import matplotlib.pyplot as plt
import logging
from LinearRegresson.simple_linear_regssion1 import SimpleLinearRegssion1

logging.basicConfig(level=logging.INFO)
x = np.array([1., 2., 3., 4., 5.])
y = np.array([1., 3., 2., 3., 5.])

x_mean = np.mean(x)
y_mean = np.mean(y)

num = 0.00
d = 0.00
for x_i, y_i in zip(x, y):
    num += (x_i - x_mean) * (y_i - y_mean)
    d += (x_i - x_mean) ** 2

a = num / d
b = y_mean - a * x_mean
y_hat = a * x + b
x_prodict = 6.0
y_prodict = a * x_prodict + b

logging.info("a=%f, b=%f, y_hot=%s, y_prodict=%s", a, b, y_hat, y_prodict)

plt.scatter(x, y)
plt.axis([0, 6, 0, 6])
plt.plot(x, y_hat, color='y')
plt.show()
reg1 = SimpleLinearRegssion1()

reg1.fit(x, y)

x_hat1 = reg1.predict(np.array([x_prodict]))

logging.info(x_hat1)
