#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import print_function
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X, data_y)

print(model.predict(data_X[:4, :]))
print(data_y[:4])
"""
预测值:
[ 30.00821269  25.0298606   30.5702317   28.60814055]
真实值:
[ 24.   21.6  34.7  33.4]
"""

#创建虚拟数据
X, y = datasets.make_regression(n_samples=100, n_features=1,
                                n_targets=1, noise=8)

plt.scatter(X, y)
plt.show()