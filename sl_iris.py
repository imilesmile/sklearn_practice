#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#创建数据
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print iris_X[:2,:]
"""
[[ 5.1  3.5  1.4  0.2]
 [ 4.9  3.   1.4  0.2]]
"""
print iris_y
"""
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
"""

X_train, X_test, y_train, y_test = train_test_split(
    iris_X, iris_y, test_size = 0.3
)

print y_train
"""
[2 0 0 1 1 2 2 1 1 2 0 1 0 1 1 2 2 0 2 0 0 2 2 0 1 0 2 1 0 1 1 2 2 0 0 0 1
 0 2 1 2 2 2 2 2 2 0 0 0 1 1 2 2 0 1 2 2 0 1 0 0 2 1 1 2 1 0 0 1 1 2 0 1 1
 0 1 1 1 0 0 1 0 1 1 1 2 1 0 2 2 1 0 0 1 1 2 1 1 1 2 0 1 0 0 0]
"""

knn = KNeighborsClassifier()
#训练
knn.fit(X_train, y_train)
print knn.predict(X_test)
print y_test