# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot


def featureNormalize(X):

    # 每个feature的取值范围均不一致，故皆按列处理
    mu = X.mean(0)
    X_norm = X - mu

    sigma = X_norm.std(0)
    X_norm = X_norm / sigma

    return X_norm, mu, sigma


# ========================== Feature Normalization ==========================
#: Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = len(X)

#: Print out some data points
print 'First 10 examples from the dataset: \n'
for i in range(10):
    print ' x = [%.0f %.0f], y = %.0f \n' % (X[i][0], X[i][1], y[i])

#: Scale features and set them to zero mean
print 'Normalizing Features ...\n'
X, mu, sigma = featureNormalize(X)

#: Add intercept term to X
X = np.c_[np.ones(m), X]

# ============================= Gradient Descend =============================
print 'Running gradient descent ...\n'

#: Choose some alpha value
alpha = 0.01
num_iters = 400

#: Init Theta and Run Cradient Descent
