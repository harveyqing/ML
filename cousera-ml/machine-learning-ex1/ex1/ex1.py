# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot


def plotData(X, y):
    pyplot.plot(X, y, 'rx', markersize=10)
    pyplot.xlabel('Population of City in 10,000s')
    pyplot.ylabel('Profit in $10,000s')


def computeCost(X, y, theta):
    m = len(y)
    return 1.0 / (2 * m) * np.sum(
        np.power(X.dot(theta) - y.reshape((m, 1)), 2))


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    j_history = np.zeros((num_iters, 1))

    for i in xrange(num_iters):
        err = X.dot(theta) - y.reshape((m, 1))
        delta = alpha * 1.0 / m * (np.transpose(X).dot(err))
        theta -= delta
        j_history[i] = computeCost(X, y, theta)

    return theta, j_history


# ============================== Plotting ==============================
print 'Plotting Data ...\n'
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(X)

#: Plot Data
plotData(X, y)
pyplot.show()

# =========================== Cradient descent ===========================
print 'Running Gradient descent ...\n'
X = np.c_[np.ones(m), data[:, 0]]  # Add a column of ones to X
theta = np.zeros((2, 1))  # initialize fitting parameters

#: Some gradient descent settings
iterations = 1500
alpha = 0.01

#: compute and display initial cost
computeCost(X, y, theta)

#: run gradient descent
theta, _ = gradientDescent(X, y, theta, alpha, iterations)

#: print theta to screen
print 'Theta found by gradient descent: %f %f \n' % (theta[0][0], theta[1][0])

#: Plot the linear fit
plotData(X[:, 1], y)
pyplot.hold(True)
pyplot.plot(X[:, 1], X.dot(theta), '-')
pyplot.legend(('Training data', 'Linear regression'))
pyplot.show()

#: Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print 'For population = 35,000, we predict a profit of %f\n' % (
    predict1 * 10000)
predict2 = np.array([1, 7]).dot(theta)
print 'For population = 70,000, we predict a profit of %f\n' % (
    predict2 * 10000)
