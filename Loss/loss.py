############################################################################################
#                                  Author: Anas AHOUZI                                     #
#                               File Name: loss.py                                         #
#                           Creation Date: September 23, 2020                              #
#                         Source Language: Python                                          #
#     Repository: https://github.com/aahouzi/Simple-Document-Classification-from-scratch   #
#                              --- Code Description ---                                    #
#                Implementation of Cross-Entropy & L2 hinge loss/gradient.                 #
############################################################################################


################################################################################
#                                   Packages                                   #
################################################################################
import numpy as np
from numpy.linalg import norm

################################################################################
#                               Main Code                                      #
################################################################################


def sigmoid(z):
    """
    Computes the sigmoid function element wise over a numpy array.
    :param z: A numpy array.
    :return: A numpy array of the same size of z.
    """
    return 1 / (1 + np.exp(-z))


def log_loss(X, Y, W, N):
    """
    Computes the log-loss function, and its gradient over a mini-batch of data.
    :param X: The feature matrix of size (N, F+1), where F is the number of features.
    :param Y: The label vector of size (N, 1).
    :param W: The weight vector of size (F+1, 1).
    :param N: The number of samples in the mini-batch.
    :return: Loss (a scalar) and gradient (a vector of size (F+1, 1)).
    """

    Y_hat = sigmoid(np.matmul(X, W))
    loss = -1 / N * (np.matmul(Y.T, np.log(Y_hat)) + np.matmul((1 - Y).T, np.log(1 - Y_hat)))
    grad = -1 / N * np.matmul(X.T, Y - Y_hat)

    return loss[0][0], grad


def l2_hinge_loss(X, Y, W, C, N):
    """
    Computes the L2 regularized Hinge Loss function, and its gradient over a mini-batch of data.
    :param X: The feature matrix of size (F+1, N).
    :param Y: The label vector of size (N, 1).
    :param W: The weight vector of size (F+1, 1).
    :param C: A hyperparameter of SVM soft margin that determines the number of observations allowed in the margin.
    :param N: The number of samples in the mini-batch.
    :return: Loss (a scalar) and gradient (a vector of size (F+1, 1)).
    """

    l = 1 / N * C
    loss = (l / 2) * norm(W) ** 2 + 1 / N * np.sum(np.maximum(np.zeros((1, N)), 1 - np.multiply(Y.T, np.matmul(W.T, X))))
    grad = l * W + 1 / N * sum([-Y[i] * X[:, i:i + 1] if Y[i] * np.matmul(W.T, X[:, i:i + 1]) < 1 else 0 for i in range(N)])

    return loss, grad
