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
import matplotlib.pyplot as plt

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


def visualizer(loss, accuracy, n_epochs):
    """
    Returns the plot of Training/Validation Loss and Accuracy.
    :param loss: A list defaultdict with 2 keys "train" and "val".
    :param accuracy: A list defaultdict with 2 keys "train" and "val".
    :param n_epochs: Number of Epochs during training.
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    x = np.arange(0, n_epochs, 1)
    axs[0].plot(x, loss['train'], 'b')
    axs[0].plot(x, loss['val'], 'r')
    axs[1].plot(x, accuracy['train'], 'b')
    axs[1].plot(x, accuracy['val'], 'r')

    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss value")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy value (in %)")

    axs[0].legend(['Training loss', 'Validation loss'])
    axs[1].legend(['Training accuracy', 'Validation accuracy'])
