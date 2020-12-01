############################################################################################
#                                  Author: Anas AHOUZI                                     #
#                               File Name: logreg_multiclass.py                            #
#                           Creation Date: September 23, 2020                              #
#                         Source Language: Python                                          #
#     Repository: https://github.com/aahouzi/Simple-Document-Classification-from-scratch   #
#                              --- Code Description ---                                    #
#               Implementation of MultiClass Logistic Regression algorithm.                #
############################################################################################


################################################################################
#                                   Packages                                   #
################################################################################

import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from Loss.loss import sigmoid, log_loss
import time

################################################################################
#                               Main Code                                      #
################################################################################


class LogisticRegression:
    def __init__(self, x_train, y_train, x_test, y_test, alpha, beta, mb, n_class, F, n_epochs, info):
        """
        This is an implementation from scratch of Multi Class Logistic Regression using One vs All strategy,
        and Momentum with SGD optimizer.

        :param x_train: Vectorized training data.
        :param y_train: Label training vector.
        :param x_test: Vectorized testing data.
        :param y_test: Label test vector.
        :param alpha: The learning rate.
        :param beta: Momentum parameter.
        :param mb: Mini-batch size.
        :param n_class: Number of classes.
        :param F: Number of features.
        :param n_epochs: Number of Epochs.
        :param info: 1 to show training loss & accuracy over epochs, 0 otherwise.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.alpha = alpha
        self.beta = beta
        self.mb = mb
        self.n_class = n_class
        self.F = F
        self.n_epochs = n_epochs
        self.info = info

    def relabel(self, label):
        """
        This function takes a class, and relabels the training label vector into a binary class,
        it's used to apply One vs All strategy.

        :param label: The class to relabel.
        :return: A new binary label vector.
        """

        y = self.y_train.tolist()
        n = len(y)
        y_new = [1 if y[i] == label else 0 for i in range(n)]

        return np.array(y_new).reshape(-1, 1)

    def momentum(self, y_relab):
        """
        This function is an implementation of the momentum with SGD optimization algorithm, and it's
        used to find the optimal weight vector of the logistic regression algorithm.
        :return:
        """

        # Initialize weights and velocity vectors
        W = np.zeros((self.F + 1, 1))
        V = np.zeros((self.F + 1, 1))

        # Store loss & accuracy values for plotting
        loss = defaultdict(list)
        accuracy = defaultdict(list)

        # Split into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(self.x_train, y_relab, test_size=0.1, random_state=42)
        n_train = len(x_train)
        n_val = len(x_val)

        for _ in range(self.n_epochs):

            start = time.time()
            train_loss = 0
            # Compute the loss and gradient over mini-batches of the training set
            for i in range(0, n_train - self.mb + 1):
                train_loss, grad = log_loss(x_train[i:i + self.mb, :], y_train[i:i + self.mb], W, self.mb)
                V = self.beta * V + grad
                W = W - self.alpha * V

            # Compute the training accuracy
            train_proba = sigmoid(np.matmul(x_train, W))
            train_class = np.array([1 if p >= 0.5 else 0 for p in train_proba])
            train_acc = round((y_train.flatten() == train_class).mean() * 100, 2)

            # Compute the loss & accuracy over the validation set
            y_hat = sigmoid(np.matmul(x_val, W))
            val_loss = -1 / n_val * (np.matmul(y_val.T, np.log(y_hat)) + np.matmul((1 - y_val).T, np.log(1 - y_hat)))[0][0]
            val_class = np.array([1 if p >= 0.5 else 0 for p in y_hat])
            val_acc = round((y_val.flatten() == val_class).mean() * 100, 2)

            end = time.time()
            duration = round(end - start, 2)

            if self.info: print("Epoch: {} | Duration: {}s | Train loss: {} |"
                                " Train accuracy: {}% | Validation loss: {} | "
                                "Validation accuracy: {}%".format(_, duration,
                                round(train_loss, 5), train_acc, round(val_loss, 5), val_acc))

            # Append training & validation accuracy and loss values to a list for plotting
            loss['train'].append(train_loss)
            loss['val'].append(val_loss)
            accuracy['train'].append(train_acc)
            accuracy['val'].append(val_acc)

        return W, loss, accuracy

    def train(self):
        """
        This function trains the model using One-vs-All strategy, and returns a weight
        matrix, to be used during inference.
        :return: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        """

        weights = []
        loss, accuracy = 0, 0
        for i in range(1, self.n_class + 1):
            print("-" * 50 + " Processing class {} ".format(i) + "-" * 50 + "\n")
            y_relab = self.relabel(i)
            W, loss, accuracy = self.momentum(y_relab)
            weights.append(W)

        # Get the weights matrix as a numpy array
        weights = np.squeeze(np.array(weights), axis=2).T

        return weights, loss, accuracy

    def test(self, weights):
        """
        This function is used to test the model over new testing data samples, using
        the weights matrix obtained after training.
        :param weights: A weight matrix of size (F+1, n_class), where F is the number of features,
        and n_class is the number of classes to predict.
        :return:
        """

        # Getting the probabilities matrix for each class over the testing set (ntest, n_class)
        proba = sigmoid(np.matmul(self.x_test, weights))

        # For each sample in the probabilities matrix, we get the index of the maximum probability
        y_hat = proba.argmax(axis=1) + 1

        # Computing the test accuracy
        test_acc = round((y_hat == np.array(self.y_test)).mean() * 100, 2)
        print("-" * 50 + "\n Test accuracy is {}%".format(test_acc))
