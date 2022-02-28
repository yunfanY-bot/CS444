"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        # TODO: implement me
        gradient = np.zeros_like(self.w)

        for i in range(X_train.shape[0]):
            correct_label = y_train[i]
            cur_image = X_train[i]
            result = np.matmul(self.w, np.transpose(cur_image))
            result -= np.max(result)
            result = np.exp(result)
            result = result / np.sum(result)

            for c in range(self.n_class):
                if c == correct_label:
                    gradient[c] = self.lr * (1 - result[c]) * cur_image
                else:
                    gradient[c] = -1 * self.lr * result[c] * cur_image

        return gradient

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = np.ones([self.n_class, X_train.shape[1] + 1])
        X_train = np.insert(arr=X_train, obj=0, values=np.transpose(np.ones(X_train.shape[0])), axis=1)

        batch_size = 2
        loops = X_train.shape[0] // batch_size
        remains = X_train.shape[0] % batch_size

        for i in range(self.epochs):
            self.lr = self.lr * np.exp(-2 * i)
            for j in range(loops):
                gradient = self.calc_gradient(X_train[j * batch_size:(j + 1) * batch_size],
                                              y_train[j * batch_size:(j + 1) * batch_size])
                self.w = self.w + gradient
            if remains != 0:
                gradient = self.calc_gradient(X_train[loops * batch_size:loops * batch_size + remains],
                                              y_train[loops * batch_size:loops * batch_size + remains])
                self.w = self.w + gradient

        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        X_test = np.insert(arr=X_test, obj=0, values=np.transpose(np.ones(X_test.shape[0])), axis=1)

        result = np.matmul(self.w, np.transpose(X_test))
        result -= np.amax(result, axis=0)
        result = np.exp(result)
        result = result / np.sum(result, axis=0)
        labels = np.argmax(result, axis=0)

        return labels
