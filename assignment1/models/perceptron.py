"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.label_set = {}

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.label_set = set(y_train)

        self.w = np.ones([self.n_class, X_train.shape[1] + 1])
        X_train = np.insert(arr=X_train, obj=0, values=1, axis=1)

        for i in range(self.epochs):
            self.lr = 0.1 * self.lr
            for j in range(X_train.shape[0]):
                correct_label = y_train[j]
                cur_image = X_train[j]
                result = np.matmul(self.w, np.transpose(cur_image))

                for n in range(self.n_class):
                    if n != correct_label and result[n] >= result[correct_label]:
                            self.w[correct_label] += self.lr * cur_image
                            self.w[n] -= self.lr * cur_image

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
        X_test = np.insert(arr=X_test, obj=0, values=1, axis=1)

        result = np.matmul(self.w, np.transpose(X_test))
        labels = np.argmax(result, axis=0)

        return labels
