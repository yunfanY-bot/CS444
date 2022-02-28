"""Logistic regression model."""

import numpy as np
from sklearn import preprocessing


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me

        return 1 / (1 + np.exp(-1*z))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        self.w = np.ones(X_train.shape[1] + 1)

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_scaled = scaler.transform(X_train)
        X_scaled = np.insert(arr=X_scaled, obj=0, values=1, axis=1)

        for i in range(self.epochs):
            self.lr = self.lr
            for j in range(X_scaled.shape[0]):
                correct_label = y_train[j]
                cur_image = X_scaled[j]
                result = np.matmul(self.w, np.transpose(cur_image))
                result = self.lr * self.sigmoid(-1 * correct_label * result) * correct_label * cur_image
                self.w += result

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

        scaler = preprocessing.StandardScaler().fit(X_test)
        X_scaled = scaler.transform(X_test)
        X_scaled = np.insert(arr=X_scaled, obj=0, values=1, axis=1)

        result = np.matmul(self.w, np.transpose(X_scaled))

        result = self.sigmoid(result)
        return result > self.threshold
