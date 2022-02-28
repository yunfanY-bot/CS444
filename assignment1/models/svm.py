"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
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
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        batch_size = 20
        gradient = np.zeros_like(self.w)
        for i in range(X_train.shape[0]):
            correct_lable = y_train[i]
            cur_image = X_train[i]
            result = np.matmul(self.w, np.transpose(cur_image))

            for c in range(self.n_class):
                if c != correct_lable and result[correct_lable] - result[c] < 1:
                    gradient[correct_lable] += self.lr * cur_image
                    gradient[c] -= self.lr * cur_image
                gradient[c] -= self.w[c] * self.lr * self.reg_const / batch_size


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
        X_train = np.insert(arr=X_train, obj=0, values=1, axis=1)

        batch_size = 20
        loops = X_train.shape[0] // batch_size
        remains = X_train.shape[0] % batch_size

        for i in range(self.epochs):
            self.lr = self.lr * np.exp(-4*i)
            for j in range(loops):
                gradient = self.calc_gradient(X_train[j*batch_size:(j+1)*batch_size], y_train[j*batch_size:(j+1)*batch_size])
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
        X_test = np.insert(arr=X_test, obj=0, values=1, axis=1)

        result = np.matmul(self.w, np.transpose(X_test))
        lables = np.argmax(result, axis=0)

        return lables
