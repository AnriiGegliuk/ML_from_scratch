import numpy as np
import pandas as pd

# creating LR class
class LinearRegression:
    def __init__(self, lr = 0.0001, iter = 1000):
        self.lr = lr
        self.iter = iter
        self.slope = None
        self.intercept = None

    # creating fit methond
    def fit(self, X, y):
        sample, features = X.shape
        self.slope = np.zeros(features)
        self.intercept = 0

        # compute gradients
        for i in range(self.iter):
            y_pred = self.predict(X)

            d_slope = (1 / sample) * np.dot(X.T, (y_pred - y))
            d_intercept = (1 / sample) * np.sum(y_pred - y)

            # update parameters
            self.slope -= self.lr * d_slope
            self.intercept -= self.lr * d_intercept

    # predicting y
    def predict(self, X):
        y_pred =  self.intercept + np.dot(X, self.slope)
        return y_pred
