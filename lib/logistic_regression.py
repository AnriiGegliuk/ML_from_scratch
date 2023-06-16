import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, learning_rate = 0.0001, num_itterations = 1000):
        self.lr = learning_rate
        self.iter = num_itterations
        self.slope = None
        self.intercept = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        sample, features = X.shape
        self.slope = np.zeros(features)
        pass

    def predict(self, X):
        pass
