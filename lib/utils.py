import pandas as pd
import numpy as np

######## Standarizing data ########

def standarizing(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std

def normalizing(X):
    min_value = np.min(X)
    max_value = np.max(X)

    return (X - min_value) / (max_value - min_value)

def robust_scaler(X):
    q1 = np.quantile(X, 25, axis = 0)
    q3 = np.quantile(X, 75, axis = 0)

    median = np.median(X, axis = 0)

    return (X - median) / (q3 - q1)

######## evaluating the performance of a regression model ########

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true, y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true, y_pred) ** 2)
    return np.sqrt(mse)
