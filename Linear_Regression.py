import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
train = pd.read_csv('Dataset/reg_train.csv', index_col=['instant', 'dteday'])
test = pd.read_csv('Dataset/reg_test.csv', index_col=['instant', 'dteday'])

# Initialize the predictors and the response
predictors = train.drop(['cnt', 'casual', 'registered'], axis=1)
response = train['cnt']

predictors_t = predictors.transpose()

b_hat = np.dot(np.dot(np.linalg.inv(np.dot(predictors_t, predictors)), predictors_t), response)
y_hat = np.mean(response) + np.dot(predictors - np.mean(predictors), b_hat)
