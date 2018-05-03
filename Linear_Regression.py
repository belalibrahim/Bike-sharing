import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data
train = pd.read_csv('Dataset/reg_train.csv', index_col=['instant', 'dteday'])
test = pd.read_csv('Dataset/reg_test.csv', index_col=['instant', 'dteday'])

# Initialize the predictors and the response
predictors = train.drop(['cnt', 'casual', 'registered'], axis=1)
response = train['cnt']

predictors = np.column_stack([np.ones([len(predictors), 1]), predictors])
predictors_t = predictors.transpose()

x_test = np.column_stack([np.ones([len(test), 1]), test])

b_hat = np.dot(np.dot(np.linalg.inv(np.dot(predictors_t, predictors)), predictors_t), response)

# Prediction of the training observation
y_hat_pre = np.dot(predictors, b_hat)

# For a future observation, the prediction:
y_hat = np.zeros([len(x_test), ])
for i in range(len(x_test) - 1):
    y_hat[i] = np.dot(x_test[i], b_hat)

# Training Loss
RSS = sum((response - y_hat_pre) ** 2)
MSE = RSS / len(response)

print("Residual Sum of Squares(RSS):", RSS)
print("Mean Square Error(MSE):", MSE)

# Set the data into the suitable format
test = test.reset_index(level=1, drop=True)
test = test.drop(['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed'], axis=1)
test['cnt'] = y_hat

# Save the data to csv format
test.to_csv('lr_output.csv', sep=',')
