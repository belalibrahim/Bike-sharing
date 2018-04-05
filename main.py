import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as k
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam, RMSprop

# For reproducibility
np.random.seed(123)

# Read the data
train = pd.read_csv('Dataset/reg_train.csv', index_col=['instant', 'dteday'])
test = pd.read_csv('Dataset/reg_test.csv', index_col=['instant', 'dteday'])

# Initialize the predictors and the response
predictors = train.drop(['cnt'], axis=1)
response = train['cnt']

# Get the number of features
n_cols = predictors.shape[1]

'''
# Build the model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(1))

# Stop the training after the loss increases 3 times in a row
early_stopping_monitor = EarlyStopping(patience=3)

# Compile, train and save the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(predictors, response, validation_split=0.2, verbose=1, epochs=50, callbacks=[early_stopping_monitor])
model.save('model.h5')
'''

# Load the model
my_model = load_model('model.h5')

# Get the predictions
predictions = np.array(my_model.predict(test), dtype='int')

# Display model summary
my_model.summary()

# Set the data into the suitable format
test = test.reset_index(level=1, drop=True)
test = test.drop(['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered'], axis=1)
test['cnt'] = predictions

# Save the data to csv format
test.to_csv('output.csv', sep=',')
