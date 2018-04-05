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
