# coding=utf-8

from keras.models import Sequential
#from sklearn.metrics import confusion_matrix
from keras.layers import Dense
from keras.layers import LSTM
import matplotlib as plt
!pip install scikit-fuzzy
!pip install utils
import skfuzzy as library

from utils import look_back_dataset, \
                  load_data, \
                  normalize, \
                  split, \
                  reshape, \
                  transform, \
                  transform_predict, \
                  rmse, \
                  plot \

#lookback (int)
look_back = 5

#load the dataset
dataset = load_data()

#normalize data
dataset = normalize(dataset)

#split to train/test data
train, test = split(dataset)

#reshape looking back for ...
X_train, Y_train = look_back_dataset(train, look_back)
X_test, Y_test = look_back_dataset(test, look_back)

#reshape to fit RNN's input shape
X_train, X_test = reshape(X_train, X_test)

classifier = Sequential()
classifier.add(LSTM(units = 4, input_shape=(1, look_back)))
classifier.add(Dense(units = 1))

classifier.compile(loss='mean_squared_error', optimizer='nadam')

classifier.fit(X_train, Y_train, epochs=55, batch_size=1)

train_pred = classifier.predict(X_train)
test_pred = classifier.predict(X_test)

train_pred = transform_predict(train_pred)
test_pred = transform_predict(test_pred)
Y_train = transform(Y_train)
Y_test = transform(Y_test)

#calculate root mean square error for train and test data
#real Y vs. predicted Y
train_score = rmse(Y_train, train_pred)
test_score = rmse(Y_test, test_pred)

print('Train Score: %.2f RMSE' % (train_score))
print('Test Score: %.2f RMSE' % (test_score))

#plot(dataset, test_pred, test_pred)
plot(dataset, train_pred, test_pred, look_back)

classifier.summary()