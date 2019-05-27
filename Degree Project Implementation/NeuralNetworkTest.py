import pandas as pd
import numpy as np
#matplotlib inline
from pandas import read_csv
from pandas import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from keras import losses

def parser(x):
	return datetime.strptime( x, '%Y-%m-%d')


def printResults(actual, predicted):
	print('The RMSE for the forecast is: ' + str(measure_rmse(actual,predicted )))
	print('The MSE for the forecast is: '+str(mean_squared_error(actual, predicted)))
	print('The MAE for the forecast is: '+str(mean_absolute_error(actual, predicted)))
	print('The MAPE for the forecast is: '+str(mean_absolute_percentage_error(actual, predicted))+'%')

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

#series = read_csv('TheData2.csv', header=0, parse_dates=[0], index_col=0,squeeze=True, date_parser=parser)
#df = series

#alternative approach
df = pd.read_csv("TheData2.csv")
df.drop(['0', '1', '2', '3', '4','5','6','7','8','9','10','11','13','14','15','16','17','18','19','20','21','22','23'], axis=1, inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index(['Date'], drop=True)
#print(df.head(10))



#print(series['Date'])

#plt.figure(figsize=(10, 6))
#df.plot()

#splitting
split_date = pd.Timestamp('2014-12-31')
stop_date = pd.Timestamp('2015-01-11')
df =  df['12']
df = df.to_frame()
train = df.loc[:'2014-12-31']
test = df.loc['2015-01-01':stop_date]
#print(train)

'''
plt.figure(figsize=(10, 6))
ax = train.plot()
test.plot(ax=ax)
plt.legend(['train', 'test'])
#plt.show()
'''

#plotting the first data.
#plt.figure(figsize=(10, 6))
#ax = train.plot()
#test.plot(ax=ax)
#plt.legend(['train', 'test'])
#plt.show()

#We scale train and test data to [-1, 1].
scaler = MinMaxScaler(feature_range=(-1, 1))
train_sc = scaler.fit_transform(train)
test_sc = scaler.transform(test)

#Get training and test data.
#X_train = train_sc[:]
#y_train = train_sc[0:]
X_train = train_sc[:-1]
y_train = train_sc[1:]
X_test = test_sc[:-1]
y_test = test_sc[1:]
#X_test = test_sc[:-1]
#y_test = test_sc[0:]

print(len(X_test))
print(len(y_test))

'''
Simple ANN for Time Series Forecasting
We create a Sequential model.
Add layers via the .add() method.
Pass an input_dim argument to the first layer.
The activation function is the Rectified Linear Unit- Relu.
Configure the learning process, which is done via the compile method.
A loss function is mean_squared_error , and An optimizer is adam.
Stop training when a monitored loss has stopped improving.
patience=2, indicate number of epochs with no improvement after which training will be stopped.
The ANN is trained for 100 epochs and a batch size of 1 is used.
'''

nn_model = Sequential()
nn_model.add(Dense(100, input_dim=1, activation='relu'))
#nn_model.add(Dense(100))
#nn_model.add(Dense(100))
#nn_model.add(Dense(100))
#nn_model.add(Dense(100))
nn_model.add(Dense(1))
nn_model.compile(loss='mean_squared_error', optimizer='sgd')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = nn_model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)

# trying this out then
y_pred_test_nn = nn_model.predict(X_test)
y_train_pred_nn = nn_model.predict(X_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_nn)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))

'''
The LSTM networks creation and model compiling is similar with those of ANN’s.

The LSTM has a visible layer with 1 input.
A hidden layer with 7 LSTM neurons.
An output layer that makes a single value prediction.
The relu activation function is used for the LSTM neurons.
The LSTM is trained for 100 epochs and a batch size of 1 is used.

'''

#LSTM as our second network to test. Doing some reshaping for the LSMT
train_sc_df = pd.DataFrame(train_sc, columns=['Y'], index=train.index)
test_sc_df = pd.DataFrame(test_sc, columns=['Y'], index=test.index)

for s in range(1,2):
    train_sc_df['X_{}'.format(s)] = train_sc_df['Y'].shift(s)
    test_sc_df['X_{}'.format(s)] = test_sc_df['Y'].shift(s)

X_train = train_sc_df.dropna().drop('Y', axis=1)
y_train = train_sc_df.dropna().drop('X_1', axis=1)

X_test = test_sc_df.dropna().drop('Y', axis=1)
y_test = test_sc_df.dropna().drop('X_1', axis=1)

X_train = X_train.as_matrix()
y_train = y_train.as_matrix()

X_test = X_test.as_matrix()
y_test = y_test.as_matrix()

X_train_lmse = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_lmse = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print('Train shape: ', X_train_lmse.shape)
print('Test shape: ', X_test_lmse.shape)

#Creating model
lstm_model = Sequential()
lstm_model.add(LSTM(100, input_shape=(1, X_train_lmse.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
#lstm_model.add(Dense(100))
#lstm_model.add(Dense(100))
#lstm_model.add(Dense(100))
#lstm_model.add(Dense(100))
lstm_model.add(Dense(1))
lstm_model.compile(loss='mean_squared_error', optimizer='sgd')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history_lstm_model = lstm_model.fit(X_train_lmse, y_train, epochs=100, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop]) #,callbacks=[early_stop]

#printing and predicting
y_pred_test_lstm = lstm_model.predict(X_test_lmse)
y_train_pred_lstm = lstm_model.predict(X_train_lmse)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))

#evaluating models
nn_test_mse = nn_model.evaluate(X_test, y_test, batch_size=1)
lstm_test_mse = lstm_model.evaluate(X_test_lmse, y_test, batch_size=1)
print('NN: %f'%nn_test_mse)
print('LSTM: %f'%lstm_test_mse)

#Predicting and plotting both network methods
#NN prediction first
nn_y_pred_test = nn_model.predict(X_test)
lstm_y_pred_test = lstm_model.predict(X_test_lmse)
plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(y_test), label='Data Set')
plt.plot(scaler.inverse_transform(y_pred_test_nn), label='Model Prediction', linestyle='dashed')
plt.title("NN's Prediction")
plt.xlabel('Days')
plt.ylabel('GW/h')
plt.legend()
plt.show()

#LSTM prediction
plt.figure(figsize=(10, 6))
plt.plot(scaler.inverse_transform(y_test), label='Data Set')
plt.plot(scaler.inverse_transform(y_pred_test_lstm), label='Model Prediction', linestyle='dashed')
plt.title("LSTM's Prediction")
plt.xlabel('Days')
plt.ylabel('GW/h')
plt.legend()
plt.show()

#printing some more results
actualValue = scaler.inverse_transform(y_test)
predictedValue=scaler.inverse_transform(y_pred_test_nn)
print(len(actualValue))
print(len(predictedValue))
print('The Results for NN ')
printResults(actualValue,predictedValue)
print('The Results for LSTM ')
actualValue = scaler.inverse_transform(y_test)
predictedValue=scaler.inverse_transform(y_pred_test_lstm)
printResults(actualValue,predictedValue)