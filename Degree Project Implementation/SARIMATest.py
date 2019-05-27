from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from numpy import ndarray
import pandas as pd
import itertools
import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np


def parser(x):
	return datetime.strptime( x, '%Y-%m-%d')

series = read_csv('TheData2.csv', header=0, parse_dates=[0], index_col=0,squeeze=True, date_parser=parser)
monthlySeries = read_csv('TheData3.csv', header=0, parse_dates=[0], index_col=0,squeeze=True, date_parser=parser )
#testseries = series['12']
#testseries.plot(figsize=(10,5), title= 'TestData', fontsize=14)
#pyplot.show()

#2192 rows in the dataset. Subsetting for trainingdata and test data.
#2007 is the last day of June, so we will forecast the next 6 months as a test.
df=series['12']
dfm = monthlySeries['12']
#df= series

trainingData = df[0:len(df)-365]
testData = df [len(df)-365:]
monthlyTraining = dfm[0:len(df)-12]
monthlyTest = dfm [len(dfm)-12:]
'''monthlyData = []
for i in range(5) :
	monthlyData.append(df['201'+str(i)+'-01-01'])
	monthlyData.append(df['201'+str(i)+'-02-01'])
	monthlyData.append(df['201'+str(i)+'-03-01'])
	monthlyData.append(df['201'+str(i)+'-04-01'])
	monthlyData.append(df['201'+str(i)+'-05-01'])
	monthlyData.append(df['201'+str(i)+'-06-01'])
	monthlyData.append(df['201'+str(i)+'-07-01'])
	monthlyData.append(df['201'+str(i)+'-08-01'])
	monthlyData.append(df['201'+str(i)+'-09-01'])
	monthlyData.append(df['201'+str(i)+'-10-01'])
	monthlyData.append(df['201'+str(i)+'-11-01'])
	monthlyData.append(df['201'+str(i)+'-12-01'])



print(monthlyData)
print(len(monthlyData))
'''
#print(df)

def calculateError(result):
    print( len(testData))
    print( len(result))
    errorArr = []
    for x in range(len(testData)):
    	errorArr.append(float(result[x])/float(testData[x]))
    sum = 0.00
    for x in range(len(errorArr)):
    	sum += errorArr[x]
    averagePercentage = sum / len(errorArr)	
    averagePercentage = 1.00 - averagePercentage
    averagePercentage = averagePercentage * 100
    returnedValue = str(averagePercentage)+ '%'
    return returnedValue

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#lets try to remove some trends and seasonality.
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced forecast
def inverse_difference(last_ob, value):
	return value + last_ob

def printResults(actual, predicted):
	print('The RMSE for the forecast is: ' + str(measure_rmse(actual,predicted )))
	print('The MSE for the forecast is: '+str(mean_squared_error(actual, predicted)))
	print('The MAE for the forecast is: '+str(mean_absolute_error(actual, predicted)))
	print('The MAPE for the forecast is: '+str(mean_absolute_percentage_error(actual, predicted))+'%')


'''
# define a dataset with a linear trend
data = trainingData
pyplot.plot(data)
pyplot.show()
# difference the dataset
diff = difference(data, 360)
#print(diff)
pyplot.plot(diff)
pyplot.show()
# invert the difference
inverted = [inverse_difference(data[i], diff[i]) for i in range(len(diff))]
pyplot.plot(inverted)
pyplot.show()
'''


#
#
#Lets do a grid search for the fitting model.
#
#
# grid search sarima hyperparameters
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# one-step sarima forecast
def sarima_forecast(history, config):
	order, sorder, trend = config
	# define model
	model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False, enforce_invertibility=False)
	# fit model
	model_fit = model.fit(disp=False)
	# make one step forecast
	yhat = model_fit.predict(len(history), len(history))
	return yhat[0]

# root mean squared error or rmse
def measure_rmse(actual, predicted):
	return sqrt(mean_squared_error(actual, predicted))

# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
	return data[:-n_test], data[-n_test:]

# walk-forward validation for univariate data
def walk_forward_validation(data, n_test, cfg):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# fit model and make forecast for history
		yhat = sarima_forecast(history, cfg)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
	# estimate prediction error
	error = measure_rmse(test, predictions)
	return error

# score a model, return None on failure
def score_model(data, n_test, cfg, debug=False):
	result = None
	# convert config to a key
	key = str(cfg)
	# show all warnings and fail on exception if debugging
	if debug:
		result = walk_forward_validation(data, n_test, cfg)
	else:
		# one failure during model validation suggests an unstable config
		try:
			# never show warnings when grid searching, too noisy
			with catch_warnings():
				filterwarnings("ignore")
				result = walk_forward_validation(data, n_test, cfg)
		except:
			error = None
	# check for an interesting result
	if result is not None:
		print(' > Model[%s] %.3f' % (key, result))
	return (key, result)

# grid search configs
def grid_search(data, cfg_list, n_test, parallel=True):
	scores = None
	if parallel:
		# execute configs in parallel
		executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
		tasks = (delayed(score_model)(data, n_test, cfg) for cfg in cfg_list)
		scores = executor(tasks)
	else:
		scores = [score_model(data, n_test, cfg) for cfg in cfg_list]
	# remove empty results
	scores = [r for r in scores if r[1] != None]
	# sort configs by error, asc
	scores.sort(key=lambda tup: tup[1])
	return scores

# create a set of sarima configs to try
def sarima_configs(seasonal=[0]):
	models = list()
	# define config lists
	p_params = [2] #[0, 1, 2]
	d_params = [1] #[0, 1]
	q_params = [1] #[0, 1, 2]
	t_params = ['n','c','t','ct']
	P_params = [2] #[0, 1, 2]
	D_params = [0] #[0, 1]
	Q_params = [2] #[0, 1, 2]
	m_params = seasonal
	# create config instances
	for p in p_params:
		for d in d_params:
			for q in q_params:
				for t in t_params:
					for P in P_params:
						for D in D_params:
							for Q in Q_params:
								for m in m_params:
									cfg = [(p,d,q), (P,D,Q,m), t]
									models.append(cfg)
	return models


'''
if __name__ == '__main__':
	# define dataset
	data = trainingData['12']
	print(data)
	# data split
	n_test = 4
	# model configs
	cfg_list = sarima_configs()
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
'''

'''
if __name__ == '__main__':
	# load dataset
	#series = read_csv('monthly-car-sales.csv', header=0, index_col=0)
	data = trainingData #trainingData['12']
	print(data.shape)
	# data split
	n_test = 12
	# model configs
	cfg_list = sarima_configs(seasonal=[7])
	# grid search
	scores = grid_search(data, cfg_list, n_test)
	print('done')
	# list top 3 configs
	for cfg, error in scores[:3]:
		print(cfg, error)
'''

'''
THese are the values
[(2, 1, 1), (2, 0, 2, 7), 'ct'] 834.3001089409712
[(2, 1, 1), (2, 0, 2, 7), 't'] 836.9687538893095
[(2, 1, 1), (2, 0, 2, 7), 'n'] 837.1173747930893

Long term forecast models
[(6, 0, 6), (1, 0, 0, 12), 'c'] 846.522172682712

'''


#'''
# SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# contrived dataset
data = trainingData
#data=monthlyTraining
#data = data.to_frame()
#print(trainingData)
# fit model
stopdate = '2015-01-05'
model = SARIMAX(data.astype(float), order=(2, 1, 1), seasonal_order=(2, 0, 2, 7), trend='ct')
model_fit = model.fit(disp=False)
# make prediction
yhat = model_fit.predict('2015-01-01', stopdate)
#trainingData.plot()
#testData.plot()
#monthlyTest.plot()
#print(yhat)
#yhat.plot()
#prediction = model.predict(25)
#prediction.plot()

''' Getting a good plot'''
RealData = testData[:stopdate]
pyplot.figure(figsize=(10, 5))
line1 = pyplot.plot(RealData.values, label='Data Set')
line2 = pyplot.plot(yhat.values, label='Model Prediction', linestyle='dashed')
#pyplot.legend((RealData, yhat),('Predicted', 'True'))
pyplot.legend()
#pyplot.xticks([])
pyplot.title("SARIMA Long-term Model")
pyplot.xlabel('Days')
pyplot.ylabel('GW/h')
pyplot.show()

#printing the data for the results.
actualData = testData[0: len(yhat)]
#actualData = monthlyTest[0:len(yhat)]
printResults(actualData, yhat)
#'''


#here is something different
#ACF and PACF with differenced values
'''
f1, (ax1, ax2, ax3) = pyplot.subplots(1, 3, figsize=(18, 5))
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
a = np.array(dfm.values)
a = a.flatten()
a = difference(a)

print ('finished the flattening')
plot_acf(a, ax=ax1, lags = range(0,50))
ax1.set_title('ACF for the power load')

plot_pacf(a, ax=ax2, lags = 50)
ax2.set_title('PACF for the power load')
print('probably finished with the plotting?')

pyplot.show()
#print(series.head(10))
'''

'''
print('The RMSE for the forecast is: ' + str(measure_rmse(actualData,yhat )))
print('The MSE for the forecast is: '+str(mean_squared_error(actualData, yhat)))
print('The MAE for the forecast is: '+str(mean_absolute_error(actualData, yhat)))
print('The MAPE for the forecast is: '+str(mean_absolute_percentage_error(actualData, yhat))+'%')
'''
