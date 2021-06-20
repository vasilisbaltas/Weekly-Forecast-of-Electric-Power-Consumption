# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 22:47:12 2021

@author: Vasileios Baltas
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
import copy
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX            # Load specific forecasting tools
from pmdarima import auto_arima                                   # for determining SARIMAX orders



df = pd.read_csv(r'C:\Users\Vasileios Baltas\Desktop\TimeSeries\Power_Usage_Prediction\final_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

df.index.freq ='D'    # since we have daily data we can change the index frequency


### turn our float values to integers so that auto_arima can work
cols = ['Global_reactive_power','Voltage','Sub_metering_1','Sub_metering_2','Sub_metering_3','sub_metering_4']
for col in cols:
    df[col] = df[col].astype(int)
    

# train-test split

train, test = df[1:-328].copy(), df[-328:-6].copy()




def optimal_parameters(data, exogenous_features):

  # the below function gives us the best configuration for Sarimax components
  print(auto_arima(data['Global_active_power'], exogenous = data[exogenous_features], stationary = True, seasonal=True, m=7).summary())       

  




def sarimax(train_set, exogenous_features, trend_elements, seasonal_elements):


  model = SARIMAX(train_set['Global_active_power'], exog = train_set[exogenous_features], order = trend_elements, seasonal_order = seasonal_elements, enforce_invertibility = False)
  sarimax = model.fit()
  print(sarimax.summary())

  return sarimax




def make_predictions(model,train, test):

  
  start = len(train)
  end = len(train)+len(test)-1
  exog_forecast = test[['Sub_metering_1','Sub_metering_3','sub_metering_4']]
  predictions = model.predict(start=start, end=end, exog = exog_forecast).rename('SARIMAX Predictions')

  predicted = np.array(np.split(predictions, len(predictions)/7))    # returns predictions in a weekly shape

  return predicted





def evaluate_forecasts(actual, predicted):
    
	scores = list()
    
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		rmse = sqrt(mse)
	
		scores.append(rmse)
        
        
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
            
			s += (actual[row, col] - predicted[row, col])**2
            
	score = sqrt(s /(actual.shape[0] * actual.shape[1]))
    
	return score, scores






if  __name__ == '__main__':
    
    
    # chosing only the exogenous features with the biggest correlation to make training faster and help the algorithm converge
    optimal_parameters(df, ['Sub_metering_1','Sub_metering_3','sub_metering_4'])
    
    # the algorithm automatically gave us the components (0,0,3)x(2,0,[1],7)
    
    # train our SARIMAX model
    model = sarimax(train, ['Sub_metering_1','Sub_metering_3','sub_metering_4'], (0,0,3), (2,0,[1],7)) 
    
    
    # obtain predictions for the test set in weekly windows
    predicted_values = make_predictions(model,train,test)
    
    
    # turn test set in a weekly shape as well in order to compare with model's predictions
    actual_values = np.array(np.split(test['Global_active_power'], len(test)/7))

    
    # obtain overall RMSE as well as for each day of the week separately
    score, scores = evaluate_forecasts(actual_values, predicted_values)

   # plot RMSE per day
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    plt.plot(days, scores, marker = 'o', label='SARIMAX')
    plt.show()

    print('Overall RMSE is:', score)













