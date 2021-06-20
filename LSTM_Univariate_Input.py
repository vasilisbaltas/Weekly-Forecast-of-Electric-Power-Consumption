import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Bidirectional


os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONHASHSEED']=str(33)
import tensorflow as tf
tf.compat.v1.set_random_seed(33)

from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)


df = pd.read_csv(r'C:\Users\Vasileios Baltas\Desktop\TimeSeries\Power_Usage_Prediction\final_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])



###    Train & Test split
    
# We will use the first three years of data for training predictive models and the final year for evaluating models
# The data in a given dataset will be divided into standard weeks - these are weeks that begin on a Sunday and end on a Saturday
# this is a realistic and useful way for using the chosen framing of the model, where the power consumption for the week ahead can be predicted

# We will split the data into standard weeks, working backwards from the test dataset
# The function split_dataset() below splits the daily data into train and test sets and organizes each into standard weeks


def split_dataset(data):
	# split into standard weeks
	train, test = data[1:-328], data[-328:-6]
	# restructure into windows of weekly data
	train = np.array(np.split(train, len(train)/7))
	test = np.array(np.split(test, len(test)/7))
	return train, test




# By splitting the dataset we obtain 159 weeks of training data which is a small sample for training a neural network
# To address this, we will use the whole training set range and create moving weeks so as to increase our sample size
# We will iterate over the time steps and divide the data into overlapping windows 
# Each iteration moves along one time step and predicts the subsequent seven days.

def to_supervised(train, n_input, n_out=7):
    
	# flatten data
	data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    
	X, y = list(), list()
	in_start = 0
    
	# step over the entire history one time step at a time
	for _ in range(len(data)):
        
		# define the end of the input sequence
		in_end = in_start + n_input
		out_end = in_end + n_out
        
		# ensure we have enough data for this instance
		if out_end <= len(data):
			x_input = data[in_start:in_end, 0]
			x_input = x_input.reshape((len(x_input), 1))
			X.append(x_input)
			y.append(data[in_end:out_end, 0])

		in_start += 1
        
	return np.array(X), np.array(y)





# We also want to know the efficiency of our model for each forecasted time step separately
# Running the below function will first return the overall RMSE regardless of day, then an array of RMSE scores for each day.
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




# The below function (evaluate_model) uses a walk-forward approach in order to evaluate a model 
# It also contains the function build_model that builds a model from training data and the function 'forecast' that makes forecasts  for each new standard week

def evaluate_model(train, test, n_input):
    

	model = build_model(train, n_input)
    
	# history is a list of weekly data
	history = [x for x in train]
    
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
        
		# predict the week
		yhat_sequence = forecast(model, history, n_input)
		predictions.append(yhat_sequence)
        
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
        
        
	# evaluate predictions days for each week
	predictions = np.array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    
	return score, scores




# train the model
def build_model(train, n_input):
    
	# prepare data
	train_x, train_y = to_supervised(train, n_input)
    
	# define some model parameters
	verbose, epochs, batch_size = 1, 70, 16
	n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    
	# define our model
	model = Sequential()
	model.add(Bidirectional(LSTM(200, activation='relu'), input_shape=(n_timesteps, n_features)))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(n_outputs))
	model.compile(loss='mse', optimizer='adam')
    

	model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    
	return model




# make  forecasts
def forecast(model, history, n_input):
    
	# flatten data
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    
	# retrieve last observations for input data
	input_x = data[-n_input:, 0]
    
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, len(input_x), 1))
    
	# forecast the next week   
	yhat = model.predict(input_x, verbose=0)
    
	# we only want the vector forecast
	yhat = yhat[0]
    
	return yhat





if  __name__ == '__main__':
    
      
   n_input = 7
   
   # split into train and test
   train, test = split_dataset(df.values)
   
   # evaluate 1st model and get scores
   score, scores = evaluate_model(train, test, n_input)

   
   # plot scores
   days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
   plt.plot(days, scores, marker = 'o')
   
   plt.show()
