import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Bidirectional
from LSTM_Univariate_Input import split_dataset, evaluate_forecasts

os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONHASHSEED']=str(33)
import tensorflow as tf
tf.compat.v1.set_random_seed(33)

from keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)



df = pd.read_csv('final_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])



# convert history into inputs and outputs
def to_supervised_2(train, n_input, n_out=7):
    
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
			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, 0])
		# move along one time step
		in_start += 1
	return np.array(X), np.array(y)



def evaluate_model_2(train, test, n_input):
    
	model = build_model_2(train, n_input)
    
	# history is a list of weekly data
	history = [x for x in train]
    
	# walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
        
		# predict the week
		yhat_sequence = forecast_2(model, history, n_input)
		predictions.append(yhat_sequence)
        
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
        
        
	# evaluate predictions days for each week
	predictions = np.array(predictions)
	score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    
	return score, scores



def build_model_2(train, n_input):
    
	# prepare data
	train_x, train_y = to_supervised_2(train, n_input)
    
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




def forecast_2(model, history, n_input):
    
	# flatten data
	data = np.array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
    
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
    
	# reshape into [1, n_input, 1]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    
	# forecast the next week   
	yhat = model.predict(input_x, verbose=0)
    
	# we only want the vector forecast
	yhat = yhat[0]
    
	return yhat




if  __name__ == '__main__':
    
      
   n_input = 7

   # split into train and test
   train_2, test_2 = split_dataset(df.values)

   # evaluate model
   score_2, scores_2 = evaluate_model_2(train_2, test_2, n_input)
   
  
   # plot scores
   days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
   plt.plot(days, scores_2, marker = 'o')

   plt.show()



