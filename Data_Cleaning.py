import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# data can be found here in a .txt format:
# https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

df = pd.read_csv('household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])


#  The data was collected between December 2006 and November 2010 and contains observations of power consumption within the household were collected every minute.
#  It is a multivariate series comprised of seven variables apart from the date and time

#  Active and reactive energy variables  refer to the technical details of alternative current.
#  Units of the total power are kilowatts


# mark all missing values
df.replace('?', np.nan, inplace=True)
# make dataset numeric
df = df.astype('float32')



# We also need to fill in the missing values now that they have been marked.
# A  simple approach would be to copy the observation from the same time the day before. We can implement this in a function named fill_missing() 
# that will take the NumPy array of the data and copy values from exactly 24 hours ago.


# fill missing values with a value at the same time one day ago
def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if np.isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]
                
       
         
fill_missing(df.values)               
                

           
# The dataset provides the active power as well as some division of the active power by main circuit in the house, specifically the kitchen, laundry, and climate control.
# These are not all the circuits in the household according to the official description of the dataset.
# The remaining watt-hours can be calculated from the active energy by first converting the active energy to watt-hours,
# then subtracting the other sub-metered active energy in watt-hours, thus we created the variable sub_metering_4


values = df.values
df['sub_metering_4'] = (values[:,0] * 1000 / 60) - (values[:,4] + values[:,5] + values[:,6])



# Finally , downsample data to daily basis because we want to frame the problem as predicting total household energy consumption for the whole day
# Also add up the power for each day

df =  df.resample('D').sum()  

# We can now save the cleaned-up version of the dataset to a new file

df.to_csv('final_power_consumption.csv')

 
