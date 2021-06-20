import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv('final_power_consumption.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])

print(df.info())



### explore spearman correlation between variables 

corrMatrix = df.corr(method ='spearman')
print (corrMatrix)

sns.heatmap(corrMatrix, annot=True)
plt.show()




### Global_active_power and Global_intensity appear to have a correlation of 1

df[['Global_active_power','Global_intensity']].plot(figsize=(12,8))

# hence we discard the Global_intensity variable

df.drop('Global_intensity',axis=1, inplace=True)





### Line plots for all columns of the dataframe

plt.figure()

for i in range(len(df.columns)):
    
	plt.subplot(len(df.columns), 1, i+1)
    
	name = df.columns[i]
    
	plt.plot(df[name])
	plt.title(name, y=0)
    
    
plt.show()




# We can create a new plot of the active power for each year to see if there are any common patterns across the years.
# The first year, 2006, has less than one month of data, so will remove it from the plot.


years = ['2007', '2008', '2009', '2010']

plt.figure()

for i in range(len(years)):
    
	# prepare subplot
	ax = plt.subplot(len(years), 1, i+1)
    
	# determine the year to plot
	year = years[i]
    
	# get all observations for the year
	result = df[str(year)]
    
	# plot the active power for the year
	plt.plot(result['Global_active_power'])
    
	# add a title to the subplot
	plt.title(str(year), y=0, loc='left')
    
    
plt.show()


# We can see some common gross patterns across the years, such as around Feb-Mar and around Aug-Sept where we see a marked decrease in consumption
# We also seem to see a downward trend over the summer months (middle of the year in the northern hemisphere) and perhaps more consumption in the winter months towards the edges of the plots
# These may show an annual seasonal pattern in consumption.





### ETS decomposition

### perform seasonal decomposition to explore the trend and seasonality components of the Global_active_power time series
from statsmodels.tsa.seasonal import seasonal_decompose


results = seasonal_decompose(df['Global_active_power'])
results.plot();                                             

results.observed.plot(figsize=(12,2)) ;      

results.trend.plot(figsize=(12,2)) ;
# there is a clear downward trend in March and August-September of each year as well as an upward trend during winter time

results.seasonal.plot(figsize=(12,2));
# there is also a clear component of seasonality but its magnitude is not so big (-100 to 200)





### Stationarity test for general power consumption

# we use the augmented Dickey-Fuller test in the below function to unedrstanf if a time series is stationary

from statsmodels.tsa.stattools import adfuller

def adf_test(series,title=''):

    
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC')              # .dropna() handles NAs in case of differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val    
    print(out.to_string())          
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")





adf_test(df['Global_active_power'])            ### Global_active_power is stationary
adf_test(df['Global_reactive_power'])          ### Global_reactive_power is stationary
adf_test(df['Voltage'])                        ### Voltage is stationary
adf_test(df['Sub_metering_1'])                 ### stationary 
adf_test(df['Sub_metering_2'])                 ### stationary
adf_test(df['Sub_metering_3'])                 ### stationary
adf_test(df['sub_metering_4'])                 ### stationary


# All in all, our features are stationary




### Time Series Distributions


# We can investigate the distributions of the data by reviewing histograms - we create a histogram for each variable in the time series.

plt.figure()

for i in range(len(df.columns)):
    
	plt.subplot(len(df.columns), 1, i+1)
    
	name = df.columns[i]
	df[name].hist(bins=100)
	plt.title(name, y=0)
    
plt.show()

## Voltage distribution seems to be Gaussian while distributions of Sub_metering 1&2 present a clear skewness
## Global_active_power and Global_reactive_power and sub_metering_4 are also skewed



# We can investigate the distribution of active power further by looking at the distribution of active power consumption for the four full years of data

years = ['2007', '2008', '2009', '2010']

plt.figure()

for i in range(len(years)):

	ax = plt.subplot(len(years), 1, i+1)
    
	# determine the year to plot
	year = years[i]

	result = df[str(year)]
	result['Global_active_power'].hist(bins=100)
    
	# zoom in on the distribution
	ax.set_xlim(1000, 5000)

	plt.title(str(year), y=0, loc='right')
    
    
plt.show()




# we can also investigate this by looking at the distribution for active power for each month in a year(i.e.2007)

months = [x for x in range(1, 13)]

plt.figure()

for i in range(len(months)):

	ax = plt.subplot(len(months), 1, i+1)
    
	# determine the month to plot
	month = '2007-' + str(months[i])

	result = df[month]
	result['Global_active_power'].hist(bins=100)
    
	# zoom in on the distribution
	ax.set_xlim(1000, 5000)

	plt.title(month, y=0, loc='right')
    
    
plt.show()

# we can see that the distribution changes significantly depending on the month - from the 6th to the 9th month most values fall in the range of 1000-2000 kilowatts


