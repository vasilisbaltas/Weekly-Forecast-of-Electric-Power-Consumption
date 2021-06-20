In this repo, electricity power usage is analysed and forecasted by using the dataset given in this link:
https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption

This data represents a multivariate time series of power-related variables that in turn could be used to model 
and forecast future electricity consumption.

The scope of this work was to practise and get hands-on with data modeling for LSTM neural networks as well as 
leverage the traditional SARIMAX algorithm.Two bidirectional LSTM models were implemented, the first one with univariate
input and the second one with multiple variables as predictors. Both of them have a quite simple architecture(tuning was 
not the point) and both of them were significantly outperformed by the SARIMAX model.

Finally, the problem was formulated as predicting tha daily power consumption of a household for each day of the week.
