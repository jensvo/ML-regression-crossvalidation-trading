#This code runs a regression using a cross validation approach for a time series and uses the parameters to predict future values.
#Disclosure: Nothing in this repository should be considered investment advice. Past performance is not necessarily indicative of future returns.
# These are general examples about how to import data using pandas for a small sample of financial data across different time intervals.
#Please note that the below code is written with the objective to explore various python features.

from datetime import date, timedelta
import quandl
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

#Download data from quandl
quandl.ApiConfig.api_key = "yourquandlapikey"
data = quandl.get(["NASDAQOMX/NDX.1"], start_date="2015-12-04", end_date=date.today())

data = data[['NASDAQOMX/NDX - Index Value']]

#Define days to predict
forecast = int(30)
data['Prediction'] = data[['NASDAQOMX/NDX - Index Value']].shift(-forecast)

#Take out prediction data and scale the data
X = np.array(data.drop(['Prediction'], 1))
X = preprocessing.scale(X)

#Set x values for prediction
X_forecast = X[-forecast:]
X = X[:-forecast]

#Set y values for prediciton
y = np.array(data['Prediction'])
y = y[:-forecast]

#Split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Training of data
model = LinearRegression()
model.fit(X_train,y_train)
#Testing of data
confidence = model.score(X_test, y_test)
print(confidence)

#Calculate predicition of data
forecast_prediction = model.predict(X_forecast)


#Copy prediction data into original dataframe
data['Prediction'].iloc[:-forecast] = np.nan
for i in range(forecast):
    data['Prediction'][-i] = forecast_prediction[-i]

#Plot the result
fig, ax = plt.subplots()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
ax.xaxis.set_tick_params(labelsize=7, labelrotation = 45)
ax.plot(data['NASDAQOMX/NDX - Index Value'], label=data['NASDAQOMX/NDX - Index Value'].name)
ax.plot(data['Prediction'], label='Predictions')
ax.legend(loc='upper left', frameon=False)

plt.show()

