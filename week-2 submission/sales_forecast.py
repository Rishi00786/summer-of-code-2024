# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NZt0k02wQR_qFQe6Wzbfi4mPJEab8EZF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

df = pd.read_csv("train 3.csv")

df.head()

df = df.drop(['Row ID'	, 'Order ID', 'Product ID' , "Customer ID"] , axis = 1)

df['Postal Code'] = df['Postal Code'].fillna(13579.0)

df.isnull().sum()

df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')

df['Ship Date'] = pd.to_datetime(df['Ship Date'] , format = '%d/%m/%Y')
df.head()

df=df.sort_values(by='Order Date', ascending=True)

df_city_sales=df[["City","Sales"]]
print(df_city_sales)

df_city_sales.groupby(["City"]).sum()

df_sales_per_segment=df[["Segment","Sales"]].groupby(["Segment"]).sum()

df_sales_per_segment.plot.pie(y='Sales', figsize=(7, 7),title="Total of Sales per Segment",autopct='%1.1f%%', shadow=True,explode=(0,0.1, 0))

df['OrderDatePeriodDay'] = df['Order Date'].dt.to_period("D")
df['OrderDatePeriodMonth'] = df['Order Date'].dt.to_period("M")
df['OrderDatePeriodYear'] = df['Order Date'].dt.to_period("Y")
df['OrderDatePeriodWeek'] = df['Order Date'].dt.to_period("W")

df['OrderDatePeriodWeek'].value_counts()

df

df_sales_per_date=df[['OrderDatePeriodYear','Sales']].groupby('OrderDatePeriodYear').sum()

df_sales_per_date=df[['OrderDatePeriodMonth','Sales']].groupby('OrderDatePeriodMonth').sum()

df_sales_per_date=df[['OrderDatePeriodDay','Sales']].groupby('OrderDatePeriodDay').sum()

df_sales_per_date.info()

df_sales_per_date.index

df_sales_per_date.plot(kind='line',y='Sales',xlabel='Order Date',title="Sales",figsize=(16,5),color = "black")

df_sales_per_date

df_train = df_sales_per_date[df_sales_per_date.index <= pd.to_datetime("2018-07-01", format='%Y-%m-%d').to_period("M")]
df_test = df_sales_per_date[df_sales_per_date.index >= pd.to_datetime("2018-07-01", format='%Y-%m-%d').to_period("M")]

print(df_train.shape)
print(df_test.shape)

df_train.index = df_train.index.to_timestamp()
df_test.index = df_test.index.to_timestamp()

import matplotlib.pyplot as plt
plt.plot(df_train, color = "black")
plt.plot(df_test, color = "red")
plt.ylabel('Sales')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Sales Data")
plt.rcParams["figure.figsize"] = (16,5)
plt.show()

from statsmodels.tsa.statespace.sarimax import SARIMAX

y=df_train['Sales']

SARIMAXmodel = SARIMAX(y, seasonal_order=(2,2,2,12))

SARIMAXmodel = SARIMAXmodel.fit()

y_pred = SARIMAXmodel.get_forecast(len(df_test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05)
y_pred_df["Predictions"] = SARIMAXmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = df_test.index
y_pred_out = y_pred_df["Predictions"]

import matplotlib.pyplot as plt
plt.plot(df_train, color = 'black', label = 'Train data')
plt.plot(df_test, color = 'red',label='Test data')
plt.plot(y_pred_out, color='green', label = 'SARIMAX Predictions')
plt.ylabel('Sales')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Sales Data")
plt.rcParams["figure.figsize"] = (16,5)
plt.legend()
plt.show()

# !pip install fbprophet

from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.offline as py
py.init_notebook_mode()

plt.style.use('fivethirtyeight')

my_model = Prophet(interval_width=0.95)

# Select only the 'Order Date' and 'Sales' columns
df_prophet = df[['OrderDatePeriodMonth','Sales']].groupby('OrderDatePeriodMonth').sum().reset_index()

# df_prophet['OrderDatePeriodMonth'] = pd.to_datetime(df_prophet['OrderDatePeriodMonth'])

# Rename the columns to 'ds' and 'y'
df_prophet.rename(columns={'OrderDatePeriodMonth': 'ds', 'Sales': 'y'}, inplace=True)

# Convert the 'ds' column to datetime format

df_prophet['ds'] = df_prophet['ds'].dt.to_timestamp()

df_prophet.head()

ax = df_prophet.set_index('ds').plot(figsize=(12, 8))
ax.set_ylabel('Monthly Sales')
ax.set_xlabel('Date')

plt.show()

my_model.fit(df_prophet)

future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
future_dates

forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

my_model.plot(forecast, uncertainty=True)

my_model.plot_components(forecast)

fig1 = my_model.plot_components(forecast)

from prophet.plot import add_changepoints_to_plot
fig = my_model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), my_model, forecast)

my_model.changepoints

pro_change= Prophet(changepoint_range=0.9)
forecast = pro_change.fit(df_prophet).predict(future_dates)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)

pro_change= Prophet(n_changepoints=20, yearly_seasonality=True)
forecast = pro_change.fit(df_prophet).predict(future_dates)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)

pro_change= Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.08)
forecast = pro_change.fit(df_prophet).predict(future_dates)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)

pro_change= Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=0.001)
forecast = pro_change.fit(df_prophet).predict(future_dates)
fig= pro_change.plot(forecast);
a = add_changepoints_to_plot(fig.gca(), pro_change, forecast)

# !pip install tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

df_prophet.head()

df_lstm = df_prophet.drop('ds' , axis = 1)
df_lstm.head()

df_lstm.info()

# Create a time series plot.
plt.figure(figsize = (15, 5))
plt.plot(df_lstm, label = "Monthly Sales")
plt.xlabel("Months")
plt.ylabel("Sales")
plt.title("Monthly Sales")
plt.legend()
plt.show()

data_raw = df_lstm.values.astype("float32")

data_raw

scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(data_raw)

print(dataset.shape)
print(dataset[0:5])

# Using 60% of data for training, 40% for validation.
TRAIN_SIZE = 0.70

train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Number of entries (training set, test set): " + str((len(train), len(test))))

# FIXME: This helper function should be rewritten using numpy's shift function. See below.
def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))

window_size = 1
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)


# Reshape the input data into appropriate form for Keras.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("New training data shape:")
print(train_X.shape)

train_Y.shape

def fit_model_original(train_X, train_Y, window_size = 1):
    model = Sequential()
    model.add(LSTM(4,
                   input_shape = (1, window_size)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error",
                  optimizer = "adam")
    model.fit(train_X,
              train_Y,
              epochs = 100,
              batch_size = 10,
              verbose = 2)
    return(model)

# Define the model.
def fit_model_new(train_X, train_Y, window_size = 1):
    model2 = Sequential()
    model2.add(LSTM(input_shape = (window_size, 1),
               units = window_size,
               return_sequences = True))
    model2.add(Dropout(0.5))
    model2.add(LSTM(256))
    model2.add(Dropout(0.5))
    model2.add(Dense(1))
    model2.add(Activation("linear"))
    model2.compile(loss = "mse",
              optimizer = "adam")
    model2.summary()

    # Fit the first model.
    model2.fit(train_X, train_Y, epochs = 100,
              batch_size = 10,
              verbose = 2)
    return(model2)
#model1=fit_model_original(train_X, train_Y)

model2=fit_model_new(train_X, train_Y)

def predict_and_score(model, X, Y):
    # Make predictions on the original scale of the data.
    pred_scaled =model.predict(X)
    pred = scaler.inverse_transform(pred_scaled)
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred, pred_scaled)

rmse_train, train_predict, train_predict_scaled = predict_and_score(model2, train_X, train_Y)
rmse_test, test_predict, test_predict_scaled = predict_and_score(model2, test_X, test_Y)

print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)

test_predict.size

#print(test_X.shape)
#print(test_X[0:1,:,:].shape)
#print(test_X[0:1,:,:])
X_single = test_X[0:1,:,:]
#print(test_Y.shape)
#print(test_Y[0:1].shape)
#print(test_Y[0:1])

# create empty array
from numpy import empty
test_predict_at_a_time = empty([test_X.size,1])

print("initial X: ", X_single)
for i in range((test_X.size)):
    Y_single = test_Y[i:i+1]
    rmse_test, predict, predict_scaled = predict_and_score(model2, X_single, Y_single)
    test_predict_at_a_time[i]= predict
    print("Test data score: %.2f RMSE" % rmse_test)
    print("predicted: ", predict[0])
    X_single = predict_scaled.copy()
    X_single=np.reshape(X_single[0], (1, 1, 1))
    print("given X: ", X_single)
test_predict_at_a_time[-3:]

# Start with training predictions.
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

# Add test predictions.
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict


# Add test predictions.
test_predict_at_a_time_plot = np.empty_like(dataset)
test_predict_at_a_time_plot[:, :] = np.nan
test_predict_at_a_time_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict_at_a_time

# Create the plot.
plt.figure(figsize = (15, 5))
plt.plot(scaler.inverse_transform(dataset), label = "True value")
plt.plot(train_predict_plot, label = "Training set prediction")
plt.plot(test_predict_plot, label = "Test set prediction")
plt.plot(test_predict_at_a_time_plot, label = "Test set prediction at a time")
plt.xlabel("Months")
plt.ylabel("Monthly Sales")
plt.title("Comparison true vs. predicted training / test")
plt.legend()
plt.show()

# !pip install pmdarima

import pandas as pd
import pmdarima as pm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.base import BaseEstimator, RegressorMixin

# Assuming the dataset is saved as 'orders.csv'
my_df = df.copy()

# Convert 'Order Date' to datetime format
my_df['Order Date'] = pd.to_datetime(my_df['Order Date'])

# Set 'Order Date' as the index
my_df.set_index('Order Date', inplace=True)

# Resample data to monthly sales
df_resampled = my_df.resample('M').agg({'Sales': 'sum'}).copy()

my_df.info()

df_resampled.head()

train, test = train_test_split(df_resampled, train_size=0.8)

print(train.head())
print(test.head())

def prepare_data(df):
    df = df.copy()
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    return df

train_prepared = prepare_data(train)
test_prepared = prepare_data(test)

print(train_prepared.head())
print(test_prepared.head())

class ARIMARegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        self.model = pm.auto_arima(
            y,
            seasonal=True,
            m=12,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        return self

    def predict(self, X):
        n_periods = len(X)
        return self.model.predict(n_periods=n_periods)

arima_regressor = ARIMARegressor()
linear_model = LinearRegression()
gb_model = GradientBoostingRegressor()

# Combine models using VotingRegressor
voting_regressor = VotingRegressor(estimators=[
    ('arima', arima_regressor),
    ('linear', linear_model),
    ('gb', gb_model)
])

voting_regressor.fit(train_prepared.drop('Sales', axis=1), train_prepared['Sales'])

voting_preds = voting_regressor.predict(test_prepared.drop('Sales', axis=1))

mape = mean_absolute_percentage_error(test_prepared['Sales'], voting_preds)
print(f'MAPE for Voting Regressor: {mape:.2%}')

import matplotlib.pyplot as plt

# Plot the actual vs predicted sales
plt.figure(figsize=(14, 7))
plt.plot(test.index, test['Sales'], label='Actual Sales', marker='o')
plt.plot(test.index, voting_preds, label='Predicted Sales', marker='x')

plt.title('Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# !pip install optuna

import optuna
from sklearn.model_selection import train_test_split

def objective(trial):
    # Suggest hyperparameters for Gradient Boosting Regressor
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 3, 10)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)

    # Define models with suggested hyperparameters
    linear_model = LinearRegression()
    gb_model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate
    )

    # Combine models using VotingRegressor
    voting_regressor = VotingRegressor(estimators=[
        ('linear', linear_model),
        ('gb', gb_model)
    ])

    # Split the training data for evaluation
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_prepared.drop('Sales', axis=1), train_prepared['Sales'], test_size=0.2, random_state=42)

    # Fit the VotingRegressor
    voting_regressor.fit(X_train, y_train)

    # Generate predictions and calculate MAPE
    preds = voting_regressor.predict(X_valid)
    mape = mean_absolute_percentage_error(y_valid, preds)

    return mape

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# Print the best hyperparameters
print('Best hyperparameters: ', study.best_params)
print('Best MAPE: ', study.best_value)

best_params = study.best_params
final_gb_model = GradientBoostingRegressor(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    learning_rate=best_params['learning_rate']
)
final_voting_regressor = VotingRegressor(estimators=[
    ('linear', LinearRegression()),
    ('gb', final_gb_model)
])

final_voting_regressor.fit(train_prepared.drop('Sales', axis=1), train_prepared['Sales'])

final_preds = final_voting_regressor.predict(test_prepared.drop('Sales', axis=1))

final_mape = mean_absolute_percentage_error(test_prepared['Sales'], final_preds)
print(f'Final MAPE after tuning: {final_mape:.2%}')

# Plot the actual vs predicted sales
plt.figure(figsize=(14, 7))
plt.plot(test.index, test['Sales'], label='Actual Sales', marker='o')
plt.plot(test.index, final_preds, label='Predicted Sales', marker='x')

plt.title('Actual vs Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.show()

# !pip install streamlit