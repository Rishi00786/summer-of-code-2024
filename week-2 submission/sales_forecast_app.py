import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import pmdarima as pm

df = pd.read_csv("train 3.csv")
df = df.drop(['Row ID', 'Order ID', 'Product ID', "Customer ID"], axis=1)
df['Postal Code'] = df['Postal Code'].fillna(13579.0)
df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d/%m/%Y')
df = df.sort_values(by='Order Date', ascending=True)
df['OrderDatePeriodMonth'] = df['Order Date'].dt.to_period("M")
df_sales_per_date = df[['OrderDatePeriodMonth', 'Sales']].groupby('OrderDatePeriodMonth').sum().reset_index()
df_sales_per_date['OrderDatePeriodMonth'] = df_sales_per_date['OrderDatePeriodMonth'].dt.to_timestamp()

st.sidebar.title("Model Selection")
model_choice = st.sidebar.selectbox("Choose a model", ["SARIMAX", "Prophet", "LSTM", "Voting Regressor"])

st.title("Sales Forecasting")
st.write("Data Preview")
st.dataframe(df.head())

if st.button("Run"):
    if model_choice == "SARIMAX":
        y = df_sales_per_date.set_index('OrderDatePeriodMonth')['Sales']
        train = y[:'2018-07-01']
        test = y['2018-07-01':]

        model = SARIMAX(train, seasonal_order=(2, 2, 2, 12)).fit()
        pred = model.get_forecast(len(test.index))
        pred_df = pred.conf_int(alpha=0.05)
        pred_df["Predictions"] = model.predict(start=pred_df.index[0], end=pred_df.index[-1])
        pred_df.index = test.index
        pred_out = pred_df["Predictions"]

        rmse = np.sqrt(mean_squared_error(test, pred_out))
        mape = mean_absolute_percentage_error(test, pred_out)

        st.write(f"RMSE: {rmse}")
        st.write(f"MAPE: {mape}")

        fig, ax = plt.subplots()
        train.plot(ax=ax, label='Train data', color='black')
        test.plot(ax=ax, label='Test data', color='red')
        pred_out.plot(ax=ax, label='SARIMAX Predictions', color='green')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title("Train/Test split for Sales Data")
        plt.legend()
        st.pyplot(fig)

    elif model_choice == "Prophet":
        df_prophet = df_sales_per_date.rename(columns={'OrderDatePeriodMonth': 'ds', 'Sales': 'y'})
        model = Prophet(interval_width=0.95)
        model.fit(df_prophet)
        future_dates = model.make_future_dataframe(periods=36, freq='MS')
        forecast = model.predict(future_dates)
        rmse = np.sqrt(mean_squared_error(df_prophet['y'], forecast['yhat'][:len(df_prophet)]))
        mape = mean_absolute_percentage_error(df_prophet['y'], forecast['yhat'][:len(df_prophet)])
        
        st.write(f"RMSE: {rmse}")
        st.write(f"MAPE: {mape}")

        fig1 = model.plot(forecast)
        st.pyplot(fig1)

    elif model_choice == "LSTM":
        df_resampled = df.set_index('Order Date').resample('ME').agg({'Sales': 'sum'}).reset_index()
        train, test = df_resampled[:int(len(df_resampled) * 0.8)], df_resampled[int(len(df_resampled) * 0.8):]

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train[['Sales']])
        test_scaled = scaler.transform(test[['Sales']])

        def create_sequences(data, seq_length):
            X = []
            y = []
            for i in range(len(data) - seq_length):
                X.append(data[i:i + seq_length])
                y.append(data[i + seq_length])
            return np.array(X), np.array(y)

        seq_length = 12  
        X_train, y_train = create_sequences(train_scaled, seq_length)
        X_test, y_test = create_sequences(test_scaled, seq_length)

        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, y_train, batch_size=1, epochs=100)

        test_preds = model.predict(X_test)
        test_preds = scaler.inverse_transform(test_preds)

        mape = mean_absolute_percentage_error(test[seq_length:]['Sales'], test_preds)
        rmse = np.sqrt(mean_squared_error(test[seq_length:]['Sales'], test_preds))

        st.write(f"RMSE: {rmse}")
        st.write(f"MAPE: {mape}")

        plt.figure(figsize=(14, 7))
        plt.plot(test['Order Date'][seq_length:], test['Sales'][seq_length:], label='Actual Sales', marker='o')
        plt.plot(test['Order Date'][seq_length:], test_preds, label='Predicted Sales', marker='x')
        plt.title('Actual vs Predicted Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        st.pyplot()

    elif model_choice == "Voting Regressor":
        df_resampled = df.set_index('Order Date').resample('M').agg({'Sales': 'sum'}).reset_index()
        train, test = df_resampled[:int(len(df_resampled) * 0.8)], df_resampled[int(len(df_resampled) * 0.8):]

        def prepare_data(df):
            df = df.copy()
            df['Month'] = df['Order Date'].dt.month
            df['Year'] = df['Order Date'].dt.year
            return df

        train_prepared = prepare_data(train)
        test_prepared = prepare_data(test)

        train_prepared = train_prepared.fillna(0)
        test_prepared = test_prepared.fillna(0)

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

        voting_regressor = VotingRegressor(estimators=[
            ('arima', arima_regressor),
            ('linear', linear_model),
            ('gb', gb_model)
        ])

        voting_regressor.fit(train_prepared[['Month', 'Year']], train_prepared['Sales'])
        voting_preds = voting_regressor.predict(test_prepared[['Month', 'Year']])

        mape = mean_absolute_percentage_error(test_prepared['Sales'], voting_preds)
        rmse = np.sqrt(mean_squared_error(test_prepared['Sales'], voting_preds))

        st.write(f"RMSE: {rmse}")
        st.write(f"MAPE: {mape}")

        plt.figure(figsize=(14, 7))
        plt.plot(test['Order Date'], test['Sales'], label='Actual Sales', marker='o')
        plt.plot(test['Order Date'], voting_preds, label='Predicted Sales', marker='x')
        plt.title('Actual vs Predicted Sales')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.grid(True)
        st.pyplot()