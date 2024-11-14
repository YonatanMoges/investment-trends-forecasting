# task_2_model_training.py

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class ModelTrainer:
    def __init__(self, data):
        self.data = data
        self.train_size = int(len(data) * 0.8)
        self.train = data[:self.train_size]
        self.test = data[self.train_size:]
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def train_arima(self, order=(5, 1, 0)):
        arima_model = ARIMA(self.train, order=order).fit()
        return arima_model

    def train_sarima(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        sarima_model = SARIMAX(self.train, order=order, seasonal_order=seasonal_order).fit()
        return sarima_model

    def prepare_lstm_data(self, sequence_length=60):
        scaled_data = self.scaler.fit_transform(np.array(self.train).reshape(-1, 1))
        X_train, y_train = [], []
        for i in range(sequence_length, len(scaled_data)):
            X_train.append(scaled_data[i-sequence_length:i, 0])
            y_train.append(scaled_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        return X_train, y_train

    def train_lstm(self, X_train, y_train, units=50, epochs=10, batch_size=32):
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        lstm_model.add(LSTM(units=units))
        lstm_model.add(Dense(1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return lstm_model

    def forecast_lstm(self, model, sequence_length=60):
        inputs = self.scaler.transform(np.array(self.data[-sequence_length:]).reshape(-1, 1))
        X_test = np.reshape(inputs, (1, sequence_length, 1))
        predictions = []
        for _ in range(len(self.test)):
            pred = model.predict(X_test)[0][0]
            predictions.append(pred)
            # Add an extra dimension to `pred` to match `X_test`
            X_test = np.append(X_test[:, 1:, :], [[[pred]]], axis=1)
        return self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    def evaluate_model(self, model, model_type='arima', metric='mae'):
        if model_type == 'lstm':
            predictions = self.forecast_lstm(model)
        else:
            predictions = model.forecast(steps=len(self.test))
        
        if metric == 'mae':
            return mean_absolute_error(self.test, predictions)
        elif metric == 'rmse':
            return np.sqrt(mean_squared_error(self.test, predictions))
        elif metric == 'mape':
            return np.mean(np.abs((self.test - predictions) / self.test)) * 100

    def plot_forecasts(self, arima_preds, sarima_preds, lstm_preds):
        plt.figure(figsize=(12, 6))
        plt.plot(self.test.index, self.test, label='Actual')
        plt.plot(self.test.index, arima_preds, label='ARIMA')
        plt.plot(self.test.index, sarima_preds, label='SARIMA')
        plt.plot(self.test.index, lstm_preds, label='LSTM')
        plt.title("Model Forecast Comparison")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()

