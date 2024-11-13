import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
import os
import pickle
from tensorflow.keras.models import save_model

warnings.filterwarnings("ignore")


import pandas as pd

class TimeSeriesForecasting:
    import pandas as pd

class TimeSeriesForecasting:
    def __init__(self, filepath):
        """Initialize the class with data loading."""
        # Load the CSV file, skipping the first row (ticker information)
        self.data = pd.read_csv(filepath, skiprows=1, header=0)

        # Check the actual columns of the dataset
        print("Columns before renaming:", self.data.columns)

        # Assign the column names based on the actual data
        if len(self.data.columns) == 8:
            self.data.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
        elif len(self.data.columns) == 9:
            self.data.columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Normalized Close']
        else:
            raise ValueError(f"Unexpected number of columns: {len(self.data.columns)}")

        # Parse dates and set the 'Date' column as the index
        self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
        self.data.set_index('Date', inplace=True)
        self.data.dropna(inplace=True)

        # Convert relevant columns to numeric
        self.data['Close'] = pd.to_numeric(self.data['Close'], errors='coerce')
        self.data['Adj Close'] = pd.to_numeric(self.data['Adj Close'], errors='coerce')

        # Check for 'Close' or 'Adj Close'
        if 'Close' in self.data.columns:
            self.data = self.data[['Close']]
        elif 'Adj Close' in self.data.columns:
            self.data = self.data[['Adj Close']]
        else:
            raise KeyError("Neither 'Close' nor 'Adj Close' column found in the dataset.")

        # Drop any remaining NaN values
        self.data.dropna(inplace=True)

        self.train = None
        self.test = None
        self.predictions = {}

    def split_data(self, test_size=0.2):
        """Split the data into training and testing sets."""
        split_index = int(len(self.data) * (1 - test_size))
        self.train = self.data.iloc[:split_index]
        self.test = self.data.iloc[split_index:]
        print("Data split into train and test sets.")

    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics: MAE, RMSE, and MAPE, handling index misalignment and division by zero errors."""

        # Ensure that the predictions and true values have the same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len].reset_index(drop=True)

        # Convert y_pred to pandas Series if it's a NumPy array
        if isinstance(y_pred, np.ndarray):
            y_pred = pd.Series(y_pred[:min_len]).reset_index(drop=True)
        else:
            y_pred = y_pred[:min_len].reset_index(drop=True)

        # Calculate MAE and RMSE
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # Calculate MAPE, handling potential division by zero
        non_zero_mask = y_true != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.nan  # If all true values are zero, MAPE cannot be computed

        return mae, rmse, mape

    def save_model(self, model, model_name):
        """Save the trained model to the ../models directory."""
        # Ensure the models directory exists
        models_dir = "../models"
        os.makedirs(models_dir, exist_ok=True)

        # Save ARIMA and SARIMA models using pickle
        if model_name in ['ARIMA', 'SARIMA']:
            model_path = os.path.join(models_dir, f"{model_name}_model.pkl")
            with open(model_path, 'wb') as file:
                pickle.dump(model, file)
            print(f"{model_name} model saved to {model_path}")

        # Save LSTM model using Keras save function
        elif model_name == 'LSTM':
            model_path = os.path.join(models_dir, "LSTM_model.h5")
            save_model(model, model_path)
            print(f"LSTM model saved to {model_path}")

        else:
            print(f"Unknown model type: {model_name}. Model not saved.")
    
    def optimize_arima(self):
        """Optimize ARIMA model parameters using auto_arima."""
        model = auto_arima(self.train['Close'], seasonal=False, trace=True)
        return model
     
    def train_arima(self):
        """Train the ARIMA model and make predictions."""
        arima_model = self.optimize_arima()
        arima_model.fit(self.train['Close'])
        forecast = arima_model.predict(n_periods=len(self.test))
        forecast = forecast[:len(self.test)]  # Trim forecast to match the test set
        self.predictions['ARIMA'] = forecast

        # Evaluate and save the model
        mae, rmse, mape = self.evaluate(self.test['Close'], forecast)
        print(f"ARIMA - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        self.save_model(arima_model, 'ARIMA')

    def optimize_sarima(self):
        """Optimize SARIMA model parameters using auto_arima with seasonality."""
        model = auto_arima(self.train['Close'], seasonal=True, m=12, trace=True)
        return model

    def train_sarima(self):
        """Train the SARIMA model and make predictions."""
        sarima_model = self.optimize_sarima()
        sarima_model.fit(self.train['Close'])
        forecast = sarima_model.predict(n_periods=len(self.test))
        forecast = forecast[:len(self.test)]  # Trim forecast to match the test set
        self.predictions['SARIMA'] = forecast

        # Evaluate and save the model
        mae, rmse, mape = self.evaluate(self.test['Close'], forecast)
        print(f"SARIMA - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        self.save_model(sarima_model, 'SARIMA')


    

    def build_lstm(self, units=50, epochs=20, batch_size=32, activation='relu', optimizer='adam', loss='mse'):
        """Build and train the LSTM model with additional options."""
        train_scaled = self.train['Close'].values.reshape(-1, 1)
        test_scaled = self.test['Close'].values.reshape(-1, 1)

        X_train, y_train = self.create_sequences(train_scaled)
        X_test, y_test = self.create_sequences(test_scaled)

        X_train = X_train.astype(float)
        y_train = y_train.astype(float)
        X_test = X_test.astype(float)
        y_test = y_test.astype(float)

        model = Sequential()
        model.add(LSTM(units, activation=activation, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))

        if optimizer == 'adam':
            model.compile(optimizer=Adam(), loss=loss)
        else:
            model.compile(optimizer=optimizer, loss=loss)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        predictions = model.predict(X_test).flatten()
        predictions = predictions[:len(y_test)]  # Trim predictions to match the test set
        self.predictions['LSTM'] = predictions

        # Evaluate and save the model
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mape = mean_absolute_percentage_error(y_test, predictions) * 100
        print(f"LSTM - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        self.save_model(model, 'LSTM')

    def create_sequences(self, data, seq_length=10):
        """Create sequences for LSTM input."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)

    def plot_results(self):
        """Plot actual vs predicted values."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.test.index, self.test['Close'], label='Actual', color='blue')
        
        for model_name, prediction in self.predictions.items():
            plt.plot(self.test.index[-len(prediction):], prediction, label=f'{model_name} Prediction')
        
        plt.title("Actual vs Predicted Stock Prices")
        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.show()

    def forecast(self):
        """Run all models and plot results."""
        self.split_data()
        self.train_arima()
        self.train_sarima()
        self.build_lstm()
        self.plot_results()
