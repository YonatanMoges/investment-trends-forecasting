import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")


import pandas as pd

class TimeSeriesForecasting:
    import pandas as pd

class TimeSeriesForecasting:
    def __init__(self, filepath):
        # Load the CSV file without renaming columns immediately
        self.data = pd.read_csv(filepath, header=0)

        # Initialize predictions as an empty dictionary
        self.predictions = {}
        
        # Display the columns to understand the current structure
        print("Columns in dataset:", self.data.columns)

        # If necessary, adjust the columns based on their count
        expected_columns = ['Date', 'Price', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Normalized Close']
        if len(self.data.columns) == 8:
            # Remove 'Normalized Close' if not present
            expected_columns = expected_columns[:-1]
        
        # Apply the column names if matched
        if len(self.data.columns) == len(expected_columns):
            self.data.columns = expected_columns
        else:
            print(f"Unexpected column structure: {self.data.columns}")

        # Parse dates and set the 'Date' column as the index
        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            self.data.set_index('Date', inplace=True)
            self.data.dropna(inplace=True)
        else:
            raise KeyError("'Date' column is missing from the dataset.")

        # Select the target column
        if 'Close' in self.data.columns:
            self.data = self.data[['Close']]
        elif 'Adj Close' in self.data.columns:
            self.data = self.data[['Adj Close']]
        else:
            raise KeyError("Neither 'Close' nor 'Adj Close' column found in the dataset.")



    def split_data(self, test_size=0.2):
        """Split the data into training and testing sets."""
        split_index = int(len(self.data) * (1 - test_size))
        self.train = self.data.iloc[:split_index]
        self.test = self.data.iloc[split_index:]
        print("Data split into train and test sets.")

    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mae, rmse, mape

    def optimize_arima(self):
        """Optimize ARIMA model parameters using auto_arima."""
        model = auto_arima(self.train['Close'], seasonal=False, trace=True)
        return model

    def train_arima(self):
        """Train the ARIMA model and make predictions."""
        arima_model = self.optimize_arima()
        arima_model.fit(self.train['Close'])
        forecast = arima_model.predict(n_periods=len(self.test))
        self.predictions['ARIMA'] = forecast
        mae, rmse, mape = self.evaluate(self.test['Close'], forecast)
        print(f"ARIMA - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    def optimize_sarima(self):
        """Optimize SARIMA model parameters using auto_arima with seasonality."""
        model = auto_arima(self.train['Close'], seasonal=True, m=12, trace=True)
        return model

    def train_sarima(self):
        """Train the SARIMA model and make predictions."""
        sarima_model = self.optimize_sarima()
        sarima_model.fit(self.train['Close'])
        forecast = sarima_model.predict(n_periods=len(self.test))
        self.predictions['SARIMA'] = forecast
        mae, rmse, mape = self.evaluate(self.test['Close'], forecast)
        print(f"SARIMA - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

    def build_lstm(self, units=50, epochs=20, batch_size=32):
        """Build and train the LSTM model."""
        train_scaled = self.train['Close'].values.reshape(-1, 1)
        test_scaled = self.test['Close'].values.reshape(-1, 1)

        # Prepare data for LSTM
        X_train, y_train = self.create_sequences(train_scaled)
        X_test, y_test = self.create_sequences(test_scaled)

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer=Adam(), loss='mse')

        # Train the model
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

        # Make predictions
        predictions = model.predict(X_test).flatten()
        self.predictions['LSTM'] = predictions
        mae, rmse, mape = self.evaluate(y_test, predictions)
        print(f"LSTM - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

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
