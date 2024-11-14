# scripts/forecasting.py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt

class LSTMForecast:
    def __init__(self, model_path, data, asset_name="Asset"):
        """
        Initialize the LSTMForecast class.
        
        Parameters:
        - model_path (str): Path to the saved LSTM model.
        - data (pd.DataFrame): Historical stock price data.
        - asset_name (str): Name of the asset for saving forecasted data.
        """
        self.model = tf.keras.models.load_model(model_path)
        self.asset_name = asset_name
        
        # Ensure 'Close' column contains numeric values only
        if data['Close'].dtype != np.float64 and data['Close'].dtype != np.float32:
            data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        
        # Drop rows with any missing values in the 'Close' column after conversion
        data = data.dropna(subset=['Close'])
        
        # Initialize scaler and fit-transform the 'Close' prices
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = data.copy()
        self.data['Close'] = self.scaler.fit_transform(self.data[['Close']])
        
        self.future_steps = 180  # Forecast for the next 6 months (approx. 180 days)

    def prepare_data(self):
        """
        Prepare the data for forecasting by taking the last 10 timesteps.
        """
        last_sequence = self.data['Close'].values[-10:]  
        last_sequence = last_sequence.reshape((1, 10, 1))  # Reshape to (1, 10, 1)
        return last_sequence

    def forecast(self):
        """
        Generate future forecasts based on the trained model.
        """
        predictions = []
        last_sequence = self.prepare_data()  # Get the last 10-timestep sequence

        for _ in range(self.future_steps):
            predicted_price = self.model.predict(last_sequence)
            predictions.append(predicted_price[0, 0])

            # Update last_sequence to include predicted_price
            predicted_price_reshaped = predicted_price.reshape((1, 1, 1))  # Reshape to (1, 1, 1)
            last_sequence = np.append(last_sequence[:, 1:, :], predicted_price_reshaped, axis=1)

        # Inverse transform predictions to original scale
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions
    

    def save_forecast(self, forecast_prices, output_path='../data/forecasted_prices.csv'):
        # Ensure the date index is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(self.data.index):
            self.data.index = pd.to_datetime(self.data.index)

        # Get the last date in the historical data
        last_date = self.data.index[-1]

        # Generate future dates
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, self.future_steps + 1)]

        # Create DataFrame for forecasted data
        forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Close': forecast_prices.flatten()})

        # Save to the specified output file
        forecast_df.to_csv(output_path, index=False)
        print(f"Forecast saved to {output_path}")

    def plot_forecast(self, forecast_prices):
        """
        Plot historical and forecasted prices.
        """
        # Ensure that the data index is a DatetimeIndex
        if not isinstance(self.data.index, pd.DatetimeIndex):
            self.data.index = pd.to_datetime(self.data.index)

        # Get the last date in the historical data
        last_date = self.data.index[-1]

        # Create a date range for the forecast
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, self.future_steps + 1)]

        # Plot historical and forecast data
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Close'], label="Historical Prices")
        plt.plot(forecast_dates, forecast_prices, label="Forecasted Prices", linestyle='--')
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.title(f"Historical and Forecasted Prices for {self.asset_name}")
        plt.legend()
        plt.show()
        
    def analyze_forecast(self, forecast_prices):
        """
        Analyze the forecast to identify trends and risks.

        Parameters:
        - forecast_prices (list): Predicted stock prices.
        """
        # Calculate basic statistics
        mean_price = np.mean(forecast_prices)
        volatility = np.std(forecast_prices)

        # Trend analysis
        if forecast_prices[-1] > forecast_prices[0]:
            trend = "Upward Trend"
        elif forecast_prices[-1] < forecast_prices[0]:
            trend = "Downward Trend"
        else:
            trend = "Stable Trend"

        # Display analysis
        print("Forecast Analysis:")
        print(f"Mean Forecast Price: {mean_price:.2f}")
        print(f"Volatility (Standard Deviation): {volatility:.2f}")
        print(f"Identified Trend: {trend}")

        # Highlight market opportunities and risks
        if volatility > 0.1 * mean_price:
            print("High volatility detected. Caution advised for investors.")
        if trend == "Upward Trend":
            print("Potential market opportunity: Expected price increase.")
        elif trend == "Downward Trend":
            print("Potential market risk: Expected price decline.")
