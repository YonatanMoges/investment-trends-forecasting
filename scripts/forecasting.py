import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import pandas as pd

class LSTMForecast:
    def __init__(self, model_path, data):
        """
        Initialize the LSTMForecast class.

        Parameters:
        - model_path (str): Path to the saved LSTM model.
        - data (pd.DataFrame): Historical stock price data.
        """
        self.model = tf.keras.models.load_model(model_path)
        
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
        Prepare the data for forecasting by normalizing the input.

        Returns:
        - last_sequence (np.array): The last sequence used for forecasting.
        """
        # Use the last 60 days of data for prediction (same as training sequence length)
        last_60_days = self.data['Close'].values[-60:]
        last_sequence = self.scaler.transform(last_60_days.reshape(-1, 1))
        last_sequence = np.expand_dims(last_sequence, axis=0)
        return last_sequence

    def forecast(self):
        """
        Generate future stock price predictions using the LSTM model.

        Returns:
        - forecast_prices (list): Predicted stock prices for the next `future_steps` days.
        """
        predictions = []
        last_sequence = self.prepare_data()

        for _ in range(self.future_steps):
            predicted_price = self.model.predict(last_sequence)
            predictions.append(predicted_price[0, 0])

            # Update the sequence with the predicted price
            new_sequence = np.append(last_sequence[0, 1:], predicted_price, axis=0)
            last_sequence = np.expand_dims(new_sequence, axis=0)

        # Inverse transform the predictions to the original scale
        forecast_prices = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        return forecast_prices

    def plot_forecast(self, forecast_prices):
        """
        Plot the forecast alongside the historical data.

        Parameters:
        - forecast_prices (list): Predicted stock prices.
        """
        # Create a date range for the forecast
        last_date = self.data.index[-1]
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, self.future_steps + 1)]

        # Plot historical and forecast data
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index, self.data['Close'], label='Historical Data', color='blue')
        plt.plot(forecast_dates, forecast_prices, label='LSTM Forecast', color='green')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('LSTM Forecast for Tesla Stock Prices')
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
