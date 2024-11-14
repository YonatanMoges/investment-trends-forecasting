import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class PortfolioOptimizer:
    def __init__(self, returns):
        """
        Initialize PortfolioOptimizer with daily returns of assets.
        """
        self.returns = returns
        self.annualized_returns = self.returns.mean() * 252  # Annualized average returns
        self.cov_matrix = self.returns.cov() * 252  # Annualized covariance matrix
    
    def calculate_portfolio_return(self, weights):
        """
        Calculate the expected portfolio return based on weights.
        """
        return np.dot(weights, self.annualized_returns)
    
    def calculate_portfolio_volatility(self, weights):
        """
        Calculate the portfolio's standard deviation (volatility) based on weights.
        """
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
    
    def calculate_sharpe_ratio(self, weights):
        """
        Calculate the Sharpe Ratio of the portfolio.
        """
        portfolio_return = self.calculate_portfolio_return(weights)
        portfolio_volatility = self.calculate_portfolio_volatility(weights)
        return portfolio_return / portfolio_volatility
    
    def portfolio_statistics(self, weights):
        """
        Wrapper function to return portfolio return, volatility, and Sharpe Ratio.
        """
        portfolio_return = self.calculate_portfolio_return(weights)
        portfolio_volatility = self.calculate_portfolio_volatility(weights)
        sharpe_ratio = self.calculate_sharpe_ratio(weights)
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def optimize_portfolio(self):
        """
        Optimize the portfolio weights to maximize the Sharpe Ratio.
        """
        num_assets = len(self.returns.columns)
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights must equal 1
        bounds = tuple((0, 1) for _ in range(num_assets))  # Weights between 0 and 1

        # Minimize the negative Sharpe Ratio
        result = minimize(
            lambda weights: -self.calculate_sharpe_ratio(weights),
            num_assets * [1. / num_assets,],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x
    
    def value_at_risk(self, asset='TSLA', confidence_level=0.05):
        """
        Calculate Value at Risk (VaR) for a given asset at the specified confidence level.
        """
        if asset not in self.returns.columns:
            raise ValueError(f"{asset} not found in returns data.")
        
        mean_return = self.returns[asset].mean()
        std_dev = self.returns[asset].std()
        var = mean_return - std_dev * np.percentile(self.returns[asset], confidence_level * 100)
        return var
    
    def calculate_cumulative_returns(self, weights):
        """
        Calculate cumulative returns for each asset and the optimized portfolio.
        """
        cumulative_returns = (1 + self.returns).cumprod()
        portfolio_cumulative_returns = (1 + self.returns.dot(weights)).cumprod()
        return cumulative_returns, portfolio_cumulative_returns
    
    def plot_cumulative_returns(self, cumulative_returns, portfolio_cumulative_returns):
        """
        Plot cumulative returns for each asset and the optimized portfolio.
        """
        plt.figure(figsize=(14, 7))
        for column in cumulative_returns.columns:
            plt.plot(cumulative_returns.index, cumulative_returns[column], label=f'{column} Cumulative Return')
        plt.plot(portfolio_cumulative_returns.index, portfolio_cumulative_returns, label='Optimized Portfolio', linewidth=2, color='black')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.title('Cumulative Return Comparison')
        plt.legend()
        plt.show()
    
    def summarize_results(self, weights):
        """
        Display the portfolio's optimized metrics and risk-return analysis.
        """
        portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_statistics(weights)
        print("Portfolio Summary:")
        print(f"  Expected Annual Return: {portfolio_return:.2f}")
        print(f"  Expected Portfolio Volatility: {portfolio_volatility:.2f}")
        print(f"  Optimized Sharpe Ratio: {sharpe_ratio:.2f}")
        
        print("\nPortfolio Weights Adjustment Rationale:")
        print("  - Allocate more to stable assets like BND if high volatility expected in TSLA.")
        print("  - Diversify with SPY to balance growth and risk.")
