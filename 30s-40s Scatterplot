!pip install yfinance pandas numpy matplotlib scipy

# Define a list of stock tickers
tickers = ['GOOGL', 'BAC', 'MSFT', 'HPK', 'AMN','ACLS','WIRE', 'CPRX','SD','EGLE']  # Example tickers # 50% large 30% mid 20% small

# Define the start and end dates for the data
start_date = '2015-01-01'
end_date = '2020-01-01'

# Load data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
log_returns = np.log(data / data.shift(1))

# Calculate mean returns and covariance matrix
mean_returns = log_returns.mean()
cov_matrix = log_returns.cov()

# Number of assets in the portfolio
num_assets = len(tickers)

# Create a function to calculate portfolio return and volatility
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return portfolio_return, portfolio_stddev
# Define optimization constraints
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Define optimization bounds (0 <= weights <= 1)
bounds = tuple((0, 1) for asset in range(num_assets))

# Initial equal weights for all assets
initial_weights = num_assets * [1. / num_assets]

# Minimize negative Sharpe ratio to maximize Sharpe ratio
def negative_sharpe(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return, p_stddev = portfolio_performance(weights, mean_returns, cov_matrix)
    return -((p_return - risk_free_rate) / p_stddev)

# Risk-free rate (you can set this to a suitable value)
risk_free_rate = 0.01

# Perform the optimization
efficient_portfolio = minimize(
    negative_sharpe,
    initial_weights,
    args=(mean_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)
# Extract optimized weights
optimized_weights = efficient_portfolio.x

# Calculate optimized portfolio performance
optimal_return, optimal_stddev = portfolio_performance(optimized_weights, mean_returns, cov_matrix)

# Plot the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(optimal_stddev, optimal_return, marker='o', color='r', label='Optimal Portfolio')
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Return')
plt.title('Scatterplot')
plt.legend()
plt.grid(True)

# Display individual assets on the plot
for i, ticker in enumerate(tickers):
    plt.scatter(np.sqrt(cov_matrix.iloc[i, i]) * np.sqrt(252), mean_returns[i] * 252, marker='x', color='b')
    plt.text(np.sqrt(cov_matrix.iloc[i, i]) * np.sqrt(252) + 0.002, mean_returns[i] * 252, ticker, verticalalignment='center')

plt.show()
