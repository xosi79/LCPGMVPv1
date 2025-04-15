import pandas as pd
import numpy as np
import cvxpy as cp
import json, requests


# ------------------------------
# Fetch Historical Price Data from CryptoCompare API
# ------------------------------
def get_crypto_prices(symbols, exchange, days):
    # API key from your original crypto.py file.
    api_key = "d4b3687111d710b5938632be4038f1bd85ea254be96abb4027ab65e1198bc44a"
    crypto_data = {}
    
    for symbol in symbols:
        # Construct API URL and fetch raw data
        api_url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={exchange}&limit={days}&api_key={api_key}"
        raw = requests.get(api_url).json()
        
        # Extract relevant data and create a DataFrame
        data = raw["Data"]["Data"]
        df = pd.DataFrame(data)[["time", "high", "low", "open", "close"]].set_index("time")
        df.index = pd.to_datetime(df.index, unit="s")
        
        crypto_data[symbol] = df

    return crypto_data

# ------------------------------
# Get Historical Daily Returns using Real Data
# ------------------------------
def get_historical_returns(symbols, exchange="GBP", days=365):
    # Fetch price data using your API function
    price_data = get_crypto_prices(symbols, exchange, days)
    combined = pd.DataFrame()
    for symbol, df in price_data.items():
        # Use the closing prices
        combined[symbol] = df["close"]
    
    # Compute daily percent change (returns)
    returns = combined.pct_change().dropna()
    return returns

# ------------------------------
# Markowitz Mean-Variance Optimization
# ------------------------------
def markowitz_optimization(returns_df, risk_free_rate=0.0):
    # Calculate expected daily returns and the covariance matrix
    mu = returns_df.mean().values  
    sigma = returns_df.cov().values  
    n = len(mu)
    
    # Define optimization variable: weights for each asset
    w = cp.Variable(n)
    # Set a simple target: the mean of expected returns
    target_return = mu.mean()
    constraints = [cp.sum(w) == 1, w >= 0, w.T @ mu >= target_return]
    
    # Objective: minimize portfolio variance
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), constraints)
    try:
        prob.solve()
    except Exception as e:
        raise Exception("Optimization failed: " + str(e))
    
    # Round optimal weights to 4 decimals
    opt_weights = np.round(w.value, 4)
    port_return = opt_weights.dot(mu)
    port_vol = np.sqrt(opt_weights.T.dot(sigma).dot(opt_weights))
    sharpe_ratio = (port_return - risk_free_rate) / (port_vol * np.sqrt(365)) if port_vol != 0 else np.nan
    
    return {
        "optimal_weights": dict(zip(returns_df.columns, opt_weights)),
        "expected_return": port_return,
        "volatility": port_vol,
        "sharpe_ratio": sharpe_ratio
    }

# ------------------------------
# Monte Carlo Simulation for Portfolio Performance (Summary Output)
# ------------------------------
def monte_carlo_simulation(returns_df, weights, sims=1000, days=252):
    n = len(weights)
    sim_results = []
    mean_daily = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    
    for i in range(sims):
        # Generate simulated daily returns using multivariate normal distribution
        simulated = np.random.multivariate_normal(mean_daily, cov_matrix, days)
        # Calculate daily portfolio returns based on the weights
        daily_returns = simulated.dot(weights)
        # Compute cumulative return for this simulation
        cum_return = np.prod(1 + daily_returns) - 1
        sim_results.append(cum_return)
        
    sim_results = np.array(sim_results)
    avg_sim = round(sim_results.mean(), 4)
    std_sim = round(sim_results.std(), 4)
    
    # Only return summary statistics, not full list
    return {"avg_simulated_return": avg_sim, "std_simulated_return": std_sim}

# ------------------------------
# Generate Final Portfolio Report by Combining the Methods
# ------------------------------
def gen_port(symbols):
    """
    Generate a portfolio report using real API data.
    1. Fetch historical returns.
    2. Run Markowitz optimization.
    3. Run Monte Carlo simulation.
    4. Return a summary dictionary.
    """
    returns_df = get_historical_returns(symbols, exchange="GBP", days=365)
    mark_res = markowitz_optimization(returns_df)
    # Use the optimal weights from the result (convert dict values to np.array)
    weights = np.array(list(mark_res["optimal_weights"].values()))
    monte_res = monte_carlo_simulation(returns_df, weights, sims=1000, days=252)
    
    port_report = {
        "optimal_weights": mark_res["optimal_weights"],
        "expected_return": round(mark_res["expected_return"], 4),
        "volatility": round(mark_res["volatility"], 4),
        "sharpe_ratio": round(mark_res["sharpe_ratio"], 4),
        "monte_carlo": monte_res
    }
    return port_report

if __name__ == "__main__":
    # Sample test with actual crypto symbols (API uses uppercase like "ADA", "ETH", "SOL")
    sample_coins = ["ADA", "ETH", "SOL"]
    result = gen_port(sample_coins)
    print("Portfolio Report:")
    print(result)
