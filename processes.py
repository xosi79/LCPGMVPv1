import requests
import pandas as pd
import numpy as np
import cvxpy as cp
import json
import random

# ------------------------------
# FETCH HISTORICAL PRICE DATA FROM CRYPTOCOMPARE API
# ------------------------------
def get_crypto_prices(symbols, exchange, days):
    api_key = "d4b3687111d710b5938632be4038f1bd85ea254be96abb4027ab65e1198bc44a"
    crypto_data = {}
    for symbol in symbols:
        api_url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym={exchange}&limit={days}&api_key={api_key}"
        raw = requests.get(api_url).json()
        data = raw["Data"]["Data"]
        df = pd.DataFrame(data)[["time", "high", "low", "open", "close"]].set_index("time")
        df.index = pd.to_datetime(df.index, unit="s")
        crypto_data[symbol] = df
    return crypto_data

# ------------------------------
# GET HISTORICAL RETURNS USING LIVE DATA
# ------------------------------
def get_historical_returns(symbols, exchange="GBP", days=365):
    price_data = get_crypto_prices(symbols, exchange, days)
    combined = pd.DataFrame()
    for symbol, df in price_data.items():
        combined[symbol] = df["close"]
    returns = combined.pct_change().dropna()
    return returns

# ------------------------------
# STANDARD MARKOWITZ OPTIMIZATION (MINIMIZE VARIANCE FOR TARGET RETURN)
# ------------------------------
def markowitz_optimization(returns_df, risk_free_rate=0.0):
    mu = returns_df.mean().values  # expected daily returns
    sigma = returns_df.cov().values  # covariance matrix
    n = len(mu)
    
    w = cp.Variable(n)  # portfolio weights variable
    target_return = mu.mean()  # simple target; adjust if needed
    constraints = [cp.sum(w) == 1, w >= 0, w.T @ mu >= target_return]
    
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, sigma)), constraints)
    try:
        prob.solve()
    except Exception as e:
        raise Exception("Optimization failed: " + str(e))
    
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
# MONTE CARLO SIMULATION (SUMMARY OUTPUT)
# ------------------------------
def monte_carlo_simulation(returns_df, weights, sims=1000, days=252):
    n = len(weights)
    sim_results = []
    mean_daily = returns_df.mean().values
    cov_matrix = returns_df.cov().values
    
    for _ in range(sims):
        simulated = np.random.multivariate_normal(mean_daily, cov_matrix, days)
        daily_returns = simulated.dot(weights)
        cum_return = np.prod(1 + daily_returns) - 1
        sim_results.append(cum_return)
    
    sim_results = np.array(sim_results)
    avg_sim = round(sim_results.mean(), 4)
    std_sim = round(sim_results.std(), 4)
    
    return {"avg_simulated_return": avg_sim, "std_simulated_return": std_sim}

# ------------------------------
# NEW FEATURE: PREDICTIVE OPTIMIZATION
# This function forecasts future returns using a simple moving average over a 'window' of recent days,
# then maximizes the forecasted return under a risk constraint.
# ------------------------------
def predictive_optimization(returns_df, window=30, risk_limit=None, risk_free_rate=0.0):
    """
    Forecast future returns using the moving average over the last 'window' days,
    then solve an optimization problem to maximize forecasted return.
    Optionally applies a risk constraint (max volatility) by ensuring that the portfolio variance
    is below risk_limit^2.
    
    Parameters:
      returns_df: DataFrame of daily returns.
      window: Number of recent days to average for the forecast.
      risk_limit: Maximum allowed portfolio volatility (if provided).
      risk_free_rate: For Sharpe ratio calculation.
      
    Returns a dict with predicted portfolio weights, forecasted return, volatility, and Sharpe ratio.
    """
    n_assets = returns_df.shape[1]
    
    # Use the moving average of the last 'window' days as the forecast
    if len(returns_df) < window:
        forecast_returns = returns_df.mean().values
    else:
        forecast_returns = returns_df.tail(window).mean().values
    
    # Historical covariance for risk estimation
    sigma = returns_df.cov().values
    
    # Optimization variable: portfolio weights for each asset
    w = cp.Variable(n_assets)
    
    # Objective: maximize the forecasted return (linear objective)
    # Note: Without risk constraint, it'll assign 100% to the highest return asset.
    objective = cp.Maximize(forecast_returns @ w)
    
    # Basic constraints: weights sum to 1, and no negative weights
    constraints = [cp.sum(w) == 1, w >= 0]
    
    # If risk_limit is given, add a risk constraint.
    # Instead of using sqrt for volatility, we square both sides:
    # sqrt(w.T @ sigma @ w) <= risk_limit  is equivalent to  (w.T @ sigma @ w) <= risk_limit^2
    if risk_limit is not None:
        port_variance = cp.quad_form(w, sigma)
        constraints.append(port_variance <= risk_limit**2)
    
    prob = cp.Problem(objective, constraints)
    
    try:
        # Use qcp=True to indicate we're solving a DQCP-compliant problem.
        prob.solve(qcp=True)
    except Exception as e:
        raise Exception("Predictive optimization failed: " + str(e))
    
    # Round our optimal weights to 4 decimal places
    pred_weights = np.round(w.value, 4)
    
    # Calculate forecasted return and volatility based on our chosen weights.
    forecasted_return = float(pred_weights.dot(forecast_returns))
    forecasted_vol = np.sqrt(float(pred_weights.T.dot(sigma).dot(pred_weights)))
    forecasted_sharpe = (forecasted_return - risk_free_rate) / (forecasted_vol * np.sqrt(365)) if forecasted_vol != 0 else np.nan
    
    return {
        "optimal_weights": dict(zip(returns_df.columns, pred_weights)),
        "forecasted_return": forecasted_return,
        "forecasted_volatility": forecasted_vol,
        "forecasted_sharpe_ratio": forecasted_sharpe
    }

# ------------------------------
# FINAL PORTFOLIO REPORT: COMBINE METHODS
# ------------------------------
def gen_port(symbols):
    """
    Generate a portfolio report using two methods:
      1. Standard Markowitz Optimization (with Monte Carlo simulation).
      2. Predictive Optimization (using recent moving average forecast).
    
    Returns a summary dictionary with both methods.
    """
    returns_df = get_historical_returns(symbols, exchange="GBP", days=365)
    
    # ----- Method 1: Markowitz Optimization -----
    mark_res = markowitz_optimization(returns_df)
    weights_mark = np.array(list(mark_res["optimal_weights"].values()))
    monte_res = monte_carlo_simulation(returns_df, weights_mark, sims=1000, days=252)
    
    # ----- Method 2: Predictive Optimization -----
    # Use the volatility from Markowitz as a risk cap for the predictive method.
    risk_limit = mark_res["volatility"]
    pred_res = predictive_optimization(returns_df, window=30, risk_limit=risk_limit, risk_free_rate=0.0)
    
    report = {
        "markowitz": {
            "optimal_weights": mark_res["optimal_weights"],
            "expected_return": round(mark_res["expected_return"], 4),
            "volatility": round(mark_res["volatility"], 4),
            "sharpe_ratio": round(mark_res["sharpe_ratio"], 4),
            "monte_carlo": monte_res
        },
        "predictive": {
            "optimal_weights": pred_res["optimal_weights"],
            "forecasted_return": round(pred_res["forecasted_return"], 4),
            "forecasted_volatility": round(pred_res["forecasted_volatility"], 4),
            "forecasted_sharpe_ratio": round(pred_res["forecasted_sharpe_ratio"], 4)
        }
    }
    return report

if __name__ == "__main__":
    # Test with sample coins—use real API symbols like "ADA", "ETH", "SOL"
    sample_coins = ["ADA", "ETH", "SOL"]
    result = gen_port(sample_coins)
    print("Portfolio Report:")
    print(result)
