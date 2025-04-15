import numpy as np
import pandas as pd
import cvxpy as cp

def sim_hist_returns(coins, days=500):
    """
    simulate historical daily returns for each coin over given number of days
    param - coins (list), days (int)
    this is a placeholder for random data
    """
    np.random.seed(42)
    #assume each coin has mean daily return 0.1% and sd of .2
    data = {coin: np.random.normal(0.001, 0.02, days) for coin in coins}
    returns_df = pd.DataFrame(data)
    return returns_df

def markowitz_optimization(returns_df, risk_free_rate=0.0):
    """
    Solving markowitz mean-variance optimization problem
    params -> dataframe of daily returns, risk free rate
    returns -> dict of optimal weights + portfolio performance metrics
    """
    #calc expectde returns and covariance matrix from sim data
    mu = returns_df.mean().values #expected daily returns for each coin
    sigma = returns_df.cov().values #cov matrix of coin returns
    n = len(mu)

    """
    objective is to maximise sharp ratio, 
    quadratic to minimize variance
    """

    #optimization variables
    w = cp.Variable(n) #weights for each asset
    target_return = mu.mean()   #adjustable
    constraints = [cp.sum(w) == 1, w >= 0, w.T @ mu >= target_return]

    obj = cp.Minimize(cp.quad_form(w, sigma))
    prob = cp.Problem(obj, constraints)

    try:
        prob.solve()
    except Exception as e:
        raise Exception(f"Optimization failed: {e}")
    
    #optimal weights as list rounded to 4 d.p
    opt_weights = np.round(w.value, 4)

    #calc expected return and variance of said weights
    port_return = opt_weights.dot(mu)
    port_volatility = np.sqrt(opt_weights.T.dot(sigma).dot(opt_weights))

    #calc simple shart using risk free
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility != 0 else np.nan

    return {
        "optimal_weights": dict(zip(returns_df.columns, opt_weights)),
        "expected_return": port_return,
        "volatility": port_volatility,
        "sharpe_ratio": sharpe_ratio
    }

def monte_carlo(returns_df, weights, sims=1000, days=252): #(trading days)
    """
    run monte carlo to estimate port performance
    params -> history returns, optimal weights, sims, days
    returns -> dict of average sim return and risk metrics
    """

    n = len(weights)
    port_results = []

    #calc stats from historical
    mean_daily_returns = returns_df.mean().values
    cov_matrix = returns_df.cov().values

    for _ in range(sims):
        #gen random daily retunrs :multivariate n.d
        sim_returns = np.random.multivariate_normal(mean_daily_returns, cov_matrix, days)
        daily_port_returns = sim_returns.dot(weights)

        #agg to get cumulative port return for sim
        cumulative_return = np.prod(1 + daily_port_returns) - 1
        port_results.append(cumulative_return)

    port_results = np.array(port_results)
    avg_return = port_results.mean()
    std_return = port_results.std()

    return {
        "avg_simulated_return": avg_return,
        "std_simulated_return": std_return,
        "simulated_return_distribution": port_results.tolist()  # For further analysis, if needed.
    }

def gen_port(coins):
    """
    gen portfolio allocaton by combining methods
    1.sim historical returns
    2.run markowitz 3.run monte carlo
    4.package outputs
    param -> coins chosen
    returns -> dict of portfolio report
    """

    returns_df = sim_hist_returns(coins, days=500)

    markowitz_res = markowitz_optimization(returns_df)

    weights = np.array(list(markowitz_res["optimal_weights"].values()))
    monte_carlo_res = monte_carlo(returns_df, weights, sims=1000, days=252)

    port_rep = {
        "optimal_weights": markowitz_res["optimal_weights"],
        "expected_return": markowitz_res["expected_return"],
        "volatility": markowitz_res["volatility"],
        "sharpe_ratio": markowitz_res["sharpe_ratio"],
        "monte_carlo": monte_carlo_res
    }

    return port_rep

if __name__ == "__main__":
    sample_coins = ["BTC", "ETH", "SOL"]
    res = gen_port(sample_coins)
    print("Portfolio Report:")
    print(res)


