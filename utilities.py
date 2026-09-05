import numpy as np
import pandas as pd
import scipy.stats as stats

from plots import *
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet



def highlight_significant(val):
    """Highlights p-values less than 0.05 in bold red."""
    color = '#d62728' if val < 0.05 else 'inherit' # Red for significant
    weight = 'bold' if val < 0.05 else 'normal'
    return f'color: {color}; font-weight: {weight}'

def get_comprehensive_stats(series, freq='W'):
    annual_factor = 52 if freq == 'W' else 12
    cum_ret = (1 + series).cumprod()
    max_dd = -(1 - cum_ret / cum_ret.cummax()).max()

    return pd.Series({
        'Annualized Return': series.mean() * annual_factor,
        'Annualized Volatility': series.std() * np.sqrt(annual_factor),
        'Sharpe Ratio': (series.mean() * annual_factor) / (series.std() * np.sqrt(annual_factor)),
        'Max Drawdown': max_dd,
        'Skewness': series.skew(),
        'Kurtosis': series.kurtosis()
    })

def format_final_table(df):
    pct_rows = ['Annualized Return', 'Annualized Volatility', 'Max Drawdown']

    # Apply styling
    return df.style.format(lambda x: f"{x*100:.2f}%" if any(r in str(df.index) for r in pct_rows) else f"{x:.2f}")


def get_fama_french_data():
    """Fetches and cleans the Fama-French 3-Factor data."""
    import urllib.request, zipfile, io, ssl
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"

    context = ssl._create_unverified_context()
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req, context=context)

    with zipfile.ZipFile(io.BytesIO(response.read())) as z:
        with z.open(z.namelist()[0]) as f:
            lines = f.readlines()
            start = next(i for i, l in enumerate(lines) if b'Mkt-RF' in l)
            f.seek(0)
            df = pd.read_csv(f, skiprows=start, index_col=0)

    # Clean and convert to PeriodIndex
    df = df[df.index.astype(str).str.strip().str.match(r'^\d{6}$')]
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format='%Y%m').to_period('M')
    return df.astype(float) / 100


def calculate_var(returns, confidence=0.01, horizon=4, method='historical'):
    """
    Calculates Value at Risk (VaR) using robust methods.

    Parameters:
    returns (array-like): Series of returns
    confidence (float): Confidence level (e.g., 0.01 for 99%)
    horizon (int): Time horizon for scaling (e.g., 4 weeks)
    method (str): 'historical', 'modified' (Cornish-Fisher), or 'gaussian'
    """
    returns_array = np.array(returns)
    if len(returns_array) == 0:
        return np.nan

    if method == 'historical':
        # Pure non-parametric approach
        var_1w = -np.percentile(returns_array, confidence * 100)

    elif method == 'modified':
        # Cornish-Fisher expansion (Accounting for Skewness and Kurtosis found in EDA)
        z = stats.norm.ppf(confidence)
        s = stats.skew(returns_array)
        k = stats.kurtosis(returns_array)

        # Cornish-Fisher Adjusted Z-score
        z_cf = (z +
                (z ** 2 - 1) * s / 6 +
                (z ** 3 - 3 * z) * k / 24 -
                (2 * z ** 3 - 5 * z) * (s ** 2) / 36)

        var_1w = -(np.mean(returns_array) + z_cf * np.std(returns_array))

    else:  # Gaussian/Parametric
        z_score = stats.norm.ppf(confidence)
        var_1w = -(np.mean(returns_array) + z_score * np.std(returns_array))

    # Scale by square root of time
    var = var_1w * np.sqrt(horizon)
    return var

def var_t(returns_series, confidence=0.01, horizon=4):
    """
    Student-t parametric VaR.
    Fits degrees of freedom from data — better for fat-tailed returns.
    """
    if len(returns_series) < 10:  # Minimum safety check for distribution fitting
        return np.nan

    df, loc, scale = stats.t.fit(returns_series)
    q = stats.t.ppf(confidence, df=df, loc=loc, scale=scale)
    return -q * np.sqrt(horizon)

def analizza_target(target_name, X, all_returns, bt, ANNUAL_FACTOR=52):
    from Backtest import RollingBacktester
    from optimization import HyperparameterOptimizer
    from Model_Runner import ModelRunner
    from Kalman_Filter import KalmanFilterStrategy
    
    print(f"\n{'='*60}")
    print(f"{target_name}")
    print(f"{'='*60}\n")
    
    y = all_returns[target_name]
    
    # ── OLS Baseline (In-Sample) ──
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X.values, y.values)
    
    ols_weights  = pd.Series(ols.coef_, index=X.columns)
    ols_replica  = X @ ols_weights
    ols_te       = (ols_replica - y).std() * np.sqrt(ANNUAL_FACTOR)
    
    print(f"=== OLS Baseline (In-Sample) ===")
    print(f"  R²               : {ols.score(X, y):.4f}")
    print(f"  Tracking Error   : {ols_te*100:.2f}%")
    print(f"  Correlation      : {ols_replica.corr(y):.4f}")
    print(f"  Gross Exposure   : {ols_weights.abs().sum():.2f}x")
    print()
    print(ols_weights.sort_values(key=abs, ascending=False).round(4).to_frame("Weight").T)
    
    plot_model_performance(y, ols_replica, ols_weights, model_name="OLS")
    plt.show()
    
    # ── Hyperparameter Search — Elastic Net ──
    from IPython.display import display
    optimizer = HyperparameterOptimizer(bt, X, y)
    
    param_grid = {
        "alpha":     [0.0001, 0.001, 0.01],
        "l1_ratio":  [0.2, 0.5, 0.8, 1.0],
    }
    
    grid_results, best_params, best_te = optimizer.grid_search(
        model_class=ElasticNet,
        param_grid=param_grid,
        n_jobs=-1,
        sample_size=None
    )
    
    print(f"\n=== Elastic Net Best Params ===")
    print(f"Best params : {best_params}")
    print(f"Best TE     : {best_te*100:.2f}%")
    
    # ── Rolling Backtest — All Models ──
    runner = ModelRunner(bt, X, y)
    models = {
        "Ridge":       Ridge(alpha=0.01, fit_intercept=False),
        "Lasso":       Lasso(alpha=0.001, fit_intercept=False, max_iter=5000),
        "Elastic Net": ElasticNet(**best_params, fit_intercept=False, max_iter=5000),
        "Kalman":      KalmanFilterStrategy(init_window=104, process_noise=1e-4,
                                            transaction_cost_bps=5.0),
    }
    
    results = runner.run_models(models)
    
    # ── Optimizing Rebalancing Frequency (Elastic Net) ──
    print("\n=== Optimizing Rebalancing Frequency (Elastic Net) ===")
    rebalance_grid = [1, 2, 4, 8, 13]
    freq_results = {}
    for freq in rebalance_grid:
        bt_freq = RollingBacktester(
            rolling_window=156,
            rebalance_every=freq,
            var_threshold=0.20,
            var_confidence=0.01,
            var_horizon=4,
            transaction_cost_bps=5.0,
            annual_factor=ANNUAL_FACTOR
        )
        runner_freq = ModelRunner(bt_freq, X, y)
        res_freq = runner_freq.run_models({"EN": ElasticNet(**best_params, fit_intercept=False, max_iter=5000)})
        freq_results[freq] = res_freq["EN"]
        print(f"  Freq={freq} -> TE: {freq_results[freq]['metrics']['Tracking Error (%)']:.2f}%")
        
    best_freq = min(freq_results.keys(), key=lambda k: freq_results[k]["metrics"]["Tracking Error (%)"])
    print(f"Best Frequency: {best_freq}")
    results[f"Elastic Net (Freq={best_freq})"] = freq_results[best_freq]

    # ── Optimizing Process Noise (Kalman Filter) ──
    print("\n=== Optimizing Process Noise (Kalman Filter) ===")
    noise_grid = [1e-5, 1e-4, 1e-3, 1e-2]
    kf_results = {}
    for pn in noise_grid:
        res_kf = runner.run_models({"KF": KalmanFilterStrategy(init_window=104, process_noise=pn, transaction_cost_bps=5.0)})
        kf_results[pn] = res_kf["KF"]
        print(f"  Noise={pn:.0e} -> TE: {kf_results[pn]['metrics']['Tracking Error (%)']:.2f}%")
        
    best_pn = min(kf_results.keys(), key=lambda k: kf_results[k]["metrics"]["Tracking Error (%)"])
    print(f"Best Noise: {best_pn:.0e}")
    results[f"Kalman (Noise={best_pn:.0e})"] = kf_results[best_pn]
    
    metrics_df = pd.DataFrame({name: res["metrics"] for name, res in results.items()}).T.round(3)
    
    print("\n=== Out-of-Sample Performance (net of transaction costs) ===")

    styled_df = metrics_df.style.apply(highlight_best, axis=0).format(precision=3)
    display(styled_df)
    fig, axes, common_idx, target_oos = plot_oos_comparison(results, y, annual_factor=ANNUAL_FACTOR)
    plt.show()
    
    # Indetify best model (lowest Tracking Error)
    best_name = metrics_df["Tracking Error (%)"].idxmin()
    best_rep  = results[best_name]["replica_net"].loc[common_idx]

    # Weights evolution
    fig_w, ax_w = plt.subplots(figsize=(14, 4))
    w = results[best_name]["weights"].loc[common_idx]
    top5 = w.abs().mean().nlargest(5).index
    w[top5].plot(ax=ax_w)
    ax_w.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_w.set_title(f"{best_name} — Weight Evolution (top 5 futures)")
    ax_w.legend(fontsize=8, ncol=3)
    plt.tight_layout()
    plt.show()

    # Return Scatter e  VaR series
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(target_oos, best_rep, alpha=0.35, s=8, color="steelblue")
    lim = max(abs(target_oos).max(), abs(best_rep).max()) * 1.1
    axes[0].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    axes[0].set_xlabel("Target Weekly Return")
    axes[0].set_ylabel("Replica Weekly Return")
    axes[0].set_title(f"Return Scatter — {best_name}")

    var_series = results[best_name]["var"].loc[common_idx].dropna()
    var_series.plot(ax=axes[1], color="darkorange", lw=1.5)
    axes[1].axhline(0.20, color="red", lw=1, linestyle="--", label="UCITS limit (20%)")
    axes[1].set_title(f"VaR Series — {best_name}")
    axes[1].set_ylabel("1M VaR (99%)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

    var = results[best_name]["var"].dropna()
    print(f"Average VaR ({best_name}): {var.mean() * 100:.2f}%")
    print(f"Maximum VaR ({best_name}): {var.max() * 100:.2f}%")
    print(f"Threshold:   {bt.var_threshold * 100:.2f}%")
    
    best_metrics = metrics_df.loc[best_name].copy()
    best_metrics["Best Model"] = best_name
    best_metrics["Average VaR (%)"] = var.mean() * 100
    best_metrics["Maximum VaR (%)"] = var.max() * 100
    return best_metrics


def get_unconstrained_markowitz_weights(returns_df, tickers, t_bill_r=0.0):
    from scipy.linalg import inv
    
    ret_data = returns_df[tickers].dropna()
    cov_mat = ret_data.cov()
    
    # Historical expected returns (annualized assuming weekly data)
    hist_exp_ret = ret_data.mean() * 52
    
    mvo_weights_unscaled = np.dot(inv(cov_mat), (hist_exp_ret - t_bill_r))
    mvo_weights_unconstrained = mvo_weights_unscaled / np.sum(mvo_weights_unscaled)
    
    return pd.Series(mvo_weights_unconstrained, index=tickers)

def get_constrained_markowitz_weights(returns_df, tickers, bounds=None, t_bill_r=0.0):
    from scipy.optimize import minimize
    
    ret_data = returns_df[tickers].dropna()
    cov_mat = ret_data.cov()
    
    # Historical expected returns (annualized assuming weekly data)
    hist_exp_ret = ret_data.mean() * 52
    
    def neg_sharpe(w):
        port_ret = np.dot(w, hist_exp_ret)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_mat, w)))
        port_vol_ann = port_vol * np.sqrt(52) 
        return -(port_ret - t_bill_r) / port_vol_ann

    num_assets = len(tickers)
    init_weights = np.repeat(1.0 / num_assets, num_assets) 
    
    if bounds is None:
        bounds = tuple((0.0001, 1.0) for _ in range(num_assets))
        
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}) 

    opt_result = minimize(neg_sharpe, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return pd.Series(opt_result.x, index=tickers)


def get_black_litterman_weights(returns_df, tickers, bench_prop_dict, bench_ret_annual=0.06, t_bill_r=0.0):
    """
    Calculates the optimal Black-Litterman weights.
    """
    from scipy.linalg import inv
    
    ret_data = returns_df[tickers].dropna()
    cov_mat = ret_data.cov()
    
    bench_prop = pd.Series(bench_prop_dict)
    bench_prop = bench_prop.reindex(tickers).fillna(0)
    
    bench_ret = bench_ret_annual / 52 
    norm_fact = (bench_ret - t_bill_r) / np.dot(np.dot(bench_prop, cov_mat), bench_prop)
    exp_ret = np.dot(cov_mat, bench_prop) * norm_fact + t_bill_r
    
    # Inject Your Views (Deltas) - currently assuming 0 for all (neutral)
    delta_port = pd.Series(0.0, index=tickers)
    track_mat = cov_mat / np.diag(cov_mat)
    
    opn_adj_ret = exp_ret + np.dot(track_mat, delta_port)
    
    bl_weights = np.dot(inv(cov_mat), (opn_adj_ret - t_bill_r))
    bl_weights = bl_weights / np.sum(bl_weights) # Normalize to 100%
    
    return pd.Series(bl_weights, index=tickers)
