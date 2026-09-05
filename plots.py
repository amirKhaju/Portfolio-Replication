import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_normalized_series(df, columns, base=100, highlight_col=None):
    """Plots multiple series normalized to a base value with optional highlighting."""
    plt.figure(figsize=(14, 7))

    for col in columns:
        if col not in df.columns:
            continue

        # Get the first non-NaN value for normalization
        first_val = df[col].dropna().iloc[0]
        series = (df[col] / first_val) * base

        # Logic for thicker line
        linewidth = 4 if col == highlight_col else 2
        alpha = 1.0 if col == highlight_col else 0.8

        plt.plot(series.index, series, linewidth=linewidth, alpha=alpha, label=col)

    plt.title(f'Historical Performance (Base {base})', fontsize=16)
    plt.axhline(y=base, color='black', linestyle='--', alpha=0.3)
    plt.ylabel(f"Value (Base {base})")
    plt.xlabel("Date")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

def get_performance_stats(returns, annual_factor=52):
    """Returns a DataFrame of key financial metrics for a returns dataframe."""
    # Annualized Return & Vol
    ann_ret = returns.mean() * annual_factor
    ann_vol = returns.std() * np.sqrt(annual_factor)

    # Max Drawdown logic
    cum_ret = (1 + returns).cumprod()
    peak = cum_ret.cummax()
    drawdown = (cum_ret - peak) / peak
    max_dd = drawdown.min()

    stats = pd.DataFrame({
        'Annualized Return': ann_ret,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': ann_ret / ann_vol,
        'Max Drawdown': max_dd,
        'Skewness': returns.skew(),
        'Kurtosis': returns.kurtosis()
    })
    return stats

def plot_correlation_heatmap(returns_df, title='Correlation Matrix'):
    """Plots a seaborn correlation heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(returns_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, fmt='.2f')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_return_distributions(returns_df, columns):
    """Plots histograms with KDE, Mean, and Std Dev annotations."""
    # Auto-calculate grid size (2 columns, dynamic rows)
    n = len(columns)
    cols = 2
    rows = (n + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if n > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Handle single column case

    for i, col in enumerate(columns):
        if col not in returns_df.columns: continue
        sns.histplot(returns_df[col].dropna(), kde=True, ax=axes[i])
        axes[i].set_title(f'Return Distribution: {col}', fontsize=14)
        axes[i].set_xlabel('Weekly Return')
        axes[i].axvline(x=0, color='red', linestyle='--')

        # Annotations
        mean, std = returns_df[col].mean(), returns_df[col].std()
        axes[i].annotate(f'Mean: {mean:.4f}\nStd: {std:.4f}',
                         xy=(0.65, 0.85), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Hide empty subplots if 'n' is odd
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_rolling_correlations(df, target_col, features_list, window=52):
    """Plots rolling correlations between a target and multiple features."""
    plt.figure(figsize=(14, 8))

    for col in features_list:
        if col not in df.columns or target_col not in df.columns:
            continue

        # Calculate rolling correlation safely
        rolling_corr = df[target_col].rolling(window=window).corr(df[col])
        plt.plot(rolling_corr.index, rolling_corr, linewidth=2, label=f'{col}')

    plt.title(f'Rolling {window}-Week Correlation: {target_col} vs Top Predictors', fontsize=16)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.ylabel('Correlation Coefficient', fontsize=14)
    plt.xlabel('Date', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12)

    # Format X-axis dates cleanly
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()



def plot_cumulative_returns(main_cum, component_returns, components_dict, dates,
                            main_label='Monster Index',
                            title='Cumulative Returns — Monster Index vs Components',
                            figsize=(14, 5)):
    """
    Plots the cumulative returns of a main index against its weighted components.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the main index
    main_cum.plot(ax=ax, color='navy', linewidth=2, label=main_label)

    # Plot each component
    for comp, weight in components_dict.items():
        # Calculate cumulative return for the specific component over the common dates
        comp_cum = (1 + component_returns[comp].loc[dates]).cumprod()
        comp_cum.plot(ax=ax, linestyle='--', alpha=0.6, label=f'{comp} (w={weight})')

    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Growth of 1 unit')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_replica_vs_target(target_returns, replica_returns,
                           target_label='Monster Index',
                           replica_label='OLS Replica',
                           title='Replica vs Target Index',
                           figsize=(14, 5)):
    """
    Plots the cumulative returns of a target index against a replicated version.
    """
    fig, ax = plt.subplots(figsize=figsize)

    (1 + target_returns).cumprod().plot(ax=ax, label=target_label, color='navy', lw=2)
    (1 + replica_returns).cumprod().plot(ax=ax, label=replica_label, color='red', lw=1.5, linestyle='--')

    ax.set_title(title, fontsize=13)
    ax.set_ylabel('Growth of 1 unit')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_qq_normality_extended(target_series, futures_df, top_futures):
    """Plots QQ plots for a target and top 5 features in a 2x3 grid."""
    plt.figure(figsize=(16, 12))

    # 1. Plot the Target (Position 1)
    plt.subplot(2, 3, 1)
    stats.probplot(target_series.dropna(), dist="norm", plot=plt)
    plt.title(f'QQ Plot: {target_series.name} vs Normal', fontsize=14)
    plt.grid(True, alpha=0.3)

    # 2. Plot the top 5 correlated futures (Positions 2 through 6)
    for i, contract in enumerate(top_futures[:5]):
        plt.subplot(2, 3, i+2)
        stats.probplot(futures_df[contract].dropna(), dist="norm", plot=plt)
        plt.title(f'QQ Plot: {contract} vs Normal', fontsize=14)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_acf_pacf(df, columns, max_lags=20):
    """Plots ACF and PACF side-by-side for given columns."""
    # Safety check: only use columns that exist in the dataframe
    valid_cols = [col for col in columns if col in df.columns]
    n = len(valid_cols)

    if n == 0:
        print("⚠️ Warning: No valid columns found for ACF/PACF plots.")
        return

    # Create grid: N rows by 2 columns (Fixed to your preferred size)
    fig, axes = plt.subplots(n, 2, figsize=(18, 4 * n))

    # Ensure axes is a 2D array even if there's only one row
    if n == 1:
        axes = np.array([axes])

    for i, col in enumerate(valid_cols):
        series = df[col].dropna()

        # Plot ACF
        plot_acf(series, lags=max_lags, ax=axes[i, 0], alpha=0.05)
        axes[i, 0].set_title(f'Autocorrelation: {col}', fontsize=14)
        axes[i, 0].set_xlabel('Lag')
        axes[i, 0].set_ylabel('Correlation')
        axes[i, 0].grid(True, alpha=0.3)

        # Plot PACF (method='ywm' suppresses the statsmodels warning)
        plot_pacf(series, lags=max_lags, ax=axes[i, 1], method='ywm', alpha=0.05)
        axes[i, 1].set_title(f'Partial Autocorrelation: {col}', fontsize=14)
        axes[i, 1].set_xlabel('Lag')
        axes[i, 1].set_ylabel('Partial Correlation')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_volatility_clustering(df, columns, max_lags=20):
    """Plots ACF of squared and absolute returns to check for volatility clustering."""
    valid_cols = [col for col in columns if col in df.columns]
    n = len(valid_cols)

    if n == 0:
        print(" Warning: No valid columns found for volatility plots.")
        return

    # Create grid: N rows by 2 columns
    fig, axes = plt.subplots(n, 2, figsize=(18, 4 * n))

    # Ensure axes is a 2D array even if there's only one row
    if n == 1:
        axes = np.array([axes])

    for i, col in enumerate(valid_cols):
        # Calculate squared and absolute returns
        sq_returns = df[col].dropna() ** 2
        abs_returns = df[col].dropna().abs()

        # Plot Squared Returns ACF (Left Column)
        plot_acf(sq_returns, lags=max_lags, ax=axes[i, 0], alpha=0.05)
        axes[i, 0].set_title(f'ACF Squared Returns: {col} (Volatility Clustering)', fontsize=14)
        axes[i, 0].set_xlabel('Lag')
        axes[i, 0].set_ylabel('Correlation')
        axes[i, 0].grid(True, alpha=0.3)

        # Plot Absolute Returns ACF (Right Column)
        plot_acf(abs_returns, lags=max_lags, ax=axes[i, 1], alpha=0.05)
        axes[i, 1].set_title(f'ACF Absolute Returns: {col}', fontsize=14)
        axes[i, 1].set_xlabel('Lag')
        axes[i, 1].set_ylabel('Correlation')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_scatter_matrix(df, target_col, features_list):
    """Plots a scatter matrix (pairplot) for the target and top features."""
    # Combine target and features, ensuring target is first
    valid_features = [col for col in features_list if col in df.columns and col != target_col]
    cols_to_plot = [target_col] + valid_features

    if len(cols_to_plot) < 2:
        print("⚠️ Warning: Not enough valid columns to create a scatter matrix.")
        return

    # Create the pairplot
    g = sns.pairplot(df[cols_to_plot], kind='scatter', diag_kind='kde',
                     plot_kws={'alpha': 0.6, 's': 20, 'edgecolor': 'k', 'linewidth': 0.5})

    # Adjust title to sit perfectly above the matrix
    g.fig.suptitle(f'Scatter Matrix: {target_col} vs Top Predictors', fontsize=16, y=1.02)
    plt.show()

def plot_replica_performance(y, replica_returns, weights, model_name="OLS"):
    """
    Plots the cumulative returns of a target vs a replica, alongside the replica's weights.

    Parameters:
    - y: pd.Series, the target asset returns.
    - replica_returns: pd.Series, the returns of the replicated portfolio.
    - weights: pd.Series, the asset weights of the replica.
    - model_name: str, name of the model for the plot titles and labels (default: "OLS").
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # 1. Cumulative Returns Plot
    (1 + y).cumprod().plot(
        ax=axes[0], label="Target", color="navy", lw=2
    )
    (1 + replica_returns).cumprod().plot(
        ax=axes[0], label=f"{model_name} (in-sample)", color="crimson", lw=1.5, linestyle="--"
    )
    axes[0].set_title(f"{model_name} Replica vs Target — Cumulative Returns")
    axes[0].legend()

    # 2. Weights Bar Chart
    weights.sort_values().plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title(f"{model_name} Weights")

    plt.tight_layout()
    plt.show()

def plot_model_performance(y, replica_returns, weights, model_name="OLS", figsize=(14, 4)):
    """
    Plots cumulative returns of a target vs. replica and the sorted model weights.

    Parameters:
    - y: pd.Series of target returns.
    - replica_returns: pd.Series of model replica returns.
    - weights: pd.Series of model weights.
    - model_name: str, name of the model to use in titles and legends (default: "OLS").
    - figsize: tuple, size of the resulting figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 1. Cumulative returns
    target_label = f"Target: {y.name}" if hasattr(y, 'name') and y.name else "Target"
    (1 + y).cumprod().plot(ax=axes[0], label=target_label, color="navy", lw=2)
    (1 + replica_returns).cumprod().plot(
        ax=axes[0],
        label=f"{model_name} (in-sample)",
        color="crimson",
        lw=1.5,
        linestyle="--"
    )
    axes[0].set_title(f"{model_name} Replica vs Target — Cumulative Returns")
    axes[0].legend()

    # 2. Weights bar chart
    weights.sort_values().plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title(f"{model_name} Weights")

    plt.tight_layout()
    plt.show()

def plot_oos_comparison(results, y, annual_factor=52):
    """
    Plots cumulative out-of-sample returns and rolling 52-week tracking error.
    """
    colors = ["steelblue", "darkorange", "seagreen", "crimson", "purple", "brown"]

    # Common date range
    common_idx = results[list(results.keys())[0]]["replica"].index
    for res in results.values():
        common_idx = common_idx.intersection(res["replica"].index)

    target_oos = y.loc[common_idx]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # ── Cumulative returns ────────────────────────────────────────────────────
    target_label = f"Target: {y.name}" if hasattr(y, 'name') and y.name else "Target"
    (1 + target_oos).cumprod().plot(ax=axes[0], color="navy", lw=2.5, label=target_label)
    for i, (name, res) in enumerate(results.items()):
        col = colors[i % len(colors)]
        rep = res["replica_net"].loc[common_idx]
        (1 + rep).cumprod().plot(ax=axes[0], lw=1.5, linestyle="--", color=col, label=name)

    axes[0].set_title(f"Cumulative Returns — Out-of-Sample (net) | {target_label}")
    axes[0].legend(fontsize=10)

    # ── Rolling tracking error (52-week) ─────────────────────────────────────
    for i, (name, res) in enumerate(results.items()):
        col = colors[i % len(colors)]
        rep = res["replica_net"].loc[common_idx]
        roll_te = (rep - target_oos).rolling(52).std() * np.sqrt(annual_factor)
        roll_te.plot(ax=axes[1], lw=1.5, color=col, label=name)

    axes[1].set_title(f"Rolling 52-Week Tracking Error (annualized) | {target_label}")
    axes[1].set_ylabel("Tracking Error")
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()

    return fig, axes, common_idx, target_oos


def highlight_best(s):

    is_max = s.name in ["Ann. Return (%)", "Correlation", "Info Ratio"]
    is_min = s.name in ["Ann. Vol (%)", "Tracking Error (%)"]
    
    if is_max:
        is_best = s == s.max()
    elif is_min:
        is_best = s == s.min()
    else:
        # Per colonne non specificate (come Gross Exposure) non applichiamo lo stile
        is_best = [False] * len(s)
        
    return ['color: red; font-weight: bold' if v else '' for v in is_best]

def plot_efficient_frontiers(cov_mat, hist_exp_ret, tickers, mvo_weights_constrained, mvo_weights_unconstrained):
    """
    Plots the constrained and unconstrained efficient frontiers.
    """
    from scipy.optimize import minimize
    def port_vol(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_mat, w))) * np.sqrt(52)

    # Calculate Risk/Return for our Constrained Model
    vol_constrained = port_vol(mvo_weights_constrained)
    ret_constrained = np.dot(mvo_weights_constrained, hist_exp_ret)

    # Calculate Risk/Return for our Unconstrained Model (The Monster)
    vol_unconstrained = port_vol(mvo_weights_unconstrained)
    ret_unconstrained = np.dot(mvo_weights_unconstrained, hist_exp_ret)

    # 1. Generate target returns (Expand range to reach the Monster!)
    min_ret = min(hist_exp_ret.min(), ret_unconstrained) * 1.2
    max_ret = max(hist_exp_ret.max(), ret_unconstrained) * 1.2
    target_returns = np.linspace(min_ret, max_ret, 80)

    frontier_vol_constrained = []
    frontier_vol_unconstrained = []

    num_assets = len(tickers)
    init_weights = np.repeat(1.0 / num_assets, num_assets)

    # 2. Set Bounds
    bounds_constrained = tuple((0.0001, 1.0) for _ in range(num_assets))
    # Fix for the missing line: Use wide bounds instead of (None, None) so solver doesn't fail
    bounds_unconstrained = tuple((-10.0, 10.0) for _ in range(num_assets)) 

    # 3. Calculate Both Frontiers
    for tr in target_returns:
        constraints_ef = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.dot(w, hist_exp_ret) - tr}
        )
        
        # Constrained (Long-Only)
        res_c = minimize(port_vol, init_weights, method='SLSQP', bounds=bounds_constrained, constraints=constraints_ef)
        frontier_vol_constrained.append(res_c.fun if res_c.success else np.nan)
        
        # Unconstrained (Shorting Allowed)
        res_u = minimize(port_vol, init_weights, method='SLSQP', bounds=bounds_unconstrained, constraints=constraints_ef)
        frontier_vol_unconstrained.append(res_u.fun if res_u.success else np.nan)

    # 4. Plotting
    plt.figure(figsize=(12, 8))

    # Plot both frontiers
    plt.plot(frontier_vol_unconstrained, target_returns, 'b-', linewidth=2, alpha=0.5, label='Unconstrained Frontier (Allows Shorting)')
    plt.plot(frontier_vol_constrained, target_returns, 'k--', linewidth=2, label='Constrained Frontier (Long-Only)')

    # Plot individual assets
    asset_vols = np.sqrt(np.diag(cov_mat)) * np.sqrt(52)
    plt.scatter(asset_vols, hist_exp_ret, marker='o', s=100, c='gray', label='Individual Assets', zorder=4)

    for i, txt in enumerate(tickers):
        plt.annotate(f"  {txt}", (asset_vols[i], hist_exp_ret.iloc[i]), fontsize=10, va='center')

    # Plot the Portfolios
    plt.scatter(vol_constrained, ret_constrained, color='green', marker='*', s=300, 
                label='Constrained MVO (Max Sharpe)', zorder=5)

    plt.scatter(vol_unconstrained, ret_unconstrained, color='red', marker='X', s=200, 
                label='Unconstrained MVO (The Monster)', zorder=5)

    # Formatting
    plt.title('Portfolio Optimization: Constrained vs. Unconstrained Efficient Frontier', fontsize=14)
    plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
    plt.ylabel('Annualized Expected Return', fontsize=12)

    # Dynamic axis limits to fit both curves
    plt.xlim(0, max(asset_vols) * 2)
    plt.ylim(min(0, min_ret), max(max(hist_exp_ret), ret_unconstrained) * 1.2)

    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()
