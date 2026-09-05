import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utilities import var_t
import scipy.stats as stats

class RollingBacktester:
    def __init__(
            self,
            rolling_window=104,
            rebalance_every=1,
            var_threshold=0.20,
            var_confidence=0.01,
            var_horizon=4,
            transaction_cost_bps=5.0,
            annual_factor=52
    ):
        self.rolling_window = rolling_window
        self.rebalance_every = rebalance_every
        self.var_threshold = var_threshold
        self.var_confidence = var_confidence
        self.var_horizon = var_horizon
        self.transaction_cost_bps = transaction_cost_bps
        self.annual_factor = annual_factor

    def _fit_model(self, model, X, y):
        scaler = StandardScaler(with_mean=False)
        X_sc = scaler.fit_transform(X)
        model.fit(X_sc, y)

        # Safe extraction: forces 1D array and provides fallback if coef_ is missing
        raw_coef = getattr(model, 'coef_', np.zeros(X.shape[1]))
        coef = np.ravel(raw_coef)

        return coef / scaler.scale_

    def _compute_var(self, returns):
        if len(returns) < 52:
            return np.nan
        return var_t(
            np.array(returns[-52:]),
            confidence=self.var_confidence,
            horizon=self.var_horizon
        )

    def run(self, X, y, model):
        X_vals = X.values
        y_vals = y.values
        dates = X.index

        n_features = X_vals.shape[1]
        current_weights = np.zeros(n_features)

        replica, replica_net = [], []
        weights_hist, var_hist = [], []

        for i in range(len(X_vals) - self.rolling_window - 1):
            t_end = i + self.rolling_window
            do_rebalance = (i % self.rebalance_every == 0)

            if do_rebalance:
                X_tr = X_vals[i:t_end]
                y_tr = y_vals[i:t_end]

                # Fit the base model to get unleveraged weights
                new_w = self._fit_model(model, X_tr, y_tr)

                # Volatility Matching (Addressing Leverage)
                # Calculate historical vol for the target and unleveraged replica
                target_vol = np.std(y_tr[-52:])
                sim_returns_unleveraged = X_tr[-52:] @ new_w
                replica_vol = np.std(sim_returns_unleveraged)

                # Calculate leverage factor to match target volatility
                # We cap this at 2.0 per UCITS/MIFID 200% leverage limits mentioned in slides
                leverage_factor = min(2.0, target_vol / replica_vol) if replica_vol > 0 else 1.0
                new_w *= leverage_factor

                # 3. Risk Management (VaR Scaling)
                # Recalculate returns with leverage for the final VaR check
                simulated_returns = X_tr[-52:] @ new_w
                port_var = self._compute_var(simulated_returns)

                # Scale back ONLY if the leveraged portfolio exceeds the 20% regulatory limit
                scale = 1.0 if np.isnan(port_var) else min(1.0, self.var_threshold / port_var)
                new_w *= scale

                # 4. Transaction Costs and History
                turnover = np.sum(np.abs(new_w - current_weights))
                tc = turnover * (self.transaction_cost_bps / 10000)

                current_weights = new_w
                var_hist.append(port_var)
            else:
                tc = 0.0
                var_hist.append(var_hist[-1] if var_hist else np.nan)

            ret = X_vals[t_end] @ current_weights

            replica.append(ret)
            replica_net.append(ret - tc)
            weights_hist.append(current_weights.copy())

        idx = dates[self.rolling_window:-1]

        return {
            "replica": pd.Series(replica, index=idx),
            "replica_net": pd.Series(replica_net, index=idx),
            "weights": pd.DataFrame(weights_hist, index=idx, columns=X.columns),
            "var": pd.Series(var_hist, index=idx)
        }

    def tracking_error(self, rep, tgt):
        return (rep - tgt).std() * np.sqrt(self.annual_factor)

