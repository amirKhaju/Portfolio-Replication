import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from Backtest import var_t


class KalmanFilterStrategy:
    """
    Kalman Filter strategy for dynamic portfolio weights.

    Idea:
    - Weights evolve over time (random walk)
    - Each step updates weights based on new observation
    - Produces a full return series (like a strategy, not just a model)
    """

    def __init__(
        self,
        init_window=104,
        process_noise=1e-4,
        obs_noise=None,
        var_threshold=0.20,
        var_confidence=0.01,
        var_horizon=4,
        transaction_cost_bps=5.0,
        annual_factor=52
    ):
        self.init_window = init_window
        self.process_noise = process_noise
        self.obs_noise = obs_noise
        self.var_threshold = var_threshold
        self.var_confidence = var_confidence
        self.var_horizon = var_horizon
        self.transaction_cost_bps = transaction_cost_bps
        self.annual_factor = annual_factor

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize_state(self, X, y):
        """
        Initialize beta (weights) using Ridge regression.
        """
        X_init = X[:self.init_window]
        y_init = y[:self.init_window]

        scaler = StandardScaler().fit(X_init)
        X_sc = scaler.transform(X_init)

        model = Ridge(alpha=1e-3, fit_intercept=False)
        model.fit(X_sc, y_init)

        beta = model.coef_ / scaler.scale_

        # Initial covariance (uncertainty about weights)
        P = np.eye(X.shape[1]) * 0.01

        return beta, P

    def _initialize_noise(self, X, y, beta):
        """
        Set process noise Q and observation noise R.
        """
        n = X.shape[1]

        Q = np.eye(n) * self.process_noise

        if self.obs_noise is None:
            residuals = y[:self.init_window] - X[:self.init_window] @ beta
            R = np.var(residuals)
        else:
            R = self.obs_noise

        return Q, R

    # ============================================================
    # KALMAN STEPS
    # ============================================================

    def _predict(self, beta, P, Q):
        """
        Prediction step (state evolves as random walk).
        """
        beta_pred = beta
        P_pred = P + Q
        return beta_pred, P_pred

    def _update(self, beta_pred, P_pred, x_t, y_t, R):
        """
        Update step using new observation.
        """
        # Prediction error
        y_hat = x_t @ beta_pred
        innovation = y_t - y_hat

        # Innovation variance
        S = x_t @ P_pred @ x_t + R

        # Kalman gain
        K = P_pred @ x_t / S

        # Update weights
        beta_new = beta_pred + K * innovation

        # Update covariance
        P_new = (np.eye(len(beta_pred)) - np.outer(K, x_t)) @ P_pred

        return beta_new, P_new

    # ============================================================
    # RISK + COST
    # ============================================================

    def _compute_var(self, returns):
        if len(returns) < 52:
            return np.nan
        return var_t(
            np.array(returns[-52:]),
            confidence=self.var_confidence,
            horizon=self.var_horizon
        )

    def _apply_var_constraint(self, beta, returns):
        """
        Scale weights if VaR exceeds threshold.
        IMPORTANT: do NOT overwrite beta (state).
        """
        port_var = self._compute_var(returns)

        if np.isnan(port_var):
            return beta.copy(), port_var

        scale = min(1.0, self.var_threshold / port_var)
        return beta * scale, port_var

    def _transaction_cost(self, w_new, w_old):
        turnover = np.sum(np.abs(w_new - w_old))
        return turnover * (self.transaction_cost_bps / 10000)

    # ============================================================
    # MAIN RUN
    # ============================================================

    def run(self, X: pd.DataFrame, y: pd.Series):
        X_np = X.values
        y_np = y.values
        dates = X.index

        # --- Initialize ---
        beta, P = self._initialize_state(X_np, y_np)
        Q, R = self._initialize_noise(X_np, y_np, beta)

        prev_weights = beta.copy()

        # Storage
        replica, replica_net = [], []
        weights_hist, var_hist, out_dates = [], [], []

        # --- Main loop ---
        for t in range(self.init_window, len(X_np) - 1):

            x_t = X_np[t]
            y_t = y_np[t]

            # Kalman steps
            beta_pred, P_pred = self._predict(beta, P, Q)
            beta, P = self._update(beta_pred, P_pred, x_t, y_t, R)

            # Apply risk constraint (on weights, not state)
            weights, port_var = self._apply_var_constraint(beta, replica)

            # Transaction cost
            tc = self._transaction_cost(weights, prev_weights)
            prev_weights = weights.copy()

            # Predict next return
            next_ret = X_np[t + 1] @ weights

            # Store
            replica.append(next_ret)
            replica_net.append(next_ret - tc)
            weights_hist.append(weights.copy())
            var_hist.append(port_var)
            out_dates.append(dates[t + 1])

        # --- Build outputs ---
        idx = pd.Index(out_dates)

        return {
            "replica": pd.Series(replica, index=idx, name="Kalman"),
            "replica_net": pd.Series(replica_net, index=idx, name="Kalman (net)"),
            "weights": pd.DataFrame(weights_hist, index=idx, columns=X.columns),
            "var": pd.Series(var_hist, index=idx)
        }