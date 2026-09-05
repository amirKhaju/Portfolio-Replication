import numpy as np
import pandas as pd
from Backtest import RollingBacktester

class ModelRunner:
    def __init__(self, backtester, X, y):
        self.bt = backtester
        self.X = X
        self.y = y

    def _calculate_metrics(self, rep, aligned_y, weights, annual_factor):
        te = self.bt.tracking_error(rep, aligned_y)
        ann_ret = rep.mean() * annual_factor * 100
        vol = rep.std() * np.sqrt(annual_factor) * 100
        corr = rep.corr(aligned_y)

        ir = ((rep.mean() - aligned_y.mean()) * annual_factor / te) if te > 0 else 0
        avg_exposure = weights.abs().sum(axis=1).mean()

        return {
            "Ann. Return (%)": ann_ret,
            "Ann. Vol (%)": vol,
            "Tracking Error (%)": te * 100,
            "Correlation": corr,
            "Info Ratio": ir,
            "Avg Gross Exposure": avg_exposure
        }

    def run_models(self, model_dict):
        results = {}

        for name, model in model_dict.items():
            if hasattr(model, "run"):
                res = model.run(self.X, self.y)
            else:
                res = self.bt.run(self.X, self.y, model)

            aligned_y = self.y.loc[res["replica"].index]
            rep = res["replica_net"]

            res["metrics"] = self._calculate_metrics(rep, aligned_y, res["weights"], self.bt.annual_factor)
            results[name] = res

        return results

    def analyze_rebalancing(self, model, freqs, rolling_window):
        results = {}

        for freq in freqs:
            bt = RollingBacktester(
                rolling_window=rolling_window,
                rebalance_every=freq,
                transaction_cost_bps=self.bt.transaction_cost_bps,
                var_threshold=self.bt.var_threshold,
                var_confidence=self.bt.var_confidence,
                var_horizon=self.bt.var_horizon,
                annual_factor=self.bt.annual_factor
            )

            res = bt.run(self.X, self.y, model)
            aligned_y = self.y.loc[res["replica"].index]
            rep = res["replica_net"]

            results[freq] = self._calculate_metrics(rep, aligned_y, res["weights"], bt.annual_factor)

        return pd.DataFrame(results).T.round(3)