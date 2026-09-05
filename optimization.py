import pandas as pd
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from itertools import product

class HyperparameterOptimizer:
    def __init__(self, backtester, X, y):
        self.bt = backtester
        self.X = X
        self.y = y

    def _evaluate(self, model_class, params, keys, sample_size=None):
        if sample_size is not None:
            X = self.X.iloc[-sample_size:]
            y = self.y.iloc[-sample_size:]
        else:
            X = self.X
            y = self.y

        # Removed warm_start=True as it is ineffective when re-instantiating the model
        try:
            model = model_class(**params, fit_intercept=False)
        except TypeError:
            model = model_class(**params)  # Fallback if model doesn't accept fit_intercept

        res = self.bt.run(X, y, model)
        aligned_y = y.loc[res["replica"].index]
        rep = res["replica"]

        te = self.bt.tracking_error(rep, aligned_y)
        return {**params, "TE": te}

    def grid_search(self, model_class, param_grid, n_jobs=-1, sample_size=None): # Adjust if necessary,
        keys = list(param_grid.keys())
        combinations = list(product(*param_grid.values()))

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self._evaluate)(
                model_class,
                dict(zip(keys, values)),
                keys,
                sample_size
            )
            for values in combinations
        )

        df = pd.DataFrame(results).sort_values("TE").reset_index(drop=True)
        best_row = df.iloc[0]
        best_score = best_row["TE"]
        best_params = {k: best_row[k] for k in keys}

        return df, best_params, best_score
