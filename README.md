# Portfolio Replica — Index & Hedge Fund Cloning via Sparse Regression

Replicates opaque target portfolios — a global hedge fund index and global equity/bond benchmarks — using a small set of liquid Futures contracts, framing index tracking as a sparse signal-extraction problem rather than a portfolio-management one.

---

## Motivation

Many alternative investments (hedge funds, illiquid strategies) disclose only their **returns**, not their holdings. This project asks: can we reconstruct the risk exposures driving those returns using a handful of liquid, tradable instruments?

This has direct practical uses:
- **Alternative investment clones** — replicate hedge-fund-like risk/return without high fees or liquidity restrictions
- **UCITS/liquid clones** — rebuild an illiquid strategy inside a regulated, liquid fund structure
- **Risk analysis** — identify the small number of liquid factors actually driving an opaque investment, useful for due diligence or scenario simulation

Framed statistically: from observed target returns alone, recover the latent weights (positions in liquid factors) that best explain them — a sparse portfolio tracking problem, and fundamentally a signal-processing exercise subject to portfolio constraints.

---

## Methodology

The notebook builds up from a simple baseline to more adaptive models, then benchmarks all of them against classical mean-variance optimization:

| Stage | Approach |
|-------|----------|
| 1. OLS Baseline | In-sample regression of target returns on futures returns — gives an optimistic ceiling and identifies which futures matter |
| 2. Elastic Net | Regularized regression (L1 + L2) with hyperparameter search, controlling for overfitting and sparsity in the replicating weights |
| 3. Rolling Backtest | Walk-forward re-estimation on a rolling window, simulating realistic deployment rather than a static in-sample fit |
| 4. Kalman Filter | State-space model treating replicating weights as time-varying latent states, updated recursively as new data arrives |
| 5. Markowitz Benchmarks | Constrained and unconstrained mean-variance optimization, included as a classical portfolio-construction baseline for comparison |

All models are evaluated on **multiple target indices** (a hedge fund index and several global equity/bond benchmarks) to check whether a given approach generalizes across target types, rather than being tuned to one series.

### Realistic Implementation Constraints

Rather than a frictionless backtest, the framework accounts for:
- **Transaction costs** — modeled in basis points per trade
- **Rebalancing frequency** — weekly through quarterly, with frequency itself treated as a variable to test
- **VaR-based leverage limits** — mirroring UCITS/MiFID constraints (e.g., max 1-month VaR at 99% confidence), not just a notional exposure cap
- **Rollover effects** — accounted for conceptually, since Futures contracts must be rolled to maintain constant exposure

---

## Data

- **Frequency:** Weekly
- **Source:** Bloomberg
- **Period:** October 2007 – April 2021 (704 weeks)
- **Targets:** Global hedge fund index (HFRXGL), global equity indices (MXWO, MXWD), global bond index (LEGATRUU)
- **Predictors:** 11 liquid Futures contracts spanning rates, equities, commodities, and currencies (e.g., bond futures, equity index futures, gold, oil)

---

## Project Structure

```
.
├── 2-Modelling.ipynb        # Main notebook: full pipeline, walkthrough, and results
├── Backtest.py              # RollingBacktester — walk-forward backtest engine
├── optimization.py          # HyperparameterOptimizer — Elastic Net hyperparameter search
├── Model_Runner.py          # ModelRunner — orchestrates models across targets
├── Kalman_Filter.py         # KalmanFilterStrategy — time-varying weight estimation
├── utilities.py             # Helper functions (target diagnostics, etc.)
├── plots.py                 # Plotting utilities
└── all_returns_data.pkl     # Weekly return data (targets + futures)
```

---

## Notebook Structure

1. Setup
2. OLS Baseline (in-sample)
3. Hyperparameter Search — Elastic Net
4. Rolling Backtest
5. Weight Evolution
6. Rebalancing Frequency Analysis
7. Kalman Filter — Process Noise Sensitivity
8–10. Model Comparison across target indices (HFRXGL, MXWO, MXWD, LEGATRUU)
11–12. Unconstrained and Constrained Markowitz benchmarks
13. Final Comparison

---

## Results

Best-performing model per target, out-of-sample:

| Target | Best Model | Ann. Return | Ann. Vol | Tracking Error | Correlation | Info Ratio |
|--------|-----------|:---:|:---:|:---:|:---:|:---:|
| HFRXGL (hedge fund index) | Elastic Net (rebalance freq. = 8) | 2.44% | 2.83% | 3.84% | 0.48 | 0.19 |
| MXWO (global equities) | Elastic Net (rebalance freq. = 2) | 11.21% | 14.73% | 3.07% | 0.98 | 0.56 |
| MXWD (global equities, incl. EM) | Kalman Filter (noise = 1e-5) | 10.44% | 14.67% | 3.69% | 0.97 | 0.42 |
| LEGATRUU (global bonds) | Kalman Filter (noise = 1e-5) | 1.71% | 3.46% | 3.62% | 0.68 | -0.26 |

**Key takeaways:**
- The equity benchmarks (MXWO, MXWD) are tracked closely — correlation above 0.97 using only liquid futures — showing that a small factor set can substitute for full index replication.
- The hedge fund index (HFRXGL) is inherently harder to replicate (lower correlation, 0.48), consistent with the fact that hedge fund strategies often contain non-linear or discretionary exposures that a linear factor model cannot fully capture.
- No single method dominates across all targets: Elastic Net wins for the equity indices, while the Kalman Filter's adaptive weighting wins where exposures likely drift over time (MXWD, LEGATRUU).

---

## Quick Start

```python
from Backtest import RollingBacktester
from Model_Runner import ModelRunner
import pandas as pd

all_returns = pd.read_pickle("all_returns_data.pkl")

FUTURES = ["RX1", "TY1", "GC1", "CO1", "ES1", "VG1", "NQ1", "LLL1", "TP1", "DU1", "TU2"]
TARGET = "HFRXGL"

y = all_returns[TARGET]
X = all_returns[FUTURES]

bt = RollingBacktester(
    rolling_window=156,
    rebalance_every=12,
    var_threshold=0.20,
    var_confidence=0.01,
    var_horizon=4,
    transaction_cost_bps=5.0,
    annual_factor=52,
)

runner = ModelRunner(bt, X, y)
# See 2-Modelling.ipynb for the full model comparison pipeline
```

---

## Requirements

```
numpy
pandas
scikit-learn
matplotlib
```

Install with:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## References

- Wu, L., Yang, Y., Liu, H. (2014). *Nonnegative-Lasso and application in index tracking*. Computational Statistics & Data Analysis, 70.
- Tibshirani, R. (1996). *Regression shrinkage and selection via the lasso*. J. R. Statist. Soc. B.
- Akansu, A. N., Kulkarni, S. R., Malioutov, D. M. (2016). *Financial Signal Processing and Machine Learning*. Wiley-IEEE Press.
- Roncalli, T., Weisang, G. (2009). *Tracking Problems, Hedge Fund Replication and Alternative Beta*. SSRN.

---

## Author

**Amirreza Khajouei**
MSc Mathematical Engineering (Quantitative Finance), Politecnico di Milano
[GitHub](https://github.com/amirKhaju) · amirreza.khajouei@mail.polimi.it
