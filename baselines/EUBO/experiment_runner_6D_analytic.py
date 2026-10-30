import os
import logging
import random
import numpy as np
import pandas as pd
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from botorch.exceptions import ModelFittingError
from torch.quasirandom import SobolEngine

from utility.utility_functions import levy6, alpine1
from utility.oracle import jnd_oracle

import warnings
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    utility_name    = "levy6"   # "levy6" or "alpine1"
    num_experiments = 30        # number of independent runs
    num_steps       = 50        # BO iterations per run
    dim             = 6         # input dimensionality
    gamma           = 0.04      # indifference threshold (JND)
    sigma           = 0.04      # perceptual noise σ
    consecutive     = True     # True: compare new vs last; False: compare new vs new
    test_size       = 25000     # Sobol grid size for inference regret
    master_seed     = 123         # global seed

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark utilities
# ─────────────────────────────────────────────────────────────────────────────
utilities = {
    "levy6":   (levy6,   torch.stack([torch.zeros(6), torch.ones(6)])),
    "alpine1": (alpine1, torch.stack([torch.zeros(6), torch.ones(6)])),
}

# ─────────────────────────────────────────────────────────────────────────────
# Single-experiment routine
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(seed, cfg, df_start, X_test):
    # seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # initial pair
    row = df_start[df_start["Run_Index"] == seed].iloc[0]
    X1 = torch.tensor(row[[f"X1_{i}" for i in range(1, cfg.dim+1)]].values, dtype=torch.float64)
    X2 = torch.tensor(row[[f"X2_{i}" for i in range(1, cfg.dim+1)]].values, dtype=torch.float64)
    X_all = torch.stack([X1, X2], dim=0)

    util_fn, bounds = utilities[cfg.utility_name]

    # first comparison (no learning yet)
    u0 = util_fn(X_all)
    comps = jnd_oracle(u0[0:1], u0[1:2], 0, 1, cfg.sigma, cfg.gamma)

    simple_regrets = []
    infer_regrets  = []

    for step in range(1, cfg.num_steps+1):
        # fit model
        try:
            model = PairwiseGP(X_all, comps, input_transform=Normalize(d=cfg.dim))
            mll   = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        except ModelFittingError:
            logging.error(f"Run {seed} failed at step {step}")
            break

        # compute regrets
        with torch.no_grad():
            best_seen = util_fn(X_all).max().item()
            simple_regrets.append(1 - best_seen)

            # approximate argmax with test grid
            pm = torch.cat([
                model.posterior(X_test[i:i+2000]).mean.squeeze(-1)
                for i in range(0, cfg.test_size, 2000)
            ])
            idx_max = int(pm.argmax().item())
            xr = X_test[idx_max].unsqueeze(0)
            infer_regrets.append(1 - util_fn(xr).item())

        logging.info(f"[Run{seed:02d} Step{step:02d}] simple={simple_regrets[-1]:.4f} infer={infer_regrets[-1]:.4f}")

        # acquisition
        if cfg.consecutive:
            acq = AnalyticExpectedUtilityOfBestOption(pref_model=model,
                                                outcome_model=None,
                                                previous_winner=X_all[-1:].unsqueeze(0))
            q = 1
        else:
            acq = AnalyticExpectedUtilityOfBestOption(pref_model=model, outcome_model=None)
            q = 2

        new_X, _ = optimize_acqf(acq, bounds=bounds, q=q, num_restarts=5, raw_samples=50)

        # simulate preference
        if cfg.consecutive:
            u_old = util_fn(X_all[-1:].unsqueeze(0))
            u_new = util_fn(new_X)
            comp  = jnd_oracle(u_old, u_new,
                        idx_a=X_all.size(0)-1, idx_b=X_all.size(0),
                        sigma=cfg.sigma, gamma=cfg.gamma)
        else:
            # compare the two new points
            u_pair = util_fn(new_X)
            comp = jnd_oracle(u_pair[0:1], u_pair[1:2],
                        idx_a=X_all.size(0), idx_b=X_all.size(0)+1,
                        sigma=cfg.sigma, gamma=cfg.gamma)

        # augment
        X_all = torch.cat([X_all, new_X], dim=0)
        comps  = torch.cat([comps, comp.long()], dim=0)

    return simple_regrets, infer_regrets

# ─────────────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config()

    # global seed
    random.seed(cfg.master_seed)
    np.random.seed(cfg.master_seed)
    torch.manual_seed(cfg.master_seed)

    # Sobol test grid
    sobol = SobolEngine(dimension=cfg.dim, scramble=False, seed=cfg.master_seed)
    X_test = sobol.draw(cfg.test_size).double()

    # load starts
    df_start = pd.read_csv("start_points/6d_start_points.csv")

    all_simple = []
    all_infer  = []
    for exp in range(1, cfg.num_experiments+1):
        s, i = run_experiment(exp, cfg, df_start, X_test)
        all_simple.append(s)
        all_infer.append(i)

    # raw results
    records = []
    for exp_idx, (s_list, i_list) in enumerate(zip(all_simple, all_infer), start=1):
        for step, (s_val, i_val) in enumerate(zip(s_list, i_list), start=1):
            records.append({
                "exp": exp_idx,
                "step": step,
                "simple_regret": s_val,
                "inference_regret": i_val
            })
    df = pd.DataFrame.from_records(records)
    out_dir = (
        f"results/{cfg.utility_name}_c{int(cfg.consecutive)}"
        f"_γ{cfg.gamma}_σ{cfg.sigma}"
        f"_exp{cfg.num_experiments}_steps{cfg.num_steps}"
    ).replace(".", "_")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f"{out_dir}/results.csv", index=False)

    # summary stats
    summary = df.groupby("step").agg(
        simple_mean=("simple_regret", "mean"),
        simple_std=("simple_regret", "std"),
        infer_mean=("inference_regret", "mean"),
        infer_std=("inference_regret", "std"),
    ).reset_index()
    summary.to_csv(f"{out_dir}/summary.csv", index=False)

    logging.info(f"Finished. Outputs in {out_dir}/{{results.csv, summary.csv}}")

if __name__ == "__main__":
    main()
