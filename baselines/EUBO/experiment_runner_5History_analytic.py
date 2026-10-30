import os
import logging
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.optim import optimize_acqf
from botorch.exceptions import ModelFittingError

from utility.utility_functions import branin, levy13, bukin6, crossintray, ackley, bohachevsky, sixhump
from utility.oracle import jnd_oracle

import warnings
warnings.filterwarnings("ignore")
# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    problem_name    = "branin"         # name of the 2D test function
    gamma           = 0.04              # JND threshold
    consecutive     = True              # True: compare new vs last; False: compare new vs new
    num_experiments = 30                # independent runs
    num_steps       = 30               # BO iterations per run
    sigma           = 0.0               # perceptual noise std
    grid_size       = 50                # resolution for regret grid
    lookback_k      = 5                 # compare new point vs last k points
    start_csv       = "start_points/2d_start_points.csv"
    out_root        = "results/2d"
    master_seed     = 123

# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────
def get_utility_fn(name: str):
    return {
        "branin":      branin,
        "levy13":      levy13,
        "bukin6":      bukin6,
        "crossintray": crossintray,
        "ackley":      ackley,
        "bohachevsky": bohachevsky,
        "sixhump":     sixhump,
    }[name]

# ─────────────────────────────────────────────────────────────────────────────
# Single experiment routine
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment(cfg: Config,
                   df_start: pd.DataFrame,
                   XY_grid: torch.Tensor,
                   grid_max: float,
                   bounds: torch.Tensor,
                   seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    util_fn = get_utility_fn(cfg.problem_name)

    row = df_start[df_start['Run_Index']==seed].iloc[0]
    X_all = torch.tensor([[row['X1'], row['Y1']],
                          [row['X2'], row['Y2']]], dtype=torch.double)

    u0 = util_fn(X_all)
    comps = jnd_oracle(u0[0:1], u0[1:2], 0, 1, cfg.sigma, cfg.gamma)

    simple_regrets = []
    infer_regrets  = []

    for step in range(1, cfg.num_steps+1):
        model = PairwiseGP(X_all, comps.long(), input_transform=Normalize(d=2))
        mll   = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        with torch.no_grad():
            best_obs = util_fn(X_all).max().item()
            simple_regrets.append(grid_max - best_obs)
            pm = model.posterior(XY_grid).mean.squeeze(-1)
            idx = int(pm.argmax())
            infer_regrets.append(grid_max - util_fn(XY_grid[idx:idx+1]).item())

        logging.info(f"[Seed {seed} Step {step}] Simple={simple_regrets[-1]:.4f} "
                     f"Inference={infer_regrets[-1]:.4f}")

        if cfg.consecutive:
            acq = AnalyticExpectedUtilityOfBestOption(
                pref_model=model, outcome_model=None,
                previous_winner=X_all[-1:].unsqueeze(0)
            )
            q = 1
        else:
            acq = AnalyticExpectedUtilityOfBestOption(pref_model=model, outcome_model=None)
            q = 2

        new_X, _ = optimize_acqf(acq, bounds=bounds, q=q,
                                 num_restarts=5, raw_samples=50)

        k = min(cfg.lookback_k, X_all.size(0))
        recent = list(range(X_all.size(0)-k, X_all.size(0)))
        u_new = util_fn(new_X)
        comps_new = []
        for idx_old in recent:
            u_old = util_fn(X_all[idx_old:idx_old+1])
            comps_new.append(
                jnd_oracle(u_old, u_new, idx_old, X_all.size(0),
                           cfg.sigma, cfg.gamma)
            )
        comps_new = torch.cat(comps_new, dim=0)

        X_all = torch.cat([X_all, new_X], dim=0)
        comps  = torch.cat([comps, comps_new], dim=0)

    # accuracy on uniform samples with random pairing; γ-zone counted wrong
    M = 5000
    X_uni = torch.rand(M, 2, dtype=torch.double)
    with torch.no_grad():
        post_mean = model.posterior(X_uni).mean.squeeze(-1)
    u_vals = util_fn(X_uni).view(-1)

    P = 100000
    i_idx = torch.randint(0, M, (P,))
    j_idx = torch.randint(0, M, (P,))
    same = (i_idx == j_idx)
    while same.any():
        j_idx[same] = torch.randint(0, M, (same.sum(),))
        same = (i_idx == j_idx)

    gt_ord   = (u_vals[i_idx] > u_vals[j_idx])
    pred_ord = (post_mean[i_idx] > post_mean[j_idx])
    ordinal_acc = float((gt_ord == pred_ord).float().mean())

    delta_u = (u_vals[i_idx] - u_vals[j_idx]).abs()
    correct = (gt_ord == pred_ord).clone()
    correct[delta_u < cfg.gamma] = False
    choice_acc = float(correct.float().mean())

    return simple_regrets, infer_regrets, ordinal_acc, choice_acc


# ─────────────────────────────────────────────────────────────────────────────
# Main entrypoint
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config()

    # prepare grid
    torch.manual_seed(cfg.master_seed)
    lin = torch.linspace(0, 1, cfg.grid_size, dtype=torch.double)
    xx, yy = torch.meshgrid(lin, lin)
    XY_grid = torch.stack([xx, yy], dim=-1).view(-1, 2)
    util_fn = get_utility_fn(cfg.problem_name)
    grid_max = util_fn(XY_grid).max().item()
    bounds = torch.stack([torch.zeros(2), torch.ones(2)]).double()

    # load starts
    df_start = pd.read_csv(cfg.start_csv)

    all_simple = []
    all_infer  = []
    all_ord_acc = []
    all_choice_acc = []

    
    seed = 1
    completed = 0
    # keep going until we collect cfg.num_experiments successful runs
    while completed < cfg.num_experiments:
        try:
            s, i, ord_acc, cho_acc = run_experiment(cfg, df_start, XY_grid, grid_max, bounds, seed)
            all_simple.append(s)
            all_infer.append(i)
            all_ord_acc.append(ord_acc)
            all_choice_acc.append(cho_acc)

            completed += 1
            logging.info(f"Seed {seed} completed ({completed}/{cfg.num_experiments})")
        except ModelFittingError:
            logging.warning(f"Seed {seed} model fit failed—skipping.")
        seed += 1

    S = np.array(all_simple)
    I = np.array(all_infer)

    outdir = f"{cfg.out_root}/{cfg.problem_name}_gamma_{cfg.gamma}_c{int(cfg.consecutive)}"
    os.makedirs(outdir, exist_ok=True)

    # save regrets
    np.savetxt(os.path.join(outdir, "simple_regrets.txt"),    S, fmt="%.6f")
    np.savetxt(os.path.join(outdir, "inference_regrets.txt"), I, fmt="%.6f")
    
    df_acc = pd.DataFrame({
    "seed":        np.arange(1, cfg.num_experiments+1),
    "ordinal_acc": all_ord_acc,
    "choice_acc":  all_choice_acc,
    })
    df_acc.to_csv(os.path.join(outdir, "accuracy_per_experiment.csv"), index=False)

    summary_acc = {
        "ordinal_mean": float(np.nanmean(all_ord_acc)),
        "ordinal_std":  float(np.nanstd(all_ord_acc)),
        "choice_mean":  float(np.nanmean(all_choice_acc)),
        "choice_std":   float(np.nanstd(all_choice_acc)),
    }
    pd.DataFrame([summary_acc]).to_csv(os.path.join(outdir, "accuracy_summary.csv"), index=False)

    # summary
    steps = np.arange(1, cfg.num_steps+1)
    summary = pd.DataFrame({
        "step":        steps,
        "mean_simple": S.mean(axis=0),
        "std_simple":  S.std(axis=0),
        "mean_infer":  I.mean(axis=0),
        "std_infer":   I.std(axis=0),
    })
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)

    # plots
    plt.errorbar(steps, summary["mean_simple"],
                 yerr=0.1*summary["std_simple"],
                 marker='o', label='Simple Regret')
    plt.xlabel('Iteration'); plt.ylabel('Simple Regret'); plt.legend()
    plt.savefig(os.path.join(outdir, 'simple_plot.png')); plt.clf()

    plt.errorbar(steps, summary["mean_infer"],
                 yerr=0.1*summary["std_infer"],
                 marker='o', color='orange', label='Inference Regret')
    plt.xlabel('Iteration'); plt.ylabel('Inference Regret'); plt.legend()
    plt.savefig(os.path.join(outdir, 'inference_plot.png'))

    logging.info(f"Done. Results saved to {outdir}")

if __name__ == "__main__":
    main()
