import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from popbo import POPBO
import sigma
import util

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
class Config:
    test_probs     = ["Branin"] # "Branin", "Levy13", "Bukin", "Cross_in_tray"
    gamma_values   = [0.04]
    num_experiments = 30            # outer loop
    num_iterations  = 30            # BO steps per run
    noise_std       = 0.0           # Thurstone noise
    grid_resolution = 50            # for f_min/f_max & inference grid
    start_csv       = "start_points/2d_start_points.csv"
    out_root        = "results"
    master_seed     = 123           # reproducibility
    N_PAIRS        = 100000

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_scalers(problem_name: str, cfg: Config):
    """Return (popbo_cfg, scale_value, inference_pts, bounds, f_min, f_max)."""
    popbo_cfg = util.get_config(f"test_func_{problem_name}", problem_dim=2)
    f = popbo_cfg["obj"]
    bounds = np.array(popbo_cfg["bounds"])  # shape (2,2)

    # dense grid to find f_min, f_max
    r = cfg.grid_resolution
    xs = np.linspace(bounds[0,0], bounds[0,1], r)
    ys = np.linspace(bounds[1,0], bounds[1,1], r)
    X1, X2 = np.meshgrid(xs, ys)
    Z = np.array([f([x,y]) for x,y in zip(X1.ravel(), X2.ravel())]).reshape(X1.shape)
    f_min, f_max = Z.min(), Z.max()

    def scale_value(v):
        return (v - f_min) / (f_max - f_min)

    # inference grid in original domain
    pts = np.stack([X1.ravel(), X2.ravel()], axis=-1)  # (r*r, 2)
    return popbo_cfg, scale_value, pts, bounds

# ─────────────────────────────────────────────────────────────────────────────
# Single‐problem, single‐experiment runner
# ─────────────────────────────────────────────────────────────────────────────
def run_problem(problem_name: str, gamma: float, cfg: Config, seed: int):
    np.random.seed(seed)

    # scalers + POPBO instance
    popbo_cfg, scale_value, grid_pts, bounds = make_scalers(problem_name, cfg)
    util_fn = popbo_cfg["obj"]
    popbo   = POPBO(popbo_cfg)

    # load initial points for this seed
    df0 = pd.read_csv(cfg.start_csv)
    row = df0[df0["Run_Index"] == seed].iloc[0]
    X1 = np.array([row["X1"], row["Y1"]])
    X2 = np.array([row["X2"], row["Y2"]])

    # initial Thurstone comparison: X1→X2
    one = sigma.jnd_oracle(
        scale_value(util_fn(X2)),
        scale_value(util_fn(X1)),
        noise_std=cfg.noise_std,
        gamma=gamma
    )
    popbo.add_new_data(x_prime=X1, x=X2, one=one)

    # track evaluated true utilities
    eval_list = [util_fn(X1), util_fn(X2)]

    # histories (length = num_iterations+1)
    simple = []
    infer  = []

    # initial simple regret
    best_true = max(eval_list)
    simple.append(1 - scale_value(best_true))

    # initial inference regret
    post = np.array([popbo.get_maxLL_min_norm_inter(pt) for pt in grid_pts])
    x_pred = grid_pts[np.argmax(post)]
    infer.append(1 - scale_value(util_fn(x_pred)))

    # BO loop
    x_prev = X2.copy()
    for it in range(cfg.num_iterations):
        # first iteration
        if it == 0:
            x_new = popbo.query_new_point(x_prime=x_prev)
        else:
            x_new = popbo.query_new_point(x_prime=x_prev)
        x_new = np.clip(x_new, bounds[:,0], bounds[:,1])

        one = sigma.jnd_oracle(
            scale_value(util_fn(x_new)),
            scale_value(util_fn(x_prev)),
            noise_std=cfg.noise_std,
            gamma=gamma
        )
        popbo.add_new_data(x_prime=x_prev, x=x_new, one=one)

        # evaluate true
        v_new = util_fn(x_new)
        eval_list.append(v_new)

        # simple regret
        best_true = max(eval_list)
        simple.append(1 - scale_value(best_true))

        # inference regret
        post = np.array([popbo.get_maxLL_min_norm_inter(pt) for pt in grid_pts])
        x_pred = grid_pts[np.argmax(post)]
        infer.append(1 - scale_value(util_fn(x_pred)))

        logging.info(f"[Seed{seed:02d} it={it+1:02d}] "
                     f"S={simple[-1]:.4f}  I={infer[-1]:.4f}")

        x_prev = x_new.copy()

    # ─── after BO: compute 3-class ties-penalized and 2-class ordinal accuracy ───

    # build the “dense” grid (already in grid_pts) and true/model values
    grid_dense = grid_pts
    u_true = np.array([util_fn(x) for x in grid_dense])
    u_pred = np.array([popbo.get_maxLL_min_norm_inter(x) for x in grid_dense])

    rng = np.random.default_rng(seed)

    # 1) 3-class “ties penalized” accuracy
    pairs_tp = rng.integers(0, len(grid_dense), size=(cfg.N_PAIRS * 2, 2))
    pairs_tp = pairs_tp[pairs_tp[:, 0] != pairs_tp[:, 1]][: cfg.N_PAIRS]

    gt_tp = []
    pred_tp = []
    for i, j in pairs_tp:
        fi, fj = u_true[i], u_true[j]
        mi, mj = u_pred[i], u_pred[j]
        if abs(fi - fj) < gamma:
            # “tie” ground truth, model only predicts binary → always wrong
            gt_tp.append(2)
            pred_tp.append(0)
        else:
            gt_tp.append(1 if fi < fj else 0)
            pred_tp.append(1 if mi < mj else 0)

    gt_tp = np.array(gt_tp)
    pred_tp = np.array(pred_tp)
    ties_penalized_acc = (gt_tp == pred_tp).mean()

    # 2) 2-class “ordinal” accuracy (ignore γ entirely)
    pairs_ord = rng.integers(0, len(grid_dense), size=(cfg.N_PAIRS * 2, 2))
    pairs_ord = pairs_ord[pairs_ord[:, 0] != pairs_ord[:, 1]][: cfg.N_PAIRS]

    gt_ord = []
    pred_ord = []
    for i, j in pairs_ord:
        fi, fj = u_true[i], u_true[j]
        mi, mj = u_pred[i], u_pred[j]
        gt_ord.append(1 if fi < fj else 0)
        pred_ord.append(1 if mi < mj else 0)

    gt_ord = np.array(gt_ord)
    pred_ord = np.array(pred_ord)
    ordinal_acc = (gt_ord == pred_ord).mean()

    return np.array(simple), np.array(infer), ordinal_acc, ties_penalized_acc

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = Config()

    for problem_name, gamma in zip(cfg.test_probs, cfg.gamma_values):
        outdir = os.path.join(cfg.out_root, f"{problem_name}_γ{gamma}")
        os.makedirs(outdir, exist_ok=True)

        all_simple, all_infer = [], []
        all_ord_acc, all_choice_acc = [], []
        seed, done = 1, 0
        while done < cfg.num_experiments:
            try:
                s, i, ord_acc, cho_acc = run_problem(problem_name, gamma, cfg, seed)
                all_simple.append(s)
                all_infer.append(i)
                all_ord_acc.append(ord_acc)
                all_choice_acc.append(cho_acc)
                done += 1
            except Exception as e:
                logging.warning(f"Seed {seed} failed: {e}; skipping.")
            seed += 1

        S = np.vstack(all_simple)
        I = np.vstack(all_infer)

        # raw histories
        np.savetxt(os.path.join(outdir,"simple_regrets.txt"),    S, fmt="%.6f")
        np.savetxt(os.path.join(outdir,"inference_regrets.txt"), I, fmt="%.6f")

        # aggregate
        its = np.arange(S.shape[1])
        df = pd.DataFrame({
            "iteration":    its,
            "mean_simple":  S.mean(axis=0),
            "std_simple":   S.std(axis=0),
            "mean_infer":   I.mean(axis=0),
            "std_infer":    I.std(axis=0),
        })
        df.to_csv(os.path.join(outdir,"regret_results.txt"), sep="\t", index=False)

        # save all per-experiment tables
        with open(os.path.join(outdir,"all_simple.txt"), "w") as f:
            f.write("exp\titer\tsimple\n")
            for e in range(S.shape[0]):
                for t in range(S.shape[1]):
                    f.write(f"{e}\t{t}\t{S[e,t]:.6f}\n")
        with open(os.path.join(outdir,"all_infer.txt"), "w") as f:
            f.write("exp\titer\tinference\n")
            for e in range(I.shape[0]):
                for t in range(I.shape[1]):
                    f.write(f"{e}\t{t}\t{I[e,t]:.6f}\n")
                    
                    
        # save per‐experiment accuracies
        pd.DataFrame({
            "seed":        np.arange(1, done+1),
            "ordinal_acc": all_ord_acc,
            "choice_acc":  all_choice_acc,
        }).to_csv(os.path.join(outdir, "accuracy_per_experiment.csv"), index=False)

        # summary accuracies
        summary_acc = {
            "ordinal_mean": np.nanmean(all_ord_acc),
            "ordinal_std":  np.nanstd(all_ord_acc),
            "choice_mean":  np.nanmean(all_choice_acc),
            "choice_std":   np.nanstd(all_choice_acc),
        }
        pd.DataFrame([summary_acc]).to_csv(
            os.path.join(outdir, "accuracy_summary.csv"), index=False
        )
                    
        # plots
        plt.errorbar(its, df["mean_simple"], yerr=0.1*df["std_simple"],
                     fmt="-o", capsize=3, label="Simple Regret")
        plt.xlabel("Iteration"); plt.ylabel("Simple Regret"); plt.legend()
        plt.savefig(os.path.join(outdir,"simple_plot.png"))
        plt.clf()

        plt.errorbar(its, df["mean_infer"], yerr=0.1*df["std_infer"],
                     fmt="-s", capsize=3, color="orange", label="Inference Regret")
        plt.xlabel("Iteration"); plt.ylabel("Inference Regret"); plt.legend()
        plt.savefig(os.path.join(outdir,"inference_plot.png"))

        logging.info(f"Finished {problem_name} (γ={gamma}). Results in {outdir}")

if __name__ == "__main__":
    main()
