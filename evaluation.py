import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from oracle import jnd_oracle
import pathlib

def simple_training_acc(
    model: torch.nn.Module,
    likelihood,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    gamma: float = None
):
    """
    Computes CGPBO training‐set accuracy by thresholding the posterior mean differences.
    Returns (accuracy, y_true, y_pred).

    - model: trained gpytorch model
    - likelihood: CPBO_likelihood instance
    - X_train: (N, D) tensor of inputs
    - y_train: (N-1,) tensor of true comparison labels in {-1,0,1}
    - gamma: optional override of likelihood.gamma
    """
    model.eval()
    likelihood.eval()

    # get gamma either from override or the likelihood
    γ = gamma if gamma is not None else float(likelihood.gamma.item())

    with torch.no_grad():
        post = model(X_train)
        μ = post.mean  # shape (N,)

        # differences between consecutive points
        diff = μ[1:] - μ[:-1]          # shape (N-1,)
        # initialize to “-1”
        y_pred = torch.full_like(diff, -1, dtype=torch.int)
        # > +γ  ⇒  +1
        y_pred[diff >  γ] = 1
        # within ±γ ⇒  0
        y_pred[(diff >= -γ) & (diff <= γ)] = 0

        # ensure y_train and y_pred are same dtype
        y_true = y_train.squeeze().to(torch.int)

        correct = (y_pred == y_true).float().mean().item()

    return correct, y_true, y_pred

def plot_gp_latent_surface(
    model: torch.nn.Module,
    scaler: StandardScaler,
    X_all: np.ndarray,
    exp_dir: Path,
    step: int,
    dim: int,
    grid_size: int = 50,
    levels: int = 50
):
    """
    If dim==2, plots the GP posterior mean (with training points overlay)
    on a uniform [0,1]^2 grid. Saves to exp_dir/step+1/gp_latent_contour.png.
    """
    if dim != 2:
        return  # only supports 2D

    # Build grid in [0,1]^2
    xs = np.linspace(0, 1, grid_size)
    ys = np.linspace(0, 1, grid_size)
    X1g, X2g = np.meshgrid(xs, ys)
    grid = np.column_stack([X1g.ravel(), X2g.ravel()])  # (grid_size^2, 2)

    # Scale
    grid_scaled = scaler.transform(grid)
    grid_t = torch.from_numpy(grid_scaled).float()

    # Posterior mean
    model.eval()
    with torch.no_grad():
        post = model(grid_t).mean.cpu().numpy()

    Z = post.reshape(grid_size, grid_size)

    # Ensure snapshot folder exists
    outdir = exp_dir / str(step+1)
    outdir.mkdir(parents=True, exist_ok=True)

    # Plot
    plt.figure(figsize=(8,6))
    cf = plt.contourf(X1g, X2g, Z, levels=levels, cmap="viridis", alpha=0.8)
    plt.colorbar(cf, label="Predicted Utility")
    plt.scatter(
        X_all[:,0], X_all[:,1],
        c="red", edgecolor="k", s=30,
        label="Training Points", zorder=3
    )
    plt.xlabel("X1 ∈ [0,1]")
    plt.ylabel("X2 ∈ [0,1]")
    plt.title("GP Posterior Mean")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "gp_latent_contour.png", dpi=150)
    plt.close()
    
    
def classification_and_ordinal_metrics_2d(
    model,
    scaler,
    utility_fn,
    gamma: float,
    grid_density: int,
    n_pairs: int,
    exp_dir: pathlib.Path,
    step: int,
):
    """
    For 2-D problems only:
      1) sample n_pairs random grid‐point pairs → get ground-truth labels under jnd_oracle
      2) predict via posterior mean + γ → compute 3-class accuracy & macro P/R/F1 + 3×3 confusion
      3) ignore γ → compute binary (↑/↓) ordinal accuracy & macro P/R/F1 + 2×2 confusion
    Writes a 'classification_metrics.txt' into exp_dir/{step+1}/ and returns a summary dict.
    """
    # build uniform grid in [0,1]^2
    u = np.linspace(0,1,grid_density)
    v = np.linspace(0,1,grid_density)
    U, V = np.meshgrid(u,v)
    grid_X = np.column_stack([U.ravel(), V.ravel()])          # (grid_density^2, 2)

    # 1) random distinct index‐pairs
    idx = np.random.randint(0, len(grid_X), size=(n_pairs*2, 2))
    # keep only pairs with distinct points, then take first n_pairs
    idx = idx[idx[:,0] != idx[:,1]][:n_pairs]

    # ground‐truth via oracle
    GT = []
    for i,j in idx:
        x_pair = torch.tensor(grid_X[[i,j]], dtype=torch.float32)
        # oracle returns a length‐1 array of labels in {-1,0,1}
        GT.append(int(jnd_oracle(x_pair, utility_fn, gamma, noise_std=0.0)[0]))
    GT = np.array(GT)

    # 2) model posterior means
    scaled = scaler.transform(grid_X)
    with torch.no_grad():
        pm = model(torch.tensor(scaled, dtype=torch.float32)).mean.cpu().numpy()

    # three‐way predictions under gamma
    P = []
    for i,j in idx:
        diff = pm[j] - pm[i]
        if diff > gamma:
            P.append(1)
        elif diff < -gamma:
            P.append(-1)
        else:
            P.append(0)
    P = np.array(P)

    # accuracy & macro‐avg P/R/F1 + confusion
    acc        = (P == GT).mean()
    cm3        = confusion_matrix(GT, P, labels=[-1,0,1])
    p3, r3, f3, _ = precision_recall_fscore_support(
        GT, P, labels=[-1,0,1], average="macro", zero_division=0
    )

    # ───── ORDINAl (ignore gamma) ─────
    # true oracle values
    with torch.no_grad():
        oracle_vals = utility_fn(torch.tensor(grid_X, dtype=torch.float32)).cpu().numpy()

    GT_ord = np.where(oracle_vals[idx[:,1]] > oracle_vals[idx[:,0]], 1, -1)
    P_ord  = np.where(pm[idx[:,1]] > pm[idx[:,0]], 1, -1)

    ord_acc   = (P_ord == GT_ord).mean()
    cm2       = confusion_matrix(GT_ord, P_ord, labels=[-1,1])
    p2, r2, f2, _ = precision_recall_fscore_support(
        GT_ord, P_ord, labels=[-1,1], average="macro", zero_division=0
    )

    # write out
    snap = exp_dir / str(step+1)
    snap.mkdir(parents=True, exist_ok=True)
    out_file = snap / "classification_metrics.txt"

    # open with utf-8
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=== Choice (gamma-threshold) Accuracy ===\n")  # plain hyphen
        f.write(f"Overall accuracy:        {acc:.4f}\n")
        f.write(f"Precision (macro):       {p3:.4f}\n")
        f.write(f"Recall (macro):          {r3:.4f}\n")
        f.write(f"F1 (macro):              {f3:.4f}\n")
        f.write("Confusion matrix (rows=true → cols=pred) [-1,0,1]:\n")
        f.write(np.array2string(cm3))
        f.write("\n\n")
        f.write("=== Ordinal (ignore gamma) Accuracy ===\n")
        f.write(f"Overall accuracy:        {ord_acc:.4f}\n")
        f.write(f"Precision (macro):       {p2:.4f}\n")
        f.write(f"Recall (macro):          {r2:.4f}\n")
        f.write(f"F1 (macro):              {f2:.4f}\n")
        f.write("Confusion matrix (rows=true → cols=pred) [-1,1]:\n")
        f.write(np.array2string(cm2))

    print(f"Saved choice/ordinal metrics to {out_file}")

    return {
        "choice_accuracy":     acc,
        "choice_precision":    p3,
        "choice_recall":       r3,
        "choice_f1":           f3,
        "ordinal_accuracy":    ord_acc,
        "ordinal_precision":   p2,
        "ordinal_recall":      r2,
        "ordinal_f1":          f2,
    }


def regret_2d(
    X_all: np.ndarray,
    model,
    utility_fn,
    scaler,
    best_simple_so_far: float,
    grid_density: int = 200,
):
    """
    Compute simple & inference regret in 2D.
    Returns (simple_regret, inference_regret, new_best_simple).
    """
    # 1) approximate true global best on a dense grid
    u = np.linspace(0, 1, grid_density)
    v = np.linspace(0, 1, grid_density)
    U, V = np.meshgrid(u, v)
    X_grid = np.column_stack([U.ravel(), V.ravel()])
    X_grid_t = torch.tensor(X_grid, dtype=torch.float32)
    with torch.no_grad():
        true_vals = utility_fn(X_grid_t).numpy()
    global_best = true_vals.max()

    # 2) simple regret on training set
    X_train_t = torch.tensor(scaler.transform(X_all), dtype=torch.float32)
    with torch.no_grad():
        pred_train = model(X_train_t).mean.numpy()
    best_train_val = utility_fn(
        torch.tensor(X_all[pred_train.argmax()], dtype=torch.float32).unsqueeze(0)
    ).item()
    new_best_simple = max(best_simple_so_far, best_train_val)
    simple_regret = 1. - new_best_simple

    # 3) inference regret on grid
    X_grid_scaled = torch.tensor(scaler.transform(X_grid), dtype=torch.float32)
    with torch.no_grad():
        pred_grid = model(X_grid_scaled).mean.numpy()
    best_grid_val = utility_fn(
        torch.tensor(X_grid[pred_grid.argmax()], dtype=torch.float32).unsqueeze(0)
    ).item()
    inference_regret = 1. - best_grid_val

    return simple_regret, inference_regret, new_best_simple


def regret_nd(
    X_all: np.ndarray,
    model,
    utility_fn,
    X_test: torch.Tensor,
    U_test: torch.Tensor,
    best_simple_so_far: float,
    scaler=None,  # unused in 6D
):
    """
    Compute simple & inference regret in DIM>2.
    Returns (simple_regret, inference_regret, new_best_simple).
    """
    # simple regret on training set
    X_all_t = torch.tensor(X_all, dtype=torch.float32)
    with torch.no_grad():
        U_train = utility_fn(X_all_t).numpy()
    new_best_simple = max(best_simple_so_far, U_train.max())
    simple_regret = 1. - new_best_simple

    # inference regret on Sobol test grid
    with torch.no_grad():
        pred_test = model(X_test).mean.numpy()
    best_idx = pred_test.argmax()
    true_best = U_test[best_idx].item()
    inference_regret = 1. - true_best

    return simple_regret, inference_regret, new_best_simple