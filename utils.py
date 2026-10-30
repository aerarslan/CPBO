import os
import utility_functions as uf
from pathlib import Path
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def save_params_to_file(file_path, params_dict):
    """
    Save the important parameters and configurations to a text file.
    
    Parameters:
        file_path (str): The path to the file where the parameters will be saved.
        params_dict (dict): A dictionary containing parameter names and their values.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w") as file:
        file.write("Model Configuration and Parameters\n")
        file.write("=" * 50 + "\n")
        
        for param_name, param_value in params_dict.items():
            file.write(f"{param_name}: {param_value}\n")
        
        file.write("=" * 50 + "\n")
        print(f"Parameters saved to {file_path}")
        

def get_utility_map():
    """
    Return a dict mapping function names to their implementations.
    """
    return {
        "branin":       uf.branin,
        "bukin6":       uf.bukin6,
        "crossintray":  uf.crossintray,
        "levy13":       uf.levy13,
        "ackley":       uf.ackley,
        "bohachevsky":  uf.bohachevsky,
        "sixhump":      uf.sixhump,
        "levy6":        uf.levy6,
        "alpine1":      uf.alpine1,
    }

def get_utility_function(name: str):
    """
    Look up a single utility by name, raising a KeyError if it doesn't exist.
    """
    mapping = get_utility_map()
    try:
        return mapping[name]
    except KeyError:
        valid = ", ".join(mapping.keys())
        raise ValueError(f"Unknown utility '{name}'. Valid names are: {valid}")
        
def make_experiment_dirs(root: str, fn_name: str, gamma: float, seed: int):
    """
    Create <root>/ and <root>/<fn_name>_γ<gamma>_exp<seed>/, return Path to exp_dir.
    """
    root = Path(root)
    exp_dir = root / f"{fn_name}_γ{gamma}_exp{seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def load_initial_pair(fn_name: str, csv_2d: str, csv_6d: str, seed: int):
    """
    Load the two starting points for a given seed from either the 2d or 6d CSV.
    Returns (X_all, dim) where X_all is a torch.FloatTensor of shape (2, dim).
    """
    is6d = fn_name in {"levy6", "alpine1"}
    csv = csv_6d if is6d else csv_2d
    df  = pd.read_csv(csv)
    row = df.loc[df["Run_Index"] == seed]
    if row.empty:
        raise ValueError(f"Seed {seed} not found in {csv}")
    row = row.iloc[0]
    dim = 6 if is6d else 2

    if is6d:
        X1 = row[[f"X1_{i}" for i in range(1, dim+1)]].values.astype(np.float32)
        X2 = row[[f"X2_{i}" for i in range(1, dim+1)]].values.astype(np.float32)
    else:
        X1 = row[["X1","Y1"]].values.astype(np.float32)
        X2 = row[["X2","Y2"]].values.astype(np.float32)

    return torch.tensor([X1, X2], dtype=torch.float32), dim

def save_experiment_params(
    exp_dir: Path,
    step: int,
    cfg,
    best_state: dict,
    training_accuracy: float,
    labels,
    X_all
):
    """
    Save a text file of all run parameters + best learned hyperparams + training accuracy.
    Writes to: exp_dir/(step+1)/model_parameters.txt
    """
    snap = exp_dir / str(step+1)
    snap.mkdir(parents=True, exist_ok=True)
    file_path = snap / "model_parameters.txt"

    params = {
        "Seed Value":                   cfg.seed_value,
        "Trainable Sigma":              cfg.bool_model_sigma,
        "Fixed Sigma Value":            cfg.fixed_model_sigma_value,
        "Trainable Gamma":              cfg.bool_model_gamma,
        "Fixed Gamma Value":            cfg.fixed_model_gamma_value,
        "Outputscale Type":             cfg.outputscale_type,
        "Fixed Outputscale Value":      cfg.fixed_outputscale_value,
        "Lengthscale Prior Trainable":  cfg.bool_lengthscale_prior,
        "Lengthscale Prior Alpha":      cfg.l_alpha,
        "Lengthscale Prior Beta":       cfg.l_beta,
        "Data Noise":                   cfg.data_sigma,
        "Data Gamma":                   cfg.data_gamma,
        "Learning Rate":                cfg.learning_rate,
        "BO Iterations":                cfg.total_steps,
        "Model Training Iterations":    cfg.num_iterations,
        "Info-Gain BO Init Steps":      cfg.info_gain_bo_init_steps,
        "Info-Gain BO Optim Steps":     cfg.info_gain_bo_optim_steps,
        "Num Inducing Points":          cfg.num_inducing_points,
        "Num Samples (ELBO)":           cfg.num_samples_training,
        # best learned hyperparams:
        "Best Gamma":                   best_state.get("gamma", "NA"),
        "Best Sigma":                   best_state.get("sigma", "NA"),
        "Best Outputscale":             best_state.get("outputscale", "NA"),
        "Best Lengthscale":             best_state.get("lengthscale", "NA"),
        # performance:,
        "Training Accuracy (%)":        f"{training_accuracy*100:.2f}",
        # raw recorded data:
        "Labels":                       labels.tolist() if hasattr(labels, "tolist") else labels,
        "Initial Points":               [tuple(x) for x in X_all],
    }

    # Write out in UTF-8
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("=== Experiment Parameters and Results ===\n\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

    print(f"Saved parameters to {file_path}")
    
def save_regret_summary(
    exp_dir: Path,
    step: int,
    inference_regret: list[float],
    simple_regret: list[float]
):
    """
    At the final BO iteration, save:
      1) inference_regret plot
      2) simple_regret plot
      3) a CSV of regret values per iteration
    
    Args:
      exp_dir: base directory for this experiment (e.g. results/branin_gamma_..._exp1)
      step:    zero‐based index of the final iteration
      inference_regret: list of inference regrets over all iterations
      simple_regret:    list of simple regrets over all iterations
    """
    snap = exp_dir / str(step+1)
    snap.mkdir(parents=True, exist_ok=True)

    # --- Inference Regret Plot ---
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(inference_regret, marker="o", linestyle="-", label="Inference Regret")
    ax.set(
        title="Inference Regret Over Iterations",
        xlabel="Iteration",
        ylabel="Inference Regret"
    )
    ax.grid(True)
    ax.legend()
    fig.savefig(snap / "inference_regret.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # --- Simple Regret Plot ---
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(simple_regret, marker="s", linestyle="-", color="orange", label="Simple Regret")
    ax.set(
        title="Simple Regret Over Iterations",
        xlabel="Iteration",
        ylabel="Simple Regret"
    )
    ax.grid(True)
    ax.legend()
    fig.savefig(snap / "simple_regret.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # --- Dump CSV ---
    out = snap / "regret_values.txt"
    with open(out, "w") as f:
        f.write("Iteration,InferenceRegret,SimpleRegret\n")
        for i, (i_r, s_r) in enumerate(zip(inference_regret, simple_regret), start=1):
            f.write(f"{i},{i_r:.6f},{s_r:.6f}\n")

    print(f"Regret values saved to {out}")