import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.quasirandom import SobolEngine
#import logging
from pathlib import Path

from config import CPBOConfig
from likelihood import CPBO_likelihood
from oracle import jnd_oracle
from cli import parse_args
import utils
from model import build_cpbo_model
from train import train_loop
import evaluation
import acquisition
# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

#logging.basicConfig(level=logging.INFO, format="%(message)s")
args = parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────────────────────────────────────
cfg = CPBOConfig()
# override with any CLI flags
cfg.seed_value                = args.seed_value
cfg.data_gamma                = args.data_gamma
cfg.bool_model_gamma          = args.bool_model_gamma
cfg.num_iterations            = args.model_iteration
cfg.info_gain_bo_init_steps   = args.bo_init
cfg.info_gain_bo_optim_steps  = args.bo_optim
cfg.total_steps               = args.total_steps

# ─────────────────────────────────────────────────────────────────────────────
# Parse Utility Function
# ─────────────────────────────────────────────────────────────────────────────

# UTILITY_FUNCTIONS = utils.get_utility_map()
function_name = args.function_name
utility_func = utils.get_utility_function(args.function_name)

# ─────────────────────────────────────────────────────────────────────────────
# Set seed
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(cfg.seed_value)
np.random.seed(cfg.seed_value)

# ─────────────────────────────────────────────────────────────────────────────
# Directory
# ─────────────────────────────────────────────────────────────────────────────
# instead just do:
exp_dir = Path(cfg.output_root) / f"{function_name}_gamma_{cfg.data_gamma}_exp{cfg.seed_value}"
exp_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load initial data
# ─────────────────────────────────────────────────────────────────────────────
X_all, DIM = utils.load_initial_pair(
    function_name,
    csv_2d="start_points/2d_start_points.csv",
    csv_6d="start_points/6d_start_points.csv",
    seed=cfg.seed_value,
)

cfg.DIM = DIM

if cfg.DIM == 6:
    # Define true utility space
    TEST_SIZE = 25000
    sobol = SobolEngine(dimension=cfg.DIM, scramble=False)
    X_test = sobol.draw(TEST_SIZE).to(torch.float32)
    with torch.no_grad():
        U_test = utility_func(X_test)

# ─────────────────────────────────────────────────────────────────────────────
# First label
# ─────────────────────────────────────────────────────────────────────────────
labels = jnd_oracle(
    X_all,
    utility_func=utility_func,
    gamma=cfg.data_gamma,
    noise_std=cfg.data_sigma,
)

# ─────────────────────────────────────────────────────────────────────────────
# History holders
# ─────────────────────────────────────────────────────────────────────────────
add_manual_data       = []
entropy_info_list     = []
inference_regret_list = []
simple_regret_list    = []
best_simple_reg_so_far    = float('-inf')

#%% TRAIN

# Compute the steps at which you want to snapshot (and make sub-dirs)
log_steps = {s for s in cfg.log_steps if s < cfg.total_steps}

# Prepare experiment directory
exp_dir = Path(cfg.output_root) / f"{function_name}_gamma_{cfg.data_gamma}_exp{cfg.seed_value}"
exp_dir.mkdir(parents=True, exist_ok=True)

for current_step in range(cfg.total_steps):
    print(f"BO iteration {current_step+1}/{cfg.total_steps} starting…")

    # Make sub-dir for this BO iteration
    if current_step in log_steps  or (current_step == cfg.total_steps - 1):
        (exp_dir / str(current_step+1)).mkdir(parents=True, exist_ok=True)

    # Data normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)          # X_all: ndarray (N × dim)
    X_train  = torch.from_numpy(X_scaled).float()   # Tensor (N × dim)
    y_train  = torch.from_numpy(labels).float()     # Tensor (N,)

    # Select inducing points
    idx = torch.randperm(X_train.size(0))[:cfg.num_inducing_points]
    inducing_points = X_train[idx]

    # Build model & likelihood & optimizer
    model      = build_cpbo_model(X_train, inducing_points, cfg)
    likelihood = CPBO_likelihood(
        bool_model_sigma=cfg.bool_model_sigma,
        fixed_model_sigma_value=cfg.fixed_model_sigma_value,
        bool_model_gamma=cfg.bool_model_gamma,
        fixed_model_gamma_value=cfg.fixed_model_gamma_value,
    )
    params = [{"params": model.parameters()},
              {"params": likelihood.parameters()}]
    optimizer = torch.optim.Adam(params, lr=cfg.learning_rate)

    # Train loop
    best_state, losses = train_loop(model, likelihood, optimizer, X_train, y_train, cfg)

    if current_step in log_steps  or (current_step == cfg.total_steps - 1):
        snap = exp_dir / str(current_step+1)
        snap.mkdir(exist_ok=True)
        # plot losses
        fig, ax = plt.subplots()
        ax.plot(losses, marker=None)
        ax.set(xlabel="Iteration", ylabel="ELBO loss", title="Training Loss")
        ax.grid(True, linestyle="--", alpha=0.3)
        fig.savefig(snap / "loss_plot.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        # save checkpoint
        torch.save(best_state, snap / "model_and_params.pth")
    
    #%% TRAINING ACCURACY
    
    model.load_state_dict(best_state["model_state_dict"])
    likelihood.load_state_dict(best_state["likelihood_state_dict"])
    gamma_value = float(likelihood.gamma.item())
    
    acc, y_true, y_pred = evaluation.simple_training_acc(model, likelihood, X_train, y_train)
    print(f"Training accuracy: {acc*100:.2f}%")
    
    #%% Latent surface creation
    
    if current_step in log_steps or (current_step == cfg.total_steps - 1):
        evaluation.plot_gp_latent_surface(
            model=model,
            scaler=scaler,
            X_all=X_all,
            exp_dir=exp_dir,
            step=current_step,
            dim=cfg.DIM,
            grid_size=50
        )
    
    #%% Log save
    
    if current_step in log_steps or (current_step == cfg.total_steps - 1):
        utils.save_experiment_params(
            exp_dir=exp_dir,
            step=current_step,
            cfg=cfg,
            best_state=best_state,
            training_accuracy=acc,
            labels=labels,
            X_all=X_all
        )
    
    #%% Acquisition
    
    best_candidate, best_info_gain = acquisition.run_info_gain_bo(
        model=model,
        likelihood=likelihood,
        scaler=scaler,
        X_train=X_train,
        calculate_info_gain=acquisition.calculate_info_gain,
        cfg=cfg,
        gamma_value=gamma_value,
        X_test=(X_test if cfg.DIM > 2 else None),
    )
    
    entropy_info_list.append(best_info_gain)
    print(f"Best candidate: {best_candidate}, Best info_gain: {best_info_gain:.4f}")
    
    #%% Expand data
    
    # Add the newly selected candidate to history
    X_all = np.vstack([X_all, best_candidate])
    add_manual_data.append(best_candidate)
    
    # Build the last‐two‐points tensor for the oracle
    last_two = torch.tensor(X_all[-2:], dtype=torch.float32)  # shape (2, DIM)
    
    # Query the oracle on that last pair
    new_label = jnd_oracle(
        last_two,
        utility_func=utility_func,
        gamma=cfg.data_gamma,
        noise_std=cfg.data_sigma
    )[0]
    
    # Append the new label
    labels = np.append(labels, new_label)
    
    
#%% Calculate regret
    
    if cfg.DIM == 2:
        simple, infer, best_simple_reg_so_far = evaluation.regret_2d(
            X_all, model, utility_func, scaler, best_simple_reg_so_far, grid_density=cfg.grid_density
        )
    else:
        simple, infer, best_simple_reg_so_far = evaluation.regret_nd(
            X_all, model, utility_func, X_test, U_test, best_simple_reg_so_far
        )
    
    simple_regret_list.append(simple)
    inference_regret_list.append(infer)
    
    print(f"Simple Regret:    {simple:.4f}")
    print(f"Inference Regret: {infer:.4f}")

#%% Calculate accuracy (choice and ordinal)

    # only for 2-D problems and at snapshot steps:
    if cfg.DIM == 2 and (current_step in log_steps or current_step == cfg.total_steps - 1):
        metrics = evaluation.classification_and_ordinal_metrics_2d(
            model=model,
            scaler=scaler,
            utility_fn=utility_func,
            gamma=cfg.data_gamma,
            grid_density=cfg.grid_density,
            n_pairs=cfg.n_choice_pairs,
            exp_dir=exp_dir,
            step=current_step,
        )
        print(f"Choice‐accuracy: {metrics['choice_accuracy']*100:.2f}%  |  Ordinal‐accuracy: {metrics['ordinal_accuracy']*100:.2f}%")


#%% Save results

    if current_step == cfg.total_steps - 1:
        utils.save_regret_summary(
            exp_dir=exp_dir,
            step=current_step,
            inference_regret=inference_regret_list,
            simple_regret=simple_regret_list
        )