import copy
import torch
import matplotlib.pyplot as plt
import gpytorch
from likelihood import CPBO_likelihood
import pathlib

def train_loop(
    model: gpytorch.models.ApproximateGP,
    likelihood: CPBO_likelihood,
    optimizer: torch.optim.Optimizer,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    cfg,
):
    """
    Train model + likelihood for cfg.num_iterations.
    Returns (best_state, loss_history).
    """
    model.train()
    likelihood.train()

    loss_history = []
    min_loss     = float('inf')
    best_state   = {}

    for it in range(cfg.num_iterations):
        optimizer.zero_grad()

        # 1) forward & ELBO
        f_dist = model(X_train)
        log_lik = likelihood.expected_log_prob(
            f_dist, y_train, cfg.num_samples_training
        )
        kl = model.variational_strategy.kl_divergence().sum() / y_train.size(0)

        # 2) loss
        if cfg.bool_lengthscale_prior == "no":
            loss = -log_lik + kl
        else:
            prior      = model.covar_module.base_kernel.lengthscale_prior
            ls         = model.covar_module.base_kernel.lengthscale
            logp_prior = prior.log_prob(ls).sum()
            loss       = -log_lik + kl - logp_prior

        # 3) backward + step
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        loss_history.append(cur_loss)

        # 4) track best
        if cur_loss < min_loss:
            min_loss = cur_loss
            best_state = {
                "model_state_dict":      copy.deepcopy(model.state_dict()),
                "likelihood_state_dict": copy.deepcopy(likelihood.state_dict()),
                "optimizer_state_dict":  copy.deepcopy(optimizer.state_dict()),
                "gamma":       float(likelihood.gamma.item()),
                "sigma":       float(getattr(likelihood, "sigma", likelihood.fixed_sigma)),
                "outputscale": float(model.covar_module.outputscale.item()),
                "lengthscale": model.covar_module.base_kernel.lengthscale.detach().cpu().numpy(),
            }

        # 5) logging
        if it % 250 == 0 or it == cfg.num_iterations - 1:
            print(f"Iter {it+1}/{cfg.num_iterations} - Loss: {cur_loss:.6f}")
            print(f"  Gamma:       {best_state['gamma']:.6f}")
            print(f"  Sigma:       {best_state['sigma']:.6f}")
            print(f"  OutputScale: {best_state['outputscale']:.6f}")
            print(f"  Lengthscale: {best_state['lengthscale']}")

    return best_state, loss_history
