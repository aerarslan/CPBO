import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import norm
import time
from torch.distributions import Normal
import GPyOpt

def run_info_gain_bo(
    model,
    likelihood,
    scaler,
    X_train: torch.Tensor,
    calculate_info_gain,
    cfg,
    gamma_value: float,
    X_test: np.ndarray = None,  # only needed for DIM > 2
):
    """
    Runs a GPyOpt BO loop to maximize information gain.
    Returns (best_candidate, best_info_gain) as numpy.ndarray and float.
    """
    def objective_function(x):
        cand = x[0]
        info_gain, _, _ = calculate_info_gain(
            next_candidate=cand,
            model=model,
            likelihood=likelihood,
            scaler=scaler,
            X_train=X_train,
            cfg=cfg,
            gamma_value=gamma_value,
            X_test=X_test
        )
        # make sure it’s a Python float
        return -float(info_gain)

    # domain
    if cfg.DIM == 2:
        bounds = [
            {'name': 'x1', 'type': 'continuous', 'domain': (0.0, 1.0)},
            {'name': 'x2', 'type': 'continuous', 'domain': (0.0, 1.0)},
        ]
    else:
        bounds = [
            {'name': f'x{i}', 'type': 'continuous', 'domain': (0.0, 1.0)}
            for i in range(cfg.DIM)
        ]

    bo = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=bounds,
        acquisition_type='EI',
        initial_design_numdata=cfg.info_gain_bo_init_steps,
        random_state=cfg.gpyopt_random_seed,
        verbose=False,
    )
    bo.run_optimization(max_iter=cfg.info_gain_bo_optim_steps)

    best_candidate = bo.x_opt
    best_info_gain = float(-bo.fx_opt)
    return best_candidate, best_info_gain

def calculate_info_gain(
    next_candidate: np.ndarray,
    model,
    likelihood,
    scaler,
    X_train: torch.Tensor,
    cfg,
    gamma_value: float,
    X_test: np.ndarray = None,
):
    """
    Calculates information gain for a single candidate using all the settings in cfg.
    Returns (info_gain, full_input_batch, feature_ranges).
    """
    # unpack everything from cfg
    DIM               = cfg.DIM
    grid_density      = cfg.grid_density
    num_x_star_samples= cfg.num_x_star_samples
    elbo_samples      = cfg.elbo_samples
    x_star_bins       = cfg.x_star_bins
    batch_size        = cfg.valid_batch_size

    # 1) ensure shape (1,DIM)
    x_cand = np.array(next_candidate).reshape(1, -1)
    # 2) scale & to-tensor
    x_cand_scaled = scaler.transform(x_cand)
    x_cand_t      = torch.tensor(x_cand_scaled, dtype=torch.float32)

    # 3) last observed state
    x_prev = X_train[-1].unsqueeze(0)

    # 4) unconditional entropy
    H_Y_t = calculate_unconditional_entropy(
        model, x_prev, x_cand_t, gamma_value, likelihood.sigma, num_samples=elbo_samples
    )

    # 5) build feature grid
    if DIM == 2:
        x1 = np.linspace(0, 1, grid_density)
        x2 = np.linspace(0, 1, grid_density)
        X1g, X2g = np.meshgrid(x1, x2)
        feature_grid = np.stack([X1g.ravel(), X2g.ravel()], axis=1)
        feature_ranges = (x1, x2)
    else:
        if X_test is None:
            raise ValueError("X_test must be provided when DIM>2")
        feature_grid = X_test
        feature_ranges = None

    # 6) scale & assemble full batch
    fg_scaled     = scaler.transform(feature_grid)
    full_input_np = np.vstack([fg_scaled, x_prev.numpy(), x_cand_scaled])
    full_input    = torch.tensor(full_input_np, dtype=torch.float32)

    # 7) sample x* & histogram weights
    x_stars, _ = generate_x_stars(model, full_input, num_x_star_samples)
    counts, edges = np.histogram(x_stars, bins=x_star_bins)
    weights       = counts / counts.sum()
    mids          = 0.5 * (edges[:-1] + edges[1:])
    x_star_mids   = torch.tensor(mids)
    x_star_weights= torch.tensor(weights)

    # 8) generate valid distributions
    valid = generate_all_valid_distributions_sequential(
        model,
        torch.cat([x_prev, x_cand_t], dim=0),
        x_star_mids,
        num_samples=elbo_samples,
        batch_size=batch_size,
    )

    # 9) conditional entropy
    H_Y_t_given = calculate_conditional_entropy(
        model, valid, gamma_value, likelihood.sigma, x_star_weights
    )

    # 10) info gain
    info_gain = H_Y_t - H_Y_t_given
    return info_gain, full_input, feature_ranges

def calculate_unconditional_entropy(model, current_state_tensor, candidate_point_tensor, gamma_value, likelihood_sigma, num_samples=1000):
    # Set the GP model to evaluation mode
    model.eval()

    # Combine the two points (current state and candidate point) into a batch
    full_input_batch = torch.cat([current_state_tensor, candidate_point_tensor], dim=0)

    with torch.no_grad():
        # Predict the posterior for the two points
        posterior = model(full_input_batch)
        samples = posterior.rsample(torch.Size([num_samples]))  # Sample outcomes for the batch of two points

        # Calculate the mean of each distribution (each row)
        means = samples.mean(dim=1, keepdim=True)
        
        # Subtract the mean from each distribution to make it zero mean
        samples = samples - means

        prediction_x = samples[:, 0]
        prediction_x_prime = samples[:, 1]

        # Define the normal distribution with combined noise
        combined_sigma = torch.sqrt(torch.tensor(2.0)) * likelihood_sigma
        normal_dist = torch.distributions.Normal(torch.tensor(0.0), combined_sigma)

        # Calculate probabilities for each label (improvement, no change, worsening)
        p_y_eq_minus_1 = normal_dist.cdf(prediction_x - prediction_x_prime - gamma_value).mean(dim=0)
        p_y_eq_0 = torch.clamp(normal_dist.cdf(prediction_x - prediction_x_prime + gamma_value) -
                               normal_dist.cdf(prediction_x - prediction_x_prime - gamma_value), min=0).mean(dim=0)
        p_y_eq_1 = normal_dist.cdf(prediction_x_prime - prediction_x - gamma_value).mean(dim=0)

        # Clamping all probabilities to avoid very small values
        eps = 1e-5
        p_y_eq_minus_1 = torch.clamp(p_y_eq_minus_1, min=eps)
        p_y_eq_0 = torch.clamp(p_y_eq_0, min=eps)
        p_y_eq_1 = torch.clamp(p_y_eq_1, min=eps)

        # Calculate unconditional entropy H(Y_t)
        H_Y_t = -(p_y_eq_1 * torch.log(p_y_eq_1) +
                  p_y_eq_0 * torch.log(p_y_eq_0) +
                  p_y_eq_minus_1 * torch.log(p_y_eq_minus_1))

    return H_Y_t


def generate_x_stars(model, full_input_batch, num_samples):
    """
    Generate a list of x* values using Gumbel sampling from the GP posterior.
    This function uses a pre-prepared batch that includes both the feature grid and the latest known configuration.

    Parameters:
    - model: The trained model used to sample from the posterior.
    - full_input_batch: torch.Tensor, pre-prepared batch including the feature grid and the latest configuration.
    - num_samples: int, the number of x* values to generate using Gumbel sampling.
    
    Returns:
    - x_stars: torch.Tensor, a tensor containing x* values, each sampled from the Gumbel distribution.
    """
    with torch.no_grad():
        # Predict the posterior for the pre-prepared input batch
        posterior = model(full_input_batch)
        # Sample outcomes for the entire batch
        samples = posterior.rsample(torch.Size([1000]))
        
        # Calculate the mean of each sample
        sample_means = samples.mean(dim=1, keepdim=True)
        
        # Zero-mean each sample individually
        zero_mean_samples = samples - sample_means
        
        # Calculate the new mean and std from the zero-meaned samples
        new_mean = zero_mean_samples.mean(dim=0).numpy()
        new_std = zero_mean_samples.std(dim=0).numpy()
        
        # Use Brent's method to find the empirical quantiles more accurately
        def find_quantile(p, mu, sigma):
            lo = (mu - 3 * sigma).min()
            hi = (mu + 5 * sigma).max()
            N = norm(loc=mu, scale=sigma)
            return brentq(lambda y: np.exp(np.sum(N.logcdf(y))) - p, lo, hi)

        # Approximate percentiles for r1 = 0.25 and r2 = 0.75
        r1, r2, r3 = 0.25, 0.75, 0.50

        # Calculate the empirical percentiles y1 and y2 from the zero-mean adjusted GP predictions
        y1 = torch.tensor(find_quantile(r1, new_mean, new_std)) # q25
        y2 = torch.tensor(find_quantile(r2, new_mean, new_std)) # q50
        y3 = torch.tensor(find_quantile(r3, new_mean, new_std)) # q75

        # Calculate b
        b = (y1 - y2) / (torch.log(-torch.log(torch.tensor(r2))) - torch.log(-torch.log(torch.tensor(r1))))

        # Calculate a
        a = y3 + b * torch.log(-torch.log(torch.tensor(r3)))

        x_stars = []
        r_values = []
        stabilizer_value = 0.01
        for _ in range(num_samples):
            # Sample uniformly from [0.01, 0.99] for numerical stability
            r = torch.FloatTensor(1).uniform_(0.0 + stabilizer_value, 1.0 - stabilizer_value).item()
            r_values.append(r)
            # Sample x* using the Gumbel distribution parameters a and b
            r_tensor = torch.tensor(r)
            x_star = a - b * r_tensor.log().mul(-1.0).log()
            x_stars.append(x_star.item())
        
        x_stars = torch.tensor(x_stars)

    return x_stars, r_values


def generate_valid_distributions(model, input_batch_2_points, x_star, num_samples, batch_size):
    """
    Generate valid distributions for a given x* such that x* is greater than f(x_1) and f(x_2) 
    (i.e., the two points being compared), by initially sampling a large batch and filtering valid samples.

    Parameters:
    - model: The trained model used to sample from the posterior.
    - input_batch_2_points: torch.Tensor, batch including the two points being compared.
    - x_star: float, the specific x* value for which to generate valid distributions.
    - num_samples: int, number of valid distributions to generate for the given x*.
    - batch_size: int, number of samples to generate in each batch.

    Returns:
    - valid_samples: torch.Tensor, tensor containing valid distributions where x* is greater than both f(x_1) and f(x_2).
    """
    valid_samples = []  # Initialize the list to store valid samples
    samples_collected = 0  # Counter for the number of valid samples collected

    start_time = time.time()
    
    while samples_collected < num_samples:
        if time.time() - start_time > 30: # ! BREAK TIME LIMIT ! PUHTI
            print(f"Timeout exceeded while generating valid distributions for x_star: {x_star}")
            break
        
        with torch.no_grad():
            # Sample a large batch from the model's posterior using the input batch of the two points
            samples = model(input_batch_2_points).rsample(torch.Size([batch_size]))
            
            # Subtract the mean from each distribution to make it zero mean
            samples = samples - samples.mean(dim=1, keepdim=True)
            
            # Extract the function values at the two points (x_1 and x_2)
            f_x1 = samples[:, 0]  # Function value at point 1
            f_x2 = samples[:, 1]  # Function value at point 2
            
            # Calculate the maximum function value between the two points for each sample
            max_values = torch.max(f_x1, f_x2)
            
            # Filter samples where x_star is greater than or equal to the maximum of the two function values
            valid_indices = max_values <= x_star
            valid_batch = samples[valid_indices]
            
            # Collect valid samples and concatenate only once after the loop
            valid_samples.append(valid_batch)
            samples_collected += valid_batch.size(0)

    # Concatenate all valid samples in one go and slice to the exact num_samples if oversampled
    valid_samples = torch.cat(valid_samples, dim=0)
    
    # If no valid sample was found at all, raise an error.
    if valid_samples.size(0) == 0:
        sample_dim = valid_samples.shape[1] if valid_samples.dim() > 1 and valid_samples.shape[1] > 0 else 2  # default dimension (adjust if needed)
        valid_samples = torch.zeros(num_samples, sample_dim)
    
    # If we didn't collect enough samples, pad the tensor until it reaches num_samples rows
    if valid_samples.size(0) < num_samples:
        pad_count = num_samples - valid_samples.size(0)
        # Repeat the last valid sample to pad the tensor.
        pad_tensor = valid_samples[-1:].repeat(pad_count, 1)
        valid_samples = torch.cat([valid_samples, pad_tensor], dim=0)
    else:
        # Otherwise, just take the first num_samples rows
        valid_samples = valid_samples[:num_samples]
    
    return valid_samples


# Function for processing a single x_star (for parallel execution)
def process_x_star(model, input_batch_2_points, x_star, num_samples, batch_size):
    return generate_valid_distributions(model, input_batch_2_points, x_star, num_samples, batch_size)

def generate_all_valid_distributions_sequential(model, input_batch_2_points, x_stars, num_samples, batch_size):
    all_valid_distributions = []  # List to store valid distributions for all x*

    # Loop through each x_star and generate valid distributions
    for idx, x_star in enumerate(x_stars):
        # No threading, just directly call the function to generate valid distributions
        valid_distributions = generate_valid_distributions(model, input_batch_2_points, x_star, num_samples, batch_size)
        all_valid_distributions.append(valid_distributions)  # Append the valid distributions

    # Stack all the distributions for all x* if needed
    all_valid_distributions = torch.stack(all_valid_distributions)

    return all_valid_distributions

def calculate_conditional_entropy(model, all_valid_distributions, gamma_value, likelihood_sigma, x_star_weights):
    num_x_stars = all_valid_distributions.size(0)
    num_samples = all_valid_distributions.size(1)
    num_features = all_valid_distributions.size(2) - 1  # remove current state

    # Calculate the combined sigma
    combined_sigma = torch.sqrt(torch.tensor(2.0)) * likelihood_sigma
    normal_dist = Normal(torch.tensor(0.0), combined_sigma)

    # Expected entropy storage for each feature
    expected_entropies = torch.zeros(num_features)

    # Vectorize the process for all x_stars
    for i in range(num_x_stars):
        distributions = all_valid_distributions[i]  # Shape [1000, 31] for 1000 valid distributions
        
        # Skip this x_star if all values in distributions are zeros
        if torch.all(distributions == 0): # ! PUHTI
            print("\n\n\n\n\n************ Y STAR SKIPPED ****************\n\n\n\n\n")
            continue
        
        # Current state (last in each distribution)
        prediction_x = distributions[:, 0].unsqueeze(1)  # Shape: [1000, 1]

        # Candidate points (all except the last)
        prediction_x_prime = distributions[:, 1].unsqueeze(1)  # Shape: [1000, num_features]

        # Calculate probabilities in one step for all samples (no loop over j)
        p_y_eq_minus_1 = normal_dist.cdf(prediction_x - prediction_x_prime - gamma_value)
        p_y_eq_0 = torch.clamp(normal_dist.cdf(prediction_x - prediction_x_prime + gamma_value) -
                               normal_dist.cdf(prediction_x - prediction_x_prime - gamma_value), min=0)
        p_y_eq_1 = normal_dist.cdf(prediction_x_prime - prediction_x - gamma_value)

        # Clamping all probabilities to avoid very small values
        eps = 1e-5
        p_y_eq_minus_1 = torch.clamp(p_y_eq_minus_1, min=eps)
        p_y_eq_0 = torch.clamp(p_y_eq_0, min=eps)
        p_y_eq_1 = torch.clamp(p_y_eq_1, min=eps)

        # Vectorized entropy calculation for all samples
        entropy = -(p_y_eq_1 * torch.log(p_y_eq_1) +
                    p_y_eq_0 * torch.log(p_y_eq_0) +
                    p_y_eq_minus_1 * torch.log(p_y_eq_minus_1))

        # Average entropy over all samples for this x_star
        mean_entropy_for_x_star = entropy.mean(dim=0)
        
        expected_entropies += x_star_weights[i] * mean_entropy_for_x_star
        
    return expected_entropies