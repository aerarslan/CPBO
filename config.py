from dataclasses import dataclass

@dataclass
class CPBOConfig:
    # Data‐generation settings
    seed_value: int = 1
    data_sigma: float = 0.0
    data_gamma: float = 0.04
    DIM:        int = 2
    
    # Model‐sigma settings
    bool_model_sigma: str = "no"           # yes | no
    fixed_model_sigma_value: float = 0.04  # used if bool_model_sigma=="no"

    # Model‐gamma settings
    bool_model_gamma: str = "yes"          # yes | no
    fixed_model_gamma_value: float = 0.0   # used if bool_model_gamma=="no"

    # Kernel hyperparameters
    outputscale_type: str = "fixed"    # fixed | trainable
    fixed_outputscale_value: float = 10.0

    bool_lengthscale_prior: str = "no"     # yes | no
    l_alpha: float = 2.0
    l_beta: float = 3.0

    # Optimization
    learning_rate: float = 0.01
    num_iterations: int = 2000
    total_steps: int = 30
    log_every: int = 10
    
    # Info‐gain
    info_gain_bo_init_steps: int = 10
    info_gain_bo_optim_steps: int = 25
    gpyopt_random_seed: int = 42
    grid_density: int = 100
    num_x_star_samples: int = 25000
    elbo_samples: int = 1000
    x_star_bins: int = 20
    valid_batch_size: int = 10000

    # Variational GP
    num_inducing_points: int = 100
    num_samples_training: int = 50

    # Misc
    output_root: str = "results"
    n_choice_pairs     = 10000
    
    @property
    def log_steps(self):
        # 0, 9, 19, 29, ... ≤ total_steps-1
        return [0] + [self.log_every - 1 + i*self.log_every 
                      for i in range((self.total_steps + self.log_every - 1)//self.log_every)]