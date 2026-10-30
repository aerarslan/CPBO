import gpytorch
from torch import Tensor
from typing import Tuple

class CPBO_model(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, mean_module, covar_module):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        covar_x = covar_x.add_jitter(1e-4)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def build_cpbo_model(
    X_train: Tensor,
    inducing_points: Tensor,
    cfg
) -> gpytorch.models.ApproximateGP:

    # common mean
    mean_module = gpytorch.means.ConstantMean()

    # make base RBF (with or without lengthscale prior)
    if cfg.bool_lengthscale_prior == "yes":
        prior = gpytorch.priors.GammaPrior(cfg.l_alpha, cfg.l_beta)
        base_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims = X_train.shape[1],
            lengthscale_prior = prior
        )
    else:
        base_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims = X_train.shape[1]
        )

    # wrap in a ScaleKernel
    covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    # fixed vs trainable outputscale
    if cfg.outputscale_type == "fixed":
        covar_module.initialize(outputscale=cfg.fixed_outputscale_value)
        # freeze raw_outputscale
        covar_module.raw_outputscale.requires_grad_(False)

    # instantiate
    return CPBO_model(inducing_points, mean_module, covar_module)