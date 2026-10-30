import torch
from torch.distributions import Normal
from gpytorch.likelihoods import Likelihood

class CPBO_likelihood(Likelihood):
    def __init__(self, bool_model_sigma="no", fixed_model_sigma_value=1.0, bool_model_gamma="no", fixed_model_gamma_value = 1.0):
        super().__init__()        
        
        self.bool_model_gamma = bool_model_gamma
        self.bool_model_sigma = bool_model_sigma
        
        if bool_model_gamma == "yes":
            # Initialize sigma as a trainable parameter
            self.register_parameter(name="raw_gamma", parameter=torch.nn.Parameter(torch.randn(1)))
        else:
            # Set sigma as a fixed value (non-trainable)
            self.fixed_gamma = torch.tensor(fixed_model_gamma_value)
        
        if bool_model_sigma == "yes":
            self.register_parameter(name="raw_sigma", parameter=torch.nn.Parameter(torch.randn(1)))
        else:
            self.fixed_sigma = torch.tensor(fixed_model_sigma_value)
        
    @property
    def gamma(self):
        if self.bool_model_gamma == "yes": 
            return torch.exp(self.raw_gamma)
        else:
            return self.fixed_gamma
    
    @property
    def sigma(self):
        if self.bool_model_sigma == "yes":
            return torch.exp(self.raw_sigma)
        else:
            return self.fixed_sigma
        
    def forward(self, sample, y):
        sigma_parameter = self.sigma
        combined_sigma = torch.sqrt(torch.tensor(2)) * sigma_parameter
        normal_dist = Normal(torch.tensor(0.0), combined_sigma)
        gamma = self.gamma # maybe some modifications here
        
        prediction_x = sample[:-1]
        prediction_x_prime = sample[1:]
        
        # Calculate probabilities for each label
        prob_y_minus_1 = normal_dist.cdf(prediction_x - prediction_x_prime - gamma)
           
        # P(∣x−x′∣<γ)=Φ(γ/σ​)−Φ(−γ/σ​), P(γ>∣f(x′)−f(x)∣)
        prob_y_0 = torch.clamp(normal_dist.cdf(prediction_x - prediction_x_prime + gamma) - \
                                normal_dist.cdf(prediction_x - prediction_x_prime - gamma), min=0)
            
        prob_y_1 = normal_dist.cdf(prediction_x_prime - prediction_x - gamma)
        
        # Ensure y is a flat vector
        y_flat = y.squeeze()
        
        eps = 1e-5
        
        # Calculate log-probabilities (apply log to each probability safely)
        log_prob_y_minus_1 = torch.log(prob_y_minus_1 + eps)
        log_prob_y_0 = torch.log(prob_y_0 + eps)
        log_prob_y_1 = torch.log(prob_y_1 + eps)
        
        # Use torch.where to select the appropriate log-probability
        log_likelihood = torch.where(
            y_flat == -1, log_prob_y_minus_1, 
            torch.where(y_flat == 0, log_prob_y_0, log_prob_y_1)
        )
        
        # Sum the log-likelihoods to get total log-likelihood
        total_log_likelihood = log_likelihood.sum()

        return total_log_likelihood

    def expected_log_prob(self, function_dist, observations, num_samples_training, *args, **kwargs):        
        # Generate samples from the function_dist, which is a distribution over functions
        samples = function_dist.rsample(torch.Size([num_samples_training]))
        
        log_likelihoods = []
        for i in range(num_samples_training):
            # Extract the i-th sample across all data points
            sample = samples[i]
            
            # Calculate log likelihood for the i-th sample
            log_likelihood = self.forward(sample, observations, **kwargs)
            
            # see pytorchGPU\Lib\site-packages\gpytorch\mlls\_approximate_mll.py forward function
            normalized_log_likelihood = log_likelihood / (observations.size(0))  # Assuming observations are N-1 and func_dist is N
            log_likelihoods.append(normalized_log_likelihood)
    
        # Stack the log likelihoods and compute the mean across the first dimension (samples)
        log_likelihoods = torch.stack(log_likelihoods)
        expected_log_likelihood = log_likelihoods.mean(dim=0)
    
        return expected_log_likelihood 