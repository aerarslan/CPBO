import numpy as np

# def sigma(y):
#     return 1 / (1 + np.exp(-y))

# def Bernoulli_sample(p):
#     return np.random.binomial(1, p)

# def pref_oracle(fx, fx_prime):
#     p = sigma(fx - fx_prime)
#     xwin = Bernoulli_sample(p)
#     return xwin

def jnd_oracle(fx, fx_prime, noise_std=0.1, gamma=0.0):
    """
    Thurstone pairwise comparison oracle with a gamma threshold applied after adding noise.
    
    Parameters:
        fx : float
            True value of the first sample.
        fx_prime : float
            True value of the second sample.
        noise_std : float, optional
            Standard deviation of the additive Gaussian noise (default is 1.0).
        gamma : float, optional
            Threshold for indecision after noise is added. If the absolute difference between the noisy values 
            is less than gamma, a random binary outcome is returned (default is 0.0).
    
    Returns:
        int: 1 if the noisy first sample is greater than the noisy second sample, or a random binary outcome
             if their difference is within the gamma threshold.
    """
    # Add Gaussian noise to both function values
    noise1 = np.random.normal(0, noise_std)
    noise2 = np.random.normal(0, noise_std)
    perceived_fx = fx + noise1
    perceived_fx_prime = fx_prime + noise2
    
    #print(f"\n\n ** {fx} ---- {fx_prime} ** \n\n")
    
    # Check if the noisy difference is within the threshold gamma
    if np.abs(perceived_fx - perceived_fx_prime) < gamma:
        return np.random.binomial(1, 0.5)
    
    # Otherwise, decide based on which noisy value is greater
    return 1 if perceived_fx > perceived_fx_prime else 0

# # Example usage:
# fx = 1.0
# fx_prime = 0.8
# result = thurstone_pref_oracle(fx, fx_prime, noise_std=0.5, gamma=0.3)
# print("Result:", result)
