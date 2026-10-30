import torch
import numpy as np

def jnd_oracle(
    X_all: torch.Tensor,
    utility_func,
    gamma: float = 0.1,
    noise_std: float = 0.01
) -> np.ndarray:
    """
    Compares consecutive points X_all[i], X_all[i+1] by their utility_func values,
    returns labels in {-1,0,1} depending on difference & gamma.
    """
    with torch.no_grad():
        # Evaluate the chosen utility function in [0,1]^2
        true_values = utility_func(X_all)

    noise = torch.randn_like(true_values) * noise_std
    noisy_values = true_values + noise

    labels_list = []
    for i in range(len(noisy_values) - 1):
        diff = noisy_values[i+1] - noisy_values[i]
        if diff > gamma:
            labels_list.append(1)
        elif diff < -gamma:
            labels_list.append(-1)
        else:
            labels_list.append(0)

    return np.array(labels_list, dtype=np.int64)