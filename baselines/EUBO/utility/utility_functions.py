import torch
import numpy as np

def bohachevsky(X: torch.Tensor) -> torch.Tensor:
    """
    Negative-normalized Bohachevsky function on [0,1]^2 -> [0,1].
    We use the form:
      f(x,y) = x^2 + 2y^2 - 0.3 cos(3πx) - 0.4 cos(4πy) + 0.7
    on the domain (x,y) in [-100,100]^2.

    Global min: f_min = 0 at (x,y) = (0,0).
    We pick f_max ~ 30000 as a rough upper bound in [-100,100]^2.
    
    Utility in [0,1], with 1 at the global minimum.
    """
    # Convert to double precision
    X = X.to(torch.float64)

    # 1) Map [0,1]^2 -> [-100,100]^2
    x = -100.0 + 200.0 * X[..., 0]
    y = -100.0 + 200.0 * X[..., 1]

    # 2) Compute the Bohachevsky function
    f_val = x*x + 2.0*y*y \
            - 0.3*torch.cos(3.0*np.pi*x) \
            - 0.4*torch.cos(4.0*np.pi*y) \
            + 0.7

    # 3) Negative-normalize => best = 1 at f_min = 0, worst = 0 at f_max ~30000
    f_min = 0.0
    f_max = 30000.0
    u = (f_max - f_val) / (f_max - f_min)

    # 4) Clamp numerically into [0,1]
    return u.clamp_(0.0, 1.0)

def ackley(X: torch.Tensor) -> torch.Tensor:
    """
    Negative-normalized 2D Ackley function on [0,1]^2.

    Domain: (x, y) ∈ [-5, 5]^2.

    Classic Ackley (with a=20, b=0.2, c=2π):
      f(x, y) = -a * exp(-b * sqrt(0.5*(x^2 + y^2)))
                - exp(0.5*(cos(c*x) + cos(c*y)))
                + a + e

    Global minimum => f_min = 0 at (x,y) = (0,0).
    We'll pick f_max = 22 as a "worst" bound in [-5,5]^2.

    Mapping [0,1]^2 -> [-5,5]^2:
      x = -5 + 10*X[...,0]
      y = -5 + 10*X[...,1]

    Negative normalization => best=1 at f_min=0, worst=0 at f_max=22:
      u(x) = (f_max - f_val) / (f_max - f_min).
    """
    # Ensure double precision
    X = X.to(torch.float64)

    # 1) Map [0,1]^2 to [-5,5]^2
    x = -5.0 + 10.0 * X[..., 0]
    y = -5.0 + 10.0 * X[..., 1]

    # 2) Compute the classic Ackley function in 2D
    a = 20.0
    b = 0.2
    c = 2.0 * np.pi

    # Sub-terms
    sum_sq = x*x + y*y
    term1 = -a * torch.exp(-b * torch.sqrt(0.5 * sum_sq))
    term2 = -torch.exp(0.5 * (torch.cos(c*x) + torch.cos(c*y)))

    f_val = term1 + term2 + a + np.e  # Ackley formula

    # 3) Negative-normalize => best=1 at f_min=0, worst=0 at f_max=22
    f_min = 0.0
    f_max = 22.0
    u = (f_max - f_val) / (f_max - f_min)

    # 4) Clamp to [0,1]
    return u.clamp_(0.0, 1.0)

def branin(X: torch.Tensor) -> torch.Tensor:
    """
    Negative (normalized) Branin-Hoo function on [0,1]^2.
    Returns values in [0,1], where 1 corresponds to the global min of the
    original Branin (~0.398) and 0 corresponds to the max (~308) in domain.

    Normalization: u = (f_max - f_val)/(f_max - f_min)
        => best=1 at f_val=f_min ~0.398
        => worst=0 at f_val=f_max ~308

    Domain: [0,1]^2 mapped to x1 in [-5,10], x2 in [0,15].
    """
    # Ensure double precision if you want
    X = X.to(torch.float64)

    # Rescale to actual Branin domain
    x1 = -5.0 + 15.0 * X[..., 0]
    x2 =  0.0 + 15.0 * X[..., 1]

    # Branin constants
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0/(8.0*np.pi)

    # Original Branin f(x)
    f_val = a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1.-t)*torch.cos(x1) + s

    # Known approximate bounds
    f_min = 0.397887
    f_max = 308.0

    # Negative-normalized => best=1 at global min, worst=0 at max
    u = (f_max - f_val)/(f_max - f_min)
    return u.clamp_(0.0, 1.0)

def crossintray(X: torch.Tensor) -> torch.Tensor:
    """
    'Positive' Cross-in-Tray on [-10, 10]^2, with a global MAX ~ 2.06261.
    We map [0,1]^2 -> [-10,10]^2, compute f(x,y) in double precision, then
    normalize so that f_min -> 0 in utility, f_max -> 1 in utility.

    The function is:
        f(x,y) = 1e-4 * (| sin(x)*sin(y)*exp(|100 - r/pi|) | + 1)^0.1
        where r = sqrt(x^2 + y^2).

    Because we multiply by +1e-4 (rather than -1e-4), this version
    *maximizes* at ~2.06261, not -2.06261.
    """
    # Ensure double precision to avoid overflow from exp(100) in float32
    X = X.to(torch.float64)

    # Map from [0,1]^2 to [-10,10]^2
    x = -10.0 + 20.0 * X[..., 0]
    y = -10.0 + 20.0 * X[..., 1]

    # Compute the "positive" Cross-in-Tray
    r = torch.sqrt(x*x + y*y)
    inside = torch.sin(x) * torch.sin(y) * torch.exp(torch.abs(100.0 - r / np.pi))
    f_xy = 1e-4 * (torch.abs(inside) + 1.0) ** 0.1

    # Empirical bounds in [-10,10]^2 for the positive version
    f_max = 2.06261  # global maximum
    f_min = 0.0      # near-min is close to zero

    # Normalize into [0,1], best=1 at f_max
    u = (f_xy - f_min) / (f_max - f_min)

    # Clamp to protect against minor numerical overshoot
    return u.clamp_(0.0, 1.0)

def bukin6(X: torch.Tensor) -> torch.Tensor:
    """
    Normalized Bukin Function N.6 on [0,1]^2.
    ...
    """
    # Map [0,1]^2 → [-15, -5] × [-3, 3]
    x1 = -15.0 + 10.0 * X[..., 0]
    x2 =  -3.0 +  6.0 * X[..., 1]

    # Bukin-6 function
    f_val = 100.0 * torch.sqrt(torch.abs(x2 - 0.01 * x1**2)) \
            + 0.01 * torch.abs(x1 + 10.0)

    # Known global min/max
    f_min = 0.0
    f_max = 230.0

    # Normalize and clamp
    u = (f_max - f_val) / (f_max - f_min)
    return u.clamp_(0.0, 1.0)

def levy13(X: torch.Tensor) -> torch.Tensor:
    """
    Negative-normalized Lévy Function N.13 on [0,1]^2.
    Domain: x,y ∈ [-10,10].
    
    f(x,y) = sin^2(3π x)
           + (x-1)^2 [1 + sin^2(3π y)]
           + (y-1)^2 [1 + sin^2(2π y)]
    Global min = 0 at (1,1).

    We map [0,1]^2 -> [-10,10]^2:
      x = -10 + 20 * X[...,0]
      y = -10 + 20 * X[...,1]

    Negative normalization => best=1 at f_min=0, worst=0 at f_max=~500
      u = (f_max - f_val)/(f_max - f_min).
    """
    # Use double precision for accuracy
    X = X.to(torch.float64)

    # Map [0,1]^2 -> [-10,10]^2
    x = -10.0 + 20.0 * X[..., 0]
    y = -10.0 + 20.0 * X[..., 1]

    # Compute Lévy N.13
    # sin^2(3πx)
    term1 = torch.sin(3.0 * np.pi * x)**2
    # (x-1)^2 [1 + sin^2(3πy)]
    term2 = (x - 1.0)**2 * (1.0 + torch.sin(3.0 * np.pi * y)**2)
    # (y-1)^2 [1 + sin^2(2πy)]
    term3 = (y - 1.0)**2 * (1.0 + torch.sin(2.0 * np.pi * y)**2)

    f_val = term1 + term2 + term3

    # f_min=0 at (1,1), pick a rough f_max=500 for [-10,10]^2
    f_min = 0.0
    f_max = 500.0

    u = (f_max - f_val) / (f_max - f_min)
    return u.clamp_(0.0, 1.0)

def sixhump(X: torch.Tensor) -> torch.Tensor:
    """
    Six-Hump Camel function on [0,1]^2 → [0,1].

    Native domain:
        x1 ∈ [-3, 3],  x2 ∈ [-2, 2]

    Function:
        f(x1,x2) = (4 - 2.1 x1^2 + x1^4 / 3) * x1^2
                   + x1*x2
                   + (-4 + 4 x2^2) * x2^2

    Global minima (two, symmetric):
        f_min ≈ -1.031628 at (x1, x2) ≈ ( 0.0898, -0.7126) and (-0.0898, 0.7126)

    Normalization (minimization → larger is better):
        u = (f_max - f_val) / (f_max - f_min),  then clamp to [0,1].

    Notes:
        - We use float64 for numerical stability.
        - f_max is a conservative bound over the recommended domain.
    """
    # Double precision
    X = X.to(torch.float64)

    # Map [0,1]^2 → [-3,3]×[-2,2]
    x1 = -3.0 + 6.0 * X[..., 0]
    x2 = -2.0 + 4.0 * X[..., 1]

    # Six-Hump Camel
    f_val = (4.0 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2 \
            + x1 * x2 \
            + (-4.0 + 4.0 * x2**2) * x2**2

    # Negative-normalize to [0,1]
    f_min = -1.031628  # ≈ global minimum
    f_max = 200.0      # conservative upper bound on [-3,3]×[-2,2]
    u = (f_max - f_val) / (f_max - f_min)

    # Clamp to [0,1]
    return u.clamp_(0.0, 1.0)

def levy6(X: torch.Tensor) -> torch.Tensor:
    """
    Negative-normalized Lévy N-D function in 6 dimensions, mapping [0,1]^6 → [0,1].

    Domain mapping:
        x_i = -10 + 20 * X[..., i],  for i=0..5

    Define w_i = 1 + (x_i - 1) / 4.

    Lévy N-D:
        f(x) = sin^2(π w_1)
             + Σ_{i=1 to 5} (w_i - 1)^2 [1 + 10 sin^2(π w_{i+1})]
             + (w_6 - 1)^2

    Global minimum f_min = 0 at x = (1,…,1), so utility = 1 there.
    We pick an approximate worst-case f_max ≈ 430 over [-10,10]^6.
    Utility u = (f_max - f(x)) / (f_max - f_min), clamped to [0,1].
    """
    X = X.to(torch.float64)

    # 1) Map [0,1]^6 -> [-10,10]^6
    x = -10.0 + 20.0 * X  # shape (..., 6)

    # 2) Compute w_i
    w = 1.0 + (x - 1.0) / 4.0  # shape (..., 6)

    # 3) Lévy N-D formula
    term1 = torch.sin(np.pi * w[..., 0])**2

    # sum over i=1..5 of (w_i - 1)^2 [1 + 10 sin^2(pi w_{i+1})]
    wi   = w[..., :-1]  # (...,5), w_1..w_5
    wip1 = w[..., 1:]   # (...,5), w_2..w_6
    sum_terms = ((wi - 1.0)**2 * (1.0 + 10.0 * torch.sin(np.pi * wip1)**2)).sum(dim=-1)

    term_last = (w[..., -1] - 1.0)**2

    f_val = term1 + sum_terms + term_last  # shape (...)

    # 4) Normalize to [0,1]
    f_min = 0.0
    f_max = 430.0  # approximate upper bound on [-10,10]^6
    u = (f_max - f_val) / (f_max - f_min)

    return u.clamp_(0.0, 1.0)

def alpine1(X: torch.Tensor) -> torch.Tensor:
    """
    Negative‐normalized Alpine-1 function in 6D, mapping [0,1]^6 → [0,1].
    """
    X = X.to(torch.float64)
    x = 10.0 * X                          # map [0,1]^6 to [0,10]^6
    f_val = torch.abs(x * torch.sin(x) + 0.1 * x).sum(dim=-1)
    f_min, f_max = 0.0, 66.0
    u = (f_max - f_val) / (f_max - f_min)
    return u.clamp_(0.0, 1.0)