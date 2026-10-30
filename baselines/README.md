# Baseline Implementations

**Disclaimer:** We do *not* claim ownership of the methods implemented in this folder. These are baseline algorithms used for comparison in our experiments. Please refer to the original sources for full details and licensing information.

## Overview

This folder contains adaptations of Preferential PBO methods used as baselines in our work:

* **EUBO**: `Expected Utility of the Best Option` from BoTorch
* **POP-BO**: `Principled Preferential Bayesian Optimization` from PREDICT-EPFL

Our implementations illustrate how to integrate these methods into the **Consecutive Preferential Bayesian Optimization (CPBO)** framework for comparison. For detailed verbal descriptions and experimental protocols, see the **Supplementary Material** of our paper.

## EUBO

* **Source:** [BoTorch EUBO documentation (Accessed 21/05/2025 16:00)](https://botorch.readthedocs.io/en/stable/_modules/botorch/acquisition/preference.html#AnalyticExpectedUtilityOfBestOption)
* **Adaptation:** We configure both EUBO (analytic) and qEUBO (MC) to propose a single new candidate (`q=1`) and force it to compare against the previously produced candidate, enforcing the consecutive-comparison constraint.
* **Run:** Any `experiment_runner` can be run based on the required output (e.g., "2D, 6D and look 5 back"). The required parameters can be edited via the `Config` class.

### Installation

Install BoTorch and dependencies:

```bash
pip install botorch
```

---

## POP-BO

* **Source:** [POP-BO GitHub repository (Accessed 21/05/2025 16:00)](https://github.com/PREDICT-EPFL/POP-BO)
* **Adaptation:** We retain the native consecutive-comparison design of POP-BO but replace its original Bradley–Terry preference model with our Thurstone-style oracle to ensure consistency across baselines.
* **Run:** Run `experiment_runner.py`. The required parameters can be edited via the `Config` class.

### Installation

Clone the repository and follow author's instructions:

```bash
git clone https://github.com/PREDICT-EPFL/POP-BO.git
```
