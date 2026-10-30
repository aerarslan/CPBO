import argparse
import logging
import sys

MANUAL_DEFAULTS = {
    "seed_value":      1,
    "data_gamma":      0.04,
    "bool_model_gamma": "yes",
    "model_iteration": 2000,
    "bo_init":         10,
    "bo_optim":        25,
    "function_name":   "branin", # "branin, bukin6, crossintray, levy13, ackley, bohachevsky, sixhump, levy6, alpine1"
    "total_steps":     30,
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="CPBO run: specify seed, gammas, iterations, BO steps, and function."
    )
    parser.add_argument("--seed-value",
                        type=int,
                        default=MANUAL_DEFAULTS["seed_value"],
                        help=f"Seed for randomness (default: {MANUAL_DEFAULTS['seed_value']})")
    parser.add_argument("--data-gamma",
                        type=float,
                        default=MANUAL_DEFAULTS["data_gamma"],
                        help=f"Data‐generation JND threshold γ (default: {MANUAL_DEFAULTS['data_gamma']})")
    parser.add_argument("--bool-model-gamma",
                        choices=["yes","no"],
                        default=MANUAL_DEFAULTS["bool_model_gamma"],
                        help=f"Whether γ is trainable (yes|no, default: {MANUAL_DEFAULTS['bool_model_gamma']})")
    parser.add_argument("--model-iteration",
                        type=int,
                        default=MANUAL_DEFAULTS["model_iteration"],
                        help=f"Number of training iterations (default: {MANUAL_DEFAULTS['model_iteration']})")
    parser.add_argument("--bo-init",
                        type=int,
                        default=MANUAL_DEFAULTS["bo_init"],
                        help=f"Info‐gain BO initial design points (default: {MANUAL_DEFAULTS['bo_init']})")
    parser.add_argument("--bo-optim",
                        type=int,
                        default=MANUAL_DEFAULTS["bo_optim"],
                        help=f"Info‐gain BO optimization steps (default: {MANUAL_DEFAULTS['bo_optim']})")
    parser.add_argument("--function-name",
                        type=str,
                        default=MANUAL_DEFAULTS["function_name"],
                        help=f"Test function to optimize (default: {MANUAL_DEFAULTS['function_name']})")
    parser.add_argument("--total-steps",
                    type=int,
                    default=MANUAL_DEFAULTS["total_steps"],
                    help=f"Total BO outer‐loop iterations (default: {MANUAL_DEFAULTS['total_steps']})")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        logging.info("No command-line arguments detected; using defaults:")
        for k, v in MANUAL_DEFAULTS.items():
            logging.info(f"  {k}: {v}")

    args.function_name = args.function_name.lower()
    return args
