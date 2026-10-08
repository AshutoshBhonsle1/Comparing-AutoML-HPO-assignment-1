"""Optional interface for Random Search in the common forest space.

Implement this loop, connect a package, or use another organisation. Choose how
to retain the results needed to analyse search progress and computational effort.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from random_forest import Config, Evaluator, sample_configuration


def optimise_random_search(
    evaluator: Evaluator,
    n_trials: int,
    n_trees: int,
    seed: int,
) -> tuple[Config, Any]:
    """randomly sample and evaluate up to n_trials configurations.

    Use the shared search space and train each forest with n_trees trees.
    Select the best configuration using the validation objective, respecting
    whether higher or lower values are better.
    Return the selected configuration and results needed for your analysis.
    """
    #Sample and evaluate n_trials configurations using the common maximum trees.
    rng = np.random.default_rng(seed)
    history: list[dict[str, Any]] = []
    best_config: Config | None = None
    best_objective = float("-inf")

    for trial in range(1, int(n_trials) + 1):
        config = sample_configuration(rng)
        result = evaluator(config, n_trees, seed)
        result["trial"] = trial
        result["method"] = "random_search"
        history.append(result)
        if result["objective"] > best_objective:
            best_objective = result["objective"]
            best_config = dict(config)

    if best_config is None:
        raise ValueError("n_trials must be at least 1")
    return best_config, history
