"""Optional interface for a pre-trained model on the largest selected dataset.

Use a suitable package directly, adapt this interface, or organise your own experiment.
"""

from __future__ import annotations

from typing import Any

from data_loading import DataSplits

# Maximum in-context learning samples supported by the TabPFN model version
TABPFN_MAX_CONTEXT_SAMPLES = 10000


def run_foundation_model(splits: DataSplits, seed: int) -> dict[str, Any]:
    """Evaluate TabPFN on the largest dataset using in-context learning.

    Args:
        splits: DataSplits containing X_train, y_train, X_valid, y_valid, X_test, y_test.
        seed: Random seed for reproducible subsampling of context data.

    Returns:
        A dictionary containing test metrics, objective value, runtime, and setup metadata.
    """
    rng = np.random.default_rng(seed)

    # 1. Combine non-test data (train + validation) to maximize in-context learning budget
    X_non_test = np.vstack([splits.X_train, splits.X_valid])
    y_non_test = np.concatenate([splits.y_train, splits.y_valid])

    # 2. Subsample context data if dataset size exceeds TabPFN's context window limit
    n_samples = len(X_non_test)
    if n_samples > TABPFN_MAX_CONTEXT_SAMPLES:
        # Stratified or uniform subsampling to stay within the budget (Section 3.5)
        indices = rng.choice(n_samples, size=TABPFN_MAX_CONTEXT_SAMPLES, replace=False)
        X_context = X_non_test[indices]
        y_context = y_non_test[indices]
    else:
        X_context = X_non_test
        y_context = y_non_test

    # 3. Fit in-context learning and score on the COMPLETE held-out test set
    start_time = perf_counter()

    # Initialize TabPFNClassifier (device='cpu' or 'cuda' depending on hardware setup)
    model = TabPFNClassifier(device="auto", random_state=seed)

    # TabPFN "fit" step loads the context samples into memory (no gradient backpropagation)
    model.fit(X_context, y_context)

    # Compute predictions on the FULL held-out test set using your shared predictive_metrics
    metrics = predictive_metrics(model, splits.X_test, splits.y_test)
    objective = validation_objective(metrics)

    elapsed_sec = float(perf_counter() - start_time)

    return {
        "model_name": "TabPFN",
        "context_samples": len(X_context),
        "total_test_samples": len(splits.y_test),
        "metrics": metrics,
        "objective": float(objective),
        "elapsed_sec": elapsed_sec,
    }