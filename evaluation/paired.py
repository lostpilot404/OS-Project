"""Paired, family-stratified bootstrap statistics shared by both experiments.

The offline policy-selection study and the self-healing study compare several methods on
the *same* generated workloads. Both need the same statistic: paired differences with
family-stratified, crossed resampling of independent model seeds and of workload
repetitions within each family. The implementation lives here once so the two experiments
cannot drift apart, and the intervals keep the same declared meaning:

* cells are paired by ``(family, repetition, workload fingerprint)``, so a method is only
  ever compared with another method on an identical workload;
* model-seed variability and workload variability are resampled independently (crossed),
  which is what makes the interval reflect both a different training run and a different
  workload draw;
* families are stratified, so a family with more repetitions cannot dominate the estimate.

The intervals are conditional on the declared seed and family design; they are not a claim
about a universal workload population.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from errors import ValidationError

__all__ = ["paired_matrices", "paired_bootstrap_difference"]

_PAIR_COLUMNS = ("repetition", "workload_fingerprint")


def paired_matrices(
    frame: pd.DataFrame,
    method: str,
    metric: str,
    *,
    method_column: str = "method",
    seed_column: str = "training_seed",
) -> Tuple[List[str], Dict[str, np.ndarray], List[Any]]:
    """Return one ``(seed, workload)`` matrix per family for a method/metric pair.

    Raises:
        ValidationError: If the method is absent or a family has incomplete pairings.
    """
    for column in (method_column, seed_column, "family", *_PAIR_COLUMNS, metric):
        if column not in frame.columns:
            raise ValidationError(f"metrics frame is missing column {column!r}")
    subset = frame[frame[method_column] == method].copy()
    if subset.empty:
        raise ValidationError(f"no rows for method {method!r}")
    seed_values = sorted(subset[seed_column].dropna().unique().tolist())
    if not seed_values:
        seed_values = [None]
        subset["_seed_key"] = 0
        seed_keys: List[Any] = [None]
    else:
        subset["_seed_key"] = subset[seed_column]
        seed_keys = seed_values
    family_names = sorted(subset["family"].unique().tolist())
    matrices: Dict[str, np.ndarray] = {}
    for family in family_names:
        family_frame = subset[subset["family"] == family]
        workload_keys = sorted(set(zip(family_frame["repetition"], family_frame["workload_fingerprint"])))
        matrix = np.empty((len(seed_keys), len(workload_keys)), dtype=float)
        for seed_index, seed in enumerate(seed_keys):
            seed_key = 0 if seed is None else seed
            seed_frame = family_frame[family_frame["_seed_key"] == seed_key]
            lookup = {
                (row.repetition, row.workload_fingerprint): float(getattr(row, metric))
                for row in seed_frame.itertuples()
            }
            if set(workload_keys) - set(lookup):
                raise ValidationError(
                    f"method {method!r} has incomplete paired workloads in {family!r}"
                )
            matrix[seed_index, :] = [lookup[key] for key in workload_keys]
        matrices[family] = matrix
    return family_names, matrices, seed_keys


def paired_bootstrap_difference(
    frame: pd.DataFrame,
    target: str,
    reference: str,
    metric: str,
    *,
    replicates: int,
    seed: int,
    method_column: str = "method",
    seed_column: str = "training_seed",
) -> Dict[str, float]:
    """Family-stratified, crossed workload/model-seed paired bootstrap interval.

    Args:
        frame: Tidy metric frame with one row per method/seed/workload.
        target: Method name in the numerator (``target - reference``).
        reference: Method name subtracted from the target.
        metric: Column compared.
        replicates: Number of bootstrap resamples.
        seed: Seed of the resampling stream (deterministic).
        method_column: Column holding method names.
        seed_column: Column holding independent model seeds (may be all missing).

    Returns:
        ``{"mean_difference", "ci95_low", "ci95_high"}``.

    Raises:
        ValidationError: If the two methods do not cover the same paired workloads.
    """
    if replicates < 1:
        raise ValidationError(f"replicates must be >= 1, got {replicates}")
    target_families, target_data, target_seeds = paired_matrices(
        frame, target, metric, method_column=method_column, seed_column=seed_column
    )
    ref_families, ref_data, ref_seeds = paired_matrices(
        frame, reference, metric, method_column=method_column, seed_column=seed_column
    )
    if target_families != ref_families:
        raise ValidationError("comparison methods do not cover the same workload families")
    rng = np.random.default_rng(seed)
    observed_family_means: List[float] = []
    for family in target_families:
        if target_data[family].shape[1] != ref_data[family].shape[1]:
            raise ValidationError("comparison methods do not cover paired workload counts")
        observed_family_means.append(
            float(target_data[family].mean()) - float(ref_data[family].mean())
        )
    observed = float(np.mean(observed_family_means))

    draws = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        target_seed_indices = rng.integers(0, len(target_seeds), size=len(target_seeds))
        ref_seed_indices = rng.integers(0, len(ref_seeds), size=len(ref_seeds))
        family_draws: List[float] = []
        for family in target_families:
            workload_count = target_data[family].shape[1]
            sampled_workloads = rng.integers(0, workload_count, size=workload_count)
            target_values = target_data[family][target_seed_indices][:, sampled_workloads]
            reference_values = ref_data[family][ref_seed_indices][:, sampled_workloads]
            family_draws.append(float(np.mean(target_values - reference_values)))
        draws[replicate] = float(np.mean(family_draws))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_difference": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def percentile(values: List[float], fraction: float) -> float:
    """Nearest-rank percentile on the integer grid (used for tail-waiting statistics)."""
    if not values:
        return float("nan")
    if not 0.0 < fraction <= 1.0:
        raise ValidationError(f"fraction must lie in (0, 1], got {fraction}")
    ordered = sorted(values)
    index = max(0, int(np.ceil(fraction * len(ordered))) - 1)
    return float(ordered[index])


def mean_or_nan(values: List[float]) -> Optional[float]:
    """Mean of a list, or ``None`` when it is empty (kept explicit for artifacts)."""
    if not values:
        return None
    return float(sum(values) / len(values))
