"""Tabular state encoding of workload fingerprints for the risk-aware selector.

The encoder discretises a declared subset of the interpretable fingerprint features
(:data:`config.KNOWN_FINGERPRINT_STATE_FEATURES`) into a mixed-radix tabular state.
Like :class:`rl.state.StateEncoder`, this class is the single owner of the state-space
layout: the Q-learning agent only ever sees integer indices.

The encoder accepts only a :class:`workload.fingerprint.WorkloadFingerprint`, which is
itself a pure function of the declared process table (see ``docs/METHODOLOGY.md``,
"Information boundary").  No schedule outcome, metric or counterfactual value can enter
the state.
"""

from __future__ import annotations

from typing import Dict, Tuple

from config import FingerprintStateConfig
from errors import ValidationError
from rl.state import LinearBinner
from workload.fingerprint import WorkloadFingerprint

__all__ = ["FingerprintStateEncoder"]


class FingerprintStateEncoder:
    """Turns a :class:`WorkloadFingerprint` into a Q-table index and back.

    Features are encoded in significance order (first feature is most significant),
    exactly like :class:`rl.state.StateEncoder`.
    """

    def __init__(self, config: FingerprintStateConfig) -> None:
        """
        Args:
            config: Discretisation configuration.

        Raises:
            ValidationError: If the configuration is not a
                :class:`config.FingerprintStateConfig`.
        """
        if not isinstance(config, FingerprintStateConfig):
            raise ValidationError(
                f"config must be a FingerprintStateConfig, got {type(config).__name__}"
            )
        self._config = config
        self._features = tuple(config.state_features)
        ranges = config.value_ranges()
        self._binners: Dict[str, LinearBinner] = {
            feature: LinearBinner(ranges[feature][0], ranges[feature][1], config.bins_per_variable)
            for feature in self._features
        }

    @property
    def config(self) -> FingerprintStateConfig:
        """The discretisation configuration."""
        return self._config

    @property
    def features(self) -> Tuple[str, ...]:
        """State features in significance order (first is most significant)."""
        return self._features

    @property
    def n_states(self) -> int:
        """Number of states in the state space."""
        return self._config.n_states

    @property
    def n_bins(self) -> int:
        """Number of bins per feature."""
        return self._config.bins_per_variable

    def binner(self, feature: str) -> LinearBinner:
        """Return the binner of one feature.

        Raises:
            ValidationError: If the feature is not part of the state.
        """
        if feature not in self._binners:
            raise ValidationError(
                f"feature {feature!r} is not part of the state; state uses "
                f"{list(self._features)}"
            )
        return self._binners[feature]

    def encode(self, fingerprint: WorkloadFingerprint) -> int:
        """Encode a fingerprint as a Q-table index in ``[0, n_states)``.

        Raises:
            ValidationError: If ``fingerprint`` is not a
                :class:`workload.fingerprint.WorkloadFingerprint`.
        """
        if not isinstance(fingerprint, WorkloadFingerprint):
            raise ValidationError(
                "fingerprint must be a WorkloadFingerprint, "
                f"got {type(fingerprint).__name__}"
            )
        values = fingerprint.as_dict()
        index = 0
        for feature in self._features:
            index = index * self.n_bins + self._binners[feature].index(values[feature])
        return index

    def decode(self, index: int) -> Dict[str, int]:
        """Return the bin index of every feature for a flat state index.

        Raises:
            ValidationError: If the index is outside the state space.
        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValidationError(f"state index must be an integer, got {index!r}")
        if not 0 <= index < self.n_states:
            raise ValidationError(
                f"state index must lie in [0, {self.n_states - 1}], got {index}"
            )
        bins_by_feature: Dict[str, int] = {}
        remainder = index
        for feature in reversed(self._features):
            bins_by_feature[feature] = remainder % self.n_bins
            remainder //= self.n_bins
        return {feature: bins_by_feature[feature] for feature in self._features}

    def describe(self) -> Dict[str, str]:
        """Return the bin-edge description of every state feature."""
        return {feature: self._binners[feature].describe() for feature in self._features}

    def __repr__(self) -> str:
        return (
            f"FingerprintStateEncoder(features={list(self._features)}, "
            f"bins={self.n_bins}, n_states={self.n_states})"
        )
