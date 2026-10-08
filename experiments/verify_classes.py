"""Pre-training verification of the workload classes.

Before the Q-learning agent is trained, this module measures -- on workloads drawn from
a dedicated seed stream that is disjoint from both the training and the evaluation
streams -- **which conventional policy each workload class actually favours** under the
declared reward.  The question it answers is the one the experiment ultimately rests
on: do the configured workload classes place the four conventional policies in
genuinely different regimes, so that a workload-aware selection can exist at all?

For every class and every probe workload it runs all four conventional policies on the
*same* workload, computes the reward of each of the four actions, and records

* the reward-argmax (the policy the reward actually prefers) per workload,
* the winner counts and shares per class, and the mean reward of every action,
* the encoded state of every workload.

From the state/argmax pairs it also measures the **state-conditional consistency**: the
share of probe workloads whose reward-argmax equals the modal argmax of their encoded
state.  A high value means the state representation carries the information the agent
needs (the optimal action is a function of the state); a low value would mean the
classes cannot be told apart before execution, which would cap what any agent could
learn.  This measurement is a property of the workload set and the state representation
-- it does not involve the agent and never writes to any Q-table.

The result is written to ``results/class_verification.json`` and summarised in
``results/summary.json``; the test suite asserts the design property it establishes
(the default classes have at least two distinct modal winners).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from config import ACTION_NAMES, ExperimentConfig, derive_seed
from evaluation.metrics import compute_metrics
from rl.reward import compute_reward
from rl.state import StateEncoder, observe_workload_state
from scheduler import POLICY_CLASSES
from workload.generator import WorkloadGenerator
from workload.models import Workload

__all__ = ["ClassVerification", "VerificationResult", "verify_classes"]

#: Seed-derivation tag of the verification workload stream.
_WORKLOAD_STREAM_TAG = 1


@dataclass(frozen=True)
class ClassVerification:
    """What the pre-training probe measured about one workload class.

    Attributes:
        family: Name of the workload class.
        workloads: Number of probe workloads measured.
        winner_counts: Number of probe workloads whose reward-argmax was each policy.
        winner_shares: Winner counts divided by ``workloads``.
        modal_winner: The policy with the most reward-argmax workloads (ties broken by
            action order, exactly like the agent's greedy tie-break).
        mean_reward_per_action: Mean reward of each action over the probe workloads.
        distinct_states: Number of distinct encoded states the class occupied.
        states: The encoded states the class occupied, ascending.
    """

    family: str
    workloads: int
    winner_counts: Mapping[str, int]
    winner_shares: Mapping[str, float]
    modal_winner: str
    mean_reward_per_action: Mapping[str, float]
    distinct_states: int
    states: Tuple[int, ...]

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable record of this class's verification."""
        return {
            "family": self.family,
            "workloads": self.workloads,
            "winner_counts": {name: self.winner_counts.get(name, 0) for name in ACTION_NAMES},
            "winner_shares": {name: self.winner_shares.get(name, 0.0) for name in ACTION_NAMES},
            "modal_winner": self.modal_winner,
            "mean_reward_per_action": {
                name: self.mean_reward_per_action.get(name, 0.0) for name in ACTION_NAMES
            },
            "distinct_states": self.distinct_states,
            "states": list(self.states),
        }


@dataclass(frozen=True)
class VerificationResult:
    """The complete pre-training verification of the workload classes.

    Attributes:
        seed: Master seed of the verification workload stream.
        repetitions: Probe workloads per class.
        classes: One :class:`ClassVerification` per configured family, in configuration
            order.
        states_observed: Number of distinct encoded states over all probe workloads.
        states_with_conflicting_argmax: Number of states whose probe workloads did not
            all have the same reward-argmax.
        workloads_matching_state_modal_argmax: Number of probe workloads whose
            reward-argmax equals the modal argmax of their state.
        workloads_total: Total number of probe workloads.
        modal_winners: Distinct modal winners across the classes, in action order.
        fingerprints: Fingerprints of every probe workload (used by the tests to prove
            the stream is disjoint from training and evaluation).
    """

    seed: int
    repetitions: int
    classes: Tuple[ClassVerification, ...]
    states_observed: int
    states_with_conflicting_argmax: int
    workloads_matching_state_modal_argmax: int
    workloads_total: int
    modal_winners: Tuple[str, ...]
    fingerprints: Tuple[str, ...]

    @property
    def state_conditional_consistency(self) -> float:
        """Share of probe workloads matching their state's modal reward-argmax."""
        if not self.workloads_total:
            return 0.0
        return self.workloads_matching_state_modal_argmax / self.workloads_total

    def summary(self) -> Dict[str, object]:
        """Return a JSON-serialisable summary of the verification."""
        return {
            "seed": self.seed,
            "repetitions_per_family": self.repetitions,
            "workloads_total": self.workloads_total,
            "states_observed": self.states_observed,
            "states_with_conflicting_argmax": self.states_with_conflicting_argmax,
            "workloads_matching_state_modal_argmax": self.workloads_matching_state_modal_argmax,
            "state_conditional_consistency": self.state_conditional_consistency,
            "modal_winners": list(self.modal_winners),
            "classes": [entry.as_dict() for entry in self.classes],
        }

    def to_json(self, path: Path) -> Path:
        """Write the full verification to ``path`` as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        return path


def verify_classes(
    config: ExperimentConfig,
    generator: Optional[WorkloadGenerator] = None,
    encoder: Optional[StateEncoder] = None,
) -> VerificationResult:
    """Measure which conventional policy each workload class favours, before training.

    Args:
        config: The experiment configuration (workload families, reward, state and the
            verification seed/repetitions).
        generator: Optional pre-built workload generator (one is created if omitted).
        encoder: Optional pre-built state encoder (one is created if omitted).

    Returns:
        The verification result.

    Raises:
        ConfigurationError: If the configuration is inconsistent.
    """
    config.validate()
    generator = generator or WorkloadGenerator(config.families)
    encoder = encoder or StateEncoder(config.state)
    policies = [policy_class(config.scheduler) for policy_class in POLICY_CLASSES]

    class_winners: List[Counter] = []
    class_reward_sums: List[Dict[int, float]] = []
    class_states: List[set] = []
    state_argmax: Dict[int, Counter] = defaultdict(Counter)
    state_modal_match = 0
    workloads_total = 0
    fingerprints: List[str] = []

    for family_index, family in enumerate(config.families):
        winners: Counter = Counter()
        reward_sums: Dict[int, float] = {action: 0.0 for action in range(len(ACTION_NAMES))}
        states: set = set()
        for repetition in range(config.evaluation.verification_repetitions):
            seed = derive_seed(
                config.evaluation.verification_seed,
                _WORKLOAD_STREAM_TAG,
                family_index,
                repetition,
            )
            workload: Workload = generator.generate(family.name, seed)
            fingerprints.append(workload.fingerprint)
            attempts = {
                action: compute_metrics(policy.run(workload))
                for action, policy in enumerate(policies)
            }
            rewards = [
                compute_reward(config.reward, attempts, action).reward
                for action in range(len(ACTION_NAMES))
            ]
            # Ties are broken by the lowest action index, exactly like the agent's
            # greedy tie-break, so the probe's argmax is the agent's best possible pick.
            best_action = max(range(len(ACTION_NAMES)), key=lambda a: (rewards[a], -a))
            winners[best_action] += 1
            for action in range(len(ACTION_NAMES)):
                reward_sums[action] += rewards[action]
            state = encoder.encode(observe_workload_state(workload, config.state))
            states.add(state)
            state_argmax[state][best_action] += 1
            workloads_total += 1

        repetitions = config.evaluation.verification_repetitions
        class_winners.append(winners)
        class_reward_sums.append(
            {action: total / repetitions for action, total in reward_sums.items()}
        )
        class_states.append(states)

    # State-conditional consistency: does the encoded state determine the reward-argmax?
    for state, counts in state_argmax.items():
        modal_action = max(counts, key=lambda a: (counts[a], -a))
        state_modal_match += counts[modal_action]

    classes: List[ClassVerification] = []
    for family, winners, reward_means, states in zip(
        config.families, class_winners, class_reward_sums, class_states
    ):
        modal_action = max(winners, key=lambda a: (winners[a], -a))
        classes.append(
            ClassVerification(
                family=family.name,
                workloads=sum(winners.values()),
                winner_counts={ACTION_NAMES[a]: winners.get(a, 0) for a in range(4)},
                winner_shares={
                    ACTION_NAMES[a]: winners.get(a, 0) / max(1, sum(winners.values()))
                    for a in range(4)
                },
                modal_winner=ACTION_NAMES[modal_action],
                mean_reward_per_action={
                    ACTION_NAMES[a]: reward_means[a] for a in range(4)
                },
                distinct_states=len(states),
                states=tuple(sorted(states)),
            )
        )

    modal_winners = tuple(
        ACTION_NAMES[action]
        for action in range(len(ACTION_NAMES))
        if any(entry.modal_winner == ACTION_NAMES[action] for entry in classes)
    )
    return VerificationResult(
        seed=config.evaluation.verification_seed,
        repetitions=config.evaluation.verification_repetitions,
        classes=tuple(classes),
        states_observed=len(state_argmax),
        states_with_conflicting_argmax=sum(1 for c in state_argmax.values() if len(c) > 1),
        workloads_matching_state_modal_argmax=state_modal_match,
        workloads_total=workloads_total,
        modal_winners=modal_winners,
        fingerprints=tuple(fingerprints),
    )
