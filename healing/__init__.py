"""Self-healing CPU scheduling: detection, diagnosis, RL recovery, verification, rollback.

The public surface of the package is re-exported here so that the experiment layer, the
tests and a reader of the repository can see the whole loop in one place.
"""

from healing.actions import (
    GROW_QUANTUM,
    HEALING_ACTION_NAMES,
    HOLD,
    REVERT_TO_LAST_ACCEPTED,
    SET_FCFS,
    SET_PRIORITY,
    SET_ROUND_ROBIN,
    SET_SJF,
    SHRINK_QUANTUM,
    HealingActionOutcome,
    SchedulerSettings,
    apply_healing_action,
)
from healing.config import HealingConfig
from healing.controller import (
    EpisodeRecord,
    IncidentRecord,
    SelfHealingController,
    available_actions,
    replay_action_at_incident,
    replay_hold_counterfactual,
)
from healing.detector import DegradationDetector, DetectionOutcome
from healing.diagnosis import (
    DIAGNOSIS_CATEGORIES,
    Diagnosis,
    ReadyEvidence,
    build_ready_evidence,
    diagnose,
)
from healing.burst_estimator import BurstEstimator, EstimationError, NoisyBurstObservation, perturb_observation
from healing.monitor import CausalMonitor, MonitorSnapshot
from healing.policies import (
    FAIRNESS_RULE_MAPPING,
    WAITING_TIME_RULE_MAPPING,
    HealingContext,
    HealingContextEncoder,
    HealingPolicy,
    HoldOnlyPolicy,
    MappedRulePolicy,
    PolicyChoice,
    QLearningHealingPolicy,
    RuleHealingPolicy,
    RuleWaitingTimePolicy,
)

__all__ = [
    "CausalMonitor",
    "MonitorSnapshot",
    "DegradationDetector",
    "DetectionOutcome",
    "DIAGNOSIS_CATEGORIES",
    "Diagnosis",
    "ReadyEvidence",
    "build_ready_evidence",
    "diagnose",
    "BurstEstimator",
    "EstimationError",
    "NoisyBurstObservation",
    "perturb_observation",
    "HealingContext",
    "HealingContextEncoder",
    "HealingPolicy",
    "HoldOnlyPolicy",
    "PolicyChoice",
    "QLearningHealingPolicy",
    "RuleHealingPolicy",
    "RuleWaitingTimePolicy",
    "MappedRulePolicy",
    "FAIRNESS_RULE_MAPPING",
    "WAITING_TIME_RULE_MAPPING",
    "HealingConfig",
    "SelfHealingController",
    "IncidentRecord",
    "EpisodeRecord",
    "replay_hold_counterfactual",
    "replay_action_at_incident",
    "available_actions",
    "HealingActionOutcome",
    "SchedulerSettings",
    "apply_healing_action",
    "HEALING_ACTION_NAMES",
    "HOLD",
    "SET_FCFS",
    "SET_SJF",
    "SET_ROUND_ROBIN",
    "SET_PRIORITY",
    "SHRINK_QUANTUM",
    "GROW_QUANTUM",
    "REVERT_TO_LAST_ACCEPTED",
]
