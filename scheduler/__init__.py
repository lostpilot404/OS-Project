"""The four conventional CPU scheduling policies.

``POLICY_CLASSES`` lists them in action order, which ties the schedulers to the
Q-learning action space exactly once; ``ACTION_NAMES`` in :mod:`config` must stay in
sync with it (asserted by the test suite).
"""

from scheduler.base import NonPreemptiveReadyQueuePolicy, SchedulingPolicy, Timeline
from scheduler.fcfs import FCFS
from scheduler.priority import Priority
from scheduler.round_robin import RoundRobin
from scheduler.sjf import SJF

#: The four policies, indexed by their Q-learning action.
POLICY_CLASSES = (FCFS, SJF, RoundRobin, Priority)

__all__ = [
    "SchedulingPolicy",
    "NonPreemptiveReadyQueuePolicy",
    "Timeline",
    "FCFS",
    "SJF",
    "RoundRobin",
    "Priority",
    "POLICY_CLASSES",
]
