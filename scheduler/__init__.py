"""Conventional policies and the event-driven runtime simulator.

``POLICY_CLASSES`` lists the four fixed policies in action order. ``RuntimeSimulator``
reuses their policy choices inside one evolving trace without changing the standalone
implementations.
"""

from scheduler.base import NonPreemptiveReadyQueuePolicy, SchedulingPolicy, Timeline
from scheduler.fcfs import FCFS
from scheduler.priority import Priority
from scheduler.round_robin import RoundRobin
from scheduler.runtime import RuntimeSimulator
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
    "RuntimeSimulator",
]
