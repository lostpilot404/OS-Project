"""Exception types shared across the project.

The project distinguishes two failure modes:

* :class:`ValidationError` -- project data (workloads, schedules, metrics) violates a
  documented invariant.  Raised by the data model and by the schedulers.
* :class:`ConfigurationError` -- a configuration object contains an invalid or
  inconsistent value.  Raised while building or validating an
  :class:`config.ExperimentConfig`.

Both derive from :class:`ValueError` so callers that only care about "bad input" can
catch a single built-in exception type.
"""

from __future__ import annotations

__all__ = ["ValidationError", "ConfigurationError"]


class ValidationError(ValueError):
    """Raised when project data violates a documented invariant."""


class ConfigurationError(ValidationError):
    """Raised when a configuration object holds an invalid or inconsistent value."""
