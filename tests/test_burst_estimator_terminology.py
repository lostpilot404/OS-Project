"""Terminology consistency: error_cv is log-space sigma, not CV."""

import pathlib


def test_burst_estimator_doc_clarifies_error_cv_is_log_space_sigma():
    text = pathlib.Path("healing/burst_estimator.py").read_text(encoding="utf-8")
    low = text.lower()
    # Must state log-space standard deviation and backward compat (allow line break)
    assert "log-space standard deviation" in low
    assert "retained for backward" in low and "compatibility" in low
    # Must mention actual CV formula to distinguish
    assert "sqrt(exp(sigma^2)-1)" in text or "sqrt(exp(sigma^2) - 1)" in text
    # Must not still claim error_cv is the multiplier's CV without qualification
    assert "Coefficient of variation of multiplicative log-normal noise" not in text
    # Class doc should contain sampling formula with sigma
    assert "exp(N(bias, sigma^2))" in text


def test_healing_config_doc_clarifies_error_cv():
    text = pathlib.Path("healing/config.py").read_text(encoding="utf-8")
    assert "log-space standard deviation" in text
    assert "burst_estimator_error_cv" in text
    assert "sqrt(exp(sigma^2)-1)" in text or "sqrt(exp(sigma" in text


def test_readme_does_not_claim_calibrated_or_realistic_estimator():
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")
    # Old unsupported claims must be gone
    assert "calibrated prediction errors" not in readme
    assert "Realistic uncertainty about CPU bursts" not in readme
    # New clarification must be present (allow markdown formatting)
    assert "calibrated or validated against real workload traces" in readme
    assert "oracle baseline, not an equal-information" in readme
    assert "sigma = error_cv" in readme
    assert "sqrt(exp(sigma" in readme


def test_architecture_and_limitations_use_correct_terminology():
    for fname in ["docs/ARCHITECTURE.md", "docs/LIMITATIONS.md", "docs/DESIGN_AND_CHOICES.md"]:
        text = pathlib.Path(fname).read_text(encoding="utf-8")
        # No unqualified CV claim for error_cv
        assert "error_cv` = coefficient of variation" not in text
        # Must contain clarified phrase
        assert "log-space standard deviation" in text


def test_experiment_report_estimator_note_is_precise():
    text = pathlib.Path("experiments/drift_uncertainty_experiment.py").read_text(encoding="utf-8")
    assert "log-normal multiplicative noise" in text
    assert "log-space standard deviation" in text
    assert "not mathematically identical to the multiplier's CV" in text


def test_sampling_behavior_unchanged():
    # Numerical behavior must be preserved: error_cv still used as sigma in rng.normal
    import inspect
    import healing.burst_estimator as m
    src = inspect.getsource(m.BurstEstimator.predict_with_noise)
    assert "rng.normal(0, self.error_cv)" in src
    # Config default unchanged (parameter name and value retained)
    from healing.config import HealingConfig
    cfg = HealingConfig()
    assert cfg.burst_estimator_error_cv == 0.30
