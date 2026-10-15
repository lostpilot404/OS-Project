# Generated study results

> Generated from `summary.json` by `experiments.study.render_report`; do not edit result values by hand.

## Scope

This is an **offline policy-selection and evaluation system**: one policy per complete workload, selected before execution from the declared process table. It is not a runtime OS scheduler and never switches policies during a schedule.

- Primary declared objective: `risk_aware_v1` (weights {'context_switches': 0.15, 'response': 0.25, 'tail_waiting': 0.25, 'waiting': 0.35}, tail quantile 0.95).
- Training: 5 seeds [42, 43, 44, 45, 46] x 3600 episodes.
- Held-out test split: 240 workloads (30 per family x 8 families), master seed 19501; includes out-of-distribution families ['poisson_arrivals', 'heavy_tail_stress'] that never appear in training.
- Best-fixed baseline policy (training workloads only): SJF.

## Test-split results by arm

Cost is the declared objective (lower is better); regret is the hindsight gap to the best of the four fixed policies on the same workload. `poor` = regret > 0, `severe` = regret > 0.1.

| Arm | Mean cost | Mean regret | Hindsight-best % | Poor % | Severe % | Mean wait | Mean response | Ctx switches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best fixed policy (training-only) | 0.8419 | 0.1040 | 77.7 | 22.3 | 21.5 | 78.174 | 78.174 | 15.25 |
| Fingerprint rule (predeclared) | 0.7752 | 0.0372 | 83.7 | 16.3 | 9.8 | 79.194 | 78.465 | 16.98 |
| Original Q-learning selector | 0.7963 | 0.0524 | 85.6 | 14.4 | 13.5 | 78.204 | 77.829 | 16.53 |
| Ablation: original objective, fingerprint state | 0.7838 | 0.0405 | 84.5 | 15.5 | 13.0 | 79.356 | 78.386 | 16.87 |
| Risk-aware Q-learning selector | 0.7953 | 0.0574 | 82.5 | 17.5 | 15.8 | 81.791 | 76.317 | 18.55 |
| Ablation: risk objective, original state | 0.8253 | 0.0874 | 76.4 | 23.6 | 22.9 | 80.943 | 77.897 | 18.03 |
| Random policy (seeded) | 0.9778 | 0.2398 | 26.4 | 73.6 | 67.6 | 110.388 | 75.657 | 30.00 |

## Per-family mean declared-objective cost (test split)

| Arm | cpu_bursty | heavy_tail_stress | long_jobs | mixed | poisson_arrivals | priority_skewed | short_jobs | staggered_interactive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Best fixed policy (training-only) | 0.8513 | 0.9188 | 0.7650 | 0.7845 | 0.7974 | 0.8015 | 0.6915 | 1.1255 |
| Fingerprint rule (predeclared) | 0.8951 | 0.8883 | 0.7650 | 0.7845 | 0.7974 | 0.8015 | 0.7158 | 0.5538 |
| Original Q-learning selector | 0.8513 | 1.0717 | 0.7650 | 0.7845 | 0.8222 | 0.8068 | 0.7153 | 0.5538 |
| Ablation: original objective, fingerprint state | 0.8597 | 0.9027 | 0.7740 | 0.8053 | 0.8170 | 0.8085 | 0.7495 | 0.5538 |
| Risk-aware Q-learning selector | 0.8702 | 0.9236 | 0.7822 | 0.8133 | 0.8290 | 0.8145 | 0.7760 | 0.5538 |
| Ablation: risk objective, original state | 0.8513 | 1.1077 | 0.7759 | 0.7845 | 0.9697 | 0.8169 | 0.7428 | 0.5538 |
| Random policy (seeded) | 0.9638 | 0.9916 | 0.9644 | 0.9636 | 0.9975 | 0.9657 | 0.9897 | 0.9863 |

## Distribution shift (test split)

| Arm | In-dist mean cost | OOD mean cost | In-dist regret | OOD regret | OOD poor % |
|---|---:|---:|---:|---:|---:|
| Best fixed policy (training-only) | 0.8366 | 0.8581 | 0.0989 | 0.1191 | 33.7 |
| Fingerprint rule (predeclared) | 0.7526 | 0.8428 | 0.0150 | 0.1039 | 29.3 |
| Original Q-learning selector | 0.7461 | 0.9470 | 0.0089 | 0.1829 | 46.5 |
| Ablation: original objective, fingerprint state | 0.7585 | 0.8598 | 0.0217 | 0.0970 | 35.7 |
| Risk-aware Q-learning selector | 0.7683 | 0.8763 | 0.0307 | 0.1373 | 38.9 |
| Ablation: risk objective, original state | 0.7542 | 1.0387 | 0.0166 | 0.2997 | 74.6 |
| Random policy (seeded) | 0.9722 | 0.9945 | 0.2346 | 0.2555 | 73.0 |

## Paired comparison against the risk-aware selector (test split)

Positive differences mean the risk-aware selector has the higher (worse) declared-objective cost; intervals are workload-level percentile bootstrap 95% CIs (2000 resamples, seed 424242).

| Compared arm | Mean cost difference (ql_risk - arm) | 95% CI | Workloads |
|---|---:|---:|---:|
| Best fixed policy (training-only) | -0.0433 | [-0.0713, -0.0171] | 240 |
| Fingerprint rule (predeclared) | +0.0181 | [+0.0095, +0.0275] | 240 |
| Original Q-learning selector | +0.0000 | [-0.0141, +0.0142] | 240 |
| Ablation: original objective, fingerprint state | +0.0120 | [+0.0067, +0.0182] | 240 |
| Ablation: risk objective, original state | -0.0293 | [-0.0467, -0.0120] | 240 |
| Random policy (seeded) | -0.1758 | [-0.1944, -0.1570] | 240 |

## Training (mean terminal reward per arm, last-100 window)

| Arm | Mean reward, last 100 (across seeds) |
|---|---:|
| Original Q-learning selector | 0.2635 |
| Ablation: original objective, fingerprint state | 0.2511 |
| Risk-aware Q-learning selector | -0.7689 |
| Ablation: risk objective, original state | -0.7527 |

## Artifacts

- `decisions.csv` - one row per selection (all arms, validation and test).
- `counterfactuals.csv` - four-policy outcomes per workload (both objectives).
- `robustness.csv` - per arm x seed x family variability, regret and poor-selection rates.
- `training_history.csv` - per-episode training records with objective values.
- `explanations/` - structured explanations for every selection (JSONL, gzip).
- `explanations_examples.json` - three fully worked examples chosen by the documented fixed rule.

All numbers in this report are regenerated from the recorded seeds; explanation logs and the per-episode training table are reproducible and intentionally not tracked in git.
