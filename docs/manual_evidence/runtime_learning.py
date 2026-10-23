"""Small real non-healing runtime training + held-out inference, not the healing demo.

Run: PYTHONPATH=. .venv/bin/python docs/manual_evidence/runtime_learning.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from experiments.runtime_config import RuntimeExperimentConfig
from experiments.runtime_experiment import run_runtime_experiment

config = RuntimeExperimentConfig()
families = config.families[:2]
config = RuntimeExperimentConfig(families=families,
    training_families=tuple(f.name for f in families), training_seeds=(1001,),
    training_episodes_per_seed=30, validation_seed=2001, final_test_seed=3001,
    validation_repetitions=2, final_test_repetitions=3, bootstrap_replicates=100)
output = Path('/tmp/os-teacher-runtime')
artifacts = run_runtime_experiment(config, output)
print('SMOKE DESIGN: 30 train episodes, one seed 1001; validation 2001; test 3001; two families')
print('Split fingerprint overlap:', artifacts.summary['split_fingerprint_overlap'])
print('Final-test unique workloads:', artifacts.summary['final_test_unique_workloads'])
print('Methods:', sorted(artifacts.final_test_metrics.method.unique()))
print('Final-test rows:', len(artifacts.final_test_metrics))
print(artifacts.final_test_metrics.groupby('method').avg_waiting_time.mean().to_string())
for name in ['training_metrics', 'q_table', 'state_action_coverage']:
    path = artifacts.paths[name]
    frame = pd.read_csv(path)
    print(f'ARTIFACT {path.name}: {len(frame)} rows; columns={list(frame.columns)}')
    print(frame.head(2).to_string(index=False))
payload = json.loads(artifacts.paths['learned_switch_demo'].read_text())
print('Validation-only selected demo:', payload['status'], 'switches=',payload['policy_switch_count'])
print('Illustrative, not representative:', payload['illustrative_not_representative'])
print('Demo decision keys:', sorted(payload['decisions'][0]))
print('Demo first four decisions:', json.dumps(payload['decisions'][:4], sort_keys=True))
print('Report:', artifacts.paths['report'])
print(next(line for line in artifacts.paths['report'].read_text().splitlines() if line.startswith('Q-learning rows average')))
# CSV is an output/audit artifact, not a loader. The experiment evaluates the
# trained in-memory controller; confirm this by inspecting its selected decision Qs.
print('Q CSV is exported; pipeline consumes trained in-memory models, not a CSV reload.')
assert not artifacts.summary['split_fingerprint_overlap']
assert len(artifacts.final_test_metrics) == 36
assert payload['source_split'] == 'validation'
print('RUNTIME TRAINING AND INFERENCE COMPLETE (no superiority claim)')

# Independent direct-controller check: training changes table; inference consumes
# visited Q values but must leave both Q and visit counts unchanged.
from rl.runtime_controller import RuntimeQController
from rl.runtime_state import RuntimeStateEncoder
from scheduler import RuntimeSimulator
from workload.generator import WorkloadGenerator
encoder = RuntimeStateEncoder(config.scheduler.round_robin_quantum)
controller = RuntimeQController(encoder, config.q_learning, seed=1001)
gen = WorkloadGenerator(config.families)
sim = RuntimeSimulator(config.scheduler)
for episode in range(10):
    sim.run(gen.generate('short_jobs', 4000 + episode), controller, training=True, episode=episode)
before_q = controller.agent.q_table.copy()
before_counts = controller.agent.state_action_visit_counts.copy()
run = sim.run(gen.generate('short_jobs', 5001), controller, training=False)
assert np.array_equal(before_q, controller.agent.q_table)
assert np.array_equal(before_counts, controller.agent.state_action_visit_counts)
assert np.count_nonzero(before_q) > 0
assert all(not d.decision.explored for d in run.decisions)
expected_reward = -sum(o.waiting_time for o in run.result.outcomes) / config.scheduler.round_robin_quantum
assert run.total_reward == expected_reward
print('DIRECT CONTROLLER: nonzero trained Q entries=',np.count_nonzero(before_q),
      '; inference decisions=',len(run.decisions),'; Q and visits unchanged=True')
print('Inference total reward=',run.total_reward,'; -sum waiting / q=',expected_reward)
print('Inference state/action/reason:', [(d.decision.state_index,d.decision.action,d.decision.fallback_reason) for d in run.decisions[:5]])
