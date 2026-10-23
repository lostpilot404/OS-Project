"""Recompute fresh-run metrics, inspect outcomes, and replay an actual fallback trace.

Run after teacher demo, full healing study and drift smoke commands in the guide.
PYTHONPATH=. .venv/bin/python docs/manual_evidence/output_integrity.py
"""
import json
from pathlib import Path
import pandas as pd
from config import build_default_config
from experiments.healing_config import HealingExperimentConfig
from experiments.healing_experiment import build_healing_workloads
from healing import SelfHealingController
from healing.policies import RuleHealingPolicy
from scheduler import RuntimeSimulator
from workload.generator import WorkloadGenerator

root = Path('/tmp/os-teacher-healing')
for name in ['healing_metrics_final_test.csv','healing_incidents_final_test.csv','healing_comparisons.csv']:
    fresh = pd.read_csv(root/name)
    frozen = pd.read_csv(Path('results/healing')/name)
    print(name, fresh.shape, 'matches committed table=', fresh.equals(frozen))
    assert fresh.equals(frozen)
metrics = pd.read_csv(root/'healing_metrics_final_test.csv')
print('Recomputed mean waiting:'); print(metrics.groupby('arm').avg_waiting_time.mean().to_string())
ledger = pd.read_csv(root/'healing_incidents_final_test.csv')
print('Preserved ledger verdicts:'); print(ledger.groupby(['arm','verdict']).size().to_string())
print('FULL DEFAULT-THRESHOLD FALLBACK REPLAY (scenario found in study, not selected as performance evidence)')
config = HealingExperimentConfig()
gen = WorkloadGenerator(config.families)
hw = next(w for w in build_healing_workloads('final_test',config,gen) if w.family=='poisson_arrivals' and w.repetition==21)
ctrl = SelfHealingController(config.healing,config.scheduler,RuleHealingPolicy(config.healing))
ctrl.begin_episode(201)
run = RuntimeSimulator(config.scheduler).run(hw.workload,ctrl,training=False)
episode = ctrl.end_episode()
print('family=',hw.family,'rep=',hw.repetition,'fingerprint=',hw.fingerprint)
for i in episode.incidents:
    print('trigger',i.decision_index,i.simulated_time,'diagnosis=',i.diagnosis_statement,
          'action=',i.action_name,'verdict=',i.verdict,'residual=',i.closure_residual,
          'confirmed failures=',i.consecutive_failures,'window end=',i.window_end_time,
          'fallback=',i.fallback_triggered,'rollback=',i.rollback_target)
assert episode.fallback_triggered
last = next(i for i in episode.incidents if i.fallback_triggered)
tail = [r for r in run.decisions if r.start_time >= last.window_end_time]
print('Tail policy IDs after fallback:', [r.decision.action for r in tail])
assert tail and all(r.decision.action==episode.settings_final.policy for r in tail)
for fixture in json.loads(Path('/tmp/os-teacher-demo/demo.json').read_text())['fixtures']:
    waits = [o['completion_time']-o['arrival_time']-o['burst_time'] for o in fixture['outcomes']]
    total = sum(waits)
    interval_total = sum(d['waiting_increment'] for d in fixture['decisions'])
    print('DEMO seed=',fixture['seed'],'process waiting sum=',total,'interval waiting sum=',interval_total,
          'mean=',total/len(waits),'verdict=',fixture['incidents'][0]['verdict'])
    assert total==interval_total and total/len(waits)==fixture['metrics']['avg_waiting_time']
path=Path('/tmp/os-teacher-drift/drift_phase_transition_estTrue_cv0.3')
detail=pd.read_csv(str(path)+'_per_trace.csv'); summary=pd.read_csv(str(path)+'_summary.csv')
print('DRIFT detail rows=',len(detail),'summary rows=',len(summary),'columns=',detail.columns.tolist())
for row in summary.to_dict('records'):
    actual=detail[detail.arm==row['arm']].mean_waiting.mean()
    assert abs(actual-row['mean_waiting'])<1e-9
print('All drift summary mean waits equal grouped detail CSV values')
