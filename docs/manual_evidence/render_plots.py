"""Render fresh-run tables; no training, historical artifact replacement or tuning."""
from pathlib import Path
import pandas as pd
from visualization.healing_plots import plot_healing_figures
root=Path('/tmp/os-teacher-healing')
paths=plot_healing_figures(Path('/tmp/os-teacher-healing-figures'),
    pd.read_csv(root/'healing_metrics_final_test.csv'),pd.read_csv(root/'healing_comparisons.csv'),
    pd.read_csv(root/'healing_detection_summary.csv'),pd.read_csv(root/'healing_audit.csv'),
    pd.read_csv(root/'healing_action_probe.csv'))
for name,path in paths.items():
    print(name,path,'bytes=',path.stat().st_size)
print('Rendered all three figures from fresh-run CSV data; historical figures unchanged.')
