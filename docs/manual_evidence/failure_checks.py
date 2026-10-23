"""Deliberate failure probes; optional import failure is simulated, not uninstalled."""
from pathlib import Path
import contextlib
import importlib.abc
import io
import pandas as pd
from errors import ValidationError
from visualization.healing_plots import plot_healing_figures

path=Path('/tmp/os-teacher-invalid-plot')
assert not path.exists(), 'Choose a fresh path before rerunning this probe'
try:
    plot_healing_figures(path, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                         pd.DataFrame(), pd.DataFrame({'reward_band_units':[1]}))
except ValidationError as exc:
    print('INVALID PLOT:', type(exc).__name__, str(exc))
else:
    raise AssertionError('legacy input unexpectedly accepted')
print('Invalid plot output directory created:',path.exists())
assert not path.exists()
assert not Path('/tmp/os-teacher-missing.pdf').exists()
print('Missing source report: no PDF was written')

# Use a fresh interpreter: earlier invalid-plot probe imported the real plotting module.
import subprocess
import sys
code = r"""
import sys, importlib.abc, contextlib, io
from pathlib import Path
class NoMatplotlib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'matplotlib' or fullname.startswith('matplotlib.'):
            raise ModuleNotFoundError('simulated missing optional plotting dependency: matplotlib')
        return None
sys.meta_path.insert(0, NoMatplotlib())
from main import main
with contextlib.redirect_stdout(io.StringIO()):
    status = main(['demo', '--results-dir', '/tmp/os-teacher-no-matplotlib'])
print('SIMULATED missing matplotlib: real demo exit=', status)
assert status == 0
try:
    from visualization.healing_plots import plot_healing_figures
except ModuleNotFoundError as exc:
    print('SIMULATED plotting import:', str(exc))
else:
    raise AssertionError('plotting unexpectedly imported')
print('Missing optional dependency was simulated before its first import; no packages removed.')
"""
subprocess.run([sys.executable, '-c', code], check=True)
print('FAILURE PROBES COMPLETE')
