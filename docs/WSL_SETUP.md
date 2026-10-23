# WSL2 Ubuntu installation and execution

## 1. Scope and verified environment

Target: WSL2 Ubuntu, Bash, x86-64. Execution was tested on Debian 12 Linux,
Python 3.11.2, not inside Windows/WSL. These Ubuntu installation instructions
have been reviewed but have not been executed in WSL. No GUI is required.

`requirements.lock` records Python **3.11.2** and pins the simulation/test packages.
There is no pyproject declaring a broader supported Python range. Do not assume
Ubuntu's default Python (3.10 on 22.04 or 3.12 on 24.04) reproduces that environment.
The commands below install the recorded interpreter privately without replacing
Ubuntu's Python. Python 3.11.2 is an old patch release: use this isolated environment
for local reproduction only, not an exposed production service. Newer patched
3.11 interpreters are sensible for future maintenance but were not tested here.

## 2. Install WSL (Windows PowerShell as administrator)

```powershell
wsl --install -d Ubuntu-24.04
wsl --set-default-version 2
wsl --list --verbose
```

Restart Windows if requested, launch Ubuntu, and create a Linux username/password.
The list should show version 2 for Ubuntu. If an existing distribution is version 1:

```powershell
wsl --set-version Ubuntu-24.04 2
```

Use the installed distribution's exact name from `wsl --list --verbose`.
All subsequent commands run in the Ubuntu terminal, not PowerShell.

## 3. Install system packages and the recorded Python

```bash
sudo apt update
sudo apt install -y git ca-certificates build-essential pkg-config \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libffi-dev liblzma-dev libncurses-dev uuid-dev tk-dev
mkdir -p "$HOME/src"
git clone --depth 1 --branch v3.11.2 \
  https://github.com/python/cpython.git "$HOME/src/cpython-3.11.2"
cd "$HOME/src/cpython-3.11.2"
./configure --prefix="$HOME/.local/python-3.11.2" --with-ensurepip=install
make -j2
make install
"$HOME/.local/python-3.11.2/bin/python3.11" --version
```

Expected interpreter version: `Python 3.11.2`. Compilation is a one-time installation
step and may take several minutes. No sudo is used for `make install`. If that source
folder already exists, do not clone over it; inspect it and resume only after verifying
`git -C "$HOME/src/cpython-3.11.2" describe --tags --exact-match` is `v3.11.2`.
If Python 3.11.2 is already installed, its absolute executable path can replace the
private-install path below. On the tested Debian host it was `/usr/bin/python3.11`.

## 4. Checkout and isolated dependencies

Keep the project on the Linux filesystem, for example `~/OS-Project`, not `/mnt/c`.
For a fresh clone:

```bash
cd "$HOME"
git clone https://github.com/lostpilot404/OS-Project.git
cd "$HOME/OS-Project"
git fetch origin
git switch main
```

For an existing checkout:

```bash
cd "$HOME/OS-Project"
git status --short
# Preserve/commit your own changes before switching; do not reset or clean.
git fetch origin
git switch main
```

For an existing checkout, after switching to `main`, update with
`git pull --ff-only origin main`. If no local `main` exists, use
`git switch --track origin/main` instead. The work originally prepared on
`arena/49fb0ff9-os-project` is consolidated into `main`; use `main` for normal work.
Check the delivered commit with `git log -1 --oneline`. Do not switch branches during
a presentation.

```bash
git branch --show-current
"$HOME/.local/python-3.11.2/bin/python3.11" -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -r requirements.lock
python -m pip check
python main.py --help
```

Do not reuse an existing `.venv` made by another interpreter. Rename it first if it
contains work you need. Do not use `sudo pip` or install into the system Python.
`requirements.txt` lists direct pins; `requirements.lock` is the reproduction input.
No pip upgrade is required. Network access is needed for packages, not for the demo.

## 5. Quick presentation and tests

```bash
cd "$HOME/OS-Project"
source .venv/bin/activate
python main.py demo
python -m pytest -q tests/test_demo.py
python -m pytest -q
```

The demo runs two real `mixed` workloads (generator seeds 3 and 13), prints inputs,
arrived-ready observations, service segments, diagnosis, encoded state, rule action,
verification and measured metrics. It uses the default self-healing configuration.
It does not fit Q-learning, regenerate the study, or use final-test data.
`results/demo/demo.json` contains the full structured trace and is gitignored.
A repeat overwrites only this disposable demo output. To choose another location:

```bash
python main.py demo --results-dir "$HOME/os-demo-output"
```

Paths with spaces must be quoted. A writable output directory is created automatically.
See `docs/DEMO_SCRIPT_FOR_LECTURER.md` for the speaking guide and actual captured output.

## 6. Optional report dependencies and repeatable PDF build

```bash
sudo apt install -y fonts-dejavu-core
cd "$HOME/OS-Project"
source .venv/bin/activate
python -m pip install -r requirements-report.txt
python tools/build_plain_report.py \
  --source docs/OS_Project_Final_Report.md \
  --output docs/OS_Project_Final_Report.pdf
```

ReportLab and pypdf support this builder; python-docx is used only by the separate
Review 2 DOCX builder. The existing report requirements bundle includes all three.
These packages are not needed to run the simulation. The builder reads the Markdown
as UTF-8, embeds local fonts, prints Mermaid as source, and does not run experiments.
Never manually patch the PDF. The original Review 2 and plain reports remain untouched.

## 7. Full research runs (not presentation commands)

Use fresh output paths outside the committed evidence directories:

```bash
python main.py healing-study \
  --results-dir "$HOME/os-runs/healing-new" \
  --figures-dir "$HOME/os-runs/healing-new-figures" --no-figures
```

This trains five agents with 1,200 traces each, evaluates validation/final-test splits,
runs bootstraps, audits and the default compact probe, and writes CSV/JSON/Markdown.
It is much more expensive than the demo. Seed 23819 remains frozen; do not tune on it.
Omit `--no-figures` only if optional PNG figures are wanted. `--no-probe` skips the
probe, not the other study work. Default CLI paths can overwrite committed evidence;
always use new paths for reproduction.

The default probe is 168 rows with the corrected reward. It is not a replacement for
the canonical full-validation 784-row/98-incident probe or the historical raw-proxy
168-row/21-incident archive. The canonical regeneration procedure is in
`docs/ENGINEERING_COMPLETION_REPORT.md`.

The independently configured non-healing runtime workflow is optional:

```bash
python main.py runtime-experiment \
  --results-dir "$HOME/os-runs/runtime-new"
```

It uses final-test seed 19301, 162 states and four policy actions. It is not the
healing experiment (648 states, eight recovery actions, final-test seed 23819).
`study`, `experiment`, and `train` are preserved offline workflows, not the quick demo.
Running `python main.py` with no command starts the full healing study; avoid that
mistake during the presentation.

## 8. Troubleshooting

- `python: command not found` or missing numpy/pandas: activate `.venv`; run
  `which python`, `python --version`, then install `requirements.lock` with that Python.
- `No matching distribution`: check Python version, CPU architecture, pip index and
  network. The locked setup was verified on Linux x86-64/Python 3.11.2; ARM was not tested.
- `No module named venv` in Ubuntu's default Python: do not accidentally use that
  interpreter; the private CPython build includes ensurepip and venv. An Ubuntu
  bootstrap environment instead needs `sudo apt install python3-venv`.
- Permission or Windows drive issues: run from `~/OS-Project`; choose a writable
  `--results-dir`. Do not run the project as root.
- Missing report fonts: install `fonts-dejavu-core`. Report generation does not require
  TeX, Pandoc, Mermaid CLI, a browser or internet access after dependencies are installed.
- Accidentally interrupted demo: rerun it. It has no model checkpoint or training state.
  Do not delete `results/healing`, change thresholds or launch the full study to fix a demo.
