# Repository Guidelines

## Project Structure & Module Organization
- Core CLI and orchestrator live in `run_testbench.py`; Python modules under `src/gathering/` handle dataset loading (`spider1.py`), prompting (`prompting_strategy.py`), model configs (`config.py`), evaluation (`spider1_testbench.py`), and Novita client calls.
- Datasets are expected in `spider/` and `test-suite-sql-eval/`; they are read in place and not tracked in git.
- Experiment outputs (JSON summaries, evaluation artifacts) land in `results/`; generated visuals belong in `plots/`.
- Utility scripts: `startup.sh` bootstraps a virtualenv and deps; `run_all_models.sh` drives a full sweep across the model list.

## Build, Test, and Development Commands
- Environment setup:  
  ```bash
  ./startup.sh
  source .venv/bin/activate
  ```
- Smoke run on a single model:  
  ```bash
  python run_testbench.py quick --model meta-llama/llama-3.2-3b-instruct --examples 5
  ```
- Compare multiple models (small batch):  
  ```bash
  python run_testbench.py compare-models --models meta-llama/llama-3.2-3b-instruct qwen/qwen3-coder-30b-a3b-instruct --examples 20
  ```
- Full sweep (long-running): `./run_all_models.sh`
- Inspect dataset stats only: `python run_testbench.py info`

## Coding Style & Naming Conventions
- Python, PEP 8 aligned: 4-space indentation, snake_case functions/variables, UpperCamelCase for classes.
- Prefer type hints, concise docstrings, and f-strings; keep async flows intact when touching Novita calls.
- Keep configuration in `config.py` or CLI args instead of hard-coded values; ensure outputs write under `results/`/`plots/` to avoid polluting the repo root.

## Testing Guidelines
- Use `quick` runs (5–10 examples) as fast regression checks before pushing; include at least one `compare-models` call when touching prompting or evaluation logic.
- Confirm result artifacts are produced and parsable (e.g., JSON in `results/`); spot-check a few entries for correct model/strategy metadata.
- No formal unit test suite exists yet—favor small, targeted runs tied to the code you changed.

## Commit & Pull Request Guidelines
- Follow existing history: short, imperative subjects (e.g., “refactor src codebase”, “switch back to spider1”).
- In PR descriptions, summarize scope, list commands run, and attach key metrics or plot paths from `results/`/`plots/` when relevant.
- Ensure diffs exclude datasets, virtualenvs, and large artifacts; check `.gitignore` coverage (`.venv/`, `secrets.txt`, `NOVITA_API_KEY`).

## Security & Configuration Tips
- Provide `NOVITA_API_KEY` via `secrets.txt` or environment; never commit keys (the files are ignored, but double-check before pushing).
- Protect external service usage: keep example counts low in smoke runs to avoid unnecessary API spend; prefer async paths already in the codebase.
- When adding new configs or scripts, document required env vars and expected output locations to keep runs reproducible.
