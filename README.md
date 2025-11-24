# NL2SQL Testbench

Lightweight harness for generating and evaluating Text-to-SQL predictions on Spider via Novita AI and test-suite-sql-eval.

## Quick Start
- Install deps: `./startup.sh`
- Add API key: `echo "NOVITA_API_KEY=your_key" > NOVITA_API_KEY`
- Activate env: `source .venv/bin/activate`

Generate only (no evaluation):
```bash
python run_testbench.py run_test --model google/gemma-3-12b-it --examples 20
```

Re-evaluate an existing run directory:
```bash
python run_testbench.py evaluate_run --run-dir results/<test_to_evaluate>
```

List configured models:
```bash
python run_testbench.py list-models
```

## Project Layout
```
run_testbench.py          # CLI entrypoint
run_all_models.sh         # Batch runner (generation only)
src/gathering/            # Core logic
  spider1_testbench.py    # Run orchestration, eval hooks
  spider1.py              # Data loading, test-suite integration
  prompting_strategy.py   # Prompt generation and SQL extraction
test-suite-sql-eval/      # Upstream evaluation scripts
spider/                   # Spider dataset (dev.json, tables.json, dbs)
results/                  # Per-run outputs (predictions, gold, results.json)
```

## Key Commands
- Single run (with immediate eval): `python run_testbench.py run_test --model <model> --examples 50 --evaluate`
- Batch (models in run_all_models.sh): `./run_all_models.sh`
- Re-eval with custom options: `python run_testbench.py evaluate_run --run-dir <dir> --spider-path spider --test-suite-path test-suite-sql-eval`

## Outputs
Each run creates `results/<name>_<id>/` containing:
- `<model>_<strategy>/predictions.txt` and `gold.txt` (tab-separated gold SQL and db_id)
- `<model>_<strategy>/evaluation_output.txt` (if evaluated)
- `results.json` (summaries and evaluation details)
- `metadata.json` per model/strategy

## Notes
- Models are hardcoded in `src/gathering/config.py`; unavailable models should be removed to avoid 404s.
- Long LLM responses are truncated when saved to keep result files small; raw requests are not stored.
- test-suite evaluation uses `--etype all --plug_value --keep_distinct false` by default; adjust via `evaluation` options when re-evaluating. 
