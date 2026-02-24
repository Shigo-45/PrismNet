# Repository Guidelines

## Project Structure & Module Organization
- `prismnet/` contains core library code (models, training loop, utilities).
- `prismnet_eval/` hosts ablation and evaluation helpers.
- `tools/` provides CLI-style scripts for training, evaluation, and analysis.
- `exp/` contains runnable experiment shell scripts (train/eval/infer/saliency).
- `data/` holds datasets (TSV/HDF5). `motif_construct/` holds Perl/R motif scripts.
- `docs/plans/` tracks experiment plans. Use `tmp/` for one-off scripts or notes.

## Build, Test, and Development Commands
- Install dependencies (UV or pip):
  - `uv sync` (uses `pyproject.toml`/`uv.lock`) or
  - `pip install -r requirements.txt && pip install -e .`
- Prepare datasets: `tools/gdata_bin.sh` (TSV → HDF5 in `data/clip_data/`).
- Train: `exp/prismnet/train.sh <PROTEIN> <DATA_DIR>`
- Evaluate: `exp/prismnet/eval.sh <PROTEIN> <DATA_DIR>`
- Inference: `exp/prismnet/infer.sh <PROTEIN> /path/to/input.tsv`

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8 naming (`snake_case` for functions/vars, `CamelCase` for classes).
- Keep scripts simple and CLI-driven; configuration is typically passed via shell scripts in `exp/`.
- Prefer clear, self-documenting names; add brief comments only for non-obvious logic.

## Testing Guidelines
- No dedicated test framework is configured yet. If you add tests, prefer `pytest` and place them in `tests/` with `test_*.py` naming.
- For model changes, add at least one quick smoke check (e.g., a minimal forward pass) and document how to run it.

## Commit & Pull Request Guidelines
- Commit messages follow a conventional format: `<type>: <summary>` (e.g., `feat:`, `fix:`, `chore:`, `analysis:`).
- If using PRs, include: goal summary, key commands run (with outputs/paths), and links to any generated artifacts in `out/` or `docs/`.

## Security & Configuration Tips
- Check GPU/RAM before training: `nvidia-smi` and `free -h`.
- Avoid running more than 3 training jobs concurrently; PrismNet training is resource-intensive.
