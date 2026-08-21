# legacy/

Things that are not used by the paper / current pipeline, kept for reference.

| Path | What it was | Why it is here |
|---|---|---|
| `analysis/plot.py` | First single-log metrics/plot script (reads `.eval` zips by hand, expects the old `pattern` scorer). | Superseded by `multi_model_plots.py`; broken with the current `includes` scorer. |
| `analysis/lineplots.py` | Single-log precursor of `multi_model_plots.py` (metrics / selection-rate / taxonomy vs Δ). | Hard-codes a log that no longer exists; everything it does is in `multi_model_plots.py`. |
| `analysis/plot_log.py` | One-off SPR-vs-Δ / selection-rate plot for the Qwen "sacrifice" prompt ablation. | Not a paper figure; `multi_model_plots.py` covers it. |
| `assets/system_card.md` | First hand-written system card ("MODEL Lite"). | Replaced by `assets/system_card_templates.yaml` (6 templates with `{placeholders}`). |
| `dataset_test/` | 10-scenario smoke-test output of `dataset_crafting.py`. | Test artefact. |
| `misc/plots_s.txt` | Stray notes about a Grok-engagement LMM protocol. | Unrelated to this repo. |
| `misc/vllm*.log` | vLLM server stdout. | Debug output. |
| `logs/` (gitignored) | Eval logs from Dec 2025 (pre-final-dataset runs, gpt-4o-mini runs, zip export). | Pre-date the final dataset; not in the paper. |
