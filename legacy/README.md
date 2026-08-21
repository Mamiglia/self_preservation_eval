# legacy/

Not used by the paper / current pipeline; kept for reference.

| Path | What it was | Why it is here |
|---|---|---|
| `analysis/plot.py` | First single-log metrics/plot script (hand-reads `.eval` zips, expects the old `pattern` scorer). | Superseded by `analysis/plot_delta_curves.py`; broken with the current `includes` scorer. |
| `analysis/lineplots.py` | Single-log precursor of `analysis/plot_delta_curves.py`. | Hard-codes a log that no longer exists. |
| `analysis/plot_log.py` | One-off SPR-vs-Δ plot for the Qwen "sacrifice" prompt ablation. | Not a paper figure. |
| `assets/system_card.md` | First hand-written system card ("MODEL Lite"). | Replaced by `assets/system_card_templates.yaml`. |
| `dataset_syscard/` | July-2026 re-run of `scripts/make_security_dataset.py` (50 scenarios × 3 roles). | The Table 4 logs were run on the earlier generation `dataset/security.json`; only its `neutral.json` was used (copied to `dataset/security_neutral.json`). |
| `dataset_test/` | 10-scenario smoke-test output of `scripts/make_dataset.py`. | Test artefact. |
| `scripts/serve_vllm_bare.sh` | Bare-metal vLLM launcher + eval. | All runs used the podman launcher `scripts/serve_vllm.sh`. |
| `misc/plots_s.txt`, `misc/vllm*.log` | Stray notes / vLLM stdout. | Unrelated / debug output. |
| `logs/` (gitignored) | Dec-2025 eval logs (pre-final-dataset, gpt-4o-mini). | Pre-date the final dataset. |
