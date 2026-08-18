# conformity-llms-facebook-posts

Do vision-language models (VLMs) show social-conformity bias — favoring a popular-but-incorrect
post over an accurate-but-unpopular one when acting as a synthetic social-media user? This repo
holds the full pipeline: synthetic stimulus generation, a model-agnostic experiment harness, and
the statistical analysis, for a study spanning eight VLMs.

This file documents how to reproduce or extend the pipeline. It does not restate the study's
methodology or findings in full — that lives in the thesis document itself (not part of this
repository).

## Repository map

| Path | What it is |
|---|---|
| `utils/` | Shared post-generation code (`post_generator_all_visible_emojis.py` — builds the HTML/CSS Facebook-style post, given a chart image, claim text, profile, and engagement counts) and chart-creation notebooks. |
| `spotify_pie_plot/` | Stimulus generation for the main study: the pie-chart pool, the correct/incorrect claim variants, and the engagement-scaled post pool. |
| `benchmarking/`, `benchmarking_simple_plot/`, `benchmarking_simple_plot_bigfont/` | Phase 1 perception validation (can each model read the chart correctly?) plus the organized, final stimulus image tree the main experiment (`experiments/e1/`) actually reads from. |
| `experiments/e1/` | The main experiment. One subfolder per model (`gemma4-12b/`, `qwen3-vl-8b/`, etc.), each a self-contained notebook + `outputs/` folder. `experiments/e1/e1_utils/` is the shared harness (see below) imported by every model's notebook. |
| `experiments/e1_simple_plot/`, `experiments/e1_simple_plot_bigfont/` | Chart-legibility pilots (simplified chart, larger font) on a subset of models. |
| `experiments/e1_climate/` | Generalization pilot run notebooks (chart type + content domain) — see `climate_pilot/` below for the stimulus generation half. |
| `climate_pilot/` | Stimulus generation for the generalization pilot: a fabricated solar-vs-wind investment line chart, deliberately independent of the main study's pool. Self-contained, documented at the top of `generate_climate_stimuli.py`. |
| `statistical_analysis/` | Cross-model formal statistics (GEE regression, `gee_analysis.ipynb`), grid overview figures, and supporting analysis scripts. |

## Environment: this pipeline spans two separate machines

**Stimulus rendering** (HTML → PNG screenshot) needs Python + Selenium + a working Chromium
binary. This is normally run on a machine that already has Chromium installed (referred to as
`scc2` in this project) — it does **not** need a GPU.

**Model inference** (loading and running each VLM) needs a CUDA GPU + PyTorch + `transformers`
(+ model-specific extras, e.g. `qwen_vl_utils` for Qwen). This does **not** need Chromium or
Selenium.

Trying to run a rendering step on the GPU machine, or an inference step on the rendering
machine, will fail on a missing dependency that installing more Python packages won't fix (no
Chromium binary on the GPU machine, no GPU on the rendering machine). Match the step to the
machine.

## Reproducing the main study

1. **Generate stimuli** (rendering machine): pie-chart pool + engagement-scaled HTML posts —
   see the notebooks under `spotify_pie_plot/` and `utils/`, then render with
   `utils/html_to_png.ipynb`. This step is normally already done; the resulting PNGs live under
   `benchmarking/{correct,incorrect}/remy-ashford/`.
2. **Run Phase 1 perception benchmarking** (GPU machine, optional — validates the model can read
   the stimulus before trusting Phase 2 results): `benchmarking/<model>-benchmarking.ipynb`.
3. **Run Phase 2 (the actual experiment)** (GPU machine): open `experiments/e1/<model>/e1-<model>.ipynb`
   and run its cells top to bottom. Each notebook:
   - loads its model,
   - builds (or reuses) a reproducible 50/100-image sample via `e1_utils.sampling.build_paired_sample`
     (seeded, cached to `selected_images.json` so every model sees the identical sample),
   - runs the single-image (baseline/likes-only/metrics) and paired A/B forced-choice protocols
     via `e1_utils.e1_optimized`,
   - writes results incrementally to `outputs/e1_results_*.json` (resumable — safe to stop and
     restart; already-completed trials are skipped).
4. **Analyze**: `e1_utils.e1_analysis_optimized` functions (`analyse_single`, `analyse_paired`,
   `analyse_metrics_paired`, ...) summarize a given model's `outputs/*.json` into accuracy
   tables/plots. `statistical_analysis/gee_analysis.ipynb` runs the cross-model formal
   statistical tests once every model's `metrics`/`likes_only`/`likes_only_noise` grids exist.

## Reproducing (or extending) the generalization pilot

Self-contained, does not touch the main study's data:

```bash
# Rendering machine
python3 climate_pilot/generate_climate_stimuli.py       # chart + HTML generation, no GPU/browser needed
python3 climate_pilot/render_climate_html_to_png.py      # HTML -> PNG, needs Selenium + Chromium

# GPU machine, per model
# open experiments/e1_climate/<model>/e1-climate-<model>.ipynb and run top to bottom
```

See the module docstring at the top of `climate_pilot/generate_climate_stimuli.py` for the
design rationale (why a line chart, why fabricated data, why the claim is phrased the way it
is) before changing the stimulus design.

## Adding a new model

1. Add a `run_inference_<model>(messages, model, processor, device) -> str` function to
   `experiments/e1/e1_utils/inference_<model>.py`, matching the signature every other adapter
   in that folder already uses.
2. Copy an existing model's notebook (pick one with a similar loading pattern — e.g. another
   `AutoModelForImageTextToText` model) as a starting point rather than writing one from
   scratch; only the model-loading cell and the `inference_fn=` argument in each run cell need
   to change.
3. Point it at a fresh `outputs/` folder; `build_paired_sample` will reuse the existing
   `selected_images.json` in `experiments/e1/` automatically, so the new model sees the same
   image sample as every other one.

## Common pitfalls

- **`ModuleNotFoundError: No module named 'selenium'` / missing Chromium** — you're likely on
  the wrong machine for this step (see "Environment" above), or just need
  `python3 -m pip install --user selenium` on the rendering machine.
- **A notebook run silently does nothing / prints "Skipping"** — this is the harness's
  resume logic working as intended (results already exist for that trial in the corresponding
  `outputs/e1_results_*.json`); delete or move that file if you want to force a clean re-run.
- **Different models need different `transformers`/`torch` pins** — each model's notebook has
  its own setup cell for this reason; don't assume one environment works for all models
  without checking that cell first.
