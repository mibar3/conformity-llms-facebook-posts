# Testing Conformity in Large Language Models Using Simulated Social Media Posts (Master Thesis)

Do vision-language models (VLMs) show social-conformity bias: favoring a popular-but-incorrect
post over an accurate-but-unpopular one when acting as a synthetic social-media user? This repo
holds the full pipeline: synthetic stimulus generation, a model-agnostic experiment harness, and
the statistical analysis, for a study spanning eight VLMs. It does not restate the study's
methodology or findings; those live in the thesis document itself (not part of this repository).

**All stimulus data -- every chart, post, and rendered PNG -- is already generated and tracked
directly in this repository.** To run benchmarking or the experiments, nothing needs to be
created first; open a notebook and run it. Chart creation, post creation, and HTML->PNG
conversion only matter if you want to extend the study or regenerate something from scratch,
see "The pipeline, step by step" below. For the exact run order and per-machine requirements,
see [`REPRODUCE.md`](REPRODUCE.md).

## Where the findings actually are

Each of these is self-contained: it reads a tree's own `outputs/*.json` and generates its own
tables and figures. This is what the thesis draws from.

| Notebook | What's in it |
|---|---|
| `benchmarking/1-overview-findings-original-plot.ipynb` | Phase 1 perception results, original chart, all 6 models |
| `benchmarking_simple_plot/1-overview-findings-simple-plot.ipynb` | Phase 1, simplified chart pilot |
| `benchmarking_simple_plot_bigfont/1-overview-findings-simple-plot_big_font.ipynb` | Phase 1, simplified chart + larger font pilot |
| `experiments/e1/1-overview-findings-e1.ipynb` | Main conformity experiment (E1), all 6 models: the core result |
| `experiments/e1_authority_grid/1-overview-findings-e1-authority.ipynb` | Authority pilot (credentialed "Dr." profile vs. engagement) |
| `experiments/e1_climate/1-overview-findings-e1-climate.ipynb` | Generalization pilot (climate chart/domain) |
| `experiments/e1_simple_plot_bigfont/1-overview-findings-e1-simple-plot-bigfont.ipynb` | E1 on the simplified chart, both with and without the larger font, for the two models piloted on it |

The neutral-claim pilot (`neutral_pilot/`, `experiments/e1_neutral/`) has no overview notebook of
its own; its one model's results are only in that tree's raw `outputs/*.json`.

## The pipeline, step by step

Not required to reproduce existing results (see above); this is for extending the study or
regenerating stimuli from scratch. Each step skips whatever already exists, so re-running any of
it is safe.

1. **Create the pie charts.** `spotify_pie_plot/100_versions_pie_plots.ipynb` (original 9-slice
   chart), `100_versions_pie_plot_simple.ipynb` (simplified), `100_versions_pie_plot_simple_bigfont.ipynb`
   (simplified, larger font). Each writes 100 chart PNGs.
2. **Turn a chart into a post.** `utils/generate_profile_posts.py --slug <name>` builds the main
   study's posts for any profile: `remy-ashford` (the default) or `dr-remy-ashford` (the
   authority/verified-badge condition), or a new one. The two simple-plot/bigfont pilots are
   built by their own notebooks instead: `utils/updated_post_generator_all_visible_emojis.ipynb`
   and its `_bigfont` counterpart, since no `.py` script covers those chart pools.
3. **Convert HTML to PNG.** Needs Chromium, not a GPU: a different machine than the rest of the
   pipeline (see `REPRODUCE.md`). `utils/render_profile_html_to_png.py --slug <name>` for the
   profile-based posts from step 2; `utils/render_html_to_png.py <target>` for everything else
   (`--list` shows every target).
4. **Log in to Hugging Face once.** `utils/huggingface_login.ipynb`, before step 5 or 6, on
   whichever machine runs them: see `REPRODUCE.md` for why and how.
5. **Run Phase 1 perception benchmarking**: `benchmarking/<model>-benchmarking.ipynb` (optional,
   but recommended before trusting Phase 2: validates the model can read the chart at all).
6. **Run Phase 2, the actual experiment**: `experiments/e1/<model>/e1-<model>.ipynb`, top to
   bottom. Each notebook loads its model, builds (or reuses) the seeded image sample via
   `e1_utils.sampling.build_paired_sample`, runs the isolated-judgment and paired A/B protocols
   via `e1_utils.e1_optimized`, and writes results incrementally to `outputs/e1_results_*.json`
   (safe to stop and restart: already-completed trials are skipped).
7. **The pilots**, each self-contained, none touch the main study's data:

   | Pilot | Stimuli | Experiment notebooks |
   |---|---|---|
   | Simplified chart | `spotify_pie_plot/pie_plot_posts/metrics_simple_plot/` | `experiments/e1_simple_plot/` |
   | Simplified chart + larger font | `..._simple_plot_bigfont/` | `experiments/e1_simple_plot_bigfont/` |
   | Authority (credentialed profile) | `benchmarking/{correct,incorrect}/dr-remy-ashford/` | `experiments/e1_authority_grid/` |
   | Generalization (climate chart) | `climate_pilot/` (`generate_climate_stimuli.py` + `render_climate_html_to_png.py`) | `experiments/e1_climate/` |
   | Neutral claim | `neutral_pilot/` (`generate_neutral_stimuli.py` + `render_neutral_html_to_png.py`) | `experiments/e1_neutral/` |

8. **Analyze**: the overview-findings notebooks listed at the top.

## Repository map

| Path | What it is |
|---|---|
| `REPRODUCE.md` | Run order for the whole pipeline, per stage and per machine. |
| `reproducibility/` | Author-side verification: `prepare_stimuli.py`, `verify_stimuli.py`, `compare_trees.py`, and the fingerprint manifest they check against. Not needed to reproduce results. |
| `utils/` | Shared post-generation code, the chart-creation and post-generator notebooks, the HTML->PNG renderers, and `huggingface_login.ipynb`. |
| `spotify_pie_plot/` | Chart pool and engagement-scaled post generation for the main study and the simple-plot/bigfont pilots. |
| `benchmarking/`, `benchmarking_simple_plot/`, `benchmarking_simple_plot_bigfont/` | Phase 1 perception validation, plus the stimulus tree `experiments/e1/` actually reads from. |
| `experiments/e1/` | The main experiment. One subfolder per model, each self-contained. `e1_utils/` is the shared harness every model's notebook imports. |
| `experiments/e1_simple_plot/`, `experiments/e1_simple_plot_bigfont/`, `experiments/e1_authority_grid/`, `experiments/e1_climate/`, `experiments/e1_neutral/` | The pilots, see the table above. |
| `climate_pilot/`, `neutral_pilot/` | Stimulus generation for those two pilots, independent of the main study's pool. |

## Models

All model inference (Phase 1 benchmarking and the Phase 2 experiments) was run on an **NVIDIA
H100 80GB HBM3**, CUDA 12.4, driver 550.127.08. Every model fits comfortably within a **40GB MIG
partition** on its own; none needed more. The full 80GB card shows up in some notebooks' own
`nvidia-smi` cell only because multiple models were sometimes run concurrently, sharing the
undivided card rather than separate MIG slices: that reflects scheduling, not any individual
model's memory requirement.

Exact `nvidia-smi` output is captured in each model's own notebook (an early cell in
`experiments/e1/<model>/e1-<model>.ipynb`).

## Adding a new model

1. Add a `run_inference_<model>(messages, model, processor, device) -> str` function to
   `experiments/e1/e1_utils/inference_<model>.py`, matching the signature every other adapter
   in that folder already uses.
2. Copy an existing model's notebook (pick one with a similar loading pattern, e.g. another
   `AutoModelForImageTextToText` model) as a starting point rather than writing one from
   scratch; only the model-loading cell and the `inference_fn=` argument in each run cell need
   to change.
3. Point it at a fresh `outputs/` folder; `build_paired_sample` will reuse the existing
   `selected_images.json` in `experiments/e1/` automatically, so the new model sees the same
   image sample as every other one.

## Common pitfalls

- **`ModuleNotFoundError: No module named 'selenium'` / missing Chromium**: rendering needs
  both, and the GPU server has neither by design. Run `python3 utils/render_html_to_png.py
  --check-env` on the machine you meant to render on; it names what is missing.
- **A notebook run silently does nothing / prints "Skipping"**: this is the harness's
  resume logic working as intended (results already exist for that trial in the corresponding
  `outputs/e1_results_*.json`); delete or move that file if you want to force a clean re-run.
- **Different models need different `transformers`/`torch` pins**: each model's notebook has
  its own setup cell for this reason; don't assume one environment works for all models
  without checking that cell first.
- **`RuntimeError: Not logged in to Hugging Face`**: run `utils/huggingface_login.ipynb` once
  first; see `REPRODUCE.md`.
