# Reproducing this study

Start here. `README.md` describes what the repository contains; this file is the
order to run it in.

The pipeline has three stages, and they do not run on the same machine:

| Stage | What it needs | Roughly |
|---|---|---|
| 1. Stimuli | Python + Pillow | minutes |
| 2. Rendering HTML to PNG | a Chromium binary, no GPU | only if a tree is short |
| 3. Experiments | an NVIDIA GPU, no Chromium | hours per model |

Stage 2 is the one that usually trips people up: the GPU server has no browser,
and the rendering machine has no GPU. The images are therefore rendered on one
machine and copied to the other. **Stage 2 is normally not needed at all** — the
rendered PNGs are already in the repository.

## Before any stage: install the base dependencies

```bash
python3 -m pip install --user -r requirements.txt
```

Covers stimulus generation, rendering, the reproducibility scripts, and the
analysis/overview-findings notebooks. It does **not** cover `torch`/`transformers` —
those are pinned per model and installed by each model's own notebook (stage 3),
because different models need different versions. See `requirements.txt`'s own
header comment, or README.md's "Common pitfalls", for why.

---

## Stage 1: put the stimuli in place

```bash
python3 reproducibility/prepare_stimuli.py
```

(Needs Pillow, already covered by `requirements.txt` above.)

That does three things:

- unpacks the 200 baseline (zero-engagement) PNGs from their tracked archives;
- installs the main study's stimuli into
  `benchmarking/{correct,incorrect}/remy-ashford/`, the tree the e1 and phase 1
  notebooks actually open. That tree is gitignored, because the same images are
  already tracked once under `spotify_pie_plot/`, so a fresh clone has every image
  but an empty experiment tree and every notebook fails on file-not-found until
  this runs. Hard links, so the second copy costs no disk;
- prints one table: every stimulus tree, HTML count against PNG count, and whether
  the experiment tree is ready.

If it says every tree is fully rendered, **skip stage 2 entirely** and go to
stage 3. Re-running is safe: nothing already in place is overwritten.

The pilots (simplified chart, larger font, climate, neutral) read straight out of
`spotify_pie_plot/` and `climate_pilot/` and need nothing installed.

Gitignored here never means missing. The baselines are two tracked zips rather
than 200 loose files, and the experiment tree is a second arrangement of images
that are already tracked; neither is a re-rendering.

To confirm that for yourself, at any point:

```bash
python3 reproducibility/verify_stimuli.py --manifest    # all 14,516 images vs. the pre-refactor fingerprints
python3 reproducibility/verify_stimuli.py --survivors   # baselines vs. the copies that survived loose in the tree
python3 reproducibility/verify_stimuli.py --baselines   # all 200 baselines vs. the metrics stimuli, structurally
```

`--manifest` is the one that answers "are these the images the results came
from?". It compares against fingerprints taken at the tag `pre-path-refactor`.

### Generating stimuli from scratch instead

Not required, and it will not reproduce the exact images unless your Chromium and
fonts match the ones the study used (`--selftest` below tells you whether they
do). The generators:

```bash
python3 utils/generate_baseline_stimuli.py       # main study, zero-engagement HTML
python3 utils/generate_profile_posts.py --check  # re-derive the main study's posts and diff them
python3 climate_pilot/generate_climate_stimuli.py
```

`generate_profile_posts.py` is also how a new poster-profile condition is made
(`--profile-name`, `--verified`, `--slug`); see its docstring.

The engagement-scaled posts come from the notebooks under `spotify_pie_plot/`
(`100_versions_pie_plots.ipynb` builds the chart pool; the
`updated_post_generator_*` notebooks build the posts).

---

## Stage 2: render HTML to PNG

Only if stage 1 reported a tree as short, or you regenerated HTML yourself. On
the machine with Chromium:

```bash
python3 utils/render_html_to_png.py --check-env    # is this machine set up?
python3 utils/render_html_to_png.py --selftest     # does it reproduce the study's pixels?
python3 utils/render_html_to_png.py --list         # what can be rendered
```

`--selftest` renders one post whose original PNG is in the archive and compares
the two byte for byte. If it says identical, anything you render will match the
study's images exactly.

Then render whatever is short, by name or by path:

```bash
python3 utils/render_html_to_png.py main-metrics --dry-run   # show the plan, render nothing
python3 utils/render_html_to_png.py main-metrics
python3 utils/render_html_to_png.py all
```

Existing PNGs are skipped, so an interrupted run resumes by re-running the same
command, and nothing already in the repository is overwritten. Copy the resulting
PNG trees to the GPU server before stage 3.

This replaces the hand-edited cells in `utils/html_to_png.ipynb`. That notebook
still works and is what produced the study's images; it is kept for the record,
but it needs a block edited per condition and hardcodes one machine's paths.

---

## Stage 3: run the experiments

On the GPU machine. Phase 1 validates that a model can read the chart at all;
phase 2 is the experiment itself.

```
benchmarking/<model>-benchmarking.ipynb           # phase 1, optional but recommended
experiments/e1/<model>/e1-<model>.ipynb           # phase 2
```

Run a notebook top to bottom. Each one:

- pins its own `transformers` / `torch` versions in its setup cell — the models do
  not share an environment, so check that cell before assuming yours works;
- builds or reuses the seeded 50/100-image sample via
  `e1_utils.sampling.build_paired_sample`, cached in `selected_images.json` so
  every model sees the identical sample;
- writes results incrementally to `outputs/e1_results_*.json` and skips trials
  that are already there, so a run is resumable. A cell that prints "Skipping" and
  does nothing has already finished; delete the output file to force a re-run.

Then the analysis:

```
statistical_analysis/gee_analysis.ipynb           # cross-model formal tests
```

`e1_utils.e1_analysis_optimized` has the per-model summaries
(`analyse_single`, `analyse_paired`, `analyse_metrics_paired`).

### The pilots

Self-contained, none of them touch the main study's data:

| Pilot | Stimuli | Notebooks |
|---|---|---|
| Simplified chart | `metrics_simple_plot` | `experiments/e1_simple_plot/` |
| Larger chart font | `metrics_simple_plot_bigfont` | `experiments/e1_simple_plot_bigfont/` |
| Climate generalization | `climate_pilot/` | `experiments/e1_climate/` |

---

## If something looks wrong

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: selenium`, or no Chromium | You are on the GPU machine. Rendering belongs on the other one. |
| A notebook prints "Skipping" and does nothing | Resume logic working. Results already exist in `outputs/`. |
| `--selftest` says the pixels differ | Your Chromium or fonts differ from the study's. The PNGs in the repository are still the authoritative ones; render nothing and use them. |
| `main-baselines` lists 3 files to render | Three leftover HTML drafts sit in the baselines tree (`dr-remy-ashford...`, and the one `--selftest` uses). Nothing reads them; rendering them is harmless, ignoring them is fine. |
| A tree looks short by a few hundred files | Folders named `...ignore` are abandoned drafts no condition reads, and are excluded from every count. `--include-dead` renders them anyway. |
| `FileNotFoundError` under `benchmarking/...` in an e1 notebook | The experiment tree was never installed. Run `python3 reproducibility/prepare_stimuli.py`. |
| `verify_stimuli.py --manifest` reports a change | An image was edited or re-rendered locally. `git checkout pre-path-refactor` returns the tree to the state the results came from. |
