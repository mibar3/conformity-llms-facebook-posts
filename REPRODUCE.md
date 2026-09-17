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

### You need your own Hugging Face token, and accepted model licenses

Every model here loads via `from_pretrained(MODEL_ID, ...)`, and several (Gemma
especially) are **gated**: Hugging Face requires an account, the model's license
accepted on its own page, and a personal access token — none of that can be
skipped or worked around from this repo, it's how gated-model downloads work on
HF's side.

**Run `utils/huggingface_login.ipynb` once, before any other notebook.** It
prompts for your token (hidden input, never typed as literal code) and caches
it to `~/.cache/huggingface/token` on this machine — every other notebook
checks that cache automatically, so this runs once per machine/account, not
once per model.

Each model notebook itself only has a lean check for this (`whoami()`, raising
a clear error naming that notebook if you haven't run it) — the old
hand-editable auth cell some notebooks had (importing a personal, untracked
`config_hf_token.py`) is gone, replaced by this shared, portable path.

If `huggingface_login.ipynb` won't run for some reason, the equivalent from a
plain terminal:

```bash
huggingface-cli login          # one time, caches your token the same way
# or: hf auth login --force    # newer huggingface_hub versions rename the CLI
```

If a model load still fails with a gated-repo/license error after logging in,
that's HF telling you to accept *that specific model's* license on its own
page — a separate one-time step per model, not fixed by logging in again.

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

Then the analysis: each model's `1-overview-findings-*.ipynb` (self-contained, generates its
own tables/figures from that tree's `outputs/*.json` — nothing external to read first) is what
the thesis actually draws from. `e1_utils.e1_analysis_optimized` has the underlying per-model
functions (`analyse_single`, `analyse_paired`, `analyse_metrics_paired`) those notebooks call.

`statistical_analysis/` (formal cross-model GEE/Wald tests) is untracked as of 2026-09-18 —
confirmed unused by the current thesis, which cites no value from it and whose overview
notebooks have zero references to that folder. Still on disk/in git history if ever needed again.

### The pilots

Self-contained, none of them touch the main study's data:

| Pilot | Stimuli | Notebooks |
|---|---|---|
| Simplified chart | `metrics_simple_plot` | `experiments/e1_simple_plot/` |
| Larger chart font | `metrics_simple_plot_bigfont` | `experiments/e1_simple_plot_bigfont/` |
| Climate generalization | `climate_pilot/` | `experiments/e1_climate/` |
| Neutral claim | `neutral_pilot/` | `experiments/e1_neutral/` |
| Authority (verified "Dr." profile) | `benchmarking/{correct,incorrect}/dr-remy-ashford/` | `experiments/e1_authority_grid/` |

---

## Quick reference: I want to work on a specific topic, what files does that touch?

The stages above are the order to run things in; this is the index to find your way back into
one part without re-reading the whole file. Every path is relative to the repo root.

### Chart pool creation (the pie chart itself, before any post is built)

| What | Notebook | Output |
|---|---|---|
| Original 9-slice chart | `spotify_pie_plot/100_versions_pie_plots.ipynb` | `spotify_pie_plot/pie_visualizations/pop_23_5_latin_11/` |
| Simplified chart | `spotify_pie_plot/100_versions_pie_plot_simple.ipynb` | `spotify_pie_plot/100_pie_charts_simple/` |
| Simplified, larger font | `spotify_pie_plot/100_versions_pie_plot_simple_bigfont.ipynb` | `spotify_pie_plot/100_pie_charts_simple_bigfont/` |

### Post (stimulus) creation — turning a chart into a Facebook-style post

| What | Script/notebook | Covers |
|---|---|---|
| Any profile, main study's chart, baseline/realistic/likes_only/likes_only_noise | `utils/generate_profile_posts.py --slug <name>` | `remy-ashford`, `dr-remy-ashford`, or a new profile — see its own docstring |
| Simplified-chart pilot | `utils/updated_post_generator_all_visible_emojis.ipynb` | The only source for this chart pool; no `.py` script covers it |
| Bigfont pilot | `utils/updated_post_generator_all_visible_emojis_bigfont.ipynb` | Same, for the bigfont chart pool |
| Climate generalization pilot | `climate_pilot/generate_climate_stimuli.py` | Independent chart + claim, own module docstring has the design rationale |
| Neutral-claim pilot | `neutral_pilot/generate_neutral_stimuli.py` | |
| Shared HTML/CSS template | `utils/post_generator_all_visible_emojis.py` | `generate_facebook_post()`, called by everything above |

### HTML → PNG rendering (needs Chromium, see Stage 2 above)

| What | Script |
|---|---|
| Any profile in `benchmarking/` (main study, any `--slug`) | `utils/render_profile_html_to_png.py --slug <name>` |
| Any tree under `spotify_pie_plot/`, by name or path | `utils/render_html_to_png.py <target>` (`--list` shows targets) |
| Climate pilot | `climate_pilot/render_climate_html_to_png.py` |
| Neutral pilot | `neutral_pilot/render_neutral_html_to_png.py` |

### Phase 1 — perception benchmarking (does the model read the chart correctly?)

| What | Where |
|---|---|
| Per-model notebooks | `benchmarking/<model>-benchmarking.ipynb`, `benchmarking_simple_plot/<model>-benchmarking.ipynb`, `benchmarking_simple_plot_bigfont/<model>-benchmarking.ipynb` |
| Shared scoring/analysis code | `benchmarking/utils/quanti_benchmarking_{1,2,3,4}_analysis.py`, `quali_benchmarking.py` |
| Summaries | Each tree's own `1-overview-findings-*.ipynb` |

### Phase 2 — the main experiment (E1)

| What | Where |
|---|---|
| 9 model folders — 6 main roster, 2 appendix-only (`mistral-small-3.1-24b`, `pixtral-12b`), 1 fully excluded from the study (`ovis2.5-9b`, folder exists but not reported anywhere) | `experiments/e1/<model>/e1-<model>.ipynb` |
| Shared harness | `experiments/e1/e1_utils/` — `sampling.py` (seeded sample), `e1_optimized.py` (the run functions), `inference_<model>.py` (per-model adapters), `e1_analysis_optimized.py` (per-model summaries) |
| Simplified-chart pilot | `experiments/e1_simple_plot/{gemma4-e4b,qwen3-vl-8b}/` |
| Bigfont pilot | `experiments/e1_simple_plot_bigfont/{gemma4-e4b,qwen3-vl-8b}/` |
| Climate pilot | `experiments/e1_climate/<model>/`, all 6 main-roster models |
| Neutral pilot | `experiments/e1_neutral/gemma4-12b/` |
| Authority pilot | `experiments/e1_authority_grid/<model>/`, all 6 main-roster models — shares `e1_utils/e1_profile_grid.py` |

### Analysis / results

| What | Where |
|---|---|
| Per-tree summary (what the thesis draws from) | Each tree's own `1-overview-findings-*.ipynb` (`benchmarking/`, `benchmarking_simple_plot/`, `benchmarking_simple_plot_bigfont/`, `experiments/e1/`, `experiments/e1_authority_grid/`, `experiments/e1_climate/`) — **`experiments/e1_simple_plot/` (non-bigfont) has no notebook of its own; its findings are combined into `experiments/e1_simple_plot_bigfont/1-overview-findings-e1-simple-plot-bigfont.ipynb`**, which covers both chart-legibility variants together for the two models tested on both (Gemma-E4B, Qwen3-VL-8B). |

### Verifying nothing has drifted

| What | Script |
|---|---|
| Every stimulus tree, HTML vs. PNG counts | `reproducibility/prepare_stimuli.py --report` |
| Image bytes vs. the pre-refactor fingerprints | `reproducibility/verify_stimuli.py --manifest` |
| `benchmarking/` vs. `spotify_pie_plot/` byte-for-byte | `reproducibility/compare_trees.py` |

---

## If something looks wrong

| Symptom | Cause |
|---|---|
| `RuntimeError: Not logged in to Hugging Face`, or a 401/`RepositoryNotFoundError` on model load | Run `utils/huggingface_login.ipynb` once. If it still fails afterward, that specific model's license likely isn't accepted on your account yet — check its page on huggingface.co. |
| `ModuleNotFoundError: selenium`, or no Chromium | You are on the GPU machine. Rendering belongs on the other one. |
| A notebook prints "Skipping" and does nothing | Resume logic working. Results already exist in `outputs/`. |
| `--selftest` says the pixels differ | Your Chromium or fonts differ from the study's. The PNGs in the repository are still the authoritative ones; render nothing and use them. |
| `main-baselines` lists 3 files to render | Three leftover HTML drafts sit in the baselines tree (`dr-remy-ashford...`, and the one `--selftest` uses). Nothing reads them; rendering them is harmless, ignoring them is fine. |
| A tree looks short by a few hundred files | Folders named `...ignore` are abandoned drafts no condition reads, and are excluded from every count. `--include-dead` renders them anyway. |
| `FileNotFoundError` under `benchmarking/...` in an e1 notebook | The experiment tree was never installed. Run `python3 reproducibility/prepare_stimuli.py`. |
| `verify_stimuli.py --manifest` reports a change | An image was edited or re-rendered locally. `git checkout pre-path-refactor` returns the tree to the state the results came from. |
