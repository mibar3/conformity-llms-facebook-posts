# Reproducibility artifacts

Run order for the whole pipeline is in [`../REPRODUCE.md`](../REPRODUCE.md).

## prepare_stimuli.py

Unpacks the archived baseline PNGs into the folders the notebooks read, then
reports every stimulus tree, HTML count against PNG count, so what is left to
render is one table instead of a hunt. Needs only Pillow. Safe to re-run: it
writes nothing over an existing file, and `--verify` checksums what is already
there against the archive.

## verify_stimuli.py

Three independent checks that the images in the tree are the images the results
came from: `--manifest` (all 14,516 against the fingerprints below),
`--survivors` (baselines against the copies that survived loose in the tree,
whole-image), `--baselines` (all 200 baselines against the metrics stimuli,
structurally, above the engagement bar).

## stimuli_manifest_before.json

MD5 of every stimulus PNG in the repository, captured on the commit tagged
`pre-path-refactor`, i.e. the state that produced all thesis results, before any
stimulus paths were changed.

14,516 images across:

| Tree | Images |
|---|---|
| `spotify_pie_plot/pie_plot_posts` | 11,212 |
| `benchmarking/correct` | 1,302 |
| `benchmarking/incorrect` | 1,302 |
| `climate_pilot/posts` | 350 |
| `neutral_pilot/posts` | 350 |

Purpose: after the stimulus-path refactor, every image a notebook reads must still
be byte-identical to the image it read when the results were produced. Re-run the
fingerprinting and diff against this file to prove that, rather than assuming it.

To go back to the pre-refactor state entirely:

    git checkout pre-path-refactor
