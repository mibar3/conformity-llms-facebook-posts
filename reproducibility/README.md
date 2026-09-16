# Reproducibility artifacts

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
