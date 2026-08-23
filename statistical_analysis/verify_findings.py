"""
Recomputes every headline number reported for the generalization pilot and the cross-model
statistics, directly from the raw result JSONs — so each claim can be checked independently
rather than trusted.

Usage (from the repo root):
    python3 statistical_analysis/verify_findings.py            # everything
    python3 statistical_analysis/verify_findings.py authority  # one section
    python3 statistical_analysis/verify_findings.py 4.10       # by thesis section
    python3 statistical_analysis/verify_findings.py --list     # section names

The argument is matched case-insensitively against the section headings, so "auth", "two-step",
"phase 1" and "4.11" all work. From a notebook cell:  %run ../statistical_analysis/verify_findings.py

Pairs with `docs/FINDINGS_AND_VERIFICATION.md`, which lists the same findings with the file
each one comes from. Every number printed here should match that document exactly.

Needs only the standard library, except:
  - scipy   (for the sign tests and Spearman correlations) — skipped with a notice if absent
  - matplotlib (only to import the stimulus generator for the chart-reading ground truth)
    — that one check is skipped with a notice if absent
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCALES = [0, 10, 100, 1000, 10000, 100000, 1000000]
MAIN_ROSTER = ["gemma4-12b", "gemma4-e4b", "qwen3-vl-4b", "qwen3-vl-8b",
               "ministral-3-8b", "ministral-3-14b"]


def load(rel):
    p = ROOT / rel
    if not p.exists():
        return None
    return json.loads(p.read_text())


# Section filtering. Each section is a top-level block introduced by hdr(); shadowing `print` at
# module level lets an unmatched section run its (cheap, file-reading) code while emitting nothing,
# which keeps the script a single flat script rather than forcing a refactor into functions.
_ARG = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
_FILTER = _ARG.lower() if _ARG else None
_LIST_ONLY = "--list" in sys.argv
_SECTIONS = []
_ON = True
_emit = print


def print(*args, **kwargs):          # noqa: A001 - deliberate module-level shadow
    if _ON:
        _emit(*args, **kwargs)


def hdr(title):
    global _ON
    _SECTIONS.append(title)
    _ON = (_FILTER is None) or (_FILTER in title.lower())
    if _LIST_ONLY:
        _ON = False
        _emit(f"  {title}")
        return
    if _ON:
        _emit(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def sub(title):
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# 1. Paired A/B: diagonal (competence) vs. pressure vs. support
# ---------------------------------------------------------------------------
def paired_summary(rel):
    data = load(rel)
    if data is None:
        return None
    n = len(data)
    diag = [d for d in data if d["correct_scale"] == d["incorrect_scale"]]
    press = [d for d in data if d["incorrect_scale"] > d["correct_scale"]]
    supp = [d for d in data if d["correct_scale"] > d["incorrect_scale"]]
    pct = lambda g: sum(1 for d in g if d["liked_variant"] == "correct") / len(g) * 100
    return dict(n=n, overall=pct(data), diagonal=pct(diag), pressure=pct(press),
                support=pct(supp), n_diag=len(diag))


def show_paired(label, rel):
    s = paired_summary(rel)
    if s is None:
        print(f"  {label:38s}  [FILE NOT FOUND: {rel}]")
        return
    print(f"  {label:38s}  n={s['n']:5d}  diagonal={s['diagonal']:6.2f}%  "
          f"pressure={s['pressure']:6.2f}%  support={s['support']:7.2f}%")


hdr("1. GENERALIZATION PILOT — paired A/B (the conformity measure)")
print("\ndiagonal = engagement tied (competence) | pressure = incorrect post has MORE engagement")
print("support  = correct post has more engagement")

sub("Climate pilot, ORIGINAL unlabeled chart (these are the primary reported figures)")
show_paired("Qwen3-VL-8B", "experiments/e1_climate/qwen3-vl-8b/outputs/pre_label_backup/e1_results_metrics_paired.json")
show_paired("Ministral-3-14B", "experiments/e1_climate/ministral-3-14b/outputs/pre_label_backup/e1_results_metrics_paired.json")
show_paired("Gemma-E4B", "experiments/e1_climate/gemma4-e4b/outputs/e1_results_metrics_paired.json")
show_paired("Gemma-12B", "experiments/e1_climate/gemma4-12b/outputs/e1_results_metrics_paired.json")

sub("Climate pilot, VALUE-LABELED chart (robustness check)")
show_paired("Qwen3-VL-8B", "experiments/e1_climate/qwen3-vl-8b/outputs/e1_results_metrics_paired.json")
show_paired("Ministral-3-14B", "experiments/e1_climate/ministral-3-14b/outputs/e1_results_metrics_paired.json")

sub("MAIN STUDY (pie chart) — same models, for comparison")
for m in ["gemma4-12b", "gemma4-e4b", "ministral-3-14b"]:
    show_paired(m, f"experiments/e1/{m}/outputs/e1_results_metrics_paired.json")

sub("Remaining roster models")
for m in ["qwen3-vl-4b", "ministral-3-8b"]:
    show_paired(m, f"experiments/e1_climate/{m}/outputs/e1_results_metrics_paired.json")


# ---------------------------------------------------------------------------
# 1b. Is tied-engagement accuracy REAL judgment, or a position default?
# ---------------------------------------------------------------------------
CLIMATE_PAIRED = [
    ("ministral-3-14b", "experiments/e1_climate/ministral-3-14b/outputs/pre_label_backup/e1_results_metrics_paired.json"),
    ("ministral-3-8b",  "experiments/e1_climate/ministral-3-8b/outputs/e1_results_metrics_paired.json"),
    ("qwen3-vl-8b",     "experiments/e1_climate/qwen3-vl-8b/outputs/pre_label_backup/e1_results_metrics_paired.json"),
    ("qwen3-vl-4b",     "experiments/e1_climate/qwen3-vl-4b/outputs/e1_results_metrics_paired.json"),
    ("gemma4-e4b",      "experiments/e1_climate/gemma4-e4b/outputs/e1_results_metrics_paired.json"),
    ("gemma4-12b",      "experiments/e1_climate/gemma4-12b/outputs/e1_results_metrics_paired.json"),
]


def lower_slot(x):
    """Which slot holds the post with FEWER reactions."""
    a = x["correct_scale"] if x["post_a_variant"] == "correct" else x["incorrect_scale"]
    b = x["correct_scale"] if x["post_b_variant"] == "correct" else x["incorrect_scale"]
    return "A" if a < b else "B"


hdr("1b. Real judgment, or a position default? (+ direction of the engagement effect)")
print("""
A model that always answers the same slot scores ~50% when engagement is tied -- not because
it judged badly, but because the correct post sat in that slot about half the time. The
per-image sign test separates the two: genuine judgment is consistent image to image, a
positional default is not.

'picks-less-popular' is the direction of the engagement effect: LOW = conformity (goes with
the crowd), HIGH = anti-conformity.
""")
try:
    from scipy.stats import binomtest as _bt

    print(f"  {'model':17s}{'tied-acc':>9s}{'answers-A':>11s}{'sign-test p':>13s}  "
          f"{'verdict':<24s}{'picks-less-popular':>19s}")
    for m, rel in CLIMATE_PAIRED:
        d = load(rel)
        if d is None:
            print(f"  {m:17s} [FILE NOT FOUND]")
            continue
        v = [x for x in d if x["answer"] in ("A", "B")]
        diag = [x for x in v if x["correct_scale"] == x["incorrect_scale"]]
        off = [x for x in v if x["correct_scale"] != x["incorrect_scale"]]
        per = defaultdict(list)
        for x in diag:
            per[x["num"]].append(1 if x["liked_variant"] == "correct" else 0)
        hi = sum(1 for g in per.values() if sum(g) / len(g) > 0.5)
        lo = sum(1 for g in per.values() if sum(g) / len(g) < 0.5)
        p = _bt(hi, hi + lo, 0.5).pvalue if hi + lo else float("nan")
        acc = sum(1 for x in diag if x["liked_variant"] == "correct") / len(diag) * 100
        a_rate = sum(1 for x in diag if x["answer"] == "A") / len(diag) * 100
        picked_lower = sum(1 for x in off if x["answer"] == lower_slot(x)) / len(off) * 100
        verdict = "REAL judgment" if (p < 0.05 and acc > 50) else "position default"
        print(f"  {m:17s}{acc:8.1f}%{a_rate:10.1f}%{p:13.2e}  {verdict:<24s}{picked_lower:18.1f}%")
except ImportError:
    print("  [SKIPPED — needs scipy]")


# ---------------------------------------------------------------------------
# 2. Gemma-12B decomposition — position vs. engagement
# ---------------------------------------------------------------------------
hdr("2. GEMMA-12B — position vs. engagement decomposition (the corrected finding)")
print("\nThis is the check that overturned the original 'position bias' reading.")
print("The point: conditioning on position ALONE is not enough — it must be crossed")
print("against which post carries more engagement.")


def decompose(label, rel):
    data = load(rel)
    if data is None:
        print(f"  [FILE NOT FOUND: {rel}]")
        return
    valid = [d for d in data if d["answer"] in ("A", "B")]
    diag = [d for d in valid if d["correct_scale"] == d["incorrect_scale"]]
    off = [d for d in valid if d["correct_scale"] != d["incorrect_scale"]]

    a_rate = lambda g: sum(1 for d in g if d["answer"] == "A") / len(g) * 100
    corr_a = lambda g: sum(1 for d in g if d["post_a_variant"] == "correct") / len(g) * 100
    corr = lambda g: sum(1 for d in g if d["liked_variant"] == "correct") / len(g) * 100

    def lower_slot(d):
        a = d["correct_scale"] if d["post_a_variant"] == "correct" else d["incorrect_scale"]
        b = d["correct_scale"] if d["post_b_variant"] == "correct" else d["incorrect_scale"]
        return "A" if a < b else "B"

    la = [d for d in off if lower_slot(d) == "A"]
    lb = [d for d in off if lower_slot(d) == "B"]
    picked_lower = sum(1 for d in off if (d["answer"] == lower_slot(d))) / len(off) * 100

    print(f"\n  {label}")
    print(f"    TIED engagement    n={len(diag):4d}   answered-A={a_rate(diag):5.1f}%   "
          f"(correct sat in A {corr_a(diag):5.1f}% — balance check)   liked_correct={corr(diag):5.1f}%")
    print(f"    OFF-DIAGONAL       lower-engagement post in slot A: answered-A={a_rate(la):5.1f}%  (n={len(la)})")
    print(f"                       lower-engagement post in slot B: answered-A={a_rate(lb):5.1f}%  (n={len(lb)})")
    print(f"                       -> swing attributable to engagement = {abs(a_rate(la)-a_rate(lb)):4.1f} points")
    print(f"                       picked the LOWER-engagement post {picked_lower:5.1f}% of the time")


# Label corrected 2026-08-23: on the value-labelled chart this model RETAINS competence
# (67.2% at tied engagement, per-image sign test p = 0.015) and still prefers the less popular
# post 84.5% of the time. The earlier "loses content competence" reading came from data
# collected before the chart was relabelled.
decompose("CLIMATE  (RETAINS content competence; prefers the LESS popular post)",
          "experiments/e1_climate/gemma4-12b/outputs/e1_results_metrics_paired.json")
decompose("MAIN STUDY  (tracks content; engagement barely matters)",
          "experiments/e1/gemma4-12b/outputs/e1_results_metrics_paired.json")


# ---------------------------------------------------------------------------
# 3. Single-image protocol — the flat/saturated responses
# ---------------------------------------------------------------------------
hdr("3. SINGLE-IMAGE PROTOCOL — saturated, content-independent responses")


def flat(label, rel_answers, rel_logprobs):
    ans = load(rel_answers)
    lp = load(rel_logprobs)
    if ans is None or lp is None:
        print(f"  {label:46s} [FILE NOT FOUND]")
        return
    uniq = sorted(set(a["answer"] for a in ans))
    vals = [d["candidates"]["like"]["prob_forced_choice"] for d in lp]
    print(f"  {label:46s} n={len(ans):4d}  answers={uniq}  P(like) {min(vals):.6f} – {max(vals):.6f}")


print()
flat("Gemma-12B climate baseline",
     "experiments/e1_climate/gemma4-12b/outputs/e1_results_baseline.json",
     "experiments/e1_climate/gemma4-12b/outputs/e1_results_baseline_logprobs.json")
flat("Gemma-12B climate metrics (6 scales)",
     "experiments/e1_climate/gemma4-12b/outputs/e1_results_metrics.json",
     "experiments/e1_climate/gemma4-12b/outputs/e1_results_metrics_logprobs.json")
flat("Gemma-12B MAIN-STUDY baseline",
     "experiments/e1/gemma4-12b/outputs/e1_results_baseline.json",
     "experiments/e1/gemma4-12b/outputs/e1_results_baseline_logprobs.json")
flat("Gemma-12B MAIN-STUDY metrics",
     "experiments/e1/gemma4-12b/outputs/e1_results_metrics.json",
     "experiments/e1/gemma4-12b/outputs/e1_results_metrics_logprobs.json")
flat("Gemma-12B neutral-caption diagnostic",
     "experiments/e1_climate/gemma4-12b/outputs/diagnostic_neutral_caption/e1_results_baseline.json",
     "experiments/e1_climate/gemma4-12b/outputs/diagnostic_neutral_caption/e1_results_baseline_logprobs.json")
flat("Gemma-E4B climate baseline",
     "experiments/e1_climate/gemma4-e4b/outputs/e1_results_baseline.json",
     "experiments/e1_climate/gemma4-e4b/outputs/e1_results_baseline_logprobs.json")
flat("Gemma-E4B climate metrics (6 scales)",
     "experiments/e1_climate/gemma4-e4b/outputs/e1_results_metrics.json",
     "experiments/e1_climate/gemma4-e4b/outputs/e1_results_metrics_logprobs.json")


# ---------------------------------------------------------------------------
# 4. Chart-reading perception checks
# ---------------------------------------------------------------------------
hdr("4. CHART-READING PERCEPTION CHECKS (does the model actually read these charts?)")
try:
    sys.path.insert(0, str(ROOT))
    from climate_pilot.generate_climate_stimuli import generate_solar_wind_series
    import re

    gt = {}
    for num in range(1, 11):
        s, w, _ = generate_solar_wind_series(num)
        gt[f"{num:03d}"] = (s[-1], w[-1])

    print()
    for ct in ["bar_grouped", "bar_single_year", "area_overlay"]:
        base = f"climate_pilot/chart_type_benchmark/{ct}/outputs"
        ans = load(f"{base}/e1_results_baseline.json")
        claim = load(f"{base}/e1_results_claim_verify.json")
        extract = load(f"{base}/e1_results_extract_values.json")
        if ans is None:
            print(f"  {ct:18s} [FILE NOT FOUND]")
            continue
        claim_acc = sum(1 for d in claim if d["answer"] == d["variant"]) / len(claim) * 100
        ok = 0
        for d in extract:
            ts, tw = gt[d["image"].split("_")[0]]
            nums = re.findall(r"-?\d+\.?\d*", d["answer"])
            if len(nums) < 2:
                continue
            gs, gw = float(nums[0]), float(nums[1])
            if abs(gs - ts) / ts <= 0.15 and abs(gw - tw) / tw <= 0.15:
                ok += 1
        print(f"  {ct:18s} like/scroll={sorted(set(a['answer'] for a in ans))}  "
              f"claim-verify={claim_acc:5.1f}%  value-extraction={ok/len(extract)*100:5.1f}%  (n={len(claim)})")
except ImportError as e:
    print(f"\n  [SKIPPED — needs matplotlib to import the stimulus generator: {e}]")


# ---------------------------------------------------------------------------
# 5. Cross-model statistics
# ---------------------------------------------------------------------------
hdr("5. CROSS-MODEL STATISTICS (main study, confirmed 6-model roster)")

sub("Diagonal vs. above-diagonal collapse")
rows = []
for m in MAIN_ROSTER:
    data = load(f"experiments/e1/{m}/outputs/e1_results_metrics_paired.json")
    if data is None:
        print(f"  {m:18s} [FILE NOT FOUND]")
        continue
    diag = [d for d in data if d["correct_scale"] == d["incorrect_scale"]]
    above = [d for d in data if d["incorrect_scale"] > d["correct_scale"]]
    dp = sum(1 for d in diag if d["liked_variant"] == "correct") / len(diag) * 100
    ap = sum(1 for d in above if d["liked_variant"] == "correct") / len(above) * 100
    rows.append((m, dp, ap, dp - ap))
    print(f"  {m:18s} diagonal={dp:5.1f}%   above-diagonal={ap:5.1f}%   collapse={dp-ap:6.1f} pt")

try:
    from scipy.stats import binomtest, spearmanr

    sub("Per-image paired sign test (diagonal vs. above-diagonal)")
    for m in MAIN_ROSTER:
        data = load(f"experiments/e1/{m}/outputs/e1_results_metrics_paired.json")
        if data is None:
            continue
        per = defaultdict(lambda: {"d": [], "a": []})
        for d in data:
            ok = 1 if d["liked_variant"] == "correct" else 0
            if d["correct_scale"] == d["incorrect_scale"]:
                per[d["num"]]["d"].append(ok)
            elif d["incorrect_scale"] > d["correct_scale"]:
                per[d["num"]]["a"].append(ok)
        down = up = tie = 0
        for v in per.values():
            if not v["d"] or not v["a"]:
                continue
            dm, am = sum(v["d"]) / len(v["d"]), sum(v["a"]) / len(v["a"])
            if am < dm: down += 1
            elif am > dm: up += 1
            else: tie += 1
        p = binomtest(down, down + up, 0.5).pvalue if down + up else float("nan")
        print(f"  {m:18s} declined={down:3d}  rose={up:3d}  tied={tie:3d}   p={p:.3e}")

    sub("Spearman: does baseline competence predict collapse size?")
    if len(rows) == len(MAIN_ROSTER):
        r_all = spearmanr([r[1] for r in rows], [r[3] for r in rows])
        five = [r for r in rows if r[0] != "gemma4-12b"]
        r_five = spearmanr([r[1] for r in five], [r[3] for r in five])
        print(f"  all six models              rho={r_all.statistic:+.3f}  p={r_all.pvalue:.3f}")
        print(f"  five below-threshold models rho={r_five.statistic:+.3f}  p={r_five.pvalue:.3f}")
except ImportError:
    print("\n  [sign tests + Spearman SKIPPED — needs scipy]")

sub("Correct-vs-correct control (both posts correct; only engagement differs)")
print("  Off-diagonal ONLY is the correct measure: on the diagonal both scales are equal, and the")
print("  harness sets liked_higher_engagement = (liked_scale == max(a,b)), which is trivially True")
print("  for every tied trial. Including the diagonal adds 700 free successes and inflates the rate.")
print()
print(f"  {'model':18s}{'condition':20s}{'off-diagonal':>14s}{'incl-diagonal':>15s}")
import glob as _glob
for _p in sorted(_glob.glob(str(ROOT / "experiments/e1/*/outputs/e1_results_*_correct_vs_correct_paired.json"))):
    _rel = Path(_p).relative_to(ROOT)
    _m = _rel.parts[2]
    _cond = Path(_p).name.split("e1_results_")[1].split("_correct_vs_correct")[0]
    _d = json.loads(Path(_p).read_text())
    _off = [x for x in _d if x["post_a_scale"] != x["post_b_scale"]]
    _r = lambda g: sum(1 for x in g if x["liked_higher_engagement"]) / len(g) * 100
    print(f"  {_m:18s}{_cond:20s}{_r(_off):13.1f}%{_r(_d):14.1f}%")

sub("GEE joint Wald tests (read from the saved notebook output)")
wt = ROOT / "statistical_analysis/outputs/wald_tests.txt"
print(wt.read_text().rstrip() if wt.exists() else "  [wald_tests.txt not found — re-run gee_analysis.ipynb]")

sub("Opposite-corner ratio + diagonal-above-chance (saved CSVs)")
for f in ["opposite_corner_ratio_summary.csv", "diagonal_above_chance_test.csv"]:
    p = ROOT / "statistical_analysis/outputs" / f
    print(f"  {f:38s} {'present' if p.exists() else 'MISSING'}  ({p.relative_to(ROOT)})")


# ---------------------------------------------------------------------------
# 6. AUTHORITY PILOT  (Section 4.11)
# ---------------------------------------------------------------------------
hdr("6. AUTHORITY PILOT — does a credential outrank the crowd? (Section 4.11)")
print("""
Every figure here is reported WITH the position split, because the aggregate alone cannot tell a
profile preference from a slot habit: under seed 42 the "Dr." post sits in slot A on 56 of the 100
sampled posts, so a model answering "A" every time scores 56% "authority preference" without ever
looking at a profile.
""")

_AUTH_MODELS = [("Gemma-12B", "gemma4-12b"), ("Qwen3-VL-8B", "qwen3-vl-8b"),
                ("Ministral-3-14B", "ministral-3-14b"), ("Gemma-E4B", "gemma4-e4b"),
                ("Qwen3-VL-4B", "qwen3-vl-4b"), ("Ministral-3-8B", "ministral-3-8b")]
_SCALES = [0, 10, 100, 1000, 10000, 100000, 1000000]
_GRID = "e1_results_authority_grid_metrics.json"


def _halves(rows, pred):
    """(rate when the Dr post sat in slot A, rate when it sat in B, |gap|)."""
    out = []
    for slot in ("A", "B"):
        h = [r for r in rows if (r["post_a_profile"] == "dr") == (slot == "A")]
        out.append(sum(1 for r in h if pred(r)) / len(h) * 100 if h else float("nan"))
    return out[0], out[1], abs(out[0] - out[1])


sub("Stage 1 — zero engagement, four pairings (n=100 each)")
_PAIRINGS = [("1_authority_vs_correctness", "Dr+wrong vs plain+right"),
             ("2_pure_authority_both_correct", "both correct"),
             ("3_pure_authority_both_incorrect", "both incorrect"),
             ("4_both_cues_agree", "both cues agree")]
print(f"  {'model':17s}{'pairing':26s}{'%Dr':>7s}{'Dr@A':>7s}{'Dr@B':>7s}{'gap':>7s}  verdict")
for _lab, _slug in _AUTH_MODELS:
    for _fn, _disp in _PAIRINGS:
        _f = ROOT / f"experiments/e1_authority/{_slug}/outputs/e1_results_authority_{_fn}.json"
        if not _f.exists():
            print(f"  {_lab:17s}{_disp:26s}  [FILE NOT FOUND]")
            continue
        _v = [r for r in json.loads(_f.read_text()) if r["answer"] in ("A", "B")]
        _p = sum(1 for r in _v if r["liked_profile"] == "dr") / len(_v) * 100
        _a, _b, _g = _halves(_v, lambda r: r["liked_profile"] == "dr")
        _verdict = ("POSITION drove it" if _g >= 60 else
                    "mixed" if _g >= 25 else
                    "real Dr preference" if min(_a, _b) >= 60 else
                    "real: avoids Dr" if max(_a, _b) <= 40 else "no preference")
        print(f"  {_lab:17s}{_disp:26s}{_p:6.1f}%{_a:6.1f}%{_b:6.1f}%{_g:6.1f}p  {_verdict}")

sub("Stage 2 — the exchange rate: % chose the Dr post as the plain post gains engagement")
print("  Both posts carry the same CORRECT claim, so only the two social cues differ.")
print(f"  {'model':17s}" + "".join(f"{h:>9s}" for h in
      ["tied", "10x", "100x", "1Kx", "10Kx", "100Kx", "1Mx"]))
_idx = {s: i for i, s in enumerate(_SCALES)}
for _lab, _slug in _AUTH_MODELS:
    _f = ROOT / f"experiments/e1_authority_grid/{_slug}/outputs/{_GRID}"
    if not _f.exists():
        print(f"  {_lab:17s} [FILE NOT FOUND]")
        continue
    _v = [r for r in json.loads(_f.read_text())
          if r["answer"] in ("A", "B") and r["plain_scale"] >= r["dr_scale"]]
    _row = []
    for _step in range(7):
        _c = [r for r in _v if _idx[r["plain_scale"]] - _idx[r["dr_scale"]] == _step]
        _row.append(f"{sum(1 for r in _c if r['liked_dr']) / len(_c) * 100:7.1f}%" if _c else "      -")
    print(f"  {_lab:17s}" + "".join(f"{x:>9s}" for x in _row))

sub("Stage 2 — the authority effect, against the no-authority control")
print("  control = correct_vs_correct (two plain posts, same engagement contrasts).")
print("  A large negative number means the credential pulled the model off the crowd.\n")
print(f"  {'model':17s}{'control':>10s}{'with Dr.':>11s}{'effect':>10s}   position gap")
for _lab, _slug in _AUTH_MODELS:
    _cf = ROOT / f"experiments/e1/{_slug}/outputs/e1_results_metrics_correct_vs_correct_paired.json"
    _gf = ROOT / f"experiments/e1_authority_grid/{_slug}/outputs/{_GRID}"
    if not (_cf.exists() and _gf.exists()):
        print(f"  {_lab:17s} [FILE NOT FOUND]")
        continue
    _c = [r for r in json.loads(_cf.read_text())
          if r["answer"] in ("A", "B") and r["scale_a"] != r["scale_b"]]
    _ctl = sum(1 for r in _c if r["liked_higher_engagement"]) / len(_c) * 100
    _g = [r for r in json.loads(_gf.read_text())
          if r["answer"] in ("A", "B") and r["region"] == "conflict"]
    _grid = sum(1 for r in _g if not r["liked_dr"]) / len(_g) * 100
    _, _, _gap = _halves(_g, lambda r: r["liked_dr"])
    print(f"  {_lab:17s}{_ctl:9.1f}%{_grid:10.1f}%{_grid - _ctl:+9.1f}p{_gap:14.1f}p")

sub("Stage 2 — is the effect conditional on the Dr post's OWN engagement? (Gemma-12B)")
print("  The ratio-based table above pools cells at a constant ratio, hiding this.")
_f = ROOT / f"experiments/e1_authority_grid/gemma4-12b/outputs/{_GRID}"
if _f.exists():
    _v = [r for r in json.loads(_f.read_text())
          if r["answer"] in ("A", "B") and r["dr_scale"] < r["plain_scale"]]
    print(f"\n  {'Dr post shows':>15s}{'n':>7s}{'chose Dr':>10s}{'position gap':>15s}")
    for _s in _SCALES[:-1]:
        _c = [r for r in _v if r["dr_scale"] == _s]
        if not _c:
            continue
        _p = sum(1 for r in _c if r["liked_dr"]) / len(_c) * 100
        _, _, _g = _halves(_c, lambda r: r["liked_dr"])
        print(f"  {_s:>15,d}{len(_c):7d}{_p:9.1f}%{_g:14.1f}p")
    _z = [r for r in _v if r["dr_scale"] == 0]
    _nz = [r for r in _v if r["dr_scale"] > 0]
    print(f"\n  Dr at ZERO engagement : {sum(1 for r in _z if r['liked_dr']) / len(_z) * 100:5.1f}%  (n={len(_z)})")
    print(f"  Dr at ANY engagement  : {sum(1 for r in _nz if r['liked_dr']) / len(_nz) * 100:5.1f}%  (n={len(_nz)})")

sub("Stage 2 (0,0) cell should reproduce Stage 1 'both correct' EXACTLY")
print("  Same images, same prompt, and a seed formula that reduces to Stage 1's at (0,0).")
for _lab, _slug in _AUTH_MODELS:
    _gf = ROOT / f"experiments/e1_authority_grid/{_slug}/outputs/{_GRID}"
    _sf = ROOT / f"experiments/e1_authority/{_slug}/outputs/e1_results_authority_2_pure_authority_both_correct.json"
    if not (_gf.exists() and _sf.exists()):
        continue
    _g00 = {r["num"]: r for r in json.loads(_gf.read_text())
            if r["dr_scale"] == 0 and r["plain_scale"] == 0}
    _s1 = [r for r in json.loads(_sf.read_text()) if r["answer"] in ("A", "B")]
    _same = sum(1 for r in _s1 if _g00.get(r["num"], {}).get("answer") == r["answer"])
    print(f"  {'OK ' if _same == len(_s1) else 'MISMATCH '}{_lab:17s}{_same}/{len(_s1)} trials identical")



# ---------------------------------------------------------------------------
# 7. PHASE 1 PERCEPTION  (Sections 5.1, 4.10)
# ---------------------------------------------------------------------------
hdr("7. PHASE 1 PERCEPTION — does a better chart make the model read it better? (Section 5.1)")
print("""
Claim-verification accuracy pooled over prompt versions v1-v4 and tests 2 + 4, the two tests that
ask the model directly whether the caption matches the chart. n = 2,800 per cell.

Scored with the same last-verdict-wins extraction as rescore_claim_verdict.py, NOT exact string
match: the original scripts compared the model's entire raw response against "correct"/"incorrect",
so a model that reasoned before answering was scored zero for not being terse. That artifact cost
Gemma-E4B 514 responses in the bigger-font tree alone.
""")

import re as _re
_VERDICT_RE = _re.compile(r"\b(in)?correct\b")
_P1_TESTS = ("test-2-gn-claim-only", "test-4-metrics-realistic-claim-only")
_P1_TREES = [("original", "benchmarking"),
             ("simplified", "benchmarking_simple_plot"),
             ("bigger font", "benchmarking_simple_plot_bigfont")]


def _verdict(raw):
    txt = str(raw).strip().lower()
    if txt in ("correct", "incorrect"):
        return txt
    m = _VERDICT_RE.findall(txt)
    return None if not m else ("incorrect" if m[-1] == "in" else "correct")


def _phase1(tree, slug, versions=("v1", "v2", "v3", "v4")):
    """Pooled accuracy over tests 2+4. Ground truth is the variant token in the filename:
    `072_incorrect` for test 2, `068_correct_10000` for test 4 -- hence parts[1], not a suffix
    check, which silently drops all 2,400 test-4 files."""
    k = n = 0
    for test in _P1_TESTS:
        for ver in versions:
            base = ROOT / tree / "outputs" / slug / "quantitative" / test / ver
            if not base.is_dir():
                continue
            for f in base.rglob("*.json"):
                d = json.loads(f.read_text())
                parts = d.get("image", "").split("_")
                gt = parts[1] if len(parts) > 1 and parts[1] in ("correct", "incorrect") else None
                if gt is None:
                    continue
                n += 1
                k += (_verdict(d.get("answers", {}).get("post_claim_correct", "")) == gt)
    return (k / n * 100 if n else float("nan")), n


sub("Chart redesign vs. claim-verification accuracy (the two models run on all three charts)")
print(f"  {'model':14s}" + "".join(f"{lab:>16s}" for lab, _ in _P1_TREES))
for _lab, _slug in (("Qwen3-VL-8B", "qwen3-vl-8b"), ("Gemma-E4B", "gemma-e4b")):
    _cells = []
    for _t, _tree in _P1_TREES:
        _a, _n = _phase1(_tree, _slug)
        _cells.append(f"{_a:9.1f}% n={_n}" if _n else "         --")
    print(f"  {_lab:14s}" + "".join(f"{c:>16s}" for c in _cells))
print("\n  Simplifying the chart helps one model and not the other, and the bigger font adds a")
print("  little more for Qwen. Neither shows up in the E1 diagonal (Section 5.1) -- which is the")
print("  point: perception improved measurably while the liking decision did not move.")

sub("Main chart, whole roster (is perception ever the bottleneck?)")
print(f"  {'model':17s}{'pooled v1-v4':>14s}{'n':>8s}")
for _slug in ["gemma4-12b", "gemma-e4b", "qwen3-vl-4b", "qwen3-vl-8b",
              "ministral-3-8b", "ministral-3-14b"]:
    _a, _n = _phase1("benchmarking", _slug)
    print(f"  {_slug:17s}{_a:13.1f}%{_n:8d}" if _n else f"  {_slug:17s}  [NO DATA]")


# ---------------------------------------------------------------------------
# 8. TWO-STEP PROMPT PILOT  (Section 5.3)
# ---------------------------------------------------------------------------
hdr("8. TWO-STEP PROMPT PILOT — does reasoning first unlock judgment? (Section 5.3)")
print("""
All figures are the tied-engagement (diagonal) cells only, n = 700 per design.

Two columns matter and they say different things. 'overall' counts a refusal as a failure, which
is the headline figure. 'valid only' scores just the trials where the model actually answered A
or B. Where those diverge, the intervention changed how often the model ANSWERED, not how often
it was right -- which is exactly what happened to Qwen3-VL-8B under v4 wording.
""")

_TS_DESIGNS = [("single-step baseline", None),
               ("paired_verdict", "paired_verdict"),
               ("isolated_verdict", "isolated_verdict"),
               ("isolated_verdict_v4", "isolated_verdict_v4")]
_TS_MODELS = [("Qwen3-VL-8B", "qwen3-vl-8b"), ("Gemma-E4B", "gemma4-e4b"),
              ("Qwen3-VL-4B", "qwen3-vl-4b"), ("Ministral-3-8B", "ministral-3-8b"),
              ("Ministral-3-14B", "ministral-3-14b"), ("Gemma-12B", "gemma4-12b")]

print(f"  {'model':17s}{'design':22s}{'n':>5s}{'overall':>9s}{'valid only':>12s}{'refusals':>10s}")
for _lab, _slug in _TS_MODELS:
    _any = False
    for _dl, _fn in _TS_DESIGNS:
        if _fn is None:
            _p = ROOT / f"experiments/e1/{_slug}/outputs/e1_results_metrics_paired.json"
            if not _p.exists():
                continue
            _d = [r for r in json.loads(_p.read_text())
                  if r["correct_scale"] == r["incorrect_scale"]]
        else:
            _p = ROOT / f"experiments/e1/{_slug}/outputs/e1_results_metrics_diagonal_twostep_{_fn}.json"
            if not _p.exists():
                continue
            _d = json.loads(_p.read_text())
        _any = True
        _ref = sum(1 for r in _d if r["answer"] not in ("A", "B"))
        _ok = sum(1 for r in _d if r.get("liked_variant") == "correct")
        _val = [r for r in _d if r["answer"] in ("A", "B")]
        _vo = sum(1 for r in _val if r["liked_variant"] == "correct") / len(_val) * 100 if _val else float("nan")
        print(f"  {_lab:17s}{_dl:22s}{len(_d):5d}{_ok / len(_d) * 100:8.1f}%{_vo:11.1f}%{_ref:10d}")
    if not _any:
        print(f"  {_lab:17s}{'(not run -- see below)':22s}")
    print()

print("  Gemma-12B was deliberately excluded from this pilot. It asks whether reasoning unlocks")
print("  judgment the terse single-token format was suppressing; Gemma-12B's judgment is already")
print("  visible at 82.7% on the diagonal, so there is nothing to unlock. Read its absence as a")
print("  scope decision, not a coverage gap.")


if _LIST_ONLY:
    pass
elif _FILTER and not any(_FILTER in s.lower() for s in _SECTIONS):
    _emit(f"\nNo section matched {_ARG!r}. Available:")
    for _s in _SECTIONS:
        _emit(f"  {_s}")
else:
    _emit(f"\n{'=' * 78}\nDone. Compare against docs/FINDINGS_AND_VERIFICATION.md\n{'=' * 78}")
