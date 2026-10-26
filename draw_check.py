#!/usr/bin/env python3
"""Is a new run a real change, or the same model with a new seed?

    draw_check.py candidate.py baseline.py [--csv cand.csv baseline.csv] [--pub 0.73 0.74 ...]

Compares two training scripts after normalising seeds and random_state values. If the
only differences are seeds, folds, shuffles or numeric hyperparameters, the new run is a
re-draw of the same recipe: its score will differ, but its expected score will not.

Why it matters: when a held-out leaderboard keeps only your best visible score, submitting
several reseeds of one recipe raises the visible number through selection while the hidden
score stays where it was. Averaging those seeds inside one script is the better use of them.

--csv reports how many prediction rows differ (informational: differing rows measure noise,
not improvement). --pub takes visible scores of several draws of one recipe and reports the
selection inflation, max - mean.
"""
import re
import sys
import difflib
import statistics

SEED_CONTEXTS = [
    r'(?:^|\W)(?:SEED|seed|random_state|bagging_seed|feature_fraction_seed)\s*=\s*-?\d+',
    r'(?:manual_seed|default_rng|np\.random\.seed|seed_everything|set_seed)\s*\(\s*-?\d+',
    r'for\s+\w+\s+in\s*\(\s*-?\d+(?:\s*,\s*-?\d+)*\s*,?\s*\)',
    r'for\s+\w+\s+in\s*\[\s*-?\d+(?:\s*,\s*-?\d+)*\s*,?\s*\]',
]
NUM = re.compile(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?')
# a changed line that only re-rolls randomisation: seeds, folds, shuffles, permutations
RANDOMISATION = re.compile(
    r'\b(?:seed|random_state|fold|folds|shuffle|permutation|permute|rng|KFold|'
    r'train_test_split|bootstrap|bagging)\b', re.I)


def normalize(path):
    out = []
    for line in open(path, encoding='utf-8'):
        line = line.split('#')[0].rstrip()
        if not line.strip():
            continue
        for pat in SEED_CONTEXTS:
            line = re.sub(pat, lambda m: NUM.sub('S', m.group(0)), line)
        out.append(line)
    return out


def main(argv):
    if len(argv) < 3 or argv[1] in ('-h', '--help'):
        print(__doc__)
        return 2

    cand, baseline = normalize(argv[1]), normalize(argv[2])
    rest = argv[3:]

    if cand == baseline:
        verdict, code = 'SEED-ONLY', 1
    else:
        changed = [l for l in difflib.unified_diff(baseline, cand, lineterm='', n=0)
                   if l[:1] in '+-' and l[:3] not in ('---', '+++')]
        print(f"source: {len(changed)} changed lines after seed-normalisation")
        for l in changed[:12]:
            print('   ', l[:110])
        rnd = [bool(RANDOMISATION.search(l)) for l in changed]
        plus = {NUM.sub('N', l[1:].strip()) for l in changed if l[0] == '+'}
        minus = {NUM.sub('N', l[1:].strip()) for l in changed if l[0] == '-'}
        if all(rnd):
            verdict, code = 'RESEED-EQUIVALENT', 1
        elif plus == minus:
            verdict, code = 'HYPERPARAM-ONLY', 1
        else:
            verdict, code = 'STRUCTURAL', 0

    if '--csv' in rest:
        i = rest.index('--csv')
        try:
            import pandas as pd
            a = pd.read_csv(rest[i + 1])
            b = pd.read_csv(rest[i + 2])
            k, v = a.columns[0], a.columns[1]
            a = a.sort_values(k).reset_index(drop=True)
            b = b.sort_values(k).reset_index(drop=True)
            n = int((a[v].values != b[v].values).sum())
            print(f"\nrows differing: {n}/{len(a)} ({100.0 * n / len(a):.1f}%)"
                  "   <- INFORMATIONAL ONLY. Differing rows measure NOISE, not usefulness.")
        except Exception as e:
            print(f"\ncsv compare skipped: {e!r}")

    if '--pub' in rest:
        i = rest.index('--pub')
        s = [float(x) for x in rest[i + 1:] if NUM.fullmatch(x)]
        if len(s) >= 2:
            infl = max(s) - statistics.mean(s)
            print(f"\ndraws of this recipe: n={len(s)} mean={statistics.mean(s):.4f} "
                  f"max={max(s):.4f} sd={statistics.pstdev(s):.4f}")
            print(f"SELECTION INFLATION = max - mean = {infl:+.4f}")
            print(f"your TRUE level is about {max(s) - infl:.4f}, not {max(s):.4f}. "
                  "Report that number, and do not treat the visible rank as settled.")

    print(f"\nVERDICT: {verdict}")
    if verdict == 'SEED-ONLY':
        print("  Identical after seed normalisation: a re-draw of the same recipe.")
        print("  Average these seeds inside one script instead of comparing single runs.")
    elif verdict == 'RESEED-EQUIVALENT':
        print("  Every changed line is a randomisation knob (seed / fold / shuffle / split).")
        print("  A re-rolled fold is a reseed too. Average the draws inside one script.")
    elif verdict == 'HYPERPARAM-ONLY':
        print("  Only numeric literals changed. Treat it as a new run only if a holdout")
        print("  that matches the test conditions says it is better, not just different.")
    else:
        print("  Real structural change. Before trusting it, check:")
        print("   1. a measured gain on a holdout that reproduces the test's distribution shift")
        print("   2. for a new ensemble member, its out-of-fold correlation with existing members")
        print("      (~0.8 across model families helps; >0.95 is the same model twice)")
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv))
