#!/usr/bin/env python3
"""Draw gate (mistake-ledger M20, ~$616). Run BEFORE spending a submission credit.

    draw_check.py candidate.py banked.py [--csv cand.csv banked.csv] [--pub 0.73 0.74 ...]

THE PLATFORM GRADES PRIVATELY THE SUBMISSION YOUR PUBLIC SCORE SELECTED. Verified on
the closed private boards of Neuromorphic Readout and Material Spectrum: for all ~32
solver rows the public column on the PRIVATE board equals that solver's BEST-PUBLIC
score. There is no private best-of-N and no floor.

So a draw is worth a credit only if it is BETTER, not merely DIFFERENT. A reseed gives
public mu+max(a) and private mu+b at an index random w.r.t. b: public rises, private
expectation does not move.

This supersedes rowdiff.py's verdict. Material Spectrum attempts 2-6 were byte-identical
except the seed pair, differed on 12-23% of ROWS, and rowdiff.py called every one of them
"DECORRELATED - worth a credit". Five credits, one recipe, public #1 -> private #3.
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

    cand, banked = normalize(argv[1]), normalize(argv[2])
    rest = argv[3:]

    if cand == banked:
        verdict, code = 'SEED-ONLY', 1
    else:
        changed = [l for l in difflib.unified_diff(banked, cand, lineterm='', n=0)
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
                  "Report that number, and do not treat the public rank as banked.")

    if code == 1 and '--roster' in rest:
        print(f"\nVERDICT: {verdict}, OVERRIDDEN by --roster.")
        print("  Allowed: public rank decides lockdown roster survival and the 12 continuing")
        print("  slots, which is worth money at zero private value. Valid ONLY while a listed")
        print("  -score cutoff is live. Do not use it to 'defend' a rank that is already safe.")
        return 0

    print(f"\nVERDICT: {verdict}")
    if verdict in ('SEED-ONLY', 'RESEED-EQUIVALENT', 'HYPERPARAM-ONLY'):
        print("  WHY THIS IS NOT FORFEITING UPSIDE: you get exactly ONE private draw no")
        print("  matter how many attempts you submit, because only the best-PUBLIC one is")
        print("  graded. K reseeds do not give K chances at a lucky private score - they")
        print("  give one, chosen by public noise. Expected private is identical at K=1.")
        print("  LEGITIMATE OVERRIDE (--roster): public rank decides lockdown roster")
        print("  survival and the 12 continuing slots, and that is worth money even at zero")
        print("  private value. If you are fighting for a listed-score cutoff, re-roll.")
    if verdict == 'SEED-ONLY':
        print("  Identical after seed-normalisation. DO NOT SPEND A CREDIT.")
        print("  Fold these seeds INTO the script and ship ONE averaged run: internal")
        print("  averaging raises the floor of the single graded run; external re-rolling")
        print("  only re-rolls the public lottery. All 7 harvested winners did the former.")
    elif verdict == 'RESEED-EQUIVALENT':
        print("  Every changed line is a randomisation knob (seed / fold / shuffle / split).")
        print("  A re-rolled fold is a reseed wearing a different hat. DO NOT SPEND A CREDIT.")
        print("  Fold the seeds INTO the script and ship ONE averaged run.")
    elif verdict == 'HYPERPARAM-ONLY':
        print("  Only numeric literals changed. That is within-frame noise, not a level")
        print("  change. Spend the credit only if a shift-matched holdout says this config")
        print("  is BETTER - not just different.")
    else:
        print("  Real structural change. Still required before the credit:")
        print("   1. a measured number on a holdout that reproduces the TEST's shift (G-SHIFT)")
        print("   2. if it adds an ensemble member, OOF correlation vs the existing member")
        print("      (~0.8 across families pays; >0.95 is the same model twice)")
        print("   3. leak_check.py + hardcode scan + style_check.py")
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv))
