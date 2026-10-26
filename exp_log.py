# exp_log.py - the experiment ledger: one change per row, measured on the SEALED holdout, keep/kill with the evidence.
#   python exp_log.py init "<project>" --metric auc --direction max
#   python exp_log.py add  "<project>" --id E3 --parent E1 --change "mixup a=0.4" --holdout 0.7312 [--t 1.4] [--verdict keep]
#                          [--kill "FT=y SCHED=y HEADLR=y LEN=y CARD=y"] [--cost "18m 3090"] [--note "..."]
#   python exp_log.py show "<project>"     table with delta vs parent, the KEEP chain, and changes killed without the five checks
#   python exp_log.py best "<project>"     best KEPT row (the one the next experiment must be paired against)
import argparse, csv, datetime, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
COLS = ["id", "parent", "change", "holdout", "t", "verdict", "kill", "cost", "note", "when"]
FIVE = ["FT", "SCHED", "HEADLR", "LEN", "CARD"]


def path(ch):
    d = ch if os.path.isdir(ch) else os.path.join(ROOT, ch)
    if not os.path.isdir(d):
        sys.exit(f"no project dir '{d}'")
    return os.path.join(d, "experiments.tsv")


def read(p):
    if not os.path.exists(p):
        sys.exit(f"{p} missing - run: exp_log.py init \"<project>\" --metric <m> --direction max|min")
    with open(p, encoding="utf-8") as f:
        head = f.readline().strip()
        m = re.match(r"# metric=(\S+) direction=(max|min)", head)
        rows = list(csv.DictReader(f, delimiter="\t"))
    return (m.group(1), m.group(2)) if m else ("?", "max"), rows


def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None


ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["init", "add", "show", "best"]); ap.add_argument("challenge")
ap.add_argument("--metric"); ap.add_argument("--direction", choices=["max", "min"])
for c in COLS[:-1]:
    ap.add_argument(f"--{c}")
a = ap.parse_args()
p = path(a.challenge)

if a.cmd == "init":
    if os.path.exists(p):
        sys.exit(f"{p} exists - not overwriting")
    if not (a.metric and a.direction):
        sys.exit("init needs --metric and --direction (write the direction down BEFORE the first experiment)")
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(f"# metric={a.metric} direction={a.direction}\n"); f.write("\t".join(COLS) + "\n")
    print(f"ledger {p} | {a.metric} {a.direction}imise"); sys.exit(0)

(metric, direction), rows = read(p)
sgn = 1 if direction == "max" else -1
byid = {r["id"]: r for r in rows}

if a.cmd == "add":
    if not a.id or a.holdout is None or not a.change:
        sys.exit("add needs --id, --change and --holdout (a measured held-out score)")
    if a.id in byid:
        sys.exit(f"id {a.id} already logged")
    if a.parent and a.parent not in byid:
        sys.exit(f"parent {a.parent} not in ledger")
    if len(re.split(r";|\s\+\s|, and ", a.change)) > 1:
        print("WARNING: this row changes more than one thing - you will not know which part moved the score. "
              "Split it unless it is a deliberate combined test.")
    v = (a.verdict or "pending").lower()
    if v == "kill":
        flags = dict(re.findall(r"(\w+)=([yn])", a.kill or ""))
        miss = [k for k in FIVE if flags.get(k) != "y"]
        if miss:
            print(f"kill check: {', '.join(miss)} not 'y' -> logged as UNTESTED, not killed. A lever is dead only when tested "
                  f"fine-tuned, at the shipped schedule, head-LR swept, full input length, on the target hardware.")
            v = "untested"
    row = {c: (getattr(a, c) or "") for c in COLS[:-1]}
    row.update(verdict=v, when=datetime.datetime.now().strftime("%m-%d %H:%M"))
    with open(p, "a", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, COLS, delimiter="\t").writerow(row)
    par = byid.get(a.parent or "")
    if par and fnum(par["holdout"]) is not None:
        d = sgn * (float(a.holdout) - float(par["holdout"]))
        print(f"{a.id}: {a.holdout} vs {a.parent} {par['holdout']} -> {'+' if d >= 0 else ''}{d:.5f} favourable | {v}")
        if v == "keep" and not a.t:
            print("  no paired t logged - run paired_delta.py against the parent's stored holdout predictions before KEEP")
    sys.exit(0)

if a.cmd == "best":
    kept = [r for r in rows if r["verdict"] == "keep" and fnum(r["holdout"]) is not None]
    if not kept:
        print("no KEEP rows yet"); sys.exit(1)
    b = max(kept, key=lambda r: sgn * float(r["holdout"]))
    print(f"best KEEP: {b['id']} {b['holdout']} ({b['change']}) - pair every new candidate against its holdout preds")
    sys.exit(0)

print(f"{metric} ({direction}imise) - {p}")
print(f"{'id':6} {'parent':6} {'holdout':>9} {'delta':>9} {'t':>6} {'verdict':9} change")
for r in rows:
    h = fnum(r["holdout"]); par = byid.get(r["parent"])
    d = sgn * (h - float(par["holdout"])) if par and h is not None and fnum(par["holdout"]) is not None else None
    print(f"{r['id']:6} {r['parent']:6} {r['holdout']:>9} {('%+.5f' % d) if d is not None else '':>9} "
          f"{r['t']:>6} {r['verdict']:9} {r['change'][:70]}")
unt = [r["id"] for r in rows if r["verdict"] == "untested"]
pend = [r["id"] for r in rows if r["verdict"] == "pending"]
if unt: print(f"UNTESTED (killed on a weak instance - still in the lever queue): {', '.join(unt)}")
if pend: print(f"PENDING verdicts: {', '.join(pend)}")
