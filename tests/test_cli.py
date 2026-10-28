import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def run(script, *args, cwd=None):
    return subprocess.run([sys.executable, str(ROOT / script), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd)


def test_paired_delta_detects_a_real_improvement(tmp_path):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    noise = rng.normal(0, 0.15, 400)
    base = np.clip(0.5 + 0.10 * (2 * y - 1) + noise, 0, 1)
    better = np.clip(0.5 + 0.25 * (2 * y - 1) + noise, 0, 1)
    pd.DataFrame({"id": range(400), "y": y}).to_csv(tmp_path / "truth.csv", index=False)
    pd.DataFrame({"id": range(400), "p": base}).to_csv(tmp_path / "base.csv", index=False)
    pd.DataFrame({"id": range(400), "p": better}).to_csv(tmp_path / "cand.csv", index=False)
    r = run("paired_delta.py", tmp_path / "cand.csv", tmp_path / "base.csv",
            "--truth", tmp_path / "truth.csv", "--target", "y", "--metric", "logloss")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "VERDICT: BETTER" in r.stdout
    swapped = run("paired_delta.py", tmp_path / "base.csv", tmp_path / "cand.csv",
                  "--truth", tmp_path / "truth.csv", "--target", "y", "--metric", "logloss")
    assert swapped.returncode == 1 and "NOT BETTER" in swapped.stdout


def test_draw_check_separates_reseeds_from_real_changes(tmp_path):
    (tmp_path / "a.py").write_text("SEED = 1\nmodel = fit(depth=6)\n")
    (tmp_path / "b.py").write_text("SEED = 2\nmodel = fit(depth=6)\n")
    (tmp_path / "c.py").write_text("SEED = 1\nmodel = fit_ensemble(depth=6)\n")
    assert "SEED-ONLY" in run("draw_check.py", tmp_path / "b.py", tmp_path / "a.py").stdout
    r = run("draw_check.py", tmp_path / "c.py", tmp_path / "a.py")
    assert r.returncode == 0 and "STRUCTURAL" in r.stdout


def test_draw_check_csv_compare(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    pd.DataFrame({"id": [2, 1, 3], "v": [1, 0, 1]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"id": [1, 2, 3], "v": [0, 0, 1]}).to_csv(tmp_path / "b.csv", index=False)
    r = run("draw_check.py", tmp_path / "a.py", tmp_path / "a.py", "--csv", tmp_path / "a.csv", tmp_path / "b.csv")
    assert "rows differing: 1/3" in r.stdout


def test_overlap_check_flags_sliding_windows(tmp_path):
    series = np.random.default_rng(0).normal(size=3000).cumsum()
    windows = np.stack([series[i:i + 32] for i in range(0, 2900, 2)])
    np.savez(tmp_path / "win.npz", X=windows)
    assert run("overlap_check.py", tmp_path / "win.npz", "--rows", 300).returncode == 1
    np.savez(tmp_path / "iid.npz", X=np.random.default_rng(1).normal(size=(1000, 32)))
    assert run("overlap_check.py", tmp_path / "iid.npz", "--rows", 300).returncode == 0


def test_exp_log_roundtrip(tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    assert run("exp_log.py", "init", proj, "--metric", "auc", "--direction", "max").returncode == 0
    run("exp_log.py", "add", proj, "--id", "E1", "--change", "baseline", "--holdout", "0.70", "--verdict", "keep")
    r = run("exp_log.py", "add", proj, "--id", "E2", "--parent", "E1", "--change", "mixup", "--holdout", "0.72",
            "--verdict", "keep", "--t", "2.1")
    assert "+0.02000" in r.stdout
    assert "best KEEP: E2" in run("exp_log.py", "best", proj).stdout


def test_draw_check_keeps_hash_inside_strings(tmp_path):
    (tmp_path / "a.py").write_text('SEED = 1\ntag = "#v1"  # run tag\n')
    (tmp_path / "b.py").write_text('SEED = 2\ntag = "#v2"  # run tag\n')
    r = run("draw_check.py", tmp_path / "b.py", tmp_path / "a.py")
    assert "SEED-ONLY" not in r.stdout


def test_draw_check_does_not_treat_a_hyperparameter_grid_as_seeds(tmp_path):
    (tmp_path / "a.py").write_text("for depth in (4, 6, 8):\n    fit(depth)\n")
    (tmp_path / "b.py").write_text("for depth in (4, 6, 10):\n    fit(depth)\n")
    r = run("draw_check.py", tmp_path / "b.py", tmp_path / "a.py")
    assert "SEED-ONLY" not in r.stdout and "HYPERPARAM-ONLY" in r.stdout


def test_draw_check_still_normalises_seed_loops(tmp_path):
    (tmp_path / "a.py").write_text("for seed in (0, 1, 2):\n    fit(seed)\n")
    (tmp_path / "b.py").write_text("for seed in (3, 4, 5):\n    fit(seed)\n")
    assert "SEED-ONLY" in run("draw_check.py", tmp_path / "b.py", tmp_path / "a.py").stdout


def test_oof_blend_rejects_rank_averaging_for_probability_metrics(tmp_path):
    pd.DataFrame({"id": range(10), "y": [0, 1] * 5}).to_csv(tmp_path / "truth.csv", index=False)
    pd.DataFrame({"id": range(10), "p": np.linspace(0.1, 0.9, 10)}).to_csv(tmp_path / "a.csv", index=False)
    r = run("oof_blend.py", "--truth", tmp_path / "truth.csv", "--target", "y", "--metric", "logloss",
            "--preds", tmp_path / "a.csv", tmp_path / "a.csv", "--rank")
    assert r.returncode != 0 and "--rank only makes sense" in r.stderr


def test_paired_delta_refuses_duplicate_ids(tmp_path):
    pd.DataFrame({"id": [1, 2, 2], "score": [0.1, 0.2, 0.3]}).to_csv(tmp_path / "base.csv", index=False)
    pd.DataFrame({"id": [1, 2, 3], "score": [0.2, 0.3, 0.4]}).to_csv(tmp_path / "cand.csv", index=False)
    r = run("paired_delta.py", tmp_path / "cand.csv", tmp_path / "base.csv", "--col", "score")
    assert r.returncode != 0 and "duplicate ids" in (r.stdout + r.stderr)

    pd.DataFrame({"id": [1, 2, 3], "y": [0, 1, 0]}).to_csv(tmp_path / "truth.csv", index=False)
    pd.DataFrame({"id": [1, 2, 2, 3], "p": [0.1, 0.9, 0.8, 0.2]}).to_csv(tmp_path / "dup.csv", index=False)
    r = run("paired_delta.py", tmp_path / "dup.csv", tmp_path / "cand.csv",
            "--truth", tmp_path / "truth.csv", "--target", "y", "--metric", "auc")
    assert r.returncode != 0 and "duplicate ids" in (r.stdout + r.stderr)
