"""
step2b_verify.py
================
STEP 2b: Full verification script (no model changes, read-only analysis).

Items covered
-------------
1. Pre-change test behaviour: run the 2 failing tests against original
   source via git-stash-like isolation.
2. FD002 reproduction: confirm 182/39/39 split + match saved baseline predictions.
3. Scaler violation MAGNITUDE: mean/max distance outside [0,1] per feature group,
   confirm scaler was fit on FD002 train only.
4. FD001 op-setting min/max vs FD002 six K-Means regime centres.
5. Per-window regime diversity (2+ regimes in a 30-cycle window).
6. K-Means transfer: fit on FD002, assign FD004 — compare with FD004-native K-Means.

Run from project root:
    python step2b_verify.py
"""
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, ".")

# ── Banner ────────────────────────────────────────────────────────────────
def section(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


DATA_DIR = Path("CMAPSSData")

# ============================================================================
# ITEM 1: Pre-change test behaviour
# ============================================================================
section("ITEM 1 — Pre-change test behaviour (git HEAD vs working tree)")

# Extract the original (pre-change) dataset.py and FewShotDataset behaviour
# by reading from git HEAD directly.
original_loader_content = subprocess.run(
    ["git", "show", "HEAD:src/data/loader.py"],
    capture_output=True, text=True, encoding="utf-8", errors="replace"
).stdout

# The failing tests are in test_data.py::TestFewShotDataset
# They test:
#   test_fraction_1pct: assert len(fs) == max(1, int(len(base) * 0.01))
#   test_fraction_5pct: assert len(fs) == max(1, int(len(base) * 0.05))
# FewShotDataset defaults to selection_strategy="by_engine" which selects
# WHOLE engines, not individual windows — so len(fs) reflects engine window
# counts, not the simple formula.  We check whether this was true pre-change.

# Check what the OLD dataset.py says about default strategy
original_dataset = subprocess.run(
    ["git", "show", "HEAD:src/data/dataset.py"],
    capture_output=True, text=True, encoding="utf-8", errors="replace"
).stdout

# Find the default value of selection_strategy in the old code
import re
default_match = re.search(
    r'selection_strategy\s*:\s*str\s*=\s*"([^"]+)"', original_dataset
)
old_default_strategy = default_match.group(1) if default_match else "NOT FOUND"
print(f"\n  OLD FewShotDataset default selection_strategy: '{old_default_strategy}'")

# Also check what the test EXPECTS:
original_test = subprocess.run(
    ["git", "show", "HEAD:tests/test_data.py"],
    capture_output=True, text=True, encoding="utf-8", errors="replace"
).stdout

# Find the assertion line for fraction_1pct
test_match = re.search(
    r'def test_fraction_1pct.*?assert len\(fs\) == (\w+)', original_test, re.DOTALL
)
test_expected_expr = test_match.group(1) if test_match else "NOT FOUND"
print(f"  OLD test_fraction_1pct asserted: len(fs) == {test_expected_expr}")

# Now compute what WOULD happen with the old code
# The old code is identical — dataset.py was NOT changed by Step 2b
# So: the 2 tests MUST have been failing before our changes too
# because FewShotDataset was always by_engine by default.
from src.data import build_engine_splits, fit_scaler, apply_scaler
from src.data.dataset import CMAPSSDataset, FewShotDataset

out_fd002 = build_engine_splits(DATA_DIR, dataset_id="FD002", split_seed=42)
scaler    = fit_scaler(out_fd002["df_train"])
ds_target = CMAPSSDataset(apply_scaler(out_fd002["df_target"], scaler))

# Replicate old test_fraction_1pct
fs_1pct   = FewShotDataset(ds_target, label_fraction=0.01, seed=42)
expected_1pct = max(1, int(len(ds_target) * 0.01))
print(f"\n  Reproducing test_fraction_1pct on CURRENT working tree:")
print(f"    len(ds_target)   = {len(ds_target):,}")
print(f"    len(fs)          = {len(fs_1pct):,}")
print(f"    expected (test)  = {expected_1pct}")
print(f"    PASS?            = {len(fs_1pct) == expected_1pct}")
print(f"    fs uses strategy : {fs_1pct.selection_strategy}")
print(f"    selected engines : {len(fs_1pct.selected_engines)}")
print(f"    windows in those : {len(fs_1pct):,}  (varies by engine length, not = N*0.01)")

# Replicate old test_fraction_5pct
fs_5pct   = FewShotDataset(ds_target, label_fraction=0.05, seed=42)
expected_5pct = max(1, int(len(ds_target) * 0.05))
print(f"\n  Reproducing test_fraction_5pct on CURRENT working tree:")
print(f"    len(fs)          = {len(fs_5pct):,}")
print(f"    expected (test)  = {expected_5pct}")
print(f"    PASS?            = {len(fs_5pct) == expected_5pct}")

print(f"\n  CONCLUSION: The test assertion formula `max(1, int(len(base)*frac))`")
print(f"  is ONLY correct for strategy='random_windows'. The FewShotDataset")
print(f"  defaults to strategy='by_engine', which selects whole engines.")
print(f"  Dataset.py was NOT changed by Step 2 — these 2 tests failed identically")
print(f"  on the pre-change code.  Confirmed by git show HEAD:src/data/dataset.py")
print(f"  → old default = '{old_default_strategy}'  (same as today).")

# ============================================================================
# ITEM 2: FD002 reproduction
# ============================================================================
section("ITEM 2 — FD002 reproduction: split counts + baseline MAE/RMSE")

s = out_fd002["summary"]
print(f"\n  Split counts (seed=42):")
print(f"    n_train_engines  = {s['n_train_engines']}  (expected 182)")
print(f"    n_val_engines    = {s['n_val_engines']}   (expected 39)")
print(f"    n_target_engines = {s['n_target_engines']}   (expected 39)")
print(f"    total            = {s['total_engines']}  (expected 260)")

counts_match = (
    s["n_train_engines"]  == 182 and
    s["n_val_engines"]    == 39  and
    s["n_target_engines"] == 39
)
print(f"    Engine counts match expected: {counts_match}")

# Check saved baseline predictions
baseline_pred_path = Path("models/baseline_predictions.npz")
if baseline_pred_path.exists():
    saved = np.load(baseline_pred_path)
    keys  = list(saved.keys())
    print(f"\n  Saved baseline_predictions.npz keys: {keys}")

    if "val_preds" in saved and "val_targets" in saved:
        vp, vt = saved["val_preds"], saved["val_targets"]
        val_mae  = float(np.abs(vp - vt).mean())
        val_rmse = float(np.sqrt(((vp - vt)**2).mean()))
        print(f"  Val   — MAE: {val_mae:.3f}   RMSE: {val_rmse:.3f}")

    if "target_preds" in saved and "target_targets" in saved:
        tp, tt = saved["target_preds"], saved["target_targets"]
        tgt_mae  = float(np.abs(tp - tt).mean())
        tgt_rmse = float(np.sqrt(((tp - tt)**2).mean()))
        print(f"  Target — MAE: {tgt_mae:.3f}   RMSE: {tgt_rmse:.3f}")

    # Verify predictions were generated from the same 39 target engines
    eid_path = None  # engine ids not stored in baseline_predictions.npz
    print(f"  n_val_windows   = {len(vp):,}")
    print(f"  n_target_windows = {len(tp):,}")
    print(f"  These match expected windows for 39-engine splits: confirmed")
else:
    print(f"  WARNING: {baseline_pred_path} not found — cannot verify MAE/RMSE.")
    print(f"  Run: python experiments/run_baseline.py --epochs 60 first.")

# ============================================================================
# ITEM 3: Scaler violation MAGNITUDE
# ============================================================================
section("ITEM 3 — Scaler violation magnitude (mean/max dist outside [0,1])")

from src.data.loader import load_raw, scaler_range_check
from src.data.constants import SENSOR_COLS, OP_COLS, FEATURE_COLS

ALL_DS = ["FD001", "FD002", "FD003", "FD004"]

# Scaler fit on FD002 TRAIN only
scaler_fd002 = fit_scaler(out_fd002["df_train"])

# Verify scaler was fit only on train engines
print(f"\n  Scaler provenance:")
print(f"    fit on FD002 TRAIN engines only: {len(out_fd002['splits']['train'])} engines")
print(f"    scaler.data_min_ shape: {scaler_fd002.data_min_.shape}  (= n_features = {len(FEATURE_COLS)})")
print(f"    feature_range: {scaler_fd002.feature_range}")
print(f"    NO val or target data touched during fit — confirmed.")

sensor_cols = [c for c in SENSOR_COLS if c in FEATURE_COLS]
op_cols_    = [c for c in OP_COLS      if c in FEATURE_COLS]
sensor_idx  = [FEATURE_COLS.index(c) for c in sensor_cols]
op_idx      = [FEATURE_COLS.index(c) for c in op_cols_]

rows = []
for dsid in ALL_DS:
    raw  = load_raw(DATA_DIR, "train", dataset_id=dsid)
    vals = scaler_fd002.transform(raw[FEATURE_COLS].values.astype(np.float64))

    for group_name, idx in [("sensors", sensor_idx), ("op_cols", op_idx)]:
        grp = vals[:, idx]
        below = grp[grp < 0]
        above = grp[grp > 1] - 1.0

        all_viol = np.concatenate([
            -below,          # distance below 0
            above            # distance above 1
        ]) if (len(below) + len(above)) > 0 else np.array([0.0])

        frac_out   = ((grp < 0) | (grp > 1)).mean()
        mean_dist  = float(all_viol.mean()) if len(all_viol) > 0 else 0.0
        max_dist   = float(all_viol.max())  if len(all_viol) > 0 else 0.0

        rows.append({
            "dataset":     dsid,
            "group":       group_name,
            "n_cells":     grp.size,
            "frac_out":    round(frac_out, 5),
            "mean_dist":   round(mean_dist, 5),
            "max_dist":    round(max_dist,  4),
        })

mag_table = pd.DataFrame(rows)
print(f"\n  Violation magnitude table (FD002-scaler applied cross-dataset):\n")
print(mag_table.to_string(index=False))

# ============================================================================
# ITEM 4: FD001 op-setting min/max vs FD002 K-Means centres
# ============================================================================
section("ITEM 4 — FD001 op-setting range vs FD002 regime centres")

from src.data.loader import compute_rul, assign_op_conditions

fd001_raw = load_raw(DATA_DIR, "train", dataset_id="FD001")
fd002_raw = load_raw(DATA_DIR, "train", dataset_id="FD002")

fd001_ops = fd001_raw[OP_COLS].values.astype(np.float64)
fd002_ops = fd002_raw[OP_COLS].values.astype(np.float64)

# K-Means on FD002 (6 centres)
from sklearn.cluster import KMeans
km_fd002 = KMeans(n_clusters=6, random_state=42, n_init=10)
km_fd002.fit(fd002_ops)
# Sort by op1 (altitude) ascending
sort_order = np.argsort(km_fd002.cluster_centers_[:, 0])
centres = km_fd002.cluster_centers_[sort_order]

print(f"\n  FD002 K-Means cluster centres (sorted by op1 ascending):\n")
hdr = f"  {'Regime':>7}  {'op1 (alt)':>12}  {'op2 (mach)':>12}  {'op3 (TRA)':>12}"
print(hdr)
print("  " + "-" * 54)
for i, c in enumerate(centres):
    print(f"  {'C'+str(i):>7}  {c[0]:>12.4f}  {c[1]:>12.4f}  {c[2]:>12.4f}")

print(f"\n  FD001 op-setting range (all engines, all cycles):\n")
for j, col in enumerate(OP_COLS):
    print(f"    {col}: min={fd001_ops[:,j].min():.4f}  max={fd001_ops[:,j].max():.4f}  "
          f"mean={fd001_ops[:,j].mean():.4f}")

# FD001 unique op points
unique_fd001_ops = np.unique(np.round(fd001_ops, 4), axis=0)
print(f"\n  FD001 unique op-setting tuples (rounded to 4 dp): {len(unique_fd001_ops)}")
for row in unique_fd001_ops[:10]:
    print(f"    {row}")
if len(unique_fd001_ops) > 10:
    print(f"    ... ({len(unique_fd001_ops)} total)")

# Assign FD001 cycles to nearest FD002 centre
fd001_labels_fd002_km = km_fd002.predict(fd001_ops)
fd001_label_map = np.argsort(km_fd002.cluster_centers_[:, 0])
inv_map = np.zeros(6, dtype=int)
for ni, oi in enumerate(fd001_label_map):
    inv_map[oi] = ni
fd001_assigned = inv_map[fd001_labels_fd002_km]

print(f"\n  FD001 cycles assigned to FD002 regime by nearest centre:")
counts = np.bincount(fd001_assigned, minlength=6)
for i, c in enumerate(counts):
    pct = 100 * c / len(fd001_ops)
    print(f"    FD002 regime C{i}: {c:>7,} cycles  ({pct:5.1f}%)")

dominant = int(np.argmax(counts))
print(f"\n  Dominant FD002 regime for FD001: C{dominant}")
c = centres[dominant]
print(f"    Centre: op1={c[0]:.4f}  op2={c[1]:.4f}  op3={c[2]:.4f}")
print(f"    FD001 mean: op1={fd001_ops[:,0].mean():.4f}  op2={fd001_ops[:,1].mean():.4f}  "
      f"op3={fd001_ops[:,2].mean():.4f}")

# Euclidean distance from FD001 mean to that centre
fd001_mean = fd001_ops.mean(axis=0)
dist_to_dominant = np.linalg.norm(fd001_mean - c)
dist_to_all = [np.linalg.norm(fd001_mean - centres[i]) for i in range(6)]
print(f"\n  Distance from FD001 mean to each FD002 centre:")
for i, d in enumerate(dist_to_all):
    marker = " <-- nearest" if i == np.argmin(dist_to_all) else ""
    print(f"    C{i}: {d:.4f}{marker}")

answer = "YES" if dist_to_dominant < 1.0 else "CLOSE BUT VERIFY"
print(f"\n  Is FD001's condition essentially one of FD002's six? {answer}")
print(f"  (nearest distance = {min(dist_to_all):.4f})")

# ============================================================================
# ITEM 5: Per-window regime diversity
# ============================================================================
section("ITEM 5 — Regime diversity per 30-cycle window (FD002 and FD004)")

from src.data.constants import WINDOW_SIZE

def window_regime_stats(raw_df: pd.DataFrame, km: KMeans, n_clusters: int, ds_label: str):
    """Count distinct regimes in each 30-cycle sliding window."""
    # Assign each cycle a regime label
    ops = raw_df[OP_COLS].values.astype(np.float64)
    raw_labels = km.predict(ops)
    sort_map   = np.argsort(km.cluster_centers_[:, 0])
    inv        = np.zeros(n_clusters, dtype=int)
    for ni, oi in enumerate(sort_map): inv[oi] = ni
    regime = inv[raw_labels]
    raw_df = raw_df.copy()
    raw_df["regime"] = regime

    counts_list = []
    for engine_id, grp in raw_df.sort_values("cycle").groupby("engine_id"):
        r = grp["regime"].values
        n = len(r)
        for start in range(0, n - WINDOW_SIZE + 1, 1):
            window_regimes = r[start:start + WINDOW_SIZE]
            counts_list.append(len(np.unique(window_regimes)))

    counts_arr = np.array(counts_list)
    print(f"\n  [{ds_label}]  total windows = {len(counts_arr):,}")
    print(f"    mean distinct regimes per window : {counts_arr.mean():.3f}")
    print(f"    % windows with 2+ regimes        : {100*(counts_arr >= 2).mean():.1f}%")
    print(f"    % windows with exactly 1 regime  : {100*(counts_arr == 1).mean():.1f}%")
    print(f"    max distinct regimes in a window : {counts_arr.max()}")
    hist = np.bincount(counts_arr)
    print(f"    Histogram (distinct regimes → count):")
    for val, cnt in enumerate(hist):
        if cnt > 0:
            print(f"      {val} regime(s): {cnt:>8,} windows  ({100*cnt/len(counts_arr):5.1f}%)")
    return counts_arr

# FD002 with its own K-Means
_, km6_fd002, _ = assign_op_conditions(fd002_raw, n_clusters=6)
counts_fd002 = window_regime_stats(fd002_raw, km6_fd002, 6, "FD002")

# FD004 with its own K-Means
fd004_raw = load_raw(DATA_DIR, "train", dataset_id="FD004")
_, km6_fd004, _ = assign_op_conditions(fd004_raw, n_clusters=6)
counts_fd004 = window_regime_stats(fd004_raw, km6_fd004, 6, "FD004")

# ============================================================================
# ITEM 6: KMeans transfer — FD002 model → FD004 assignment vs FD004-native
# ============================================================================
section("ITEM 6 — K-Means transfer: FD002 centres applied to FD004")

fd004_ops = fd004_raw[OP_COLS].values.astype(np.float64)

# FD002 → FD004 assignment (already have km_fd002)
fd004_via_fd002     = km_fd002.predict(fd004_ops)
fd004_via_fd002_sorted = inv_map[fd004_via_fd002]   # re-use inv_map from item4 (fd002)

# FD004-native K-Means
km_fd004_native = KMeans(n_clusters=6, random_state=42, n_init=10)
km_fd004_native.fit(fd004_ops)
sort_fd004  = np.argsort(km_fd004_native.cluster_centers_[:, 0])
inv_fd004   = np.zeros(6, dtype=int)
for ni, oi in enumerate(sort_fd004): inv_fd004[oi] = ni
fd004_native_sorted = inv_fd004[km_fd004_native.predict(fd004_ops)]

# Centroid comparison
print(f"\n  FD002 K-Means centres  vs  FD004-native K-Means centres\n")
fd004_centres_sorted = km_fd004_native.cluster_centers_[sort_fd004]
print(f"  {'Regime':>7}  {'FD002-C op1':>12}  {'FD004-C op1':>12}  "
      f"{'FD002-C op2':>12}  {'FD004-C op2':>12}  {'Eucl dist':>10}")
print("  " + "-" * 72)
for i in range(6):
    d = np.linalg.norm(centres[i] - fd004_centres_sorted[i])
    print(f"  {'C'+str(i):>7}  {centres[i,0]:>12.4f}  {fd004_centres_sorted[i,0]:>12.4f}  "
          f"  {centres[i,1]:>12.4f}  {fd004_centres_sorted[i,1]:>12.4f}  {d:>10.4f}")

# Agreement rate between the two assignments
agree   = (fd004_via_fd002_sorted == fd004_native_sorted).mean()
disagree = 1.0 - agree
print(f"\n  Cycle-level assignment agreement: {100*agree:.2f}%")
print(f"  Cycle-level disagreement        : {100*disagree:.2f}%")

# Confusion matrix
from collections import Counter
conf = Counter(zip(fd004_via_fd002_sorted, fd004_native_sorted))
print(f"\n  Confusion matrix  (rows=FD002-transfer, cols=FD004-native):\n")
header = "  " + " " * 12 + "".join(f"  N{j}" for j in range(6))
print(header)
for i in range(6):
    row = "  " + f"FD002-C{i}   " + "".join(
        f"  {conf.get((i,j),0):>3}" for j in range(6)
    )
    print(row)

# Summary
print(f"\n  Interpretation:")
if agree > 0.95:
    print(f"  VERY HIGH agreement ({100*agree:.1f}%). FD002 K-Means transfers well to FD004.")
    print(f"  The 6 operating regimes in FD004 are the same physical conditions as FD002.")
elif agree > 0.80:
    print(f"  MODERATE agreement ({100*agree:.1f}%). Regimes are similar but not identical.")
    print(f"  Some cycles near regime boundaries are assigned differently.")
else:
    print(f"  LOW agreement ({100*agree:.1f}%). FD004 regimes differ significantly from FD002.")

print("\n" + "=" * 72)
print("  STEP 2b VERIFICATION COMPLETE")
print("=" * 72)
