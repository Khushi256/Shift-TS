import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from analysis.step6_analysis import compute_naurc, get_true_rul_bin

def do_sanity():
    print("=== SANITY CHECKS (Seed 0) ===")
    df = pd.read_csv(PROJECT_ROOT / "results" / "step5_scores_seed0.csv")
    scores = ["s_rec", "s_mc", "s_maha", "s_stage", "s_input"]
    
    # Assert s_stage sign convention
    # For a scenario, check correlation with abs_err. If s_rec and s_stage both have >0 correlation, they share sign convention
    df_err = pd.read_csv(PROJECT_ROOT / "results" / "step6_error_ranking_seed0.csv")
    s_rec_sp = df_err[(df_err["scenario"]=="FAULT_1") & (df_err["score"]=="s_rec")]["sp_overall"].values[0]
    s_stage_sp = df_err[(df_err["scenario"]=="FAULT_1") & (df_err["score"]=="s_stage")]["sp_overall"].values[0]
    print(f"s_rec Spearman vs err (FAULT_1): {s_rec_sp:.4f}")
    print(f"s_stage Spearman vs err (FAULT_1): {s_stage_sp:.4f}")
    if (s_rec_sp > 0 and s_stage_sp > 0) or (s_rec_sp < 0 and s_stage_sp < 0):
        print("PASS: s_stage uses same sign convention as s_rec (correlated with error in same direction).")
    else:
        print("FAIL: s_stage sign convention mismatch!")
    
    # Shuffle checks
    rng = np.random.RandomState(0)
    scenarios = ["FAULT_1", "FAULT_2", "OPCOND_1"]
    
    for scen in scenarios:
        scen_df = df[df["scenario"] == scen].copy()
        if scen_df.empty: continue
        
        # Error ranking nAURC
        scen_df["random_score"] = rng.rand(len(scen_df))
        if "abs_err" in scen_df.columns:
            naurc = compute_naurc(scen_df["random_score"].values, scen_df["abs_err"].values)
            print(f"Random nAURC ({scen}): {naurc:.4f} (expect ~1.0)")
            
        # Shift detection AUROC
        src = scen_df.iloc[0]["source"]
        ho_scen = f"ID_{src}" if scen != "ID_FD002" else "in_domain_heldout" # simplification
        ho_rows = df[(df["source"] == src) & (df["scenario"] == ho_scen) & (df["split"] == "source_heldout")].copy()
        tgt_rows = df[(df["scenario"] == scen) & (df["split"] == "target")].copy()
        if not ho_rows.empty and not tgt_rows.empty:
            ho_rows["shift_label"] = 0
            tgt_rows["shift_label"] = 1
            comb = pd.concat([ho_rows, tgt_rows])
            comb["random_score"] = rng.rand(len(comb))
            auc = roc_auc_score(comb["shift_label"], comb["random_score"])
            print(f"Random AUROC ({scen}): {auc:.4f} (expect ~0.5)")

def do_pooled():
    print("\n=== POOLED RESULTS ===")
    try:
        err_dfs = [pd.read_csv(PROJECT_ROOT / "results" / f"step6_error_ranking_seed{s}.csv") for s in [0, 1, 2]]
        err_all = pd.concat(err_dfs)
        
        pooled_naurc = err_all.groupby(["scenario", "score"])["nAURC"].agg(["mean", "std"]).reset_index()
        print("\nPooled nAURC:")
        print(pooled_naurc.to_string(index=False))
        
        dec_dfs = [pd.read_csv(PROJECT_ROOT / "results" / f"step6_decision_seed{s}.csv") for s in [0, 1, 2]]
        dec_all = pd.concat(dec_dfs)
        wins = dec_all[dec_all["scenario"] != "OVERALL"].groupby("scenario")[["win", "win_mc", "win_stage"]].sum()
        print("\nWins per scenario (out of 3 seeds):")
        print(wins)
        
        overall_wins = (wins["win"] >= 2).sum()
        print(f"\ns_rec 'wins overall' (>= 3 scenarios in majority of seeds)? {'Yes' if overall_wins >= 3 else 'No'} ({overall_wins} scenarios won)")
        
        shift_dfs = [pd.read_csv(PROJECT_ROOT / "results" / f"step6_shift_detection_seed{s}.csv") for s in [0, 1, 2]]
        shift_all = pd.concat(shift_dfs)
        pooled_shift = shift_all.groupby(["scenario", "score"])["AUROC"].agg(["mean", "std"]).reset_index()
        print("\nPooled Shift AUROC:")
        print(pooled_shift.to_string(index=False))
    except FileNotFoundError as e:
        print("Pooled results are pending because step 5/6 for seeds 1 and 2 haven't been run yet.")
        print(e)

if __name__ == "__main__":
    do_sanity()
    do_pooled()
