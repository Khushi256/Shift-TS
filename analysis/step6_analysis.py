import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
DEFINITIONS
- Higher score = less trustworthy.
- Scenario groups. Error-ranking set: ID_FD002, ID_FD004 (split = source_heldout), FAULT_1, FAULT_2 (split = target). Stress set: OPCOND_1, OPCOND_2, COMBINED (shift detection only, because models are collapsed or degenerate).
- nAURC = (AURC - AURC_oracle) / (AURC_random - AURC_oracle). Rank windows by score, with risk = mean abs_err of the retained windows over coverage 0.05..1.0. Oracle ranks by abs_err. 0 = best, 1 = random.
- Decision rule: s_rec "wins" a scenario if its nAURC is lower than BOTH s_mc and s_stage AND the paired-bootstrap 95% CI of the difference excludes zero. s_rec "wins overall" only if it wins at least 3 of the 4 error-ranking scenarios. Otherwise report "mixed" or "does not win". Do not change this after seeing results.
"""

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_aurc(scores, errs):
    rng = np.random.RandomState(42)
    noise = rng.uniform(-1e-8, 1e-8, size=len(scores))
    idx = np.argsort(scores + noise)
    sorted_errs = errs[idx]
    cum_sum = np.cumsum(sorted_errs)
    k_arr = np.arange(1, len(sorted_errs) + 1)
    cum_mean = cum_sum / k_arr
    N = len(sorted_errs)
    min_k = max(1, int(np.ceil(0.05 * N)))
    aurc = np.mean(cum_mean[min_k-1:])
    return aurc

def compute_naurc(scores, errs):
    aurc = compute_aurc(scores, errs)
    aurc_oracle = compute_aurc(errs, errs)
    aurc_random = np.mean(errs)
    if aurc_random == aurc_oracle:
        return 0.0
    return (aurc - aurc_oracle) / (aurc_random - aurc_oracle)

def bootstrap_diff(df, score_a, score_b, err_col, n_resamples=1000, seed=0):
    rng = np.random.RandomState(seed)
    engines = df['engine_id'].unique()
    n_engines = len(engines)
    
    diffs = []
    # Pre-group by engine to speed up resampling
    grouped = {e: df[df['engine_id'] == e] for e in engines}
    
    for _ in range(n_resamples):
        sampled_engines = rng.choice(engines, size=n_engines, replace=True)
        pieces = [grouped[e] for e in sampled_engines]
        sample_df = pd.concat(pieces, axis=0)
        
        n_a = compute_naurc(sample_df[score_a].values, sample_df[err_col].values)
        n_b = compute_naurc(sample_df[score_b].values, sample_df[err_col].values)
        diffs.append(n_a - n_b)
        
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def bootstrap_naurc(df, score_col, err_col, n_resamples=1000, seed=0):
    rng = np.random.RandomState(seed)
    engines = df['engine_id'].unique()
    n_engines = len(engines)
    
    naurcs = []
    grouped = {e: df[df['engine_id'] == e] for e in engines}
    
    for _ in range(n_resamples):
        sampled_engines = rng.choice(engines, size=n_engines, replace=True)
        pieces = [grouped[e] for e in sampled_engines]
        sample_df = pd.concat(pieces, axis=0)
        n_a = compute_naurc(sample_df[score_col].values, sample_df[err_col].values)
        naurcs.append(n_a)
        
    naurcs = np.array(naurcs)
    return np.percentile(naurcs, 2.5), np.percentile(naurcs, 97.5)

def bootstrap_auroc(df, score_col, label_col, n_resamples=1000, seed=0):
    rng = np.random.RandomState(seed)
    engines = df['engine_id'].unique()
    n_engines = len(engines)
    
    aurocs = []
    grouped = {e: df[df['engine_id'] == e] for e in engines}
    
    for _ in range(n_resamples):
        sampled_engines = rng.choice(engines, size=n_engines, replace=True)
        pieces = [grouped[e] for e in sampled_engines]
        sample_df = pd.concat(pieces, axis=0)
        if len(sample_df[label_col].unique()) > 1:
            aurocs.append(roc_auc_score(sample_df[label_col], sample_df[score_col]))
        
    if len(aurocs) == 0: return np.nan, np.nan
    aurocs = np.array(aurocs)
    return np.percentile(aurocs, 2.5), np.percentile(aurocs, 97.5)

def get_true_rul_bin(rul):
    if rul <= 30: return "0-30"
    elif rul <= 60: return "30-60"
    elif rul <= 90: return "60-90"
    else: return "90+"

def rank_res(y, X):
    r_y = y.rank()
    r_X = X.rank()
    mod = LinearRegression().fit(r_X, r_y)
    return r_y - mod.predict(r_X)

def get_flag_worst(errs, q):
    threshold = np.quantile(errs, 1 - q)
    return (errs >= threshold).astype(int)

def plot_rc_curve(scores_dict, errs, title, filename):
    plt.figure()
    for name, scores in scores_dict.items():
        rng = np.random.RandomState(42)
        noise = rng.uniform(-1e-8, 1e-8, size=len(scores))
        idx = np.argsort(scores + noise)
        sorted_errs = errs[idx]
        cum_sum = np.cumsum(sorted_errs)
        k_arr = np.arange(1, len(sorted_errs) + 1)
        cum_mean = cum_sum / k_arr
        coverage = k_arr / len(sorted_errs)
        
        # Only plot >= 0.05
        mask = coverage >= 0.05
        plt.plot(coverage[mask], cum_mean[mask], label=name)
        
    plt.xlabel("Coverage")
    plt.ylabel("Risk (Mean Absolute Error)")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.close()

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    seed = args.seed
    
    s4 = pd.read_csv(PROJECT_ROOT / "results" / "step4_metrics.csv")
    s4 = s4[s4["model"] == "baseline"].copy()
    s4 = s4[["source", "target", "split", "scenario", "seed", "degenerate"]]
    s4.rename(columns={"degenerate": "model_degenerate"}, inplace=True)
    
    df = pd.read_csv(PROJECT_ROOT / "results" / f"step5_scores_seed{seed}.csv")
    df = pd.merge(df, s4, on=["source", "target", "split", "scenario", "seed"], how="left")
    
    definitions = {
        "higher_score_meaning": "less trustworthy",
        "scenario_groups": {
            "error_ranking": ["ID_FD002", "ID_FD004", "FAULT_1", "FAULT_2"],
            "stress": ["OPCOND_1", "OPCOND_2", "COMBINED"]
        },
        "n_aurc_definition": "nAURC = (AURC - AURC_oracle) / (AURC_random - AURC_oracle). Rank windows by score, with risk = mean abs_err of the retained windows over coverage 0.05..1.0. Oracle ranks by abs_err. 0 = best, 1 = random.",
        "decision_rule": "s_rec 'wins' a scenario if its nAURC is lower than BOTH s_mc and s_stage AND the paired-bootstrap 95% CI of the difference excludes zero. s_rec 'wins overall' only if it wins at least 3 of the 4 error-ranking scenarios. Otherwise report 'mixed' or 'does not win'. Do not change this after seeing results."
    }

    with open(PROJECT_ROOT / "results" / "decision_rule.json", "w") as f:
        json.dump(definitions, f, indent=2)

    with open(PROJECT_ROOT / "results" / "decision_rule.json") as f:
        defs = json.load(f)
        
    err_scenarios = defs["scenario_groups"]["error_ranking"]
    stress_scenarios = defs["scenario_groups"]["stress"]
    all_scenarios = err_scenarios + stress_scenarios
    
    scores = ["s_rec", "s_mc", "s_maha", "s_stage", "s_input"]
    
    err_res = []
    
    for scen in err_scenarios:
        scen_df = df[df["scenario"] == scen].copy()
        if scen_df.empty: continue
        scen_df['true_rul_bin'] = scen_df['true_rul'].apply(get_true_rul_bin)
        
        # Risk-coverage plot
        scores_dict = {s: scen_df[s].values for s in scores}
        scores_dict["oracle"] = scen_df["abs_err"].values
        plot_rc_curve(scores_dict, scen_df["abs_err"].values, f"RC Curve: {scen} (Seed {seed})", PROJECT_ROOT / "results" / f"step6_rc_{scen}_seed{seed}.png")
        
        flag_20 = get_flag_worst(scen_df['abs_err'], 0.2)
        flag_10 = get_flag_worst(scen_df['abs_err'], 0.1)
        
        for score in scores:
            sp_overall, _ = spearmanr(scen_df[score], scen_df['abs_err'])
            row = {"scenario": scen, "score": score, "sp_overall": sp_overall}
            
            for b in ["0-30", "30-60", "60-90", "90+"]:
                b_df = scen_df[scen_df['true_rul_bin'] == b]
                if not b_df.empty:
                    sp_b, _ = spearmanr(b_df[score], b_df['abs_err'])
                else:
                    sp_b = np.nan
                row[f"sp_{b}"] = sp_b
                
            res_score = rank_res(scen_df[score], scen_df[['pred_rul', 'true_rul']])
            res_err = rank_res(scen_df['abs_err'], scen_df[['pred_rul', 'true_rul']])
            sp_partial, _ = spearmanr(res_score, res_err)
            row["sp_partial"] = sp_partial
            
            row["auc_20"] = roc_auc_score(flag_20, scen_df[score]) if len(np.unique(flag_20)) > 1 else np.nan
            row["auc_10"] = roc_auc_score(flag_10, scen_df[score]) if len(np.unique(flag_10)) > 1 else np.nan
            
            naurc = compute_naurc(scen_df[score].values, scen_df['abs_err'].values)
            ci_low, ci_high = bootstrap_naurc(scen_df, score, 'abs_err')
            
            row["nAURC"] = naurc
            row["nAURC_ci_low"] = ci_low
            row["nAURC_ci_high"] = ci_high
            
            err_res.append(row)
            
    pd.DataFrame(err_res).to_csv(PROJECT_ROOT / "results" / f"step6_error_ranking_seed{seed}.csv", index=False)
    
    # Decisions
    decision_rows = []
    wins_count = 0
    for scen in err_scenarios:
        scen_df = df[df["scenario"] == scen].copy()
        if scen_df.empty: continue
        
        naurc_rec = compute_naurc(scen_df["s_rec"].values, scen_df["abs_err"].values)
        naurc_mc = compute_naurc(scen_df["s_mc"].values, scen_df["abs_err"].values)
        naurc_stage = compute_naurc(scen_df["s_stage"].values, scen_df["abs_err"].values)
        
        ci_rec_mc_low, ci_rec_mc_high = bootstrap_diff(scen_df, "s_rec", "s_mc", "abs_err")
        ci_rec_stage_low, ci_rec_stage_high = bootstrap_diff(scen_df, "s_rec", "s_stage", "abs_err")
        
        win_mc = (naurc_rec < naurc_mc) and (ci_rec_mc_high < 0)
        win_stage = (naurc_rec < naurc_stage) and (ci_rec_stage_high < 0)
        
        is_win = win_mc and win_stage
        if is_win: wins_count += 1
        
        decision_rows.append({
            "scenario": scen,
            "win": is_win,
            "win_mc": win_mc,
            "win_stage": win_stage,
            "diff_mc_ci": f"[{ci_rec_mc_low:.4f}, {ci_rec_mc_high:.4f}]",
            "diff_stage_ci": f"[{ci_rec_stage_low:.4f}, {ci_rec_stage_high:.4f}]"
        })
        
    decision_rows.append({
        "scenario": "OVERALL",
        "win": wins_count >= 3,
        "win_mc": "",
        "win_stage": "",
        "diff_mc_ci": "",
        "diff_stage_ci": ""
    })
    
    pd.DataFrame(decision_rows).to_csv(PROJECT_ROOT / "results" / f"step6_decision_seed{seed}.csv", index=False)
    
    # SHIFT DETECTION
    shift_res = []
    for scen in all_scenarios:
        scen_rows = df[df["scenario"] == scen]
        if scen_rows.empty: continue
        src = scen_rows.iloc[0]["source"]
        
        # Get source held-out for this source
        ho_scen = f"ID_{src}" if f"ID_{src}" in all_scenarios else "in_domain_heldout"
        ho_rows = df[(df["source"] == src) & (df["scenario"] == ho_scen) & (df["split"] == "source_heldout")].copy()
        tgt_rows = df[(df["scenario"] == scen) & (df["split"] == "target")].copy()
        
        if ho_rows.empty or tgt_rows.empty:
            continue
            
        ho_rows['shift_label'] = 0
        tgt_rows['shift_label'] = 1
        comb_df = pd.concat([ho_rows, tgt_rows], axis=0)
        
        is_deg = bool(tgt_rows.iloc[0]["model_degenerate"]) if not tgt_rows.empty else False
        
        for score in scores:
            auc = roc_auc_score(comb_df['shift_label'], comb_df[score])
            ci_low, ci_high = bootstrap_auroc(comb_df, score, 'shift_label')
            shift_res.append({
                "scenario": scen,
                "score": score,
                "AUROC": auc,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "model_degenerate": is_deg
            })
            
        # Plot histograms
        plt.figure()
        plt.hist(ho_rows["s_rec"], bins=50, alpha=0.5, label='source_heldout', density=True)
        plt.hist(tgt_rows["s_rec"], bins=50, alpha=0.5, label='target', density=True)
        if scen in stress_scenarios:
            plt.xscale('log')
        plt.legend()
        plt.title(f"s_rec shift: {scen} (Seed {seed})")
        plt.savefig(PROJECT_ROOT / "results" / f"step6_hist_{scen}_seed{seed}.png")
        plt.close()
        
    shift_df = pd.DataFrame(shift_res)
    shift_df.to_csv(PROJECT_ROOT / "results" / f"step6_shift_detection_seed{seed}.csv", index=False)
    
    # Compare s_rec against s_input in shift detection
    for scen in all_scenarios:
        scen_res = shift_df[shift_df["scenario"] == scen]
        if scen_res.empty: continue
        rec_auc = scen_res[scen_res["score"] == "s_rec"]["AUROC"].values[0]
        inp_auc = scen_res[scen_res["score"] == "s_input"]["AUROC"].values[0]
        beats = "Yes" if rec_auc > inp_auc else "No"
        print(f"Shift Detection {scen}: s_rec beats s_input? {beats} (s_rec={rec_auc:.4f}, s_input={inp_auc:.4f})")
    
    # DIAGNOSTICS
    with open(PROJECT_ROOT / "results" / f"step6_diagnostics_seed{seed}.csv", "w") as f:
        f.write("DIAGNOSTICS\n")
    
    for scen in all_scenarios:
        scen_df = df[df["scenario"] == scen]
        if scen_df.empty: continue
        corr = scen_df[scores].corr(method='spearman')
        print(f"\nCorrelation matrix {scen}:")
        print(corr)
        
    fd2_ho = df[(df["source"] == "FD002") & (df["split"] == "source_heldout")].copy()
    if not fd2_ho.empty:
        fd2_ho['true_rul_bin'] = fd2_ho['true_rul'].apply(get_true_rul_bin)
        for b in ["0-30", "30-60", "60-90", "90+"]:
            b_df = fd2_ho[fd2_ho['true_rul_bin'] == b]
            if not b_df.empty:
                sp, _ = spearmanr(b_df['s_mc'], b_df['s_maha'])
                print(f"FD002 HO bin {b}: s_mc vs s_maha spearman = {sp:.4f}")
                
    counts = df.groupby('scenario').agg(
        n_engines=('engine_id', 'nunique'),
        n_windows=('window_idx', 'count')
    )
    print("\nCounts:")
    print(counts)
    
    # Sanity check: AUROC of random score
    ho_rows = df[(df["source"] == "FD001") & (df["scenario"] == "in_domain_heldout")].copy()
    tgt_rows = df[(df["scenario"] == "FAULT_1")].copy()
    if not ho_rows.empty and not tgt_rows.empty:
        ho_rows['shift_label'] = 0
        tgt_rows['shift_label'] = 1
        comb_df = pd.concat([ho_rows, tgt_rows], axis=0)
        rng = np.random.RandomState(42)
        comb_df['random_score'] = rng.rand(len(comb_df))
        rand_auc = roc_auc_score(comb_df['shift_label'], comb_df['random_score'])
        print(f"\nSanity Check: AUROC of random score = {rand_auc:.4f} (should be ~0.5)")

if __name__ == "__main__":
    main()
