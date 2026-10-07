import pandas as pd
import numpy as np

# Load the data
df4 = pd.read_csv('results/step4_metrics.csv')
df5 = pd.read_csv('results/step5_scores_seed0.csv')

# 1. Check d passes: the MAE recomputed from abs_err matches Step 4
print("--- Check 1: MAE matches Step 4 ---")
df5_grouped = df5.groupby(['scenario', 'source', 'target', 'seed', 'split', 'predictor'])['abs_err'].mean().reset_index()
df5_grouped.rename(columns={'predictor': 'model', 'abs_err': 'MAE_recomputed'}, inplace=True)

merged = pd.merge(df4, df5_grouped, on=['scenario', 'source', 'target', 'seed', 'split', 'model'], how='inner')

merged['diff'] = np.abs(merged['MAE'] - merged['MAE_recomputed'])
max_diff = merged['diff'].max()
if max_diff < 1e-3:
    print(f"PASS: MAE matches between Step 4 and Step 5. Maximum difference is {max_diff:.5f}")
else:
    print(f"FAIL: MAE does not match. Maximum difference is {max_diff:.5f}")

# 2. s_rec is not near-identical to s_stage or s_maha
print("\n--- Check 2: Correlation between scores ---")
corr_rec_stage = df5['s_rec'].corr(df5['s_stage'])
corr_rec_maha = df5['s_rec'].corr(df5['s_maha'])
print(f"Correlation between s_rec and s_stage: {corr_rec_stage:.4f}")
print(f"Correlation between s_rec and s_maha: {corr_rec_maha:.4f}")
if abs(corr_rec_stage) > 0.95 or abs(corr_rec_maha) > 0.95:
    print("WARNING: s_rec is near-identical to s_stage or s_maha (> 0.95 correlation)")
else:
    print("PASS: s_rec provides distinct information (correlation < 0.95)")

# 3. s_rec is higher on the target than on source held-out for the FAULT scenarios
print("\n--- Check 3: s_rec higher on target than source held-out for FAULT scenarios ---")
fault_scenarios = ['FAULT_1', 'FAULT_2']

for fault in fault_scenarios:
    # Get source dataset for this fault scenario
    fault_df = df5[df5['scenario'] == fault]
    if fault_df.empty:
        continue
    source = fault_df['source'].iloc[0]
    
    # Get target s_rec
    target_s_rec = fault_df['s_rec'].mean()
    
    # Get source held-out scenario for this source
    source_heldout_df = df5[(df5['source'] == source) & (df5['split'] == 'source_heldout')]
    if source_heldout_df.empty:
        print(f"No source_heldout found for source {source}")
        continue
    
    source_heldout_s_rec = source_heldout_df['s_rec'].mean()
    
    if target_s_rec > source_heldout_s_rec:
        print(f"PASS: [{fault}] target ({target_s_rec:.5f}) > source_heldout ({source_heldout_s_rec:.5f})")
    else:
        print(f"FINDING: [{fault}] target ({target_s_rec:.5f}) <= source_heldout ({source_heldout_s_rec:.5f})")

