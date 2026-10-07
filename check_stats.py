import pandas as pd
import numpy as np
import os

df4 = pd.read_csv('results/step4_metrics.csv')
df5 = pd.read_csv('results/step5_scores_seed0.csv')

print("(b)")
for col in ['s_rec', 's_mc', 's_maha', 's_stage']:
    count = df5[col].isna().sum() + np.isinf(df5[col]).sum()
    print(f"{col}: {count}")

print("(c)")
df5_counts_baseline = df5[df5['predictor'] == 'baseline'].groupby(['source', 'split', 'scenario']).size().reset_index(name='n_windows_5')
df4_baseline = df4[df4['model'] == 'baseline'][['source', 'split', 'scenario', 'n_windows']]
merged_baseline = pd.merge(df5_counts_baseline, df4_baseline, on=['source', 'split', 'scenario'], how='outer')
mismatches = merged_baseline[merged_baseline['n_windows_5'] != merged_baseline['n_windows']]
num_mismatches = len(mismatches)
print(f"{'pass' if num_mismatches == 0 else 'fail'}, mismatches: {num_mismatches}")

print("(e)")
print(df5.groupby(['scenario', 'split'])[['s_rec', 's_mc', 's_maha', 's_stage']].mean().to_string())

print("(f)")
target_df = df5[df5['split'] == 'target']
if not target_df.empty:
    print("split: target")
    print(target_df[['s_rec', 's_mc', 's_maha', 's_stage']].corr().to_string())
else:
    print("split: all")
    print(df5[['s_rec', 's_mc', 's_maha', 's_stage']].corr().to_string())

print("(h)")
size = os.path.getsize('results/step5_scores_seed0.csv')
print(f"results/step5_scores_seed0.csv, {size} bytes")
