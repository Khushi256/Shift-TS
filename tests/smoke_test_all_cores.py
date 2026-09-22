import sys, numpy as np, torch
sys.path.insert(0, '.')

from src.data import build_engine_splits, fit_scaler, apply_scaler, CMAPSSDataset, FewShotDataset
from src.data.dataset import SSLDataset

out    = build_engine_splits('CMAPSSData')
scaler = fit_scaler(out['df_train'])
s      = out['summary']
print("[Core 1] Engines train:{} val:{} target:{}".format(
    s['n_train_engines'], s['n_val_engines'], s['n_target_engines']))
df_tr = apply_scaler(out['df_train'], scaler)
df_ta = apply_scaler(out['df_target'], scaler)

from src.models.baseline import GRUBaseline
from src.models.encoder import GRUEncoder
ds = CMAPSSDataset(df_tr)
model = GRUBaseline(input_dim=ds.n_features)
x, y, _ = ds[0]
pred = model(x.unsqueeze(0))
print("[Core 2] Baseline pred={:.2f}  nonneg={}".format(float(pred), float(pred) >= 0))

from src.models.ssl_heads import SSLModel
ssl_ds = SSLDataset(df_tr)
v1, v2, mask, y2, _ = ssl_ds[0]
ssl_m = SSLModel(input_dim=ds.n_features)
losses = ssl_m(v1.unsqueeze(0), v2.unsqueeze(0), mask.unsqueeze(0))
print("[Core 3] SSL recon:{:.4f}  contrast:{:.4f}".format(
    losses['recon_loss'].item(), losses['contrastive_loss'].item()))

from src.training.adapt import FewShotAdapter
ds_t = CMAPSSDataset(df_ta)
ds_f = FewShotDataset(ds_t, label_fraction=0.05)
print("[Core 4] FewShot 5%: {} windows from {} total".format(len(ds_f), len(ds_t)))

from src.models.uncertainty import MCDropoutWrapper
from src.evaluation.metrics import compute_all_metrics
wrapper = MCDropoutWrapper(model, n_passes=10)
mean, std = wrapper.mc_predict(x.unsqueeze(0))
print("[Core 5] MC Dropout mean={:.2f}  std={:.4f}".format(float(mean), float(std)))
dummy_m = np.random.rand(100) * 100
dummy_s = np.random.rand(100) * 10 + 1
dummy_t = np.random.rand(100) * 100
mets = compute_all_metrics(dummy_m, dummy_s, dummy_t)
print("         coverage={:.2f}  spearman_rho={:.2f}".format(
    mets['coverage_95pct'], mets['spearman_rho']))

from src.evaluation.robustness import PERTURBATIONS
sample_np = x.numpy()[np.newaxis]
for name, fn in PERTURBATIONS.items():
    out_p = fn(sample_np, seed=0)
print("[Core 6] {} perturbations OK".format(len(PERTURBATIONS)))

print("\nAll cores smoke-tested successfully!")
