import pandas as pd
from pathlib import Path
import pytest
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.constants import SCENARIOS, VALID_DATASET_IDS

METRICS_CSV = PROJECT_ROOT / "results" / "step4_metrics.csv"

def test_step4_coverage():
    """Ensure results/step4_metrics.csv contains all expected runs."""
    if not METRICS_CSV.exists():
        pytest.skip("step4_metrics.csv not found")
        
    df = pd.read_csv(METRICS_CSV)
    
    models = ["baseline", "ssl_finetuned", "constant"]
    seeds = [0, 1, 2]
    
    missing = []
    
    for src in VALID_DATASET_IDS:
        scens = [s for s, cfg in SCENARIOS.items() if cfg["source"] == src]
        for s in seeds:
            for m in models:
                if m != "constant":
                    mask_val = (df["source"] == src) & (df["seed"] == s) & (df["model"] == m) & (df["split"] == "val")
                    if not df[mask_val].shape[0]:
                        missing.append(f"{src} seed {s} model {m} split val")
                
                mask_ho = (df["source"] == src) & (df["seed"] == s) & (df["model"] == m) & (df["split"] == "source_heldout")
                if not df[mask_ho].shape[0]:
                    missing.append(f"{src} seed {s} model {m} split source_heldout")
                
                for sc in scens:
                    if SCENARIOS[sc]["shift_type"] != "in_domain":
                        mask_tgt = (df["source"] == src) & (df["seed"] == s) & (df["model"] == m) & (df["split"] == "target") & (df["scenario"] == sc)
                        if not df[mask_tgt].shape[0]:
                            missing.append(f"{src} seed {s} model {m} split target scenario {sc}")
                            
    assert not missing, f"Missing {len(missing)} rows in step4_metrics.csv: {missing}"
