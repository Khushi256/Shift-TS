import json
import pandas as pd
import numpy as np
import os

with open('results/step5_fit_meta_seed0.json', 'r') as f:
    meta = json.load(f)

print("KEYS in json:", list(meta.keys()))

if 'leak_assertion' in meta:
    print("a)", meta['leak_assertion'])
else:
    print("a) not found in meta")
