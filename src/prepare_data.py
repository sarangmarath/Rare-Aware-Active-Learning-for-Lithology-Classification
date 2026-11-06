"""Phase 1: load, clean, define rare classes, split by well, scale."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------- config ----------------
SEED = 42
RARE_THRESHOLD = 0.05      # X: class is rare if < 5% of samples
MIN_CLASS_ROWS = 500       # drop classes with fewer rows than this (whole dataset)
MIN_RARE_COUNT = 30        # each rare class needs this many in pool AND test
TEST_WELL_FRAC = 0.2
POOL_SIZE = 40000
TEST_SIZE = 40000
FEATURES = ["CALI", "RMED", "RDEP", "RHOB", "GR", "NPHI", "DTC"]
LOG_FEATURES = ["RMED", "RDEP"]
LABEL = "FORCE_2020_LITHOFACIES_LITHOLOGY"
NAMES = {30000: "Sandstone", 65030: "Sandstone/Shale", 65000: "Shale",
         80000: "Marl", 74000: "Dolomite", 70000: "Limestone",
         70032: "Chalk", 88000: "Halite", 86000: "Anhydrite",
         99000: "Tuff", 90000: "Coal", 93000: "Basement"}
# ----------------------------------------

root = Path(__file__).resolve().parents[1]
out_dir = root / "data" / "processed"
out_dir.mkdir(exist_ok=True)
(root / "results").mkdir(exist_ok=True)

# 1. load + clean
df = pd.read_csv(root / "data" / "train.csv", sep=";",
                 usecols=["WELL", LABEL] + FEATURES)
print("raw rows:", len(df), "| wells:", df["WELL"].nunique())
df = df.dropna(subset=FEATURES + [LABEL]).copy()
for c in LOG_FEATURES:
    df[c] = np.log10(df[c].clip(lower=1e-3))
print("rows after dropping missing:", len(df), "| wells:", df["WELL"].nunique())

# 1b. exclude classes that are too small to learn or evaluate
counts = df[LABEL].value_counts()
too_small = counts[counts < MIN_CLASS_ROWS].index.tolist()
print("Excluded (too few rows):",
      {NAMES.get(c, c): int(counts[c]) for c in too_small})
df = df[~df[LABEL].isin(too_small)].copy()

# 2. class distribution + rare classes
freq = df[LABEL].value_counts(normalize=True)
rare_codes = freq[freq < RARE_THRESHOLD].index.tolist()
classes = sorted(df[LABEL].unique())
code2idx = {c: i for i, c in enumerate(classes)}
print("\nClass distribution (after exclusion):")
for c in freq.index:
    print(f"  {c:>6} {NAMES.get(c, '?'):<16} {freq[c]*100:6.2f}%"
          f"{'  RARE' if c in rare_codes else ''}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar([NAMES.get(c, str(c)) for c in freq.index], freq.values * 100,
       color=["tab:red" if c in rare_codes else "tab:blue" for c in freq.index])
ax.set_yscale("log"); ax.set_ylabel("% of samples (log scale)")
ax.axhline(RARE_THRESHOLD * 100, ls="--", c="k", lw=1)
plt.xticks(rotation=45, ha="right"); plt.tight_layout()
plt.savefig(root / "results" / "class_distribution.png", dpi=200)

# 3. split by WELL; retry until every rare class is present in pool and test
wells_all = df["WELL"].unique()
for attempt in range(500):
    rng = np.random.default_rng(SEED + attempt)
    wells = rng.permutation(wells_all)
    n_test = max(1, int(len(wells) * TEST_WELL_FRAC))
    test_wells, pool_wells = wells[:n_test], wells[n_test:]
    pool = df[df["WELL"].isin(pool_wells)]
    test = df[df["WELL"].isin(test_wells)]
    pool = pool.sample(min(POOL_SIZE, len(pool)), random_state=SEED)
    test = test.sample(min(TEST_SIZE, len(test)), random_state=SEED)
    pc, tc = pool[LABEL].value_counts(), test[LABEL].value_counts()
    if all(pc.get(c, 0) >= MIN_RARE_COUNT and tc.get(c, 0) >= MIN_RARE_COUNT
           for c in rare_codes):
        break
else:
    raise RuntimeError("No valid well split found; lower MIN_RARE_COUNT.")
print(f"\nValid well split found at attempt {attempt} (split seed {SEED + attempt})")
print("Test wells:", len(test_wells), "| Pool wells:", len(pool_wells))

# 4. scale (fit on pool only)
scaler = StandardScaler().fit(pool[FEATURES])
X_pool, X_test = scaler.transform(pool[FEATURES]), scaler.transform(test[FEATURES])
y_pool = pool[LABEL].map(code2idx).to_numpy()
y_test = test[LABEL].map(code2idx).to_numpy()

# 5. save
np.savez(out_dir / "data.npz", X_pool=X_pool, y_pool=y_pool,
         wells_pool=pool["WELL"].to_numpy(), X_test=X_test, y_test=y_test,
         wells_test=test["WELL"].to_numpy())
joblib.dump(scaler, out_dir / "scaler.joblib")
meta = {"classes": [int(c) for c in classes],
        "names": [NAMES.get(c, str(c)) for c in classes],
        "rare_idx": [code2idx[c] for c in rare_codes],
        "features": FEATURES, "rare_threshold": RARE_THRESHOLD,
        "split_seed": SEED + attempt,
        "excluded_classes": [int(c) for c in too_small]}
(out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

print("\nPer-class counts (pool | test):")
for c in classes:
    i = code2idx[c]
    print(f"  {NAMES.get(c, c):<16} {np.sum(y_pool == i):>6} | {np.sum(y_test == i):>6}"
          f"{'  RARE' if c in rare_codes else ''}")
print("\nSaved to", out_dir)