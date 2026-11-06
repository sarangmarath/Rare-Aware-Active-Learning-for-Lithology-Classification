"""Phase 2: baseline SVM on the full labelled pool (upper bound) + tuning."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.svm import SVC

SEED = 42
TUNE_SIZE = 8000   # tune on a subset so the grid search stays fast

root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed"
d = np.load(proc / "data.npz", allow_pickle=True)   # wells are string arrays
meta = json.loads((proc / "meta.json").read_text())
X_pool, y_pool, wells_pool = d["X_pool"], d["y_pool"], d["wells_pool"]
X_test, y_test = d["X_test"], d["y_test"]
names, rare_idx = meta["names"], meta["rare_idx"]
labels = list(range(len(names)))
print("Pool:", X_pool.shape, "| Test:", X_test.shape)

# 1. small grid search, grouped by well so folds don't share wells
rng = np.random.default_rng(SEED)
sub = rng.choice(len(X_pool), TUNE_SIZE, replace=False)
grid = {"C": [1, 10, 100], "gamma": ["scale", 0.1, 0.5]}
gs = GridSearchCV(SVC(kernel="rbf", decision_function_shape="ovr"), grid,
                  scoring="f1_macro", cv=GroupKFold(n_splits=3), n_jobs=-1)
t = time.time()
gs.fit(X_pool[sub], y_pool[sub], groups=wells_pool[sub])
best = gs.best_params_
print(f"Best params: {best} (CV macro-F1 {gs.best_score_:.3f}, {time.time()-t:.0f}s)")
(root / "results" / "svm_params.json").write_text(json.dumps(best))

# 2. train on the FULL pool: plain vs class_weight='balanced'
def evaluate(model, tag):
    t = time.time()
    model.fit(X_pool, y_pool)
    pred = model.predict(X_test)
    rec = recall_score(y_test, pred, labels=labels, average=None, zero_division=0)
    row = {"model": tag,
           "accuracy": accuracy_score(y_test, pred),
           "macro_F1": f1_score(y_test, pred, labels=labels, average="macro", zero_division=0),
           "rare_macro_F1": f1_score(y_test, pred, labels=rare_idx, average="macro", zero_division=0),
           "dominant_macro_F1": f1_score(y_test, pred, labels=[i for i in labels if i not in rare_idx],
                                         average="macro", zero_division=0)}
    for i, n in enumerate(names):
        row[f"recall_{n}"] = rec[i]
    print(f"{tag}: trained+tested in {time.time()-t:.0f}s")
    return row

rows = [evaluate(SVC(kernel="rbf", **best), "plain"),
        evaluate(SVC(kernel="rbf", class_weight="balanced", **best), "balanced")]
res = pd.DataFrame(rows).set_index("model").T
print("\n", res.round(3))
res.to_csv(root / "results" / "baseline_svm.csv")
print("\nSaved results/baseline_svm.csv and results/svm_params.json")