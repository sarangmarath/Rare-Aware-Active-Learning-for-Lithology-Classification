"""Phase 3: active learning loop with random, uncertainty and rare-aware queries."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import SVC

root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed"
d = np.load(proc / "data.npz", allow_pickle=True)
meta = json.loads((proc / "meta.json").read_text())
PARAMS = json.loads((root / "results" / "svm_params.json").read_text())
X_pool, y_pool = d["X_pool"], d["y_pool"]
X_test, y_test = d["X_test"], d["y_test"]
NAMES, RARE = meta["names"], np.array(meta["rare_idx"])
K = len(NAMES)
DOMINANT = np.array([i for i in range(K) if i not in RARE])


# ---------- query strategies: return indices into X_cand ----------
def margin(S):
    """Gap between the top two class scores (small = uncertain)."""
    top2 = np.partition(S, -2, axis=1)
    return top2[:, -1] - top2[:, -2]


def random_query(model, X_cand, B, rng, rho):
    return rng.choice(len(X_cand), B, replace=False), 0


def uncertainty_query(model, X_cand, B, rng, rho):
    m = margin(model.decision_function(X_cand))
    return np.argsort(m)[:B], 0


def rare_aware_query(model, X_cand, B, rng, rho):
    S = model.decision_function(X_cand)
    m = margin(S)
    pred = model.classes_[S.argmax(axis=1)]
    k = int(round(rho * B))
    cand = np.where(np.isin(pred, RARE))[0]            # predicted-rare candidates
    rare_pick = cand[np.argsort(m[cand])[:k]]          # most uncertain among them
    shortfall = k - len(rare_pick)                     # fewer candidates than k
    mask = np.ones(len(X_cand), bool)
    mask[rare_pick] = False
    rest = np.where(mask)[0]
    unc_pick = rest[np.argsort(m[rest])[:B - len(rare_pick)]]
    return np.concatenate([rare_pick, unc_pick]), shortfall


STRATEGIES = {"random": random_query,
              "uncertainty": uncertainty_query,
              "rare_aware": rare_aware_query}


# ---------- helpers ----------
def init_labelled(y, rng, size, min_per_class):
    idx = []
    for c in range(K):
        ci = np.where(y == c)[0]
        idx += rng.choice(ci, min(min_per_class, len(ci)), replace=False).tolist()
    remaining = np.setdiff1d(np.arange(len(y)), idx)
    idx += rng.choice(remaining, size - len(idx), replace=False).tolist()
    return np.array(idx)


def evaluate(model):
    pred = model.predict(X_test)
    labels = list(range(K))
    rec = recall_score(y_test, pred, labels=labels, average=None, zero_division=0)
    out = {"accuracy": accuracy_score(y_test, pred),
           "macro_F1": f1_score(y_test, pred, labels=labels, average="macro", zero_division=0),
           "rare_macro_F1": f1_score(y_test, pred, labels=RARE, average="macro", zero_division=0),
           "dominant_macro_F1": f1_score(y_test, pred, labels=DOMINANT, average="macro", zero_division=0),
           "rare_macro_recall": rec[RARE].mean()}
    for i, n in enumerate(NAMES):
        out[f"recall_{n}"] = rec[i]
    return out


def run_one(strategy, seed, cfg):
    rng_init = np.random.default_rng(seed)           # same start for every strategy
    rng_q = np.random.default_rng(seed + 1000)
    labelled = init_labelled(y_pool, rng_init, cfg["init_size"], cfg["init_min_per_class"])
    is_lab = np.zeros(len(y_pool), bool)
    is_lab[labelled] = True
    query_fn = STRATEGIES[strategy]
    rows, queried_rare_total = [], 0

    for r in range(cfg["rounds"] + 1):
        model = SVC(kernel="rbf", **PARAMS)
        lab_idx = np.where(is_lab)[0]
        model.fit(X_pool[lab_idx], y_pool[lab_idx])
        row = {"strategy": strategy, "seed": seed, "round": r, "n_labels": len(lab_idx),
               "rare_in_labelled": int(np.isin(y_pool[lab_idx], RARE).sum()),
               "shortfall": 0, "rare_queried_this_round": 0}
        if r < cfg["rounds"]:
            unl = np.where(~is_lab)[0]
            if len(unl) > cfg["candidates"]:
                unl = rng_q.choice(unl, cfg["candidates"], replace=False)
            local, short = query_fn(model, X_pool[unl], cfg["batch"], rng_q, cfg["rho"])
            chosen = unl[local]
            assert len(set(chosen)) == cfg["batch"] and not is_lab[chosen].any()
            row["shortfall"] = short
            row["rare_queried_this_round"] = int(np.isin(y_pool[chosen], RARE).sum())
            is_lab[chosen] = True
        row.update(evaluate(model))
        rows.append(row)
    assert is_lab.sum() == cfg["init_size"] + cfg["rounds"] * cfg["batch"]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=25)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--init_size", type=int, default=100)
    ap.add_argument("--init_min_per_class", type=int, default=2)
    ap.add_argument("--candidates", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    cfg = vars(a)

    t = time.time()
    jobs = [(s, seed) for seed in range(a.seeds) for s in STRATEGIES]
    out = Parallel(n_jobs=a.workers, verbose=5)(
        delayed(run_one)(s, seed, cfg) for s, seed in jobs)
    res = pd.concat(out, ignore_index=True)
    fname = f"al_results_rho{a.rho}{a.tag}.csv"
    res.to_csv(root / "results" / fname, index=False)
    print(f"\nDone in {time.time() - t:.0f}s -> results/{fname}")

    last = res[res["round"] == a.rounds]
    cols = ["accuracy", "macro_F1", "rare_macro_F1", "rare_macro_recall",
            "dominant_macro_F1", "rare_in_labelled"]
    print("\nFinal-round mean over seeds:")
    print(last.groupby("strategy")[cols].mean().round(3))
    print("\nAvg rare-class shortfall per round (rare_aware):",
          round(res[res.strategy == "rare_aware"]["shortfall"].mean(), 2))