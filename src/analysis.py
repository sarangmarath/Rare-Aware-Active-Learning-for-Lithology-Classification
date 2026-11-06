"""Phase 4: statistics, tables and figures from the active learning results."""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

root = Path(__file__).resolve().parents[1]
res_dir = root / "results"
fig_dir = res_dir / "figures"
fig_dir.mkdir(exist_ok=True)
meta = json.loads((root / "data" / "processed" / "meta.json").read_text())
NAMES, RARE = meta["names"], meta["rare_idx"]
RARE_NAMES = [NAMES[i] for i in RARE]

ap = argparse.ArgumentParser()
ap.add_argument("--rho", type=float, default=None,
                help="rho for the per-class plot (default: best by rare F1)")
args = ap.parse_args()

# ---------- load all rho files ----------
files = []
for f in res_dir.glob("al_results_rho*.csv"):
    m = re.fullmatch(r"al_results_rho(.+)\.csv", f.name)
    try:
        files.append((float(m.group(1)), f))
    except ValueError:
        continue                                   # skips *_test files
files.sort()
assert files, "no al_results_rho*.csv found in results/"

frames = []
for i, (rho, f) in enumerate(files):
    d = pd.read_csv(f)
    if i > 0:                                      # random/uncertainty are identical across files
        d = d[d.strategy == "rare_aware"]
    d["method"] = np.where(d.strategy == "rare_aware", f"rare_aware_{rho}", d.strategy)
    d["rho"] = rho
    frames.append(d)
df = pd.concat(frames, ignore_index=True)

RHOS = [r for r, _ in files]
BASE = ["random", "uncertainty"]
RA = [f"rare_aware_{r}" for r in RHOS]
METHODS = BASE + RA
LAST = df["round"].max()
final = df[df["round"] == LAST]
print(f"rho values: {RHOS} | seeds: {df.seed.nunique()} | final round: {LAST}\n")

# ---------- per-seed values (rows = seeds, cols = methods) ----------
SUMMARY = ["rare_macro_F1", "rare_macro_recall", "dominant_macro_F1",
           "accuracy", "macro_F1", "rare_in_labelled", "rare_F1_AULC"]
P = {k: final.pivot(index="seed", columns="method", values=k)[METHODS]
     for k in SUMMARY if k != "rare_F1_AULC"}
# AULC = mean rare F1 over all rounds (budget-free measure of label efficiency)
P["rare_F1_AULC"] = df.groupby(["seed", "method"])["rare_macro_F1"].mean().unstack()[METHODS]

# ---------- 1. mean +- std table ----------
tab = pd.DataFrame({m: {k: f"{P[k][m].mean():.3f} +- {P[k][m].std():.3f}" for k in SUMMARY}
                    for m in METHODS}).T
tab.to_csv(res_dir / "summary_final.csv")
print("=== Final-round mean +- std over seeds ===")
print(tab.to_string(), "\n")


# ---------- 2. Wilcoxon signed-rank (paired by seed) ----------
def holm(p):
    p = np.asarray(p)
    order, n = np.argsort(p), len(p)
    adj, run = np.empty(n), 0.0
    for i, k in enumerate(order):
        run = max(run, (n - i) * p[k])
        adj[k] = min(1.0, run)
    return adj


TESTED = ["rare_macro_F1", "rare_macro_recall", "rare_F1_AULC", "dominant_macro_F1", "accuracy"]
rows = []
for k in TESTED:
    part = []
    for m in RA:
        for b in BASE:
            a_, b_ = P[k][m].values, P[k][b].values
            diff = a_ - b_
            p = 1.0 if np.allclose(diff, 0) else wilcoxon(a_, b_).pvalue
            part.append({"metric": k, "method": m, "vs": b, "mean_diff": diff.mean(),
                         "seeds_better": f"{(diff > 0).sum()}/{len(diff)}", "p": p})
    adj = holm([r["p"] for r in part])               # correct across all rho x baseline tests
    for r, pa in zip(part, adj):
        r["p_holm"] = pa
    rows += part
wt = pd.DataFrame(rows)
wt.to_csv(res_dir / "wilcoxon_tests.csv", index=False)
print("=== Wilcoxon: rare_aware minus baseline (p_holm = corrected for multiple tests) ===")
print(wt.round(4).to_string(index=False), "\n")

# ---------- 3. labels needed to match random's final rare F1 ----------
curves = df.groupby(["n_labels", "method"])["rare_macro_F1"].mean().unstack()[METHODS]
target = curves["random"].iloc[-1]
eff = {}
for m in METHODS:
    hit = curves.index[curves[m] >= target]
    eff[m] = int(hit[0]) if len(hit) else "not reached"
print(f"=== Labels to reach random's final rare F1 ({target:.3f}) ===")
print(pd.Series(eff).to_string(), "\n")

# ---------- 4. figures ----------
cmap = plt.get_cmap("viridis")
colors = {"random": "tab:gray", "uncertainty": "black"}
colors.update({m: cmap(i / max(1, len(RA) - 1) * 0.85) for i, m in enumerate(RA)})
style = {"random": "-", "uncertainty": "--"}

# 4a. learning curves
metrics = [("rare_macro_F1", "Rare-class macro-F1"), ("rare_macro_recall", "Rare-class macro-recall"),
           ("dominant_macro_F1", "Dominant-class macro-F1"), ("accuracy", "Accuracy")]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (k, title) in zip(axes.ravel(), metrics):
    g = df.groupby(["n_labels", "method"])[k]
    mean, sem = g.mean().unstack()[METHODS], g.sem().unstack()[METHODS]
    for m in METHODS:
        ax.plot(mean.index, mean[m], style.get(m, "-"), color=colors[m], label=m, lw=1.6)
        ax.fill_between(mean.index, mean[m] - sem[m], mean[m] + sem[m], color=colors[m], alpha=0.12)
    ax.set_title(title)
    ax.set_xlabel("Number of labels")
    ax.grid(alpha=0.3)
axes[0, 0].legend(fontsize=8)
fig.suptitle("Learning curves (mean +- s.e.m. over seeds)")
fig.tight_layout()
fig.savefig(fig_dir / "learning_curves.png", dpi=200)
plt.close(fig)

# 4b. rho ablation
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (k, title) in zip(axes, [("rare_macro_F1", "Rare macro-F1"),
                                 ("dominant_macro_F1", "Dominant macro-F1"),
                                 ("rare_in_labelled", "Rare samples in labelled set")]):
    ax.errorbar(RHOS, [P[k][m].mean() for m in RA], yerr=[P[k][m].std() for m in RA],
                marker="o", capsize=3, color="tab:blue", label="rare_aware")
    for b in BASE:
        ax.axhline(P[k][b].mean(), color=colors[b], ls=style[b], label=b)
    ax.set_title(title)
    ax.set_xlabel("rho (reserved fraction)")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=8)
fig.suptitle("rho ablation at the final round (error bars = std)")
fig.tight_layout()
fig.savefig(fig_dir / "rho_ablation.png", dpi=200)
plt.close(fig)

# 4c. per-class recall
best = f"rare_aware_{args.rho}" if args.rho is not None else \
    max(RA, key=lambda m: P["rare_macro_F1"][m].mean())
cls_cols = [f"recall_{n}" for n in NAMES]
pc = final.groupby("method")[cls_cols].mean()
pc.columns = NAMES
pc.loc[METHODS].round(3).to_csv(res_dir / "per_class_recall.csv")
print(f"=== Per-class recall at final round (best rho: {best}) ===")
print(pc.loc[BASE + [best]].T.round(3).to_string(), "\n")

show = BASE + [best]
x = np.arange(len(NAMES))
w = 0.8 / len(show)
fig, ax = plt.subplots(figsize=(12, 5))
for i, m in enumerate(show):
    ax.bar(x + i * w - 0.4 + w / 2, pc.loc[m].values, w, label=m, color=colors[m])
ax.set_xticks(x)
ax.set_xticklabels([n + ("*" if n in RARE_NAMES else "") for n in NAMES], rotation=30, ha="right")
ax.set_ylabel("Recall on unseen test wells")
ax.set_title("Per-class recall, final round (* = rare class)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(fig_dir / "per_class_recall.png", dpi=200)
plt.close(fig)

print("Saved: results/summary_final.csv, wilcoxon_tests.csv, per_class_recall.csv")
print("       results/figures/learning_curves.png, rho_ablation.png, per_class_recall.png")
print(f"Note: '{best}' was picked using the same test wells, so its numbers are slightly optimistic.")