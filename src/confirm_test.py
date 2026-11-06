"""Confirmatory test for rho=0.5: 50 seeds, and fresh seeds 20-49 only."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

root = Path(__file__).resolve().parents[1]
df = pd.read_csv(root / "results" / "al_results_rho0.5_50seeds.csv")
LAST = df["round"].max()

final = df[df["round"] == LAST]
P = {k: final.pivot(index="seed", columns="strategy", values=k)
     for k in ["rare_macro_F1", "rare_macro_recall", "dominant_macro_F1", "accuracy"]}
P["rare_F1_AULC"] = df.groupby(["seed", "strategy"])["rare_macro_F1"].mean().unstack()

PRIMARY = "rare_macro_F1"   # pre-specified; 2 comparisons -> Holm over 2
rows = []
for label, seeds in [("all 50 seeds", P[PRIMARY].index),
                     ("fresh seeds 20-49", P[PRIMARY].index[P[PRIMARY].index >= 20])]:
    for k, p_ in P.items():
        for b in ["random", "uncertainty"]:
            a, c = p_.loc[seeds, "rare_aware"].values, p_.loc[seeds, b].values
            d = a - c
            p = wilcoxon(a, c).pvalue
            rows.append({"subset": label, "metric": k, "vs": b, "n": len(d),
                         "mean_diff": d.mean(), "ci95_lo": d.mean() - 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                         "ci95_hi": d.mean() + 1.96 * d.std(ddof=1) / np.sqrt(len(d)),
                         "seeds_better": f"{(d > 0).sum()}/{len(d)}", "p": p})
out = pd.DataFrame(rows)
# Holm over the two primary tests within each subset
for label in out.subset.unique():
    m = (out.subset == label) & (out.metric == PRIMARY)
    p = out.loc[m, "p"].values
    out.loc[m, "p_holm"] = [min(1, x) for x in np.maximum.accumulate(
        np.sort(p) * np.array([2, 1]))[np.argsort(np.argsort(p))]]
out.to_csv(root / "results" / "confirmatory_tests.csv", index=False)
print(out.round(4).to_string(index=False))
print("\nPrimary = rare_macro_F1 (p_holm applies to it only). Other metrics are secondary.")