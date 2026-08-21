"""Where does the incumbent's predicted class marginal go wrong, on OOF and on test?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import iteration18_base as B
from evaluate import load_truth

data = B.load_all()
c = np.load(B.OUT / "incumbent_probs.npz", allow_pickle=True)
classes, y = data["classes"], data["y"]
oof = B.prior_correct(c["oof_raw"], y, classes)
test = B.prior_correct(c["test_raw"], y, classes)
oof_pred = B.decode(oof, c["oof_allow"], classes)
test_pred = B.decode(test, c["test_allow"], classes)
truth = load_truth().reindex(data["meta_test"].index.astype(str)).to_numpy()

print(f"OOF accuracy  {np.mean(oof_pred == y):.4f}")
print(f"test accuracy {np.mean(test_pred == truth):.4f}")

tr_n = pd.Series(y).value_counts().reindex(classes).fillna(0).to_numpy()
te_n = pd.Series(truth).value_counts().reindex(classes).fillna(0).to_numpy()
op_n = pd.Series(oof_pred).value_counts().reindex(classes).fillna(0).to_numpy()
tp_n = pd.Series(test_pred).value_counts().reindex(classes).fillna(0).to_numpy()
soft = test.sum(0)

tab = pd.DataFrame({"train_n": tr_n, "test_true": te_n, "test_pred": tp_n,
                    "oof_pred": op_n, "test_soft": soft}, index=classes)
tab["pred-train"] = tab.test_pred - tab.train_n
tab["pred-true"] = tab.test_pred - tab.test_true
print("\ntotal |pred - train_prior| on test:", int(np.abs(tp_n - tr_n).sum()))
print("total |pred - test_truth| on test:", int(np.abs(tp_n - te_n).sum()))
print("total |train - test_truth|      :", int(np.abs(tr_n - te_n).sum()))
print("total |soft - train_prior|      :", float(np.abs(soft - tr_n).sum()))
print("\nworst marginal offenders (test):")
print(tab.reindex(tab["pred-train"].abs().sort_values(ascending=False).index)
      .head(16).to_string())

# oracle: how much accuracy is available from fixing only the marginal?
print("\n--- per-class recall/precision on test, top errors ---")
df = pd.DataFrame({"true": truth, "pred": test_pred})
wrong = df[df.true != df.pred]
pairs = wrong.groupby(["true", "pred"]).size().sort_values(ascending=False)
print(f"{len(wrong)} errors")
for (a, b), n in pairs.head(20).items():
    print(f"  {n:4d}  {a:34s} -> {b}")

glia = data["meta_test"]["Region"].isna().to_numpy()
print(f"\nglia n={glia.sum()} acc={np.mean(test_pred[glia]==truth[glia]):.4f}")
print(f"neuron n={(~glia).sum()} acc={np.mean(test_pred[~glia]==truth[~glia]):.4f}")

# how much of the error is 'second choice was right'?
order = np.argsort(-np.where(c["test_allow"], test, -1.0), axis=1)
rank = np.array([np.where(classes[order[i]] == truth[i])[0][0] for i in range(len(truth))])
for k in [1, 2, 3, 5, 10]:
    print(f"  top-{k:<2d} accuracy {np.mean(rank < k):.4f}")
