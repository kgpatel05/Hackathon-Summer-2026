"""Iteration 8e - mouse-centred parent-atlas transfer.

WHY THIS IS DIFFERENT FROM THE EXISTING HARMONISATION TEST
-----------------------------------------------------------
The submitted model already knows ``Mouse_ID`` and iteration 7 showed that adding
mouse-centred expression to its challenge feature matrix is redundant.  The parent-atlas
transfer model is different: it is a single expression-only logistic fit across ten mice.
It has no mouse feature, so technical offsets are folded into its class coefficients.

This experiment centres every released gene within mouse *before* fitting that atlas
logistic.  Atlas statistics use only the 136,621 non-challenge cells.  Challenge train and
test matrices are centred independently within mouse, using counts and Mouse_ID only.
No challenge label, withheld gene, or recovered test label enters the feature block.

SCREEN BEFORE MODEL CV
----------------------
The transfer model is out-of-sample for all challenge cells, so its accuracy can be
measured directly on the labelled challenge training rows.  In the initial diagnostic,
mouse centring raised atlas-transfer accuracy from 0.6040 to 0.6114.  Per-mouse models
were rejected before this experiment (0.5578): splitting the atlas sacrifices too much
data.  The candidate therefore shares coefficients across mice and changes only the
normalisation.

HONEST DECISION RULE
--------------------
There is one candidate and it has the same 60 columns as the block it replaces.  Compare
it with the frozen submitted feature stack on identical 5-fold out-of-fold predictions,
including the adopted metadata mask.  A second, independently shuffled fold partition is
the confirmation.  Adopt only if the candidate gains accuracy and exact McNemar p < 0.05
on BOTH partitions.  Each partition is tested separately; repeated predictions of the
same cell are deliberately not flattened into a pseudo-replicated McNemar test.

This script never reads the recovered test labels and never overwrites the submission.
It writes a candidate prediction only if the pre-registered rule passes.
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
import iteration5_features as F
import iteration5_models as M

OUT = Path("outputs/iteration8")
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = tuple(range(10))
FOLD_SEEDS = (7, 23)
MASK_COLS = ("Region", "Excitatory_vs_Inhibitory", "Segment")

counts_train, meta_train, counts_test, meta_test = F.load_challenge()
y = meta_train[F.TARGET].astype(str).to_numpy()
CLASSES = sorted(set(y))
CLASS_ARR = np.array(CLASSES)
GENES = list(counts_train.columns)
glia = meta_train["Region"].isna().to_numpy()


def decode_categorical(handle, key):
    categories = [x.decode() for x in handle[f"obs/{key}/categories"][:]]
    codes = handle[f"obs/{key}/codes"][:]
    return np.array([categories[c] if c >= 0 else "NA" for c in codes])


def centre_within_group(matrix, groups):
    """Centre each gene within group without looking at a response label."""
    out = np.empty_like(matrix, dtype=np.float32)
    groups = np.asarray(groups, dtype=str)
    for group in np.unique(groups):
        rows = groups == group
        out[rows] = matrix[rows] - matrix[rows].mean(0)
    return out


def mouse_centred_atlas_transfer():
    """Return aligned train/test probabilities and the external reference size."""
    with h5py.File(F.PARENT_ATLAS, "r") as handle:
        ids = np.array([x.decode() for x in handle["obs/_index"][:]])
        atlas_genes = [x.decode() for x in handle["var/_index"][:]]
        lookup = {gene: i for i, gene in enumerate(atlas_genes)}
        missing = [gene for gene in GENES if gene not in lookup]
        if missing:
            raise ValueError(f"{len(missing)} released genes absent from parent atlas")

        matrix = sparse.csr_matrix(
            (handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]),
            shape=(len(ids), len(atlas_genes)),
        )[:, [lookup[gene] for gene in GENES]]
        labels = np.array([
            F._normalise_label(label)
            for label in decode_categorical(handle, "MERFISH cell type annotation")
        ])
        mice = decode_categorical(handle, "Mouse ID")

    position = {cell_id: i for i, cell_id in enumerate(ids)}
    challenge_ids = pd.Index(meta_train.index).append(pd.Index(meta_test.index)).astype(str)
    missing_ids = [cell_id for cell_id in challenge_ids if cell_id not in position]
    if missing_ids:
        raise ValueError(f"{len(missing_ids)} challenge cells absent from parent atlas")
    is_challenge = np.zeros(len(ids), bool)
    is_challenge[[position[cell_id] for cell_id in challenge_ids]] = True

    keep = (~is_challenge) & np.isin(labels, CLASSES)
    keep &= np.asarray(matrix.sum(1)).ravel() > 0
    reference = np.asarray(matrix[keep].todense(), dtype=np.float32)
    reference = centre_within_group(F.log_cpm(reference), mice[keep])
    reference_labels = labels[keep]

    model = LogisticRegression(C=0.1, max_iter=2000, n_jobs=1)
    model.fit(reference, reference_labels)

    class_index = {label: i for i, label in enumerate(CLASSES)}
    outputs = []
    for counts, meta in ((counts_train, meta_train), (counts_test, meta_test)):
        query = centre_within_group(
            F.log_cpm(counts.to_numpy()), meta["Mouse_ID"].astype(str).to_numpy()
        )
        raw = model.predict_proba(query)
        aligned = np.zeros((len(query), len(CLASSES)), dtype=np.float32)
        for j, label in enumerate(model.classes_):
            aligned[:, class_index[str(label)]] = raw[:, j]
        outputs.append(aligned)
    return outputs, len(reference)


def compatibility_mask(train_rows, meta_eval):
    """Class compatibility learned only from the fold's labelled training rows."""
    allow = np.ones((len(meta_eval), len(CLASSES)), bool)
    for column in MASK_COLS:
        train_values = meta_train[column].iloc[train_rows].astype(str).to_numpy()
        known = set(train_values)
        seen = [set(train_values[y[train_rows] == label]) for label in CLASSES]
        for i, value in enumerate(meta_eval[column].astype(str).to_numpy()):
            if value in known:
                allow[i] &= np.array([value in class_values for class_values in seen])
    allow[~allow.any(1)] = True
    return allow


def out_of_fold_predictions(matrix, fold_seed):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_seed)
    pred = np.empty(len(y), dtype=object)
    for fold, (train_rows, valid_rows) in enumerate(folds.split(y, y), start=1):
        probabilities = M.fit_extra_trees(
            matrix[train_rows], pd.Series(y[train_rows]), CLASSES,
            matrix[valid_rows], seeds=SEEDS,
        )
        probabilities = M.correct_prior(
            probabilities, M.prior_vector(pd.Series(y[train_rows]), CLASSES), 0.45
        )
        allow = compatibility_mask(train_rows, meta_train.iloc[valid_rows])
        pred[valid_rows] = CLASS_ARR[np.where(allow, probabilities, -1.0).argmax(1)]
        print(f"    partition={fold_seed} fold={fold}/5", flush=True)
    return pred


print("building mouse-centred atlas transfer", flush=True)
t0 = time.time()
(harm_train, harm_test), n_reference = mouse_centred_atlas_transfer()
cache = np.load(
    "outputs/merfish_hackathon_iteration5_full_model/feature_cache.npz",
    allow_pickle=True,
)
global_train = cache["ATL_TR"]
global_test = cache["ATL_TE"]
print(f"  reference rows={n_reference:,} build_seconds={time.time() - t0:.0f}", flush=True)

transfer_rows = []
for name, probabilities in (
    ("global atlas transfer", global_train),
    ("mouse-centred atlas transfer", harm_train),
):
    prediction = CLASS_ARR[probabilities.argmax(1)]
    row = {
        "stage": "transfer_only",
        "partition_seed": np.nan,
        "variant": name,
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "glia": accuracy_score(y[glia], prediction[glia]),
        "gain": np.nan,
        "mcnemar_p": np.nan,
        "passes": np.nan,
    }
    transfer_rows.append(row)
    print(
        f"  {name:31s} acc={row['accuracy']:.4f} "
        f"balanced={row['balanced_accuracy']:.4f} glia={row['glia']:.4f}",
        flush=True,
    )

common_train = [cache["BASE_TR"], cache["EXT_TR"], cache["SPA_TR"], cache["NIC_TR"]]
common_test = [cache["BASE_TE"], cache["EXT_TE"], cache["SPA_TE"], cache["NIC_TE"]]
baseline_train = np.hstack(common_train + [global_train]).astype(np.float32)
candidate_train = np.hstack(common_train + [harm_train]).astype(np.float32)
candidate_test = np.hstack(common_test + [harm_test]).astype(np.float32)
assert baseline_train.shape == candidate_train.shape == (len(y), 529)

cv_rows = []
all_pass = True
for fold_seed in FOLD_SEEDS:
    print(f"\n=== independent 5-fold partition seed={fold_seed} ===", flush=True)
    start = time.time()
    baseline_pred = out_of_fold_predictions(baseline_train, fold_seed)
    candidate_pred = out_of_fold_predictions(candidate_train, fold_seed)
    baseline_ok = baseline_pred == y
    candidate_ok = candidate_pred == y
    gain = candidate_ok.mean() - baseline_ok.mean()
    p_value, table = M.paired_mcnemar(candidate_ok, baseline_ok)
    passes = bool(gain > 0 and p_value < 0.05)
    all_pass &= passes
    print(
        f"  baseline={baseline_ok.mean():.4f} candidate={candidate_ok.mean():.4f} "
        f"gain={gain:+.4f} p={p_value:.4g} glia="
        f"{candidate_ok[glia].mean():.4f} table={table} ({time.time() - start:.0f}s)",
        flush=True,
    )
    cv_rows.append({
        "stage": "full_model_cv",
        "partition_seed": fold_seed,
        "variant": "mouse-centred replaces global atlas",
        "accuracy": candidate_ok.mean(),
        "balanced_accuracy": balanced_accuracy_score(y, candidate_pred),
        "glia": candidate_ok[glia].mean(),
        "gain": gain,
        "mcnemar_p": p_value,
        "passes": passes,
    })

print(
    f"\nVERDICT: {'ADOPT' if all_pass else 'DO NOT ADOPT'} "
    "(requires positive gain and p<0.05 on both independent partitions)",
    flush=True,
)

if all_pass:
    print("fitting the passed candidate on all labelled training cells", flush=True)
    probabilities = M.fit_extra_trees(
        candidate_train, pd.Series(y), CLASSES, candidate_test, seeds=tuple(range(20))
    )
    probabilities = M.correct_prior(
        probabilities, M.prior_vector(pd.Series(y), CLASSES), 0.45
    )
    allow = compatibility_mask(np.arange(len(y)), meta_test)
    prediction = CLASS_ARR[np.where(allow, probabilities, -1.0).argmax(1)]
    submission = pd.DataFrame({
        "Cell_ID": meta_test.index.astype(str),
        "MERFISH_cell_type_annotation.y": prediction,
    })
    assert len(submission) == 5000 and not submission.Cell_ID.duplicated().any()
    path = OUT / "prediction_atlas_harmonised.csv"
    path.write_text(submission.to_csv(index=False).rstrip("\n"))
    print(f"wrote {path}; submission NOT overwritten", flush=True)

pd.DataFrame(transfer_rows + cv_rows).to_csv(OUT / "atlas_harmonise.csv", index=False)
print(f"wrote {OUT / 'atlas_harmonise.csv'}", flush=True)
