"""Model, blending and evaluation helpers for iteration 5."""
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import binomtest

ET_KWARGS = dict(n_estimators=600, max_features="sqrt", min_samples_leaf=2, n_jobs=-1)


def align_proba(model, X, classes):
    index = {c: i for i, c in enumerate(classes)}
    raw = model.predict_proba(X)
    out = np.zeros((len(raw), len(classes)), np.float32)
    for j, label in enumerate(model.classes_):
        if str(label) in index:
            out[:, index[str(label)]] = raw[:, j]
    return out


def prior_vector(y, classes):
    freq = pd.Series(y).value_counts(normalize=True)
    return np.array([freq.get(c, 1e-12) for c in classes], float)


def correct_prior(probs, prior, alpha):
    adjusted = probs / (prior[None, :] ** alpha)
    total = adjusted.sum(1, keepdims=True)
    total[total == 0] = 1.0
    return adjusted / total


def fit_extra_trees(X, y, classes, X_eval, seeds=(0, 1, 2)):
    """Average several seeds - a single ET seed carries ~0.3 pt of noise."""
    stacked = np.zeros((len(X_eval), len(classes)), np.float32)
    for seed in seeds:
        model = ExtraTreesClassifier(random_state=seed, **ET_KWARGS).fit(X, y)
        stacked += align_proba(model, X_eval, classes)
    return stacked / len(seeds)


# ----------------------------------------------------------------------
# B4: glia specialist trained on challenge glia + external glia
# ----------------------------------------------------------------------
def fit_glia_specialist(expr_train, y_train, glia_mask, ref_expr, ref_labels,
                        glia_classes, expr_eval, external_weight=0.3):
    """21-class model over the non-neuronal classes, ~10x more data than iteration 3.

    External cells carry no hackathon metadata, so this uses expression only.
    They are downweighted because they come from different animals and batches.
    """
    keep_ref = np.isin(ref_labels, list(glia_classes))
    X = np.vstack([expr_train[glia_mask], ref_expr[keep_ref]])
    y = np.concatenate([y_train[glia_mask], ref_labels[keep_ref]])
    weights = np.concatenate([
        np.ones(glia_mask.sum()),
        np.full(keep_ref.sum(), external_weight),
    ])

    model = ExtraTreesClassifier(random_state=0, **ET_KWARGS)
    model.fit(X, y, sample_weight=weights)
    return align_proba(model, expr_eval, sorted(glia_classes)), sorted(glia_classes)


def blend_specialist(global_probs, specialist_probs, specialist_classes,
                     classes, rows, weight):
    """Mix specialist probabilities into `rows` of the global matrix."""
    out = global_probs.copy()
    index = {c: i for i, c in enumerate(classes)}
    expanded = np.zeros((len(rows), len(classes)), np.float32)
    for j, label in enumerate(specialist_classes):
        expanded[:, index[label]] = specialist_probs[:, j]

    mixed = (1 - weight) * out[rows] + weight * expanded
    total = mixed.sum(1, keepdims=True)
    total[total == 0] = 1.0
    out[rows] = mixed / total
    return out


# ----------------------------------------------------------------------
# B5: pairwise arbitration on the dominant confusion pairs
# ----------------------------------------------------------------------
CONFUSION_PAIRS = [
    ("oligodendrocyte_1", "oligodendrocyte_progenitor_2"),
    ("oligodendrocyte_progenitor_2", "oligodendrocyte_2"),
    ("astrocyte_1", "astrocyte_2"),
    ("endothelial", "astrocyte_1"),
    ("meninges_1", "meninges_2"),
    ("gamma_motoneuron", "alpha_motoneuron"),
    ("oligodendrocyte_precursor_cell", "oligodendrocyte_progenitor_1"),
    ("endothelial", "pericyte"),
]


def fit_pair_models(X, y, pairs=CONFUSION_PAIRS, min_cells=30):
    models = {}
    for a, b in pairs:
        mask = (y == a) | (y == b)
        if mask.sum() < min_cells or y[mask].nunique() < 2:
            continue
        models[(a, b)] = ExtraTreesClassifier(
            n_estimators=400, max_features="sqrt",
            min_samples_leaf=1, n_jobs=-1, random_state=0,
        ).fit(X[mask], y[mask])
    return models


def arbitrate(probs, X_eval, models, classes, margin=0.25):
    """Re-decide cells whose top-2 classes form a known confusion pair.

    Only fires when the global model is genuinely torn (top-2 relative gap below
    `margin`), so confident predictions are left alone.
    """
    out = probs.copy()
    order = np.argsort(-probs, axis=1)
    top1, top2 = order[:, 0], order[:, 1]
    gap = probs[np.arange(len(probs)), top1] - probs[np.arange(len(probs)), top2]

    keyed = {frozenset(k): v for k, v in models.items()}
    for pair, model in keyed.items():
        a, b = tuple(pair)
        ia, ib = classes.index(a), classes.index(b)
        rows = np.flatnonzero(
            (((top1 == ia) & (top2 == ib)) | ((top1 == ib) & (top2 == ia)))
            & (gap < margin)
        )
        if len(rows) == 0:
            continue

        decision = model.predict_proba(X_eval[rows])
        for j, label in enumerate(model.classes_):
            column = classes.index(str(label))
            # Keep the global mass but re-weight the two contenders.
            pair_mass = out[rows, ia] + out[rows, ib]
            out[rows, column] = pair_mass * decision[:, j]
    return out


# ----------------------------------------------------------------------
# B6: count-aware member
# ----------------------------------------------------------------------
def fit_multinomial_nb(counts_train, y, classes, counts_eval, alpha=0.5):
    model = MultinomialNB(alpha=alpha).fit(counts_train, y)
    return align_proba(model, counts_eval, classes)


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def paired_mcnemar(correct_a, correct_b):
    """Paired test on identical folds - the only honest way to compare two strategies."""
    table = [
        [int((correct_a & correct_b).sum()), int((correct_a & ~correct_b).sum())],
        [int((~correct_a & correct_b).sum()), int((~correct_a & ~correct_b).sum())],
    ]
    b, c = table[0][1], table[1][0]
    if b + c == 0:
        return 1.0, table
    # Exact binomial McNemar: robust at the small discordant counts we see here.
    return float(binomtest(b, b + c, 0.5).pvalue), table
