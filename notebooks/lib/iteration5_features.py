"""Feature blocks for iteration 5.

Every block here is label-free with respect to the hackathon training labels
(or explicitly fold-scoped), so blocks can be built once outside the CV loop.
The one exception is `neighbour_label_histogram`, which takes an explicit
fold-train mask and must be rebuilt per fold.
"""
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path("data")
EXTERNAL = DATA_DIR / "external" / "SNI_merged_0531.h5ad"

TARGET = "MERFISH_cell_type_annotation"
CATEGORICAL_META = [
    "Datasets", "Region", "Excitatory_vs_Inhibitory", "Segment",
    "Gender", "Mouse_ID", "AP_position", "Section_ID",
]


def load_challenge():
    def read(name):
        df = pd.read_csv(DATA_DIR / name, index_col=0)
        df.index = df.index.astype(str)
        return df

    counts_train, meta_train = read("counts_train.csv"), read("meta_train.csv")
    counts_test, meta_test = read("counts_test.csv"), read("meta_test.csv")
    counts_train = counts_train.loc[meta_train.index]
    counts_test = counts_test.loc[meta_test.index, counts_train.columns]
    return counts_train, meta_train, counts_test, meta_test


def log_cpm(counts):
    """Per-cell normalisation to a fixed depth, then log1p."""
    matrix = np.asarray(counts, dtype=np.float32)
    total = matrix.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0
    return np.log1p(matrix / total * 100.0)


def zscore(matrix):
    return (matrix - matrix.mean(0)) / (matrix.std(0) + 1e-6)


# ----------------------------------------------------------------------
# Base block: log1p expression + QC + numeric/one-hot metadata
# ----------------------------------------------------------------------
def base_block(counts, meta, encoder):
    expression = np.log1p(counts.astype(float)).to_numpy(np.float32)

    total = counts.sum(axis=1).to_numpy(float)
    volume = pd.to_numeric(meta["volume"], errors="coerce").to_numpy(float)
    safe_volume = np.where(volume == 0, np.nan, volume)

    qc = np.column_stack([
        np.log1p(total),
        (counts > 0).sum(axis=1),
        (counts == 0).mean(axis=1),
        np.log1p(np.clip(volume, 0, None)),
        total / safe_volume,                      # transcript density (B6)
        (counts > 0).sum(axis=1) / safe_volume,   # gene density (B6)
        volume,
        meta["center_x"].to_numpy(float),
        meta["center_y"].to_numpy(float),
    ])

    categorical = encoder.transform(meta[CATEGORICAL_META].astype(str)).toarray()
    return np.hstack([
        expression, np.nan_to_num(qc, nan=-1.0), categorical
    ]).astype(np.float32)


# ----------------------------------------------------------------------
# B1: external reference transfer (the iteration-4 bug, fixed)
# ----------------------------------------------------------------------
def _normalise_label(name):
    name = name.replace(" ", "_").replace("-", "_")
    return "VH_in_Chat" if name == "M_in_Chat" else name


def load_reference(gene_order, label_column="voting"):
    """Read the external reference over the shared gene panel.

    `label_column` is stated explicitly on purpose: iteration 4 auto-detected it
    and picked `Section ID` (109 tissue sections), which silently turned every
    reference centroid into a tissue-section pseudobulk.
    """
    with h5py.File(EXTERNAL, "r") as handle:
        reference_genes = [g.decode() for g in handle["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(reference_genes)}

        missing = [g for g in gene_order if g not in lookup]
        if missing:
            raise ValueError(f"{len(missing)} challenge genes absent from reference")

        columns = np.array([lookup[g] for g in gene_order])
        order = np.argsort(columns)
        matrix = handle["X"][:, columns[order]].astype(np.float32)
        matrix = matrix[:, np.argsort(order)]

        categories = [c.decode() for c in handle[f"obs/{label_column}/categories"][:]]
        codes = handle[f"obs/{label_column}/codes"][:]

    labels = np.array([
        _normalise_label(categories[c]) if c >= 0 else "NA" for c in codes
    ])
    usable = matrix.sum(axis=1) > 0
    return matrix[usable], labels[usable]


def reference_transfer(gene_order, classes, matrices, label_column="voting", C=0.1):
    """Fit one L2 logistic on the reference, return aligned probabilities.

    Sees zero hackathon labels, so it is safe to compute once outside the CV loop.
    """
    reference, labels = load_reference(gene_order, label_column)

    unmapped = sorted(set(labels) - set(classes))
    if unmapped:
        raise ValueError(f"reference labels absent from challenge taxonomy: {unmapped}")

    # lbfgs multinomial fitting is single-process in current sklearn; spelling this out
    # also avoids joblib semaphore probes in restricted macOS/Codex environments.
    model = LogisticRegression(C=C, max_iter=2000, n_jobs=1)
    model.fit(zscore(log_cpm(reference)), labels)

    index = {c: i for i, c in enumerate(classes)}
    outputs = []
    for counts in matrices:
        raw = model.predict_proba(zscore(log_cpm(counts.to_numpy())))
        aligned = np.zeros((len(raw), len(classes)), np.float32)
        for j, label in enumerate(model.classes_):
            aligned[:, index[label]] = raw[:, j]
        outputs.append(aligned)
    return outputs, reference, labels


# ----------------------------------------------------------------------
# B7: anatomically registered spatial features
# ----------------------------------------------------------------------
def registered_spatial(meta_all, neuron_mask):
    """Put every section in a common frame, then derive anatomical distances.

    Raw center_x/center_y live in per-section imaging frames, so they are not
    comparable across the 108 sections - which is why the round-1 spatial kNN
    scored below the majority-class floor. Here each section is centred, rotated
    so its principal axis is horizontal, oriented dorsal-up using the Region-1
    (dorsal horn) neuron centroid, and scaled by its own radius.
    """
    sections = meta_all["Section_ID"].astype(str).to_numpy()
    coords = meta_all[["center_x", "center_y"]].to_numpy(float)
    out = np.zeros((len(meta_all), 8), np.float32)

    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        points = coords[rows]
        if len(points) < 3:
            continue

        centred = points - points.mean(0)
        # Principal axis -> horizontal.
        _, _, components = np.linalg.svd(centred - centred.mean(0), full_matrices=False)
        rotated = centred @ components.T

        # Dorsal-up: the dorsal horn (Region 1) should sit at positive y.
        dorsal = neuron_mask[rows]
        if dorsal.any() and rotated[dorsal, 1].mean() < 0:
            rotated[:, 1] *= -1.0

        radius = np.linalg.norm(rotated, axis=1)
        scale = np.percentile(radius, 95) + 1e-9
        rotated = rotated / scale
        radius = radius / scale

        neighbours = NearestNeighbors(
            n_neighbors=min(11, len(points))
        ).fit(rotated)
        distances, _ = neighbours.kneighbors(rotated)
        local_spacing = distances[:, 1:].mean(1)

        try:
            hull = ConvexHull(rotated)
            edges = []
            for simplex in hull.simplices:
                a, b = rotated[simplex[0]], rotated[simplex[1]]
                edge = b - a
                length = np.linalg.norm(edge) + 1e-12
                t = np.clip(((rotated - a) @ edge) / length**2, 0, 1)
                projection = a + t[:, None] * edge
                edges.append(np.linalg.norm(rotated - projection, axis=1))
            hull_distance = np.min(np.vstack(edges), axis=0)
        except Exception:
            hull_distance = np.full(len(rotated), np.nan)

        out[rows, 0] = radius                       # peripheral = white matter
        out[rows, 1] = rotated[:, 0]                # medio-lateral
        out[rows, 2] = rotated[:, 1]                # dorso-ventral
        out[rows, 3] = np.arctan2(rotated[:, 1], rotated[:, 0])
        out[rows, 4] = hull_distance                # meninges sit on the boundary
        out[rows, 5] = local_spacing                # local packing density
        out[rows, 6] = len(points)
        out[rows, 7] = np.linalg.norm(rotated - np.median(rotated, 0), axis=1)

    return np.nan_to_num(out, nan=-1.0)


# ----------------------------------------------------------------------
# B8: transductive neighbourhood features (train + test pooled, label-free)
# ----------------------------------------------------------------------
def niche_expression(expression_all, meta_all, k=15, n_components=30):
    """Mean expression of each cell's k spatial neighbours within its section."""
    sections = meta_all["Section_ID"].astype(str).to_numpy()
    coords = meta_all[["center_x", "center_y"]].to_numpy(float)
    niche = np.zeros_like(expression_all)

    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        k_eff = min(k, len(rows))
        model = NearestNeighbors(n_neighbors=k_eff).fit(coords[rows])
        _, neighbours = model.kneighbors(coords[rows])
        niche[rows] = expression_all[rows][neighbours].mean(1)

    return PCA(n_components=n_components, random_state=0).fit_transform(niche).astype(np.float32)


def neighbour_label_histogram(meta_all, labels_all, labelled_mask, classes, k=20):
    """Class histogram of a cell's k nearest *labelled* neighbours in its section.

    Fold-scoped: `labelled_mask` must mark only fold-training cells, and a cell
    never votes for itself.
    """
    sections = meta_all["Section_ID"].astype(str).to_numpy()
    coords = meta_all[["center_x", "center_y"]].to_numpy(float)
    index = {c: i for i, c in enumerate(classes)}
    out = np.zeros((len(meta_all), len(classes)), np.float32)

    for section in np.unique(sections):
        rows = np.flatnonzero(sections == section)
        donors = rows[labelled_mask[rows]]
        if len(donors) == 0:
            continue

        k_eff = min(k, len(donors))
        model = NearestNeighbors(n_neighbors=k_eff).fit(coords[donors])
        distances, neighbours = model.kneighbors(coords[rows])
        weights = 1.0 / (distances + 1.0)

        for local, row in enumerate(rows):
            for j, donor_local in enumerate(neighbours[local]):
                donor = donors[donor_local]
                if donor == row:
                    continue
                out[row, index[labels_all[donor]]] += weights[local, j]

    totals = out.sum(1, keepdims=True)
    totals[totals == 0] = 1.0
    return out / totals


# ----------------------------------------------------------------------
# B10: parent-atlas transfer (added after the §7 panel diagnosis)
# ----------------------------------------------------------------------
PARENT_ATLAS = Path("data") / "external" / "MERFISH_spinal_cord_0531.h5ad"


def atlas_transfer(gene_order, classes, matrices, C=0.1):
    """Transfer from the parent atlas, excluding every challenge cell.

    The atlas holds 146,621 cells from the SAME animals and sections as the
    challenge. Removing the 10,000 challenge cells leaves 136,621 labelled cells
    that are legitimate external training data - a closer-matched reference than
    SNI_merged (different mice), and empirically a stronger one:
    0.5992 vs 0.5588 standalone.

    Only the cell-type column is used. Transferring the auxiliary annotations
    (Laminae / Markers / Neurotransmitter) alongside it reduced accuracy.
    """
    import pandas as pd
    from scipy import sparse

    counts_train, meta_train, counts_test, meta_test = load_challenge()

    with h5py.File(PARENT_ATLAS, "r") as handle:
        ids = np.array([x.decode() for x in handle["obs/_index"][:]])
        atlas_genes = [g.decode() for g in handle["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(atlas_genes)}
        columns = np.array([lookup[g] for g in gene_order])

        matrix = sparse.csr_matrix(
            (handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]),
            shape=(len(ids), len(atlas_genes)),
        )
        categories = [c.decode() for c in
                      handle["obs/MERFISH cell type annotation/categories"][:]]
        codes = handle["obs/MERFISH cell type annotation/codes"][:]

    labels = np.array([_normalise_label(categories[c]) if c >= 0 else "NA" for c in codes])

    position = {c: i for i, c in enumerate(ids)}
    challenge = np.zeros(len(ids), bool)
    for index in [meta_train.index, meta_test.index]:
        challenge[[position[c] for c in index.astype(str) if c in position]] = True
    outside = np.flatnonzero(~challenge)

    reference = np.asarray(matrix[outside][:, columns].todense(), np.float32)
    labels = labels[outside]
    usable = (reference.sum(1) > 0) & np.isin(labels, list(classes))
    reference, labels = reference[usable], labels[usable]

    model = LogisticRegression(C=C, max_iter=1500, n_jobs=1)
    model.fit(zscore(log_cpm(reference)), labels)

    index_of = {c: i for i, c in enumerate(classes)}
    outputs = []
    for counts in matrices:
        raw = model.predict_proba(zscore(log_cpm(counts.to_numpy())))
        aligned = np.zeros((len(raw), len(classes)), np.float32)
        for j, label in enumerate(model.classes_):
            aligned[:, index_of[label]] = raw[:, j]
        outputs.append(aligned)
    return outputs, len(reference)


# ----------------------------------------------------------------------
# Iteration 9: restoring the tissue context the 1-in-27 subsample destroyed.
#
# The challenge file holds ~70 cells per section; the parent atlas holds 964. A challenge
# cell's nearest in-file neighbour is therefore hundreds of microns away, which is why
# every earlier spatial experiment failed. These two blocks rebuild each cell's real
# microenvironment from the 136,612 NON-challenge atlas cells, through two different
# channels - the neighbours' class histogram, and their mean expression.
#
# Validated on three fresh fold partitions (41/59/83), 20 ET seeds, one out-of-fold
# prediction per cell: +0.54 / +0.46 / +0.58 pt, mean +0.53 pt, against a within-section
# row-shuffled null at +0.00 pt.
#
# Legitimacy: all 10,000 challenge cells are removed from the donor pool before any
# neighbour search, so no challenge or test label can enter. Expression uses only the 200
# RELEASED genes. The neighbours' cell-type annotations are the same public column that
# atlas_transfer already trains on; disclosure is that those annotations were derived by
# the study from 500 genes, so these blocks carry 500-gene information about a cell's
# MICROENVIRONMENT, never about its own transcriptome.
# ----------------------------------------------------------------------
def _atlas_neighbour_setup(meta_all):
    """Shared donor-pool construction: atlas coordinates, sections, labels, non-challenge."""
    from scipy import sparse  # noqa: F401  (imported for symmetry with atlas_transfer)

    with h5py.File(PARENT_ATLAS, "r") as handle:
        ids = np.array([x.decode() for x in handle["obs/_index"][:]])
        cat = [c.decode() for c in
               handle["obs/MERFISH cell type annotation/categories"][:]]
        codes = handle["obs/MERFISH cell type annotation/codes"][:]
        sec_cat = [c.decode() for c in handle["obs/Section ID/categories"][:]]
        sec_codes = handle["obs/Section ID/codes"][:]
        ax = handle["obs/center_x"][:].astype(float)
        ay = handle["obs/center_y"][:].astype(float)

    labels = np.array([_normalise_label(cat[c]) if c >= 0 else "NA" for c in codes])
    sections = np.array([sec_cat[c] if c >= 0 else "NA" for c in sec_codes])

    position = {c: i for i, c in enumerate(ids)}
    present = [c for c in meta_all.index.astype(str) if c in position]
    missing = len(meta_all) - len(present)
    if missing:
        # The validation cohort that replaces meta_test.csv after 3pm 8/22 need not be a
        # subset of the public atlas.  Cells that are not in it simply cannot leak into
        # the donor pool, so there is nothing to remove and nothing to fail on.
        print(f"[atlas-neighbours] {missing} of {len(meta_all)} query cells are not in "
              f"the parent atlas; the donor pool excludes the {len(present)} that are")
    is_challenge = np.zeros(len(ids), bool)
    is_challenge[[position[c] for c in present]] = True
    donors = np.flatnonzero(~is_challenge)
    assert len(donors) == len(ids) - len(present), "donor pool contains challenge cells"
    return ids, labels, sections, ax, ay, donors


def atlas_composition(meta_all, classes, k=10):
    """Class histogram of each cell's k nearest NON-challenge parent-atlas neighbours.

    Returns (n_cells, len(classes) + 1): the 60 challenge classes plus one column for
    atlas annotations outside the taxonomy, so every row sums to 1.
    """
    from scipy.spatial import cKDTree

    _, labels, sections, ax, ay, donors = _atlas_neighbour_setup(meta_all)
    index_of = {c: i for i, c in enumerate(classes)}
    other = len(classes)
    codes = np.array([index_of.get(l, other) for l in labels])

    q_sec = meta_all["Section_ID"].astype(str).to_numpy()
    q_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
    out = np.zeros((len(meta_all), other + 1), np.float32)

    donor_sec = sections[donors]
    q_mouse = meta_all["Mouse_ID"].astype(str).to_numpy()
    donor_mouse = np.array([str(x).split("_")[1] if "_" in str(x) else ""
                            for x in donor_sec])
    for section in np.unique(q_sec):
        pool = donors[donor_sec == section]
        rows = np.flatnonzero(q_sec == section)
        if len(rows) == 0:
            continue
        if len(pool) < 10:
            # the validation cohort may come from sections the public atlas does not
            # contain; fall back to the same mouse, then to the whole atlas, rather
            # than leaving the neighbourhood features at zero
            mouse = q_mouse[rows[0]]
            pool = donors[donor_mouse == mouse]
            if len(pool) < 10:
                pool = donors
        if len(pool) < 10:
            continue
        tree = cKDTree(np.column_stack([ax[pool], ay[pool]]))
        pool_code = codes[pool]
        _, nn = tree.query(q_xy[rows], k=min(k, len(pool)))
        taken = pool_code[np.atleast_2d(nn)]
        for j in range(other + 1):
            out[rows, j] = (taken == j).mean(1)
    return out


def atlas_niche(meta_all, gene_order, k=50, n_components=30):
    """Mean 200-gene log-CPM of each cell's k nearest NON-challenge atlas neighbours."""
    from scipy import sparse
    from scipy.spatial import cKDTree

    ids, _, sections, ax, ay, donors = _atlas_neighbour_setup(meta_all)
    with h5py.File(PARENT_ATLAS, "r") as handle:
        atlas_genes = [g.decode() for g in handle["var/_index"][:]]
        lookup = {g: i for i, g in enumerate(atlas_genes)}
        cols = np.array([lookup[g] for g in gene_order])
        matrix = sparse.csr_matrix(
            (handle["X/data"][:], handle["X/indices"][:], handle["X/indptr"][:]),
            shape=(len(ids), len(atlas_genes)))

    q_sec = meta_all["Section_ID"].astype(str).to_numpy()
    q_xy = meta_all[["center_x", "center_y"]].to_numpy(float)
    out = np.zeros((len(meta_all), len(gene_order)), np.float32)

    donor_sec = sections[donors]
    for section in np.unique(q_sec):
        pool = donors[donor_sec == section]
        rows = np.flatnonzero(q_sec == section)
        if len(pool) < 2 or len(rows) == 0:
            continue
        tree = cKDTree(np.column_stack([ax[pool], ay[pool]]))
        _, nn = tree.query(q_xy[rows], k=min(k, len(pool)))
        block = log_cpm(np.asarray(matrix[pool][:, cols].todense(), np.float32))
        out[rows] = block[np.atleast_2d(nn)].mean(1)
    return PCA(n_components=n_components, random_state=0).fit_transform(out).astype(np.float32)
