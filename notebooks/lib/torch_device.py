"""Shared device selection: prefer Apple MPS over CPU.

Measured on this machine (arm64 Darwin, torch 2.8.0), 200 iterations of a
(10000x529) @ (529x256) matmul + relu:

    cpu   0.97 s
    mps   0.43 s      2.3x

KNOWN MPS GAP: sparse COO tensors are unimplemented -
`aten::_sparse_coo_tensor_with_dims_and_tensors` raises NotImplementedError on the
SparseMPS backend. Graph aggregation must therefore be written as a gather + mean over a
padded neighbour index rather than `torch.sparse.mm`; see `neighbour_mean` below.

NOT APPLICABLE elsewhere in this project: scikit-learn (ExtraTrees, logistic,
HistGradientBoosting) is CPU-only, and xgboost offers CUDA but not MPS. The Extra Trees
model that produces the submission cannot use the GPU at all.
"""
import torch


def get_device(verbose=True):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if verbose:
        print(f"[device] {device}", flush=True)
    return device


def pad_neighbours(src, dst, n, fill=-1):
    """Edge list -> (n, max_degree) index matrix plus a validity mask.

    Lets neighbour aggregation run as a gather, which MPS supports, instead of a sparse
    matmul, which it does not.
    """
    import numpy as np
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    degree = np.bincount(src, minlength=n)
    width = int(degree.max()) if len(degree) else 0
    index = np.full((n, width), fill, np.int64)
    cursor = 0
    for node in range(n):
        d = degree[node]
        if d:
            index[node, :d] = dst[cursor:cursor + d]
            cursor += d
    return index, index >= 0


def neighbour_mean(h, index, mask):
    """Mean of each node's neighbours. index/mask from `pad_neighbours`, on `h.device`."""
    gathered = h[index.clamp(min=0)]                      # (n, width, d)
    gathered = gathered * mask.unsqueeze(-1)
    count = mask.sum(1, keepdim=True).clamp(min=1)
    return gathered.sum(1) / count
