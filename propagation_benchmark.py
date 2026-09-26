"""
propagation_benchmark.py
=========================
Single-file LABEL-PROPAGATION + CLUSTERING benchmark for graph-rewiring
methods.

WHAT THIS DOES
--------------
For each (dataset, method, seed) it REWIRES the graph with the method's own
rewiring / representation code (verbatim model implementations — nothing
about the models themselves is changed) and then measures:

  PROPAGATION METRICS (both original and rewired graph)
    * Propagation speed / time steps — iterations until the propagation
      process stops changing (label propagation) or stops discovering new
      nodes (BFS).
    * Coverage / reachability — fraction of nodes reached from the labelled
      (train-mask) source set, via multi-source BFS.
    * Propagation depth — the maximum hop-distance (BFS level) any reached
      node sits at from the nearest source.
    * Label-propagation convergence + test accuracy — iterations to converge
      and a cheap sanity check on how good the propagated soft-labels are.

  CLUSTERING METRICS (on the REWIRED graph's representations)
    * The model's own learned embedding — only for methods that produce one
      (glare, idgl, gadc, lpkg); NaN-filled for the structure-only methods
      (dhgr, graphite, fosr, comfy).
    * The label-propagation soft-label matrix F — produced by every method,
      so every method gets a comparable clustering score even without its
      own embedding.
    KMeans(k=num_classes) is run on each, over ALL nodes and over TEST nodes
    only, reporting external metrics (NMI, ARI, clustering accuracy via
    Hungarian matching, purity, homogeneity/completeness/V-measure) and
    internal metrics (silhouette, Davies-Bouldin, Calinski-Harabasz,
    intra/inter-cluster distance).

It does NOT train any downstream classifier beyond the generic label
propagation step, and it does not compute or store homophily measures or
raw rewired edge lists.

Methods (model architectures are reproduced VERBATIM — nothing changed):
    glare, idgl, gadc, lpkg, dhgr, graphite, fosr, comfy

Per-method notes:
    * GADC / GRAPHITE do not rewire the node-node structure (GADC diffuses
      features; GRAPHITE adds bipartite feature nodes) — the "rewired" graph
      propagation metrics equal the original graph's, and this is flagged
      (structure_changed=False) in every record.
    * IDGL learns a soft dense adjacency; a discrete rewired graph is obtained
      by top-k sparsification (dense case) or cosine-kNN over the learned
      hidden (anchor case, N>2000). Documented measurement choice, not a
      model change.

OUTPUT (folder: others_benchmark_results_propagation/)
    results.jsonl                        one JSON record per finished run
    embeddings/*.npy                     model embeddings + LP soft-label
                                          matrices, for later re-use / re-clustering
    tables/table_bfs.csv                 BFS speed / coverage, original vs rewired
    tables/table_label_prop.csv          label-propagation steps / convergence / acc
    tables/table_graph_stats.csv         edge counts, degree, structure_changed
    tables/table_clustering_model_emb.csv       clustering on model embedding, all nodes
    tables/table_clustering_model_emb_test.csv  clustering on model embedding, test nodes
    tables/table_clustering_lp_f.csv            clustering on LP-F, all nodes
    tables/table_clustering_lp_f_test.csv       clustering on LP-F, test nodes
    tables/table_clustering_comparison.csv      emb vs LP-F side by side
    tables/table_clustering_pivot_*.csv         method x dataset pivots (NMI/ACC/ARI)
    tables/all_results_flat.csv          everything in one flat table
    SUMMARY.md                           pivoted human-readable summary

RESUME / CHECKPOINTING
    Runs are keyed by (dataset, method, seed) and appended durably; interrupting
    and re-running continues exactly where it stopped ("run one, write one").

USAGE
    python propagation_benchmark.py                     # all methods, 1 seed
    python propagation_benchmark.py --smoke_test
    python propagation_benchmark.py --methods glare fosr --datasets Actor
    python propagation_benchmark.py --seeds 0 1 2 --device cuda

REQUIREMENTS
    Place the DHGR/ repo and the common/ package next to this file (as in the
    original project). torch, torch_geometric, scipy, scikit-learn, networkx,
    numpy are required.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
import warnings
import copy
import sys
import math as _math
import csv as _csv
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import networkx as nx

from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from torch_geometric.utils import degree as pyg_degree

warnings.filterwarnings("ignore")

# ── Unified-benchmark additions ──────────────────────────────────────────────
import os as _os
import random as _random

_HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path(".").resolve()
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import metrics as _metrics          # noqa: E402
from common import progress as _progress        # noqa: E402
from common import checkpoint as _checkpoint     # noqa: E402
from common import reporting as _reporting       # noqa: E402
from common import plotting as _plotting         # noqa: E402


def set_global_seed(seed: int):
    """Seed every RNG we can reach and put cuDNN in deterministic mode."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    _os.environ["PYTHONHASHSEED"] = str(seed)

PROP_OUT_DIR = Path("others_benchmark_results_propagation")
PROP_OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_DIR = PROP_OUT_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_DIR = PROP_OUT_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

def save_split(dataset_name, data, splits_dir=SPLITS_DIR):
    """
    Save the train/val/test masks used for `dataset_name` to disk.
    """
    npz_path = splits_dir / f"{dataset_name}.npz"
    pt_path  = splits_dir / f"{dataset_name}.pt"

    if npz_path.exists() and pt_path.exists():
        return

    train_mask = data.train_mask.detach().cpu()
    val_mask   = data.val_mask.detach().cpu()
    test_mask  = data.test_mask.detach().cpu()

    np.savez(
        npz_path,
        train_mask=train_mask.numpy(),
        val_mask=val_mask.numpy(),
        test_mask=test_mask.numpy(),
        num_nodes=int(data.num_nodes),
    )

    torch.save(
        {
            "dataset": dataset_name,
            "train_mask": train_mask,
            "val_mask": val_mask,
            "test_mask": test_mask,
            "num_nodes": int(data.num_nodes),
        },
        pt_path,
    )

    print(
        f"  [Splits] Saved {dataset_name} split "
        f"(train={int(train_mask.sum())}, "
        f"val={int(val_mask.sum())}, "
        f"test={int(test_mask.sum())}) -> {npz_path.name} / {pt_path.name}"
    )


# =============================================================================
#  SECTION 1 — Dataset loading 
# =============================================================================

def _safe_extract_masks(d):
    def _to_bool(m, N):
        if m is None:
            return torch.zeros(N, dtype=torch.bool)
        if m.dtype == torch.bool:
            if m.dim() == 2:
                return m[:, 0]
            return m
        mask = torch.zeros(N, dtype=torch.bool)
        mask[m] = True
        return mask
    N = d.num_nodes
    return (
        _to_bool(d.train_mask, N),
        _to_bool(d.val_mask,   N),
        _to_bool(d.test_mask,  N),
    )


def _make_ns(d, tm, vm, te):
    return SimpleNamespace(
        x=d.x, y=d.y, edge_index=d.edge_index,
        train_mask=tm, val_mask=vm, test_mask=te,
        num_nodes=d.num_nodes,
        num_classes=int(d.y.max().item()) + 1,
    )

def load_real_dataset(name, root="./data"):
    if name == "Actor":
        from torch_geometric.datasets import Actor as _Actor
        ds = _Actor(root=root)
        d  = ds[0]
        tm, vm, te = _safe_extract_masks(d)
        if int(tm.sum()) == 0:
            N   = d.num_nodes
            rng = torch.Generator(); rng.manual_seed(0)
            perm = torch.randperm(N, generator=rng)
            tm = torch.zeros(N, dtype=torch.bool); tm[perm[:int(0.6*N)]] = True
            vm = torch.zeros(N, dtype=torch.bool); vm[perm[int(0.6*N):int(0.8*N)]] = True
            te = torch.zeros(N, dtype=torch.bool); te[perm[int(0.8*N):]] = True
        return _make_ns(d, tm, vm, te)

    _wiki_map = {"Squirrel-F": "squirrel", "Chameleon-F": "chameleon"}
    if name in _wiki_map:
        wiki_name = _wiki_map[name]
        from torch_geometric.datasets import WikipediaNetwork as _Wiki
        _wiki_d = None
        for _kwargs in [
            {"geom_gcn_preprocess": True},
            {"geom_gcn_preprocess": False},
            {},
        ]:
            try:
                _ds = _Wiki(root=root, name=wiki_name, **_kwargs)
                _d  = _ds[0]
                if _d.y is None or _d.y.numel() == 0:
                    continue
                _tm, _vm, _te = _safe_extract_masks(_d)
                if int(_tm.sum()) > 0:
                    return _make_ns(_d, _tm, _vm, _te)
                _wiki_d = _d
            except Exception:
                continue
        if _wiki_d is None:
            raise RuntimeError(f"Cannot load {name}")
        N   = _wiki_d.num_nodes
        rng = torch.Generator(); rng.manual_seed(0)
        perm = torch.randperm(N, generator=rng)
        tm = torch.zeros(N, dtype=torch.bool); tm[perm[:int(0.6*N)]] = True
        vm = torch.zeros(N, dtype=torch.bool); vm[perm[int(0.6*N):int(0.8*N)]] = True
        te = torch.zeros(N, dtype=torch.bool); te[perm[int(0.8*N):]] = True
        return _make_ns(_wiki_d, tm, vm, te)

    import numpy as _np_inner
    _orig_np_load = _np_inner.load
    def _pickle_load(*args, **kwargs):
        kwargs.setdefault("allow_pickle", True)
        return _orig_np_load(*args, **kwargs)
    _np_inner.load = _pickle_load
    try:
        from torch_geometric.datasets import HeterophilousGraphDataset
        ds = HeterophilousGraphDataset(root=root, name=name)
        d  = ds[0]
    finally:
        _np_inner.load = _orig_np_load
    tm, vm, te = _safe_extract_masks(d)
    return _make_ns(d, tm, vm, te)


def generate_synthetic_dataset(name, seed=0):
    import random as _random
    rng = np.random.RandomState(seed)
    N = 20_000

    def balanced_labels(n, C):
        y = np.concatenate([
            np.full(n // C + (1 if i < n % C else 0), i) for i in range(C)
        ])
        rng.shuffle(y); return y

    def make_masks(n):
        idx  = rng.permutation(n)
        n_tr = int(0.20 * n); n_va = int(0.20 * n)
        tm = torch.zeros(n, dtype=torch.bool)
        vm = torch.zeros(n, dtype=torch.bool)
        te = torch.zeros(n, dtype=torch.bool)
        tm[idx[:n_tr]] = True
        vm[idx[n_tr:n_tr+n_va]] = True
        te[idx[n_tr+n_va:]] = True
        return tm, vm, te

    def edge_tensor(edges):
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
        E = np.unique(np.array(edges, dtype=np.int64), axis=0)
        E = E[E[:, 0] != E[:, 1]]
        rev = E[:, [1, 0]]
        E = np.unique(np.vstack([E, rev]), axis=0)
        return torch.tensor(E.T, dtype=torch.long)

    configs = {
        "HSBM-MED":  (N, 6), "STRUC-HET": (N, 8),
        "FEAT-HET":  (N, 5), "MIXED-SIG": (N, 7),
    }
    if name not in configs:
        raise ValueError(f"Unknown synthetic dataset: {name}")

    n, C = configs[name]; y = balanced_labels(n, C)
    groups = [np.where(y == c)[0] for c in range(C)]; edges = []

    if name == "HSBM-MED":
        pin, pout = 0.00025, 0.00125
        for c1 in range(C):
            for c2 in range(c1, C):
                p = pin if c1 == c2 else pout
                m = int(p * len(groups[c1]) * len(groups[c2]))
                uu = rng.choice(groups[c1], m); vv = rng.choice(groups[c2], m)
                edges.extend(zip(uu.tolist(), vv.tolist()))
        centers = rng.randn(C, 32) * 1.1; X = np.zeros((n, 128), dtype=np.float32)
        for i in range(n):
            c = y[i]; alt = rng.choice([k for k in range(C) if k != c])
            X[i, :32]  = 0.65*centers[c] + 0.35*centers[alt] + rng.randn(32)*1.35
            X[i, 32:64] = rng.randn(32); X[i, 64:] = rng.randn(64)*0.8

    elif name == "STRUC-HET":
        deg_budget = np.array([42, 36, 30, 24, 8, 7, 6, 5], dtype=float)
        hub_classes = [0,1,2,3]; peri_classes = [4,5,6,7]
        for c in hub_classes:
            t1 = peri_classes[c % 4]; t2 = peri_classes[(c+1) % 4]
            for u in groups[c]:
                reps = int(deg_budget[c] * 0.35)
                for _ in range(reps):
                    tgt = t1 if rng.rand() < 0.7 else t2
                    edges.append((u, rng.choice(groups[tgt])))
                if rng.rand() < 0.08:
                    v = rng.choice(groups[c])
                    if v != u: edges.append((u, v))
        if len(edges) > 320_000:
            edges = _random.sample(edges, 320_000)
        centers = rng.randn(C, 8) * 0.35; X = np.zeros((n, 128), dtype=np.float32)
        for i in range(n):
            c = y[i]; X[i, :8] = centers[c] + rng.randn(8)*2.3; X[i, 8:] = rng.randn(120)

    elif name == "FEAT-HET":
        pin, pout = 0.00035, 0.00145
        for c1 in range(C):
            for c2 in range(c1, C):
                p = pin if c1 == c2 else pout
                m = int(p * len(groups[c1]) * len(groups[c2]))
                uu = rng.choice(groups[c1], m); vv = rng.choice(groups[c2], m)
                edges.extend(zip(uu.tolist(), vv.tolist()))
        centers = rng.randn(C, 64) * 1.8; X = np.zeros((n, 128), dtype=np.float32)
        for i in range(n):
            c = y[i]; alt = rng.choice([k for k in range(C) if k != c])
            X[i,:64] = 0.78*centers[c] + 0.22*centers[alt] + rng.randn(64)*0.95
            X[i,64:] = rng.randn(64)*0.55

    elif name == "MIXED-SIG":
        for c in range(C):
            nxt=(c+1)%C; nxt2=(c+2)%C; prv=(c-1)%C
            for u in groups[c]:
                for _ in range(2): edges.append((u, rng.choice(groups[nxt])))
                for _ in range(1): edges.append((u, rng.choice(groups[nxt2])))
                if rng.rand() < 0.22: edges.append((u, rng.choice(groups[prv])))
                if rng.rand() < 0.12: edges.append((u, rng.choice(groups[c])))
        centers = rng.randn(C, 20) * 1.0; X = np.zeros((n, 128), dtype=np.float32)
        for i in range(n):
            c = y[i]; alt1=(c+1)%C; alt2=(c-1)%C
            mix = rng.choice([0,1,2], p=[0.55,0.25,0.20])
            proto = centers[c] if mix==0 else (centers[alt1] if mix==1 else centers[alt2])
            X[i,:20] = proto + rng.randn(20)*1.4
            X[i,20:48] = rng.randn(28); X[i,48:] = rng.randn(80)*0.6

    X = X.astype(np.float32)
    mu = X.mean(0, keepdims=True); sd = X.std(0, keepdims=True) + 1e-8
    X = (X - mu) / sd
    edge_index = edge_tensor(edges)
    train_mask, val_mask, test_mask = make_masks(n)
    return SimpleNamespace(
        x=torch.tensor(X, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=edge_index,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        num_nodes=n, num_classes=C,
    )

# =============================================================================
#  SECTION 2 — Shared helpers
# =============================================================================

def _to_scipy(edge_index, num_nodes):
    from torch_geometric.utils import to_scipy_sparse_matrix
    return to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

def _sym_norm_adj(edge_index, num_nodes, add_self_loops=True):
    A = _to_scipy(edge_index.cpu(), num_nodes).astype(np.float32)
    if add_self_loops:
        A = A + sp.eye(num_nodes, format="csr")
    deg  = np.asarray(A.sum(1)).reshape(-1)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    D    = sp.diags(dinv)
    S    = (D @ A @ D).tocoo().astype(np.float32)
    idx  = torch.tensor(np.stack([S.row, S.col]), dtype=torch.long)
    val  = torch.tensor(S.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes)).coalesce()

def _row_norm_adj(edge_index, num_nodes, add_self_loops=True):
    A = _to_scipy(edge_index.cpu(), num_nodes).astype(np.float32)
    if add_self_loops:
        A = A + sp.eye(num_nodes, format="csr")
    deg  = np.asarray(A.sum(1)).reshape(-1)
    dinv = np.where(deg > 0, 1.0 / deg, 0.0)
    D    = sp.diags(dinv)
    S    = (D @ A).tocoo().astype(np.float32)
    idx  = torch.tensor(np.stack([S.row, S.col]), dtype=torch.long)
    val  = torch.tensor(S.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes)).coalesce()


# =============================================================================
#  SECTION 3 — IDGL faithful implementation
# =============================================================================

class _IDGLGraphLearner(nn.Module):
    EPS = 1e-12

    def __init__(self, in_dim, num_pers=4, epsilon=0.5):
        super().__init__()
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_pers, in_dim)))
        self.epsilon = epsilon

    def forward(self, x):
        xw  = x.unsqueeze(0) * self.W.unsqueeze(1)
        xn  = F.normalize(xw, p=2, dim=-1)
        S   = torch.matmul(xn, xn.transpose(-1, -2)).mean(0)
        S   = S * (S > self.epsilon).float()
        rs  = S.sum(-1, keepdim=True).clamp(min=self.EPS)
        return S / rs, S


class _IDGLAnchorLearner(nn.Module):
    EPS = 1e-12

    def __init__(self, in_dim, num_anchors, num_pers=4, epsilon=0.1):
        super().__init__()
        self.W       = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_pers, in_dim)))
        self.epsilon = epsilon

    def forward(self, x, x_anc):
        xw  = x.unsqueeze(0)     * self.W.unsqueeze(1)
        aw  = x_anc.unsqueeze(0) * self.W.unsqueeze(1)
        xn  = F.normalize(xw, p=2, dim=-1)
        an  = F.normalize(aw, p=2, dim=-1)
        R   = torch.bmm(xn, an.transpose(-1, -2)).mean(0)
        R   = R * (R > self.epsilon).float()
        rs  = R.sum(-1, keepdim=True).clamp(min=self.EPS)
        return R / rs, R


def _idgl_graph_reg(A_raw, X, alpha=1.0, beta=1.0, gamma=0.5):
    N  = X.shape[0]
    A  = A_raw
    deg = A.sum(-1)
    LX  = deg.unsqueeze(-1) * X - torch.mm(A, X)
    l_smooth = alpha * (X * LX).sum() / (N * N)
    l_conn = -beta  * torch.log(deg.clamp(min=1e-12)).mean()
    l_spar =  gamma * A.pow(2).sum() / (N * N)
    return l_smooth + l_conn + l_spar


def _idgl_anchor_reg(R_raw, X_anc, alpha=1.0, beta=1.0, gamma=0.5):
    EPS = 1e-12
    s   = R_raw.size(1)
    Delta_inv = 1.0 / R_raw.sum(dim=1).clamp(min=EPS)
    B_hat = R_raw.t() @ (Delta_inv.unsqueeze(-1) * R_raw)
    deg_b = B_hat.sum(-1)
    L_b   = torch.diag(deg_b) - B_hat
    l_sm  = alpha * torch.trace(X_anc.t() @ (L_b @ X_anc)) / (2.0 * s * s)
    l_conn = -beta  * torch.log(deg_b.clamp(min=EPS)).mean()
    l_spar =  gamma * B_hat.pow(2).sum() / (s * s)
    return l_sm + l_conn + l_spar


def _idgl_anchor_mp(x, R_norm, Lam_inv, Del_inv):
    f1 = Lam_inv.unsqueeze(-1) * (R_norm.t() @ x)
    return Del_inv.unsqueeze(-1) * (R_norm @ f1)


def _sample_anchors_idgl(N, s, edge_index, device):
    from torch_geometric.utils import degree as pyg_deg
    deg  = pyg_deg(edge_index[1], num_nodes=N).cpu().numpy().astype(np.float64)
    deg  = np.maximum(deg, 1.0)
    prob = deg / deg.sum()
    idx  = np.random.choice(N, size=s, replace=False, p=prob)
    return torch.tensor(idx, dtype=torch.long, device=device)


def _sparsify_topk_from_dense(A_dense, k, num_nodes):
    """Top-k (by weight) out-neighbours per row of a dense adjacency, self
    loops removed.  Returns a directed edge_index."""
    A = A_dense.clone()
    A.fill_diagonal_(float("-inf"))
    k = max(1, min(int(k), num_nodes - 1))
    _, nbr = torch.topk(A, k=k, dim=1, largest=True, sorted=False)
    src = torch.arange(num_nodes, device=A.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nbr.reshape(-1)
    finite = torch.isfinite(A[src, dst])
    return torch.stack([src[finite], dst[finite]]).cpu()


def _knn_from_embedding(Z, k, num_nodes, block=1024):
    """Memory-safe cosine kNN edge_index over a node embedding Z [N, d]."""
    Zn = F.normalize(Z.float(), dim=1)
    k = max(1, min(int(k), num_nodes - 1))
    rows, cols = [], []
    for s in range(0, num_nodes, block):
        e = min(s + block, num_nodes)
        sim = Zn[s:e] @ Zn.t()
        local = torch.arange(e - s)
        sim[local, torch.arange(s, e)] = float("-inf")
        _, nbr = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
        src = torch.arange(s, e).unsqueeze(1).expand(-1, k).reshape(-1)
        rows.append(src)
        cols.append(nbr.reshape(-1))
    return torch.stack([torch.cat(rows), torch.cat(cols)]).cpu()


# =============================================================================
#  SECTION 4 — LPkG faithful implementation
# =============================================================================
import copy
import random

def _lpkg_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def _lpkg_normalized_adjacency(edge_index, num_nodes, add_self_loops=True, device=None):
    if device is None:
        device = edge_index.device
    edge_index = edge_index.to(device)
    row = edge_index[0]
    col = edge_index[1]
    if add_self_loops:
        loops = torch.arange(num_nodes, device=device, dtype=torch.long)
        row = torch.cat([row, loops])
        col = torch.cat([col, loops])
    values = torch.ones(row.numel(), dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(torch.stack([row, col], dim=0), values, size=(num_nodes, num_nodes), device=device).coalesce()
    idx = A.indices()
    val = A.values()
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.scatter_add_(0, idx[0], val)
    deg_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)
    norm_values = deg_inv_sqrt[idx[0]] * val * deg_inv_sqrt[idx[1]]
    A_norm = torch.sparse_coo_tensor(idx, norm_values, size=(num_nodes, num_nodes), device=device).coalesce()
    return A_norm

from torch_geometric.nn import GCNConv

class _LPkGGAE(nn.Module):
    def __init__(self, in_dim, hid_dim, lat_dim):
        super().__init__()
        self.enc1 = GCNConv(in_dim, hid_dim)
        self.enc2 = GCNConv(hid_dim, lat_dim)
        self.dec1 = nn.Linear(lat_dim, hid_dim)
        self.dec2 = nn.Linear(hid_dim, in_dim)

    def encode(self, x, edge_index):
        h = self.enc1(x, edge_index)
        h = torch.relu(h)
        z = self.enc2(h, edge_index)
        return z

    def decode(self, z):
        h = self.dec1(z)
        h = torch.sigmoid(h)
        x_hat = self.dec2(h)
        return x_hat

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_hat = self.decode(z)
        reconstruction = torch.sigmoid(x_hat)
        loss = F.mse_loss(reconstruction, x)
        return z, loss


@torch.no_grad()
def _lpkg_build_knn_graph(z, k, batch_size):
    n = z.size(0)
    if k >= n:
        raise ValueError(f"k={k} must be smaller than N={n}")
    z_norm = F.normalize(z, p=2, dim=1)
    rows = []
    cols = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        z_batch = z_norm[start:end]
        similarity = z_batch @ z_norm.t()
        local = torch.arange(end - start, device=z.device)
        global_idx = torch.arange(start, end, device=z.device)
        similarity[local, global_idx] = -float("inf")
        _, neighbors = torch.topk(similarity, k=k, dim=1, largest=True, sorted=False)
        source = torch.arange(start, end, device=z.device).unsqueeze(1).expand(-1, k).reshape(-1)
        destination = neighbors.reshape(-1)
        rows.append(source)
        cols.append(destination)
    return torch.stack([torch.cat(rows), torch.cat(cols)], dim=0)


# =============================================================================
# DHGR official implementation
# =============================================================================
def _locate_dhgr_root():
    import os
    here = Path(__file__).resolve().parent
    candidates = []
    env = os.environ.get("DHGR_ROOT")
    if env:
        candidates.append(Path(env))
    candidates += [here / "DHGR", here.parent / "DHGR", here.parent.parent / "DHGR"]
    for c in candidates:
        if (c / "GraphLearner.py").exists():
            return c
    return None

DHGR_ROOT = _locate_dhgr_root()
DHGRModelHandler = None
if DHGR_ROOT is not None:
    if str(DHGR_ROOT) not in sys.path:
        sys.path.insert(0, str(DHGR_ROOT))
    try:
        from GraphLearner import ModelHandler as DHGRModelHandler
    except Exception as e:
        DHGRModelHandler = None


# =============================================================================
# GRAPHITE — FAITHFUL IMPLEMENTATION
# =============================================================================
def graphite_transform(data, dataset_name=None, device="cpu"):
    X = data.x.detach().cpu().float()
    N, Fdim = X.shape
    unique_vals = torch.unique(X)
    is_binary = bool(torch.all((unique_vals == 0) | (unique_vals == 1)))
    if not is_binary:
        raise ValueError("GRAPHITE faithful implementation requires binary/discrete features.")
    
    X_np = X.numpy().astype(np.float32)
    feature_node_ids = N + np.arange(Fdim, dtype=np.int64)
    row, col = np.nonzero(X_np > 0)
    feat_nodes_for_edges = (N + col).astype(np.int64)
    
    forward_src = row.astype(np.int64)
    forward_dst = feat_nodes_for_edges
    backward_src = feat_nodes_for_edges
    backward_dst = row.astype(np.int64)
    
    feat_src = np.concatenate([forward_src, backward_src])
    feat_dst = np.concatenate([forward_dst, backward_dst])
    feat_ei = torch.tensor(np.stack([feat_src, feat_dst], axis=0), dtype=torch.long, device=device)
    
    X_t = torch.from_numpy(X_np)
    feature_counts = X_t.sum(dim=0, keepdim=True).t().clamp_min(1.0)
    feature_node_features = (X_t.t() @ X_t) / feature_counts
    
    graph_node_features = X_t.clone()
    ds = "" if dataset_name is None else str(dataset_name).lower().replace("_", "-").strip()
    if ds in {"squirrel", "squirrel-f"}:
        graph_node_features = torch.zeros_like(graph_node_features)
    elif ds in {"cora", "citeseer"}:
        row_sum = graph_node_features.sum(dim=1, keepdim=True)
        graph_node_features = graph_node_features / row_sum.clamp_min(1e-12)
        
    X_ext = torch.cat([graph_node_features, feature_node_features], dim=0).to(device)
    
    edge_index_np = data.edge_index.detach().cpu().numpy().astype(np.int64)
    src, dst = edge_index_np[0], edge_index_np[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    graph_pairs = np.vstack([np.stack([src, dst], axis=1), np.stack([dst, src], axis=1)])
    graph_pairs = np.unique(graph_pairs, axis=0)
    graph_ei = torch.tensor(graph_pairs.T, dtype=torch.long, device=device)
    
    return X_ext, graph_ei, feat_ei, N, Fdim


# =============================================================================
# SECTION — FoSR FAITHFUL NODE-CLASSIFICATION IMPLEMENTATION
# =============================================================================
def _fosr_choose_edge_to_add(x, edge_index, degrees):
    n = x.size
    m = edge_index.shape[1]
    y = x / np.sqrt(degrees + 1.0)
    products = np.outer(y, y)
    for e in range(m):
        u = edge_index[0, e]
        v = edge_index[1, e]
        products[u, v] = np.inf
    for i in range(n):
        products[i, i] = np.inf
    smallest = np.argmin(products)
    return smallest % n, smallest // n

def _fosr_compute_degrees(edge_index, num_nodes):
    degrees = np.zeros(num_nodes, dtype=np.float64)
    for e in range(edge_index.shape[1]):
        u = edge_index[0, e]
        degrees[u] += 1.0
    return degrees

def _fosr_add_edge(edge_index, u, v):
    new_edge = np.array([[u, v], [v, u]], dtype=np.int64)
    return np.concatenate([edge_index, new_edge], axis=1)

def _fosr_adj_matvec(edge_index, x, num_nodes):
    y = np.zeros(num_nodes, dtype=np.float64)
    for e in range(edge_index.shape[1]):
        u = edge_index[0, e]
        v = edge_index[1, e]
        y[u] += x[v]
    return y

def _fosr_rewire_once(edge_index, initial_power_iters=5):
    edge_index = np.asarray(edge_index, dtype=np.int64).copy()
    n = int(edge_index.max()) + 1
    degrees = _fosr_compute_degrees(edge_index, n)
    x = 2.0 * np.random.random(n) - 1.0
    for _ in range(initial_power_iters):
        denom = max(degrees.sum(), 1e-12)
        x = x - (np.dot(x, np.sqrt(degrees)) * np.sqrt(degrees) / denom)
        y = x + _fosr_adj_matvec(edge_index, x / np.sqrt(np.maximum(degrees, 1e-12)), n) / np.sqrt(np.maximum(degrees, 1e-12))
        norm = np.linalg.norm(y)
        if norm < 1e-12: break
        x = y / norm
    u, v = _fosr_choose_edge_to_add(x, edge_index, degrees)
    edge_index = _fosr_add_edge(edge_index, u, v)
    return edge_index

def _fosr_rewire(edge_index, num_iterations, initial_power_iters=5):
    e = edge_index.detach().cpu().numpy().astype(np.int64)
    e = np.concatenate([e, e[[1, 0], :]], axis=1)
    e = np.unique(e, axis=1)
    for _ in range(num_iterations):
        e = _fosr_rewire_once(e, initial_power_iters=initial_power_iters)
    return torch.tensor(e, dtype=torch.long)


# =============================================================================
# SECTION — ComFy FAITHFUL IMPLEMENTATION
# =============================================================================
def _comfy_rewire(data, budget_add, budget_delete, seed):
    X = data.x.detach().cpu().float()
    N = int(data.num_nodes)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    e = data.edge_index.detach().cpu().numpy()
    for u, v in zip(e[0], e[1]):
        u, v = int(u), int(v)
        if u != v:
            G.add_edge(u, v)

    communities = list(nx.community.louvain_communities(G, seed=seed))
    assigned = set()
    for c in communities: assigned.update(c)
    for node in range(N):
        if node not in assigned: communities.append({node})
    M = len(communities)

    Xn = F.normalize(X, p=2, dim=1)
    similarity = (Xn @ Xn.t()).numpy()

    pair_scores = {}
    total_score = 0.0
    for i in range(M):
        ni = len(communities[i])
        for j in range(i, M):
            nj = len(communities[j])
            score = ni * nj
            pair_scores[(i, j)] = score
            total_score += score
    if total_score <= 0: total_score = 1.0

    budgets_add = {}
    budgets_delete = {}
    for pair, score in pair_scores.items():
        frac = score / total_score
        budgets_add[pair] = int(budget_add * frac)
        budgets_delete[pair] = int(budget_delete * frac)

    edges_added = set()
    edges_deleted = set()
    for i in range(M):
        C_i = list(communities[i])
        for j in range(i, M):
            C_j = list(communities[j])
            pair = (i, j)
            add_budget = budgets_add[pair]
            del_budget = budgets_delete[pair]
            if add_budget <= 0 and del_budget <= 0: continue

            existing_edges = []
            for u in C_i:
                for v in C_j:
                    if u != v and G.has_edge(u, v):
                        existing_edges.append((u, v))
            existing_edges = list({tuple(sorted(ed)) for ed in existing_edges})
            
            if existing_edges:
                current_sim = np.mean([similarity[u, v] for u, v in existing_edges])
            else:
                current_sim = 0.0
            num_existing = len(existing_edges)

            addition_candidates = []
            if add_budget > 0:
                for u in C_i:
                    for v in C_j:
                        if u != v and not G.has_edge(u, v):
                            candidate_sim = similarity[u, v]
                            if candidate_sim > current_sim:
                                rank = (current_sim * num_existing + candidate_sim) / (num_existing + 1) if num_existing > 0 else candidate_sim
                                addition_candidates.append((rank, u, v))
            addition_candidates.sort(key=lambda z: z[0])
            for rank, u, v in addition_candidates[-add_budget:]:
                key = tuple(sorted((u, v)))
                if key not in edges_added and not G.has_edge(u, v):
                    if len(edges_added) >= budget_add: break
                    G.add_edge(u, v)
                    edges_added.add(key)

            deletion_candidates = []
            if del_budget > 0 and num_existing > 1:
                for u, v in existing_edges:
                    edge_sim = similarity[u, v]
                    if edge_sim < current_sim:
                        rank = (current_sim * num_existing - edge_sim) / (num_existing - 1)
                        deletion_candidates.append((rank, u, v))
            deletion_candidates.sort(key=lambda z: z[0])
            for rank, u, v in deletion_candidates[-del_budget:]:
                key = tuple(sorted((u, v)))
                if key not in edges_deleted and G.has_edge(u, v):
                    if len(edges_deleted) >= budget_delete: break
                    G.remove_edge(u, v)
                    edges_deleted.add(key)

    edges = list(G.edges())
    if edges:
        undirected = np.asarray(edges, dtype=np.int64)
        directed = np.concatenate([undirected, undirected[:, [1, 0]]], axis=0)
        edge_index = torch.tensor(directed.T, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return edge_index, len(edges_added), len(edges_deleted)


# =============================================================================
# GLARE faithful implementation
# =============================================================================
def compute_two_hop_edges(edge_index, num_nodes, max_edges=500_000):
    import scipy.sparse as sp
    src, dst = edge_index.cpu().numpy()
    A = sp.csr_matrix((np.ones(len(src), np.float32), (src, dst)), shape=(num_nodes, num_nodes))
    A2 = (A @ A).tocoo()
    existing = set(zip(src.tolist(), dst.tolist()))
    m = np.array([(r, c) not in existing and r != c for r, c in zip(A2.row, A2.col)])
    r2, c2 = A2.row[m], A2.col[m]
    if len(r2) > max_edges:
        idx = np.random.choice(len(r2), max_edges, replace=False); r2, c2 = r2[idx], c2[idx]
    return torch.tensor(np.stack([r2, c2]), dtype=torch.long)

class _WeightedSAGELayer(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.lin = nn.Linear(in_c * 2, out_c)
    def forward(self, x, edge_index, edge_weight=None):
        N = x.shape[0]; src, dst = edge_index
        w = edge_weight if edge_weight is not None else torch.ones(src.shape[0], device=x.device)
        deg = torch.zeros(N, device=x.device).scatter_add_(0, dst, w).clamp(min=1e-6)
        A = torch.sparse_coo_tensor(torch.stack([dst, src]), w, (N, N))
        agg = torch.sparse.mm(A, x) / deg.unsqueeze(1)
        return self.lin(torch.cat([x, agg], dim=-1))

class GLARESAGEModel(nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout=0.5):
        super().__init__()
        self.conv1 = _WeightedSAGELayer(in_c, hid_c); self.conv2 = _WeightedSAGELayer(hid_c, out_c)
        self.bn1 = nn.BatchNorm1d(hid_c)
        self.dropout = dropout
    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_weight=edge_weight)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index, edge_weight=edge_weight)

def subsample_to_density(edge_index, target_num_edges, features_or_labels, train_mask=None):
    from torch_geometric.utils import to_undirected
    src, dst = edge_index; mask = src != dst; src, dst = src[mask], dst[mask]
    E = src.shape[0]
    if E <= target_num_edges: return edge_index
    feat = features_or_labels
    if feat.dim() == 1:
        labels = feat
        if train_mask is None:
            raise ValueError("subsample_to_density: train_mask required when features_or_labels is a label tensor")
        both_train = train_mask[src] & train_mask[dst]
        same_train = (labels[src] == labels[dst]).float()
        score = torch.where(both_train, same_train, torch.zeros_like(same_train))
        N = labels.shape[0]
    else:
        Z = F.normalize(feat.float(), dim=-1)
        score = (Z[src] * Z[dst]).sum(dim=-1).clamp(-1.0, 1.0)
        N = feat.shape[0]
    order = torch.argsort(score * E - torch.arange(E, dtype=score.dtype, device=score.device), descending=True)
    keep = order[:target_num_edges]
    return to_undirected(torch.stack([src[keep], dst[keep]]), num_nodes=N)


def apply_label_mask(train_mask: torch.Tensor, label_mask_ratio: float, seed: int = 0) -> torch.Tensor:
    if label_mask_ratio == 1.0:
        return train_mask
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    n_train = len(train_indices)
    n_keep  = int(round(label_mask_ratio * n_train))
    rng = torch.Generator(); rng.manual_seed(seed)
    perm     = torch.randperm(n_train, generator=rng)
    keep_idx = train_indices[perm[:n_keep]]
    masked   = torch.zeros_like(train_mask)
    if n_keep > 0:
        masked[keep_idx] = True
    return masked

def _glare_sample_candidate_edges(edge_index, num_nodes, neg_budget, seed=0, x=None, topk=5):
    torch.manual_seed(seed)
    src, dst = edge_index; m = src < dst
    pos = torch.stack([src[m], dst[m]], dim=0)
    existing = set(zip(pos[0].tolist(), pos[1].tolist()))
    cand = [pos]
    twohop_edges = []
    try:
        budget = neg_budget // 2
        raw = compute_two_hop_edges(edge_index, num_nodes, max_edges=budget * 3)
        for i in range(raw.shape[1]):
            u, v = sorted((int(raw[0, i]), int(raw[1, i])))
            if (u, v) not in existing:
                existing.add((u, v)); twohop_edges.append((u, v))
                if len(twohop_edges) >= budget:
                    break
        if twohop_edges:
            cand.append(torch.tensor(twohop_edges, dtype=torch.long).T)
    except Exception:
        pass

    hard_neg_budget = neg_budget - len(twohop_edges)
    if x is not None and hard_neg_budget > 0:
        with torch.no_grad():
            X = F.normalize(x.float(), dim=1); N = num_nodes
            deg = torch.zeros(N).scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
            probe_size = N if N <= 3000 else min(N, max(512, int(N ** 0.5)))
            if probe_size < N:
                p = (deg + 1.0); p = p / p.sum()
                g = torch.Generator(); g.manual_seed(seed)
                probe_idx = torch.multinomial(p, probe_size, replacement=False, generator=g)
            else:
                probe_idx = torch.arange(N)
            sims = X[probe_idx] @ X.T
            knn_edges, hard_negs = [], []
            nbr = [set() for _ in range(N)]
            for s, d in zip(src.tolist(), dst.tolist()):
                nbr[s].add(d)
            for pp, i in enumerate(probe_idx.tolist()):
                row = sims[pp].clone(); row[i] = -2.0
                for j in torch.topk(row, k=topk + 1).indices.tolist():
                    if j == i:
                        continue
                    u, v = sorted((i, j))
                    if (u, v) not in existing:
                        existing.add((u, v)); knn_edges.append((u, v))
                for nb in nbr[i]:
                    row[nb] = -2.0
                for j in torch.topk(row, k=min(max(3, topk), N - 1)).indices.tolist():
                    if j == i:
                        continue
                    u, v = sorted((i, j))
                    if (u, v) not in existing:
                        existing.add((u, v)); hard_negs.append((u, v))
                        if len(hard_negs) >= hard_neg_budget:
                            break
                if len(hard_negs) >= hard_neg_budget:
                    break
            if knn_edges:
                cand.append(torch.tensor(knn_edges, dtype=torch.long).T)
            if hard_negs:
                cand.append(torch.tensor(hard_negs, dtype=torch.long).T)

    n_so_far = sum(t.shape[1] for t in cand) - pos.shape[1]
    remaining = max(0, neg_budget // 4 - n_so_far)
    neg_edges = []; attempts = 0
    while len(neg_edges) < remaining and attempts < remaining * 20:
        attempts += 1
        u, v = torch.randint(0, num_nodes, (2,)).tolist()
        if u == v: continue
        a, b = sorted((u, v))
        if (a, b) not in existing:
            existing.add((a, b)); neg_edges.append((a, b))
    if neg_edges: cand.append(torch.tensor(neg_edges, dtype=torch.long).T)
    c = torch.cat(cand, dim=1)
    return torch.cat([c, c.flip(0)], dim=1)

def _row_norm_prop(edge_index, num_nodes, device):
    ei = to_undirected(edge_index, num_nodes=num_nodes).to(device)
    src, dst = ei
    deg = torch.zeros(num_nodes, device=device).scatter_add_(0, dst, torch.ones(src.shape[0], device=device)).clamp(min=1.0)
    A = torch.sparse_coo_tensor(torch.stack([dst, src]), torch.ones(src.shape[0], device=device), (num_nodes, num_nodes))
    return lambda feat: torch.sparse.mm(A, feat) / deg.unsqueeze(1)


def compute_neighbor_distributions(edge_index, x, y, train_mask, num_nodes, num_classes, M=2, device="cpu"):
    prop = _row_norm_prop(edge_index, num_nodes, device)
    Y_train = torch.zeros(num_nodes, num_classes, device=device)
    tr = train_mask.nonzero(as_tuple=True)[0].to(device)
    if len(tr) > 0:
        Y_train[tr, y.to(device)[tr]] = 1.0
    X = x.to(device).float()
    label_dists, feat_dists = [], []
    cy, cx = Y_train, X
    for _ in range(M):
        cy, cx = prop(cy), prop(cx)
        label_dists.append(cy); feat_dists.append(cx)
    coverage = prop(train_mask.float().to(device).unsqueeze(1)).squeeze(1)
    return label_dists, feat_dists, coverage


def _decentered_cosine(a, b):
    a = a - a.mean(0, keepdim=True); b = b - b.mean(0, keepdim=True)
    return F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)


def compute_distribution_affinity(label_dists, feat_dists, coverage, src, dst, y, train_mask,
                                   chunk_size=150_000):
    device = src.device; E = src.shape[0]
    affinity = torch.empty(E, device=device); trust = torch.empty(E, device=device)
    for s in range(0, E, chunk_size):
        e = min(s + chunk_size, E)
        sc, dc = src[s:e], dst[s:e]
        ls = torch.ones(e - s, device=device)
        for Dy in label_dists:
            ls = ls * _decentered_cosine(Dy[sc], Dy[dc])
        fs = torch.ones(e - s, device=device)
        for Dx in feat_dists:
            fs = fs * _decentered_cosine(Dx[sc], Dx[dc])
        soft_trust = (coverage[sc] * coverage[dc]).clamp(0, 1)
        blended = soft_trust * ls + (1.0 - soft_trust) * fs
        same_class_visible = train_mask[sc] & train_mask[dc] & (y[sc] == y[dc])
        affinity[s:e] = torch.where(same_class_visible, torch.ones_like(blended), blended)
        trust[s:e] = torch.where(same_class_visible, torch.ones_like(soft_trust), soft_trust)
    return affinity, trust


class _SimLearner(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid); self.bn1 = nn.BatchNorm1d(hid); self.act1 = nn.PReLU()
        self.fc2 = nn.Linear(hid, hid); self.bn2 = nn.BatchNorm1d(hid); self.act2 = nn.PReLU()
        self.fc3 = nn.Linear(hid, hid); self.bn3 = nn.BatchNorm1d(hid)
        self.act_out = nn.PReLU()
        self.skip = nn.Linear(in_dim, hid)

    def forward(self, X):
        h = self.act1(self.bn1(self.fc1(X)))
        h = self.act2(self.bn2(self.fc2(h)))
        h = self.bn3(self.fc3(h))
        h = h + self.skip(X)
        return self.act_out(h)


def _feature_dropout(x, p):
    if p <= 0.0:
        return x
    keep = (torch.rand_like(x) > p).float()
    return x * keep


def _nt_xent_loss(z1, z2, temperature):
    n = z1.shape[0]
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    sim = (z @ z.T) / temperature
    sim.fill_diagonal_(-1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def _learn_similarity(x, edge_index, y, train_mask, num_nodes, cfg, device):
    x = x.to(device).float(); edge_index = edge_index.to(device); y = y.to(device); train_mask = train_mask.to(device)
    N, D = x.shape; C = cfg.num_classes
    prop = _row_norm_prop(edge_index, num_nodes, device)
    Y_train = torch.zeros(N, C, device=device)
    tr = train_mask.nonzero(as_tuple=True)[0]
    if len(tr) > 0:
        Y_train[tr, y[tr]] = 1.0
    label_dists, feat_dists = [], []
    cy, cx = Y_train, x
    for _ in range(cfg.glare18_sim_M):
        cy, cx = prop(cy), prop(cx)
        label_dists.append(cy); feat_dists.append(cx)

    mask_y = train_mask.float()

    def sim_from(dists, i, j):
        s = torch.ones(len(i), device=device)
        for Dk in dists:
            s = s * _decentered_cosine(Dk[i], Dk[j])
        return s

    learner = _SimLearner(D, cfg.glare18_sim_hidden).to(device)
    opt = Adam(learner.parameters(), lr=cfg.glare18_sim_lr, weight_decay=cfg.glare18_sim_weight_decay)

    def batch_sim(H, i, j):
        hk = H; hs = []
        for _ in range(cfg.glare18_sim_M):
            hk = prop(hk); hs.append(hk)
        s = torch.ones(len(i), device=device)
        for h_ in hs:
            s = s * _decentered_cosine(h_[i], h_[j])
        return s

    k1 = k2 = min(cfg.glare18_sim_batch_k, N)
    batch_n = min(cfg.glare18_sim_batch_k, N)
    for _ in range(cfg.glare18_sim_pretrain_epochs):
        for _ in range(cfg.glare18_sim_max_iter):
            idx = torch.randint(0, N, (batch_n,), device=device)
            x_batch = x[idx]
            v1 = _feature_dropout(x_batch, cfg.glare18_sim_aug_drop)
            v2 = _feature_dropout(x_batch, cfg.glare18_sim_aug_drop)
            z1, z2 = learner(v1), learner(v2)
            loss = _nt_xent_loss(z1, z2, cfg.glare18_sim_nce_temp)
            opt.zero_grad(); loss.backward(); opt.step()

    labeled_nodes = mask_y.nonzero(as_tuple=True)[0]
    if len(labeled_nodes) >= 2:
        for _ in range(cfg.glare18_sim_finetune_epochs):
            for _ in range(cfg.glare18_sim_max_iter):
                p1 = labeled_nodes[torch.randint(0, len(labeled_nodes), (min(k1, len(labeled_nodes)),), device=device)]
                p2 = labeled_nodes[torch.randint(0, len(labeled_nodes), (min(k2, len(labeled_nodes)),), device=device)]
                H = learner(x); s_hat = batch_sim(H, p1, p2); s_tgt = sim_from(label_dists, p1, p2).detach()
                loss = F.mse_loss(s_hat, s_tgt); opt.zero_grad(); loss.backward(); opt.step()

    with torch.no_grad():
        H = learner(x); hk = H; hs = [prop(hk)]
        for _ in range(1, cfg.glare18_sim_M):
            hs.append(prop(hs[-1]))
        Z = torch.cat(hs, dim=1)
        Zn = F.normalize(Z - Z.mean(0, keepdim=True), dim=-1)
    del learner, opt
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return Zn.detach()


def _full_knn_candidates(Zn, edge_index, num_nodes, cfg, device):
    existing = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    K = min(cfg.glare18_knn_K, max(num_nodes - 1, 1))
    add_src, add_dst = [], []
    for s in range(0, num_nodes, cfg.glare18_knn_block):
        e = min(s + cfg.glare18_knn_block, num_nodes)
        sim = Zn[s:e] @ Zn.T; sim[:, s:e].fill_diagonal_(-2.0)
        vals, idx = sim.topk(K, dim=1)
        keep = vals >= cfg.glare18_knn_epsilon
        rows = torch.arange(s, e, device=device).unsqueeze(1).expand_as(idx)
        add_src.append(rows[keep]); add_dst.append(idx[keep])
    add_src = torch.cat(add_src) if add_src else torch.empty(0, dtype=torch.long, device=device)
    add_dst = torch.cat(add_dst) if add_dst else torch.empty(0, dtype=torch.long, device=device)
    uu, vv = [], []
    for u, v in zip(add_src.tolist(), add_dst.tolist()):
        a, b = sorted((u, v))
        if a != b and (a, b) not in existing:
            existing.add((a, b)); uu.append(a); vv.append(b)
    if not uu:
        return torch.empty((2, 0), dtype=torch.long)
    c = torch.tensor([uu, vv], dtype=torch.long)
    return torch.cat([c, c.flip(0)], dim=1)


def _merge_candidates(a, b):
    def pairs(c):
        if c.numel() == 0:
            return set()
        return set((min(x, y), max(x, y)) for x, y in zip(c[0].tolist(), c[1].tolist()) if x != y)
    p = pairs(a) | pairs(b)
    if not p:
        return torch.empty((2, 0), dtype=torch.long)
    arr = torch.tensor(list(p), dtype=torch.long).T
    return torch.cat([arr, arr.flip(0)], dim=1)


class GLARE18Model(nn.Module):
    def __init__(self, x, edge_index, y, train_mask, num_nodes, cfg):
        super().__init__()
        self.cfg = cfg; self.num_nodes = num_nodes; self.device = torch.device(cfg.device)
        self.register_buffer("x_orig", x.to(self.device)); self.register_buffer("edge_index_orig", edge_index.to(self.device))

        cand_base = _glare_sample_candidate_edges(edge_index.cpu(), num_nodes, cfg.glare_neg_edge_budget, cfg.seed, x.cpu())
        Zn = None; n_extra = 0
        if cfg.glare18_use_learned_knn:
            Zn = _learn_similarity(x, edge_index, y, train_mask, num_nodes, cfg, self.device)
            extra = _full_knn_candidates(Zn, edge_index.to(self.device), num_nodes, cfg, self.device).cpu()
            n_extra = extra.shape[1] // 2
            cand = _merge_candidates(cand_base, extra)
        else:
            cand = cand_base

        n_orig = max(edge_index.shape[1] // 2, 1)
        extra_ratio = n_extra / n_orig
        self.density_adapt_scale = 1.0 / (1.0 + cfg.glare18_density_adapt_strength * extra_ratio)
        self.extra_candidate_ratio = extra_ratio

        self.register_buffer("cand_edges", cand.to(self.device))
        ei_dev = edge_index.to(self.device); cand_dev = self.cand_edges
        key_fwd = ei_dev[0].long() * num_nodes + ei_dev[1].long()
        key_bwd = ei_dev[1].long() * num_nodes + ei_dev[0].long()
        orig_keys = torch.unique(torch.cat([key_fwd, key_bwd]))
        cand_key = cand_dev[0].long() * num_nodes + cand_dev[1].long()
        self.register_buffer("orig_mask", torch.isin(cand_key, orig_keys).float())

        label_dists, feat_dists, coverage = compute_neighbor_distributions(
            edge_index, x, y, train_mask, num_nodes, cfg.num_classes, M=cfg.glare18_dist_M, device=self.device)
        affinity, trust = compute_distribution_affinity(
            label_dists, feat_dists, coverage, self.cand_edges[0], self.cand_edges[1],
            y.to(self.device), train_mask.to(self.device))

        if Zn is not None:
            E = self.cand_edges.shape[1]
            learned_aff = torch.empty(E, device=self.device)
            chunk = cfg.glare18_affinity_chunk_size
            for s in range(0, E, chunk):
                e = min(s + chunk, E)
                learned_aff[s:e] = _decentered_cosine(Zn[self.cand_edges[0][s:e]], Zn[self.cand_edges[1][s:e]])
            w = cfg.glare18_learned_affinity_weight
            affinity = w * learned_aff + (1.0 - w) * affinity
            del Zn, learned_aff
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        self.register_buffer("dist_affinity", affinity); self.register_buffer("dist_trust", trust)
        prior = torch.sigmoid(cfg.glare18_kappa_prior * self.dist_affinity)
        if cfg.glare18_orig_retain_floor > 0.0:
            floor = torch.full_like(prior, cfg.glare18_orig_retain_floor)
            prior = torch.where(self.orig_mask.bool(), torch.maximum(prior, floor), prior)
        self.register_buffer("prior", prior)
        with torch.no_grad():
            init_theta = torch.where(self.orig_mask.bool(), torch.logit(prior.clamp(0.05, 0.95)), torch.full_like(prior, -2.0))
        self.theta = nn.Parameter(init_theta.clone())

        m0 = edge_index.shape[1] / 2.0; deg0 = pyg_degree(edge_index[0], num_nodes).float()
        self.register_buffer("m0", torch.tensor(m0, dtype=torch.float32)); self.register_buffer("deg0", deg0.to(self.device))
        target_deg = (self.deg0 * cfg.glare18_density_ratio * self.density_adapt_scale).clamp(min=1.0)
        self.register_buffer("target_deg", target_deg)

    def soft_weights(self):
        return torch.sigmoid(self.theta / self.cfg.glare_temperature)

    def get_soft_graph(self):
        w = self.soft_weights(); return torch.stack(self.cand_edges.unbind(0)), w

    def get_hard_graph(self):
        w = self.soft_weights(); mask = w > self.cfg.glare_threshold
        return torch.stack([self.cand_edges[0][mask], self.cand_edges[1][mask]])

    def modularity_loss(self, emb=None, pseudo_labels=None):
        w = self.soft_weights(); src, dst = self.cand_edges
        S = self.x_orig if emb is None else emb.detach()
        E = src.shape[0]; chunk = 150_000
        sim_emb = torch.empty(E, device=src.device)
        for s in range(0, E, chunk):
            e = min(s + chunk, E)
            s1 = F.normalize(S[src[s:e]], dim=-1); s2 = F.normalize(S[dst[s:e]], dim=-1)
            sim_emb[s:e] = (s1 * s2).sum(dim=-1)
        g = self.cfg.glare18_struct_mix * sim_emb + (1.0 - self.cfg.glare18_struct_mix) * self.dist_affinity
        null_model = (self.deg0[src] * self.deg0[dst]) / (2.0 * self.m0)
        structural_term = ((g - null_model) * w).sum()
        hom_term = w.new_tensor(0.0)
        if pseudo_labels is not None:
            same = (pseudo_labels[src] * pseudo_labels[dst]).sum(-1) if pseudo_labels.dim() == 2 else (pseudo_labels[src] == pseudo_labels[dst]).float()
            trust_w = self.dist_trust
            hom_term = (trust_w * w * same).sum() / trust_w.sum().clamp(min=1.0)
        return -(structural_term / (2.0 * self.m0)) - self.cfg.glare18_lambda_hom * hom_term

    def degree_budget_loss(self):
        w = self.soft_weights(); src, _ = self.cand_edges
        deg_soft = torch.zeros(self.num_nodes, device=self.device).scatter_add_(0, src, w)
        return F.relu(deg_soft - self.target_deg).pow(2).mean()

    def regularization_loss(self):
        w = self.soft_weights(); eps = 1e-8
        entropy = -(w * (w + eps).log() + (1 - w) * (1 - w + eps).log())
        prior_loss = F.mse_loss(w, self.prior)
        return (self.cfg.glare_lambda_entropy * (-entropy.mean())
                + self.cfg.glare18_lambda_prior * prior_loss
                + self.cfg.glare18_lambda_degree * self.degree_budget_loss())

    def forward_gnn(self, gnn, hard=False):
        if hard:
            return gnn(self.x_orig, self.get_hard_graph())
        ei, ew = self.get_soft_graph()
        return gnn(self.x_orig, ei, edge_weight=ew)


def train_glare18(data, cfg, device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    x, edge_index, y = data.x.to(device), data.edge_index.to(device), data.y.to(device)
    train_mask, val_mask = data.train_mask.to(device), data.val_mask.to(device)

    try:
        glare = GLARE18Model(x, edge_index, y, data.train_mask, data.num_nodes, cfg).to(device)
    except RuntimeError as err:
        if "out of memory" not in str(err).lower() or device.type != "cuda":
            raise
        print("  [OOM] retrying with learned-KNN disabled")
        torch.cuda.empty_cache()
        cfg2 = copy.deepcopy(cfg); cfg2.glare18_use_learned_knn = 0
        glare = GLARE18Model(x, edge_index, y, data.train_mask, data.num_nodes, cfg2).to(device)
        cfg = cfg2

    gnn = GLARESAGEModel(x.shape[1], cfg.glare_gnn_hidden, data.num_classes, cfg.glare_gnn_dropout).to(device)
    best_val, best_state = 0.0, None
    t_start, t_end = max(cfg.glare_temperature * 2.0, 2.0), max(cfg.glare_temperature * 0.25, 0.25)

    for outer in range(cfg.glare_outer_loops):
        frac = outer / max(cfg.glare_outer_loops - 1, 1)
        glare.cfg.glare_temperature = t_end + 0.5 * (t_start - t_end) * (1 + np.cos(np.pi * frac))

        opt_gnn = Adam(gnn.parameters(), lr=cfg.glare_gnn_lr, weight_decay=5e-4)
        for _ in range(cfg.glare_gnn_epochs):
            gnn.train(); glare.eval(); opt_gnn.zero_grad()
            logits = glare.forward_gnn(gnn, hard=False)
            if train_mask.sum() > 0:
                loss = F.cross_entropy(logits[train_mask], y[train_mask])
                loss.backward(); torch.nn.utils.clip_grad_norm_(gnn.parameters(), 5.0); opt_gnn.step()

        gnn.eval(); glare.eval()
        with torch.no_grad():
            logits = glare.forward_gnn(gnn, hard=True)
            val_acc = (logits[val_mask].argmax(-1) == y[val_mask]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {"gnn": copy.deepcopy(gnn.state_dict()), "theta": glare.theta.detach().clone()}

        with torch.no_grad():
            ei_soft, ew_soft = glare.get_soft_graph()
            h = F.relu(gnn.bn1(gnn.conv1(x, ei_soft, edge_weight=ew_soft)))
            pseudo_soft = torch.softmax(glare.forward_gnn(gnn, hard=False), dim=-1)

        opt_theta = Adam([glare.theta], lr=cfg.glare_rewire_lr)
        for _ in range(cfg.glare_rewire_steps):
            glare.train(); opt_theta.zero_grad()
            loss = glare.modularity_loss(emb=h, pseudo_labels=pseudo_soft) + glare.regularization_loss()
            loss.backward(); torch.nn.utils.clip_grad_norm_([glare.theta], 5.0); opt_theta.step()

    if best_state is not None:
        gnn.load_state_dict(best_state["gnn"]); glare.theta.data.copy_(best_state["theta"])

    gnn.eval()
    with torch.no_grad():
        ei = glare.get_hard_graph().cpu()
        ei_dev = ei.to(device)
        h1 = F.relu(gnn.bn1(gnn.conv1(x, ei_dev)))
        prop2 = _row_norm_prop(ei_dev, x.shape[0], device)
        h2 = F.relu(prop2(h1))
        Z18 = torch.cat([h1, h2], dim=-1).cpu()
    del glare
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ei, Z18, best_val


def rewire_glare_orig(data, cfg, device, seed):
    masked_train = apply_label_mask(data.train_mask, cfg.label_mask_ratio, seed=seed)
    data_glare = SimpleNamespace(x=data.x, y=data.y, edge_index=data.edge_index,
                                  train_mask=masked_train, val_mask=data.val_mask,
                                  test_mask=data.test_mask, num_nodes=data.num_nodes,
                                  num_classes=data.num_classes)
    cfg.seed = seed; cfg.num_classes = data.num_classes
    n_full, n_mask = int(data.train_mask.sum()), int(masked_train.sum())
    print(f"  [Semi-sup] label_mask_ratio={cfg.label_mask_ratio:.2f}  visible_labels={n_mask}/{n_full}")
    rewired_ei, Z, best_val = train_glare18(data_glare, cfg, device)
    print(f"  [GLARE-18] best_val={best_val:.4f}  rewired_edges={rewired_ei.shape[1]}")
    target_edges = int(cfg.glare18_target_edge_ratio * data.edge_index.shape[1])
    ei_matched = subsample_to_density(rewired_ei, target_edges, Z)
    print(f"  [GLARE-18] final edges after {cfg.glare18_target_edge_ratio:.2f}x subsample: {ei_matched.shape[1]}")
    return ei_matched.cpu(), Z.cpu()


# =============================================================================
#  PART C — PROPAGATION EVALUATORS
# =============================================================================

def _symmetric_bool_adj(edge_index, num_nodes):
    """Undirected boolean adjacency (no self-loops) as a scipy CSR matrix."""
    ei = edge_index.cpu().numpy()
    if ei.shape[1] == 0:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=bool)
    src, dst = ei[0], ei[1]
    keep = src != dst
    src, dst = src[keep], dst[keep]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.shape[0], dtype=bool)
    A = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    A = A.tocsr()
    A.data[:] = True
    return A

def _mask_to_bool_np(mask, num_nodes):
    """Accepts a torch bool/float mask or an index tensor; returns np.bool_[N]."""
    m = mask
    if hasattr(m, "dim") and m.dim() == 2:
        m = m[:, 0]
    m = m.detach().cpu().numpy()
    if m.dtype != bool:
        m = m.astype(bool)
    if m.shape[0] != num_nodes:
        out = np.zeros(num_nodes, dtype=bool)
        out[m] = True
        return out
    return m

def bfs_propagation_metrics(edge_index, num_nodes, source_mask, max_steps=None):
    """
    Multi-source, array-based BFS reachability from `source_mask` over the
    undirected version of `edge_index`.  Returns speed / coverage / depth.
    """
    src_bool = _mask_to_bool_np(source_mask, num_nodes)
    if not src_bool.any() or num_nodes == 0:
        return {"bfs_steps": 0, "bfs_coverage": 0.0, "bfs_unreached": int(num_nodes),
                "bfs_steps_to_50": None, "bfs_steps_to_90": None, "bfs_steps_to_99": None}

    A = _symmetric_bool_adj(edge_index, num_nodes)
    max_steps = int(max_steps) if max_steps is not None else num_nodes

    visited = src_bool.copy()
    frontier = src_bool.copy()
    coverage_curve = [float(visited.mean())]
    steps = 0
    thresholds = {0.50: None, 0.90: None, 0.99: None}
    for t, v in thresholds.items():
        if coverage_curve[-1] >= t:
            thresholds[t] = 0

    while frontier.any() and steps < max_steps:
        reached = A.dot(frontier)
        new_nodes = reached & (~visited)
        if not new_nodes.any():
            break
        steps += 1
        visited |= new_nodes
        frontier = new_nodes
        cov = float(visited.mean())
        coverage_curve.append(cov)
        for t in thresholds:
            if thresholds[t] is None and cov >= t:
                thresholds[t] = steps

    return {
        "bfs_steps": steps,
        "bfs_coverage": float(visited.mean()),
        "bfs_unreached": int((~visited).sum()),
        "bfs_steps_to_50": thresholds[0.50],
        "bfs_steps_to_90": thresholds[0.90],
        "bfs_steps_to_99": thresholds[0.99],
    }

def label_propagation_run(edge_index, num_nodes, num_classes, y, train_mask,
                          test_mask=None, alpha=0.9, max_iters=100, tol=1e-4):
    """
    Zhu & Ghahramani-style hard-clamped label propagation over the
    row-normalised, self-looped, undirected adjacency of `edge_index`.
    """
    src_bool = _mask_to_bool_np(train_mask, num_nodes)
    y_np = y.detach().cpu().numpy().astype(np.int64)

    A = _symmetric_bool_adj(edge_index, num_nodes).astype(np.float32)
    A = A + sp.eye(num_nodes, format="csr", dtype=np.float32)
    deg = np.asarray(A.sum(1)).reshape(-1)
    d_inv = np.where(deg > 0, 1.0 / deg, 0.0)
    S = sp.diags(d_inv) @ A

    F0 = np.zeros((num_nodes, num_classes), dtype=np.float32)
    if src_bool.any():
        F0[src_bool, y_np[src_bool]] = 1.0

    Fc = F0.copy()
    steps, converged, delta = 0, False, float("nan")
    for step in range(1, int(max_iters) + 1):
        F_new = alpha * (S @ Fc) + (1.0 - alpha) * F0
        F_new[src_bool] = F0[src_bool]
        delta = float(np.abs(F_new - Fc).max()) if num_nodes else 0.0
        Fc = F_new
        steps = step
        if delta < tol:
            converged = True
            break

    test_acc = None
    if test_mask is not None:
        te_bool = _mask_to_bool_np(test_mask, num_nodes)
        if te_bool.any():
            pred = Fc.argmax(axis=1)
            test_acc = float((pred[te_bool] == y_np[te_bool]).mean())

    metrics = {
        "lp_steps": steps,
        "lp_converged": converged,
        "lp_final_delta": delta,
        "lp_test_acc": test_acc,
    }
    return metrics, Fc

def graph_stats(edge_index, num_nodes):
    ei = edge_index.cpu()
    E_dir = int(ei.shape[1])
    src, dst = ei
    non_self = src != dst
    self_loops = int((~non_self).sum().item())
    return {
        "edges_directed": E_dir,
        "self_loops": self_loops,
        "avg_degree": (E_dir / num_nodes) if num_nodes else float("nan"),
    }


# =============================================================================
#  SECTION D — Clustering metrics
# =============================================================================
#
#  For every (dataset, method, seed) we cluster two kinds of representation
#  with KMeans(k=num_classes):
#    * the MODEL's own learned embedding — only for methods that produce one
#      (glare, idgl, gadc, lpkg); NaN-filled for the rest.
#    * the LP soft-label matrix F from label_propagation_run() on the
#      REWIRED graph — available for every method, since every method
#      produces a rewired graph.
#  Each is evaluated on ALL nodes and on TEST nodes only, with both external
#  (vs. ground-truth labels) and internal (structure-only) metrics.
# =============================================================================

CLUST_METRIC_KEYS = [
    "nmi", "ari", "acc", "purity",
    "homogeneity", "completeness", "v_measure",
    "silhouette", "davies_bouldin", "calinski_harabasz",
    "intra_cluster_dist", "inter_cluster_dist",
    "n_samples", "n_clusters_found",
]
_NAN_CLUST = {k: float("nan") for k in CLUST_METRIC_KEYS}


def _clustering_acc_hungarian(y_true_np, y_pred_np, num_classes):
    """Clustering accuracy via Hungarian optimal cluster-to-class assignment."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return float("nan")
    Cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true_np, y_pred_np):
        if 0 <= int(t) < num_classes and 0 <= int(p) < num_classes:
            Cm[int(p), int(t)] += 1
    row_ind, col_ind = linear_sum_assignment(-Cm)
    return float(Cm[row_ind, col_ind].sum()) / max(len(y_true_np), 1)


def _cluster_purity(y_true_np, y_pred_np, num_classes):
    """Fraction of nodes assigned to their own cluster's majority class."""
    total = len(y_true_np)
    if total == 0:
        return float("nan")
    purity = 0.0
    for c in range(num_classes):
        m = y_pred_np == c
        if not m.any():
            continue
        counts = np.bincount(y_true_np[m], minlength=num_classes)
        purity += int(counts.max())
    return purity / total


def compute_clustering_metrics(X, y_true, num_classes, subset_mask=None,
                                n_init=10, seed=0, max_sil_samples=10_000):
    """
    Run KMeans(k=num_classes) on X (optionally restricted to subset_mask rows)
    and report external + internal clustering metrics.  Returns a dict with
    every key in CLUST_METRIC_KEYS (NaN-filled on any failure / degenerate
    input, so callers never need to special-case a missing key).
    """
    nan = float("nan")
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import (
            normalized_mutual_info_score, adjusted_rand_score,
            silhouette_score, davies_bouldin_score, calinski_harabasz_score,
            homogeneity_score, completeness_score, v_measure_score,
        )
    except ImportError:
        return dict(_NAN_CLUST)

    if X is None or len(X) == 0:
        return dict(_NAN_CLUST)

    X_use = X if subset_mask is None else X[subset_mask]
    y_use = y_true if subset_mask is None else y_true[subset_mask]

    n_samples = len(X_use)
    n_unique_labels = len(np.unique(y_use))
    if n_samples < num_classes or n_unique_labels < 2:
        return dict(_NAN_CLUST)

    X_arr = np.array(X_use, dtype=np.float32)
    norms = np.linalg.norm(X_arr, axis=1, keepdims=True)
    X_arr = X_arr / np.where(norms > 1e-12, norms, 1.0)   # L2-normalise

    try:
        km = KMeans(n_clusters=num_classes, n_init=n_init, random_state=seed,
                    max_iter=500)
        labels_pred = km.fit_predict(X_arr)
    except Exception:
        return dict(_NAN_CLUST)

    y_arr = np.array(y_use, dtype=np.int64)

    try:
        nmi = float(normalized_mutual_info_score(y_arr, labels_pred, average_method="arithmetic"))
    except Exception:
        nmi = nan
    try:
        ari = float(adjusted_rand_score(y_arr, labels_pred))
    except Exception:
        ari = nan
    try:
        acc = _clustering_acc_hungarian(y_arr, labels_pred, num_classes)
    except Exception:
        acc = nan
    try:
        purity = _cluster_purity(y_arr, labels_pred, num_classes)
    except Exception:
        purity = nan
    try:
        homog = float(homogeneity_score(y_arr, labels_pred))
    except Exception:
        homog = nan
    try:
        compl = float(completeness_score(y_arr, labels_pred))
    except Exception:
        compl = nan
    try:
        vm_ = float(v_measure_score(y_arr, labels_pred))
    except Exception:
        vm_ = nan

    if n_samples > max_sil_samples:
        rng = np.random.RandomState(seed)
        samp_idx = rng.choice(n_samples, max_sil_samples, replace=False)
        X_sil, lab_sil = X_arr[samp_idx], labels_pred[samp_idx]
    else:
        X_sil, lab_sil = X_arr, labels_pred
    try:
        sil = float(silhouette_score(X_sil, lab_sil, metric="euclidean"))
    except Exception:
        sil = nan
    try:
        dbi = float(davies_bouldin_score(X_arr, labels_pred))
    except Exception:
        dbi = nan
    try:
        chi = float(calinski_harabasz_score(X_arr, labels_pred))
    except Exception:
        chi = nan

    try:
        centroids = km.cluster_centers_
        n_found = int(len(np.unique(labels_pred[labels_pred >= 0])))
        intra_vals = []
        for c in range(num_classes):
            m = labels_pred == c
            if m.sum() > 0:
                intra_vals.append(float(np.linalg.norm(X_arr[m] - centroids[c], axis=1).mean()))
        intra = float(np.mean(intra_vals)) if intra_vals else nan
        if num_classes > 1:
            inter_vals = [float(np.linalg.norm(centroids[i] - centroids[j]))
                          for i in range(len(centroids)) for j in range(i + 1, len(centroids))]
            inter = float(np.mean(inter_vals)) if inter_vals else nan
        else:
            inter = nan
    except Exception:
        intra = inter = nan
        n_found = num_classes

    return {
        "nmi": nmi, "ari": ari, "acc": acc, "purity": purity,
        "homogeneity": homog, "completeness": compl, "v_measure": vm_,
        "silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi,
        "intra_cluster_dist": intra, "inter_cluster_dist": inter,
        "n_samples": int(n_samples), "n_clusters_found": int(n_found),
    }


def run_clustering_for_record(emb_np, lp_F_np, y_np, test_bool, num_classes, seed):
    """
    Cluster the model's own embedding (if present) and the LP soft-label
    matrix, each over ALL nodes and TEST nodes only.  Returns a flat dict of
    prefixed keys (clust_emb_all_*, clust_emb_test_*, clust_lp_all_*,
    clust_lp_test_*) ready to merge into a results record.
    """
    def _safe(X, mask=None):
        if X is None:
            return dict(_NAN_CLUST)
        return compute_clustering_metrics(X, y_np, num_classes, subset_mask=mask, seed=seed)

    out = {}
    for k, v in _safe(emb_np).items():
        out[f"clust_emb_all_{k}"] = v
    for k, v in _safe(emb_np, mask=test_bool).items():
        out[f"clust_emb_test_{k}"] = v
    for k, v in _safe(lp_F_np).items():
        out[f"clust_lp_all_{k}"] = v
    for k, v in _safe(lp_F_np, mask=test_bool).items():
        out[f"clust_lp_test_{k}"] = v
    return out


# =============================================================================
#  SECTION C2 — Per-method rewiring wrappers
# =============================================================================

class RewireResult:
    def __init__(self, edge_index, embedding=None, structure_changed=True, extra=None):
        self.edge_index = edge_index.cpu().long()
        self.embedding = None if embedding is None else embedding.detach().cpu().float()
        self.structure_changed = bool(structure_changed)
        self.extra = extra or {}

def rewire_glare(data, cfg, device, seed):
    if getattr(data.train_mask, "dim", lambda: 1)() == 2:
        col = seed % data.train_mask.shape[1]
        data = SimpleNamespace(
            x=data.x, y=data.y, edge_index=data.edge_index,
            train_mask=data.train_mask[:, col],
            val_mask=data.val_mask[:, col],
            test_mask=data.test_mask[:, col],
            num_nodes=data.num_nodes, num_classes=data.num_classes)
    ei, Z = rewire_glare_orig(data, cfg, device, seed)
    return RewireResult(ei, embedding=Z, structure_changed=True, extra={"embedding_dim": int(Z.shape[1])})

def rewire_fosr(data, cfg, device, seed):
    iters = int(getattr(cfg, "fosr_iterations", 10))
    ip = int(getattr(cfg, "fosr_initial_power_iters", 5))
    ei = _fosr_rewire(data.edge_index, num_iterations=iters, initial_power_iters=ip)
    return RewireResult(ei, embedding=None, structure_changed=True, extra={"fosr_iterations": iters})

def rewire_comfy(data, cfg, device, seed):
    ba = int(getattr(cfg, "comfy_budget_add", 100))
    bd = int(getattr(cfg, "comfy_budget_delete", 100))
    ei, added, deleted = _comfy_rewire(data, budget_add=ba, budget_delete=bd, seed=seed)
    return RewireResult(ei, embedding=None, structure_changed=True, extra={"edges_added": int(added), "edges_deleted": int(deleted)})

def rewire_dhgr(data, cfg, device, seed, dataset_name=None):
    if DHGRModelHandler is None:
        raise ImportError("DHGR repo not available (set $DHGR_ROOT).")
    from torch_geometric.data import Data
    x = data.x.float().to(device)
    y = data.y.long().to(device)
    tm = data.train_mask.bool().to(device)
    vm = data.val_mask.bool().to(device)
    te = data.test_mask.bool().to(device)
    if tm.dim() == 2:
        col = seed % tm.shape[1]
        tm, vm, te = tm[:, col], vm[:, col], te[:, col]
    edge_index = data.edge_index.long().to(device)
    pyg_data = Data(x=x, y=y, edge_index=edge_index, train_mask=tm, val_mask=vm, test_mask=te)
    pyg_data.num_nodes = data.num_nodes

    min_deg = int(getattr(cfg, "dhgr_min_deg", 10))
    if dataset_name == "Roman-empire":
        min_deg = int(getattr(cfg, "dhgr_roman_min_deg", 3))

    graph_handler = DHGRModelHandler(
        in_size=pyg_data.num_features, num_classes=int(data.num_classes),
        thres_min_deg=min_deg, thres_min_deg_ratio=1.0, hidden=128, device=device,
        save_dir="./dhgr_ckpt/", seed=seed, num_epoch=10, num_epoch_finetune=30,
        window_size=[5000, 5000], lr=0.001, weight_decay=5e-3, shuffle=[False, False],
        drop_last=[False, False], moment=1, use_cpu_cache=False)
    rewired = graph_handler(pyg_data, k=8, epsilon=None, embedding_post=True, cat_self=False, prunning=True, thres_prunning=0.5, load_path=None, save_path=None)
    return RewireResult(rewired.edge_index.long(), embedding=None, structure_changed=True,
                        extra={"dhgr_min_deg": min_deg})

def rewire_gadc(data, cfg, device, seed):
    """GADC — adversarial graph diffusion (Ma et al.).

    GADC does not rewire the edge SET; it builds a modified transition matrix
    on the ORIGINAL edges and diffuses features to produce a new node
    representation F = S X.  Structure is therefore unchanged; the learned
    embedding is the diffused feature matrix F.  This is the exact,
    training-free diffusion computation (no downstream classifier).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = data.num_nodes
    lam = cfg.gadc_lam
    K = cfg.gadc_K
    eps = cfg.gadc_epsilon

    X = data.x.float().cpu().numpy().astype(np.float32)

    A_sp = _to_scipy(data.edge_index.cpu(), N).astype(np.float32)
    A_sp = A_sp - sp.diags(A_sp.diagonal())              # drop self-loops

    A_tilde = A_sp + sp.eye(N, format="csr", dtype=np.float32)
    deg_t = np.asarray(A_tilde.sum(1)).reshape(-1)
    d_inv = np.where(deg_t > 0, deg_t ** -0.5, 0.0)
    D_inv = sp.diags(d_inv)
    A_sym = (D_inv @ A_tilde @ D_inv).tocoo().astype(np.float32)

    # row-normalise features
    X_d = torch.tensor(X, dtype=torch.float32, device=device)
    row_sum = torch.sum(X_d, dim=1, keepdim=True)
    row_inv = torch.where(row_sum != 0, 1.0 / row_sum, torch.zeros_like(row_sum))
    X_d = row_inv * X_d
    X = X_d.cpu().numpy()

    # Phi — cosine similarity on ORIGINAL edges
    A_orig_coo = A_sp.tocoo()
    row = A_orig_coo.row.astype(np.int64)
    col = A_orig_coo.col.astype(np.int64)
    xi, xj = X[row], X[col]
    ni = np.linalg.norm(xi, axis=1).clip(min=1e-12)
    nj = np.linalg.norm(xj, axis=1).clip(min=1e-12)
    cos = np.sum(xi * xj, axis=1) / (ni * nj)
    Phi = sp.coo_matrix((eps * cos, (row, col)), shape=(N, N),
                        dtype=np.float32).tocsr()

    # modified transition matrix  T = A_sym - eps * Phi
    T_coo = (A_sym.astype(np.float32).tocsr() - Phi).tocoo()
    neumann = lam / (1.0 + lam)
    T_torch = torch.sparse_coo_tensor(
        torch.tensor(np.stack([T_coo.row, T_coo.col]), dtype=torch.long),
        torch.tensor(T_coo.data * neumann, dtype=torch.float32),
        (N, N)).coalesce().to(device)

    F_d = X_d.clone()
    cur = X_d.clone()
    for _ in range(K):
        cur = torch.sparse.mm(T_torch, cur)
        F_d = F_d + cur
    F_d = F_d / (1.0 + lam)
    F_emb = F_d.detach().cpu()

    return RewireResult(data.edge_index, embedding=F_emb,
                        structure_changed=False,
                        extra={"note": "diffusion (structure unchanged)",
                               "embedding_dim": int(F_emb.shape[1]),
                               "gadc_K": int(K)})

def rewire_lpkg(data, cfg, device, seed, dataset_name=None):
    _lpkg_set_seed(seed)
    device = torch.device(device)
    X = data.x.float().to(device)
    edge_index = data.edge_index.long().to(device)
    gae = _LPkGGAE(in_dim=X.shape[1], hid_dim=cfg.lpkg_gae_hid, lat_dim=cfg.lpkg_gae_lat).to(device)
    opt = Adam(gae.parameters(), lr=cfg.lpkg_gae_lr)
    gae.train()
    for _ in range(cfg.lpkg_gae_epochs):
        opt.zero_grad()
        _z, loss = gae(X, edge_index)
        loss.backward(); opt.step()
    gae.eval()
    with torch.no_grad(): Z_lat, _ = gae(X, edge_index)
    knn_ei = _lpkg_build_knn_graph(Z_lat, k=cfg.lpkg_k, batch_size=512)
    return RewireResult(knn_ei, embedding=Z_lat, structure_changed=True)

def rewire_graphite(data, cfg, device, seed, dataset_name=None):
    X_ext, graph_ei, feat_ei, N_orig, N_feat = graphite_transform(data=data, dataset_name=dataset_name, device=device)
    ei = graph_ei[:, (graph_ei[0] < N_orig) & (graph_ei[1] < N_orig)].cpu()
    return RewireResult(ei, embedding=None, structure_changed=False)

def rewire_idgl(data, cfg, device, seed):
    """IDGL — iterative deep graph learning (Chen et al., NeurIPS 2020).

    IDGL learns a *soft, dense* adjacency jointly with its GNN; there is no
    natural discrete edge set.  We run IDGL's own graph learner + GNN (the
    verbatim modules `_IDGLGraphLearner` / `_IDGLAnchorLearner`) and at the
    best-validation checkpoint extract:
      * the learned node representation h -> reported as the embedding, and
      * a DISCRETE rewired graph obtained by keeping, per node, its top-k
        strongest learned neighbours (k = round(original average out-degree)).

    Full IDGL (N<=2000): top-k is taken from the dense combined adjacency A-bar.
    IDGL-ANCH  (N>2000): the combined adjacency is never materialised; the
    discrete graph is a cosine kNN over the learned h.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N, C = data.num_nodes, data.num_classes
    X = data.x.float().to(device)
    y = data.y.to(device)
    tm = data.train_mask.to(device)
    vm = data.val_mask.to(device)
    if tm.dim() == 2:
        col = seed % tm.shape[1]
        tm, vm = tm[:, col], vm[:, col]

    hidden = cfg.idgl_hidden
    num_pers = cfg.idgl_num_pers
    epsilon = cfg.idgl_epsilon
    lam = cfg.idgl_lambda
    eta = cfg.idgl_eta
    max_iter = cfg.idgl_max_iter
    eps_adj = cfg.idgl_eps_adj
    epochs = cfg.idgl_epochs
    lr = cfg.idgl_lr
    dropout = cfg.idgl_dropout
    reg_alpha = cfg.idgl_reg_alpha
    reg_beta = cfg.idgl_reg_beta
    reg_gamma = cfg.idgl_reg_gamma

    USE_ANCH = N > 2000
    s = min(getattr(cfg, "idgl_num_anchors", None) or 300, N // 4, N)
    k_target = max(1, int(round(data.edge_index.shape[1] / max(N, 1))))

    A0_sp = (_to_scipy(data.edge_index.cpu(), N).astype(np.float32)
             + sp.eye(N, format="csr"))
    deg0 = np.asarray(A0_sp.sum(1)).reshape(-1)
    d_inv = np.where(deg0 > 0, 1.0 / deg0, 0.0)
    L0_sp = (sp.diags(d_inv) @ A0_sp).tocoo().astype(np.float32)
    L0_ei = torch.tensor(np.stack([L0_sp.row, L0_sp.col]), dtype=torch.long, device=device)
    L0_val = torch.tensor(L0_sp.data, dtype=torch.float32, device=device)
    L0_t = torch.sparse_coo_tensor(L0_ei, L0_val, (N, N), device=device).coalesce()

    def sparse_mp(x):
        return torch.sparse.mm(L0_t, x)

    W1 = nn.Linear(X.shape[1], hidden, bias=False).to(device)
    W2 = nn.Linear(hidden, C, bias=False).to(device)

    if not USE_ANCH:
        L0_dense = torch.tensor((sp.diags(d_inv) @ A0_sp).toarray(),
                                dtype=torch.float32, device=device)
        gl1 = _IDGLGraphLearner(X.shape[1], num_pers, epsilon).to(device)
        gl2 = _IDGLGraphLearner(hidden, num_pers, epsilon).to(device)

        def combine(At_n, A1_n):
            return lam * L0_dense + (1.0 - lam) * (eta * At_n + (1.0 - eta) * A1_n)

        def mp(A, x):
            return torch.mm(A, x)

        opt = Adam(list(gl1.parameters()) + list(gl2.parameters())
                   + list(W1.parameters()) + list(W2.parameters()),
                   lr=lr, weight_decay=5e-4)
        best_val = -1.0
        best_Ab = None
        best_h = None
        h = W1(X)
        Z = W2(h)
        cur_Ab = L0_dense
        for epoch in range(epochs):
            gl1.train(); gl2.train(); W1.train(); W2.train()
            opt.zero_grad()
            A1n, A1r = gl1(X)
            Ab1 = combine(A1n, A1n)
            h = F.relu(F.dropout(mp(Ab1, W1(X)), dropout, training=True))
            Z = mp(Ab1, W2(h))
            loss = (F.cross_entropy(Z[tm], y[tm])
                    + _idgl_graph_reg(A1r, X, reg_alpha, reg_beta, reg_gamma))
            prev = A1r.detach()
            cur_Ab = Ab1
            iter_losses = []
            for _ in range(max_iter):
                Atn, Atr = gl2(h.detach())
                Abt = combine(Atn, A1n)
                h2 = F.relu(F.dropout(mp(Abt, W1(X)), dropout, training=True))
                Z2 = mp(Abt, W2(h2))
                l_t = (F.cross_entropy(Z2[tm], y[tm])
                       + _idgl_graph_reg(Atr, X, reg_alpha, reg_beta, reg_gamma))
                iter_losses.append(l_t)
                diff = (Atr.detach() - prev).pow(2).sum()
                denom = A1r.detach().pow(2).sum().clamp(min=1e-12)
                prev = Atr.detach()
                h = h2; Z = Z2; cur_Ab = Abt
                if (diff / denom).item() < eps_adj:
                    break
            if iter_losses:
                loss = loss + torch.stack(iter_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(list(gl1.parameters()) + list(gl2.parameters()), 1.0)
            opt.step()
            with torch.no_grad():
                va = (Z.argmax(-1)[vm] == y[vm]).float().mean().item()
                if va > best_val:
                    best_val = va
                    best_Ab = cur_Ab.detach()
                    best_h = h.detach()
        if best_Ab is None:
            best_Ab = cur_Ab.detach()
            best_h = h.detach()
        ei = _sparsify_topk_from_dense(best_Ab, k_target, N)
        emb = best_h.cpu()
        return RewireResult(ei, embedding=emb, structure_changed=True,
                            extra={"idgl_mode": "full", "best_val": float(best_val),
                                   "embedding_dim": int(emb.shape[1])})
    else:
        anchor_idx = _sample_anchors_idgl(N, s, data.edge_index, device)
        gl1 = _IDGLAnchorLearner(X.shape[1], s, num_pers, epsilon * 0.2).to(device)
        gl2 = _IDGLAnchorLearner(hidden, s, num_pers, epsilon * 0.2).to(device)
        opt = Adam(list(gl1.parameters()) + list(gl2.parameters())
                   + list(W1.parameters()) + list(W2.parameters()),
                   lr=lr, weight_decay=5e-4)
        best_val = -1.0
        best_h = None
        h = W1(X)
        Z = W2(h)
        for epoch in range(epochs):
            gl1.train(); gl2.train(); W1.train(); W2.train()
            opt.zero_grad()
            x_anc = X[anchor_idx]
            R1n, R1r = gl1(X, x_anc)
            Lam1 = R1r.sum(0).clamp(min=1e-12)
            Del1 = R1r.sum(1).clamp(min=1e-12)
            h0_lin = W1(X)
            mp1 = _idgl_anchor_mp(h0_lin, R1n, 1.0 / Lam1, 1.0 / Del1)
            h = F.relu(F.dropout(lam * sparse_mp(h0_lin) + (1.0 - lam) * mp1,
                                 dropout, training=True))
            mp2 = _idgl_anchor_mp(h, R1n, 1.0 / Lam1, 1.0 / Del1)
            Z = W2(lam * sparse_mp(h) + (1.0 - lam) * mp2)
            loss = (F.cross_entropy(Z[tm], y[tm])
                    + _idgl_anchor_reg(R1r, x_anc, reg_alpha, reg_beta, reg_gamma))
            prev_R = R1r.detach()
            iter_losses = []
            for _ in range(max_iter):
                x_anc_h = h[anchor_idx].detach()
                Rtn, Rtr = gl2(h.detach(), x_anc_h)
                Lamt = Rtr.sum(0).clamp(min=1e-12)
                Delt = Rtr.sum(1).clamp(min=1e-12)

                def comb_mp(v, _Rtn=Rtn, _Lamt=Lamt, _Delt=Delt):
                    return (lam * sparse_mp(v) + (1.0 - lam) *
                            (eta * _idgl_anchor_mp(v, _Rtn, 1.0 / _Lamt, 1.0 / _Delt)
                             + (1.0 - eta) * _idgl_anchor_mp(v, R1n, 1.0 / Lam1, 1.0 / Del1)))

                h2 = F.relu(F.dropout(comb_mp(W1(X)), dropout, training=True))
                Z2 = W2(comb_mp(h2))
                l_t = (F.cross_entropy(Z2[tm], y[tm])
                       + _idgl_anchor_reg(Rtr, x_anc_h, reg_alpha, reg_beta, reg_gamma))
                iter_losses.append(l_t)
                diff = (Rtr.detach() - prev_R).pow(2).sum()
                denom = Rtr.detach().pow(2).sum().clamp(min=1e-12)
                prev_R = Rtr.detach()
                h = h2; Z = Z2
                if (diff / denom).item() < eps_adj:
                    break
            if iter_losses:
                loss = loss + torch.stack(iter_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(list(gl1.parameters()) + list(gl2.parameters()), 1.0)
            opt.step()
            with torch.no_grad():
                va = (Z.argmax(-1)[vm] == y[vm]).float().mean().item()
                if va > best_val:
                    best_val = va
                    best_h = h.detach()
        if best_h is None:
            best_h = h.detach()
        ei = _knn_from_embedding(best_h, k_target, N)
        emb = best_h.cpu()
        return RewireResult(ei, embedding=emb, structure_changed=True,
                            extra={"idgl_mode": "anchor", "best_val": float(best_val),
                                   "embedding_dim": int(emb.shape[1])})

REWIRE_FNS = {
    "glare":    rewire_glare,
    "idgl":     rewire_idgl,
    "gadc":     rewire_gadc,
    "lpkg":     rewire_lpkg,
    "dhgr":     rewire_dhgr,
    "graphite": rewire_graphite,
    "fosr":     rewire_fosr,
    "comfy":    rewire_comfy,
}
_TAKES_DATASET_NAME = {"lpkg", "dhgr", "graphite"}


# =============================================================================
#  SECTION C3 — Output paths, tables, and the resumable harness
# =============================================================================

PROP_RESULTS_JSONL = PROP_OUT_DIR / "results.jsonl"
PROP_TABLES_DIR = PROP_OUT_DIR / "tables"
PROP_TABLES_DIR.mkdir(parents=True, exist_ok=True)
PROP_ORIG_JSON = PROP_OUT_DIR / "original_measures.json"

PROP_KEY_FIELDS = ["dataset", "method", "seed"]
PROP_SYNTHETIC = {"HSBM-MED", "STRUC-HET", "FEAT-HET", "MIXED-SIG"}
PROP_ALL_METHODS = ["glare", "idgl", "gadc", "lpkg", "dhgr", "fosr", "comfy"]

# Methods whose wrapper produces its own learned node embedding (beyond the
# generic LP soft-label matrix, which every method gets).
EMBEDDING_METHODS = {"glare", "idgl", "gadc", "lpkg"}
PROP_DEFAULT_DATASETS = ["Amazon-ratings", "Roman-empire", "Actor", "Chameleon-F", "Squirrel-F", "Tolokers"]

def _load_json(path):
    if Path(path).exists():
        try:
            with open(path) as f: return json.load(f)
        except Exception: return {}
    return {}

def _save_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2)
    _os.replace(tmp, str(path))

def _fmt(v, nd=4):
    if v is None: return ""
    try:
        if isinstance(v, float) and _math.isnan(v): return "nan"
    except Exception: pass
    if isinstance(v, float): return f"{v:.{nd}f}"
    return str(v)

def load_dataset_any(name, cfg):
    if name in PROP_SYNTHETIC: return generate_synthetic_dataset(name, seed=0)
    return load_real_dataset(name, root=cfg.data_root)

def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and _math.isnan(v))]
    return (sum(vals) / len(vals)) if vals else float("nan")

def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def write_all_tables(records, datasets, methods):
    ok = [r for r in records if r.get("status") == "ok"]

    # BFS Table
    bfs_rows = []
    for r in ok:
        bfs_rows.append([
            r["dataset"], r["method"], r["seed"],
            r.get("orig_bfs_steps"), r.get("rew_bfs_steps"),
            _fmt(r.get("orig_bfs_coverage")), _fmt(r.get("rew_bfs_coverage")),
            r.get("orig_bfs_steps_to_50"), r.get("rew_bfs_steps_to_50"),
            r.get("orig_bfs_steps_to_90"), r.get("rew_bfs_steps_to_90"),
            r.get("orig_bfs_steps_to_99"), r.get("rew_bfs_steps_to_99"),
            r.get("orig_bfs_unreached"), r.get("rew_bfs_unreached"),
        ])
    _write_csv(PROP_TABLES_DIR / "table_bfs.csv",
               ["dataset", "method", "seed",
                "orig_bfs_steps", "rew_bfs_steps",
                "orig_bfs_coverage", "rew_bfs_coverage",
                "orig_bfs_steps_to_50", "rew_bfs_steps_to_50",
                "orig_bfs_steps_to_90", "rew_bfs_steps_to_90",
                "orig_bfs_steps_to_99", "rew_bfs_steps_to_99",
                "orig_bfs_unreached", "rew_bfs_unreached"], bfs_rows)

    # LP Table
    lp_rows = []
    for r in ok:
        lp_rows.append([
            r["dataset"], r["method"], r["seed"],
            r.get("orig_lp_steps"), r.get("rew_lp_steps"),
            r.get("orig_lp_converged"), r.get("rew_lp_converged"),
            _fmt(r.get("orig_lp_test_acc")), _fmt(r.get("rew_lp_test_acc")),
        ])
    _write_csv(PROP_TABLES_DIR / "table_label_prop.csv",
               ["dataset", "method", "seed",
                "orig_lp_steps", "rew_lp_steps",
                "orig_lp_converged", "rew_lp_converged",
                "orig_lp_test_acc", "rew_lp_test_acc"], lp_rows)

    # Graph Stats Table
    grows = []
    for r in ok:
        grows.append([
            r["dataset"], r["method"], r["seed"],
            r.get("orig_edges"), r.get("rew_edges"),
            r.get("edges_delta_directed"),
            r.get("rew_self_loops"),
            _fmt(r.get("orig_avg_degree")), _fmt(r.get("rew_avg_degree")),
            r.get("structure_changed"), json.dumps(r.get("extra", {})),
        ])
    _write_csv(PROP_TABLES_DIR / "table_graph_stats.csv",
               ["dataset", "method", "seed", "orig_edges", "rew_edges",
                "edges_delta", "rew_self_loops",
                "orig_avg_degree", "rew_avg_degree",
                "structure_changed", "extra"], grows)

    # ---- Clustering: model embedding, ALL nodes --------------------------
    emb_ok = [r for r in ok if r.get("has_embedding")]
    _write_clustering_table(
        PROP_TABLES_DIR / "table_clustering_model_emb.csv",
        emb_ok, "clust_emb_all_")

    # ---- Clustering: model embedding, TEST nodes only --------------------
    _write_clustering_table(
        PROP_TABLES_DIR / "table_clustering_model_emb_test.csv",
        emb_ok, "clust_emb_test_")

    # ---- Clustering: LP soft-label F, ALL nodes, ALL methods --------------
    _write_clustering_table(
        PROP_TABLES_DIR / "table_clustering_lp_f.csv", ok, "clust_lp_all_")

    # ---- Clustering: LP soft-label F, TEST nodes only, ALL methods -------
    _write_clustering_table(
        PROP_TABLES_DIR / "table_clustering_lp_f_test.csv", ok, "clust_lp_test_")

    # ---- Clustering comparison: emb vs LP-F, key metrics side by side ----
    _write_csv(PROP_TABLES_DIR / "table_clustering_comparison.csv",
        ["dataset", "method", "seed", "has_embedding",
         "emb_all_nmi", "emb_all_ari", "emb_all_acc", "emb_all_silhouette",
         "lp_all_nmi", "lp_all_ari", "lp_all_acc", "lp_all_silhouette",
         "emb_test_nmi", "emb_test_ari", "emb_test_acc",
         "lp_test_nmi", "lp_test_ari", "lp_test_acc"],
        [[r["dataset"], r["method"], r["seed"], r.get("has_embedding"),
          _fmt(r.get("clust_emb_all_nmi")), _fmt(r.get("clust_emb_all_ari")),
          _fmt(r.get("clust_emb_all_acc")), _fmt(r.get("clust_emb_all_silhouette")),
          _fmt(r.get("clust_lp_all_nmi")), _fmt(r.get("clust_lp_all_ari")),
          _fmt(r.get("clust_lp_all_acc")), _fmt(r.get("clust_lp_all_silhouette")),
          _fmt(r.get("clust_emb_test_nmi")), _fmt(r.get("clust_emb_test_ari")),
          _fmt(r.get("clust_emb_test_acc")),
          _fmt(r.get("clust_lp_test_nmi")), _fmt(r.get("clust_lp_test_ari")),
          _fmt(r.get("clust_lp_test_acc"))] for r in ok])

    # ---- Pivoted (method x dataset) tables for the headline metrics ------
    _write_pivoted_clust(PROP_TABLES_DIR / "table_clustering_pivot_nmi_emb.csv",
                         ok, datasets, methods, "clust_emb_all_nmi")
    _write_pivoted_clust(PROP_TABLES_DIR / "table_clustering_pivot_acc_emb.csv",
                         ok, datasets, methods, "clust_emb_all_acc")
    _write_pivoted_clust(PROP_TABLES_DIR / "table_clustering_pivot_ari_emb.csv",
                         ok, datasets, methods, "clust_emb_all_ari")
    _write_pivoted_clust(PROP_TABLES_DIR / "table_clustering_pivot_nmi_lp.csv",
                         ok, datasets, methods, "clust_lp_all_nmi")
    _write_pivoted_clust(PROP_TABLES_DIR / "table_clustering_pivot_acc_lp.csv",
                         ok, datasets, methods, "clust_lp_all_acc")

    _write_markdown_summary(ok, datasets, methods)
    
    # Flat Table
    cols = (["dataset", "method", "seed", "status", "n_nodes", "n_classes",
             "structure_changed", "has_embedding",
             "orig_edges", "rew_edges", "edges_delta_directed", "orig_avg_degree", "rew_avg_degree",
             "orig_bfs_steps", "rew_bfs_steps", "orig_bfs_coverage", "rew_bfs_coverage",
             "orig_bfs_unreached", "rew_bfs_unreached",
             "orig_lp_steps", "rew_lp_steps", "orig_lp_test_acc", "rew_lp_test_acc",
             "emb_saved_path", "lp_F_saved_path", "rewire_time_s"]
            + [f"clust_emb_all_{k}" for k in CLUST_METRIC_KEYS]
            + [f"clust_emb_test_{k}" for k in CLUST_METRIC_KEYS]
            + [f"clust_lp_all_{k}" for k in CLUST_METRIC_KEYS]
            + [f"clust_lp_test_{k}" for k in CLUST_METRIC_KEYS])
    rows = []
    for r in ok:
        rows.append([_fmt(r.get(c)) if isinstance(r.get(c), float) else r.get(c) for c in cols])
    _write_csv(PROP_TABLES_DIR / "all_results_flat.csv", cols, rows)


def _write_clustering_table(path, records, prefix):
    """Write a clustering CSV using CLUST_METRIC_KEYS under the given prefix."""
    header = ["dataset", "method", "seed", "has_embedding"] + list(CLUST_METRIC_KEYS)
    rows = []
    for r in records:
        base = [r["dataset"], r["method"], r["seed"], r.get("has_embedding")]
        metrics = [_fmt(r.get(f"{prefix}{k}")) if isinstance(r.get(f"{prefix}{k}"), float)
                   else r.get(f"{prefix}{k}") for k in CLUST_METRIC_KEYS]
        rows.append(base + metrics)
    _write_csv(path, header, rows)


def _write_pivoted_clust(path, ok, datasets, methods, key):
    """Write a pivoted (method x dataset) mean-over-seeds table for `key`."""
    present_methods = [m for m in methods if any(r["method"] == m for r in ok)]
    present_ds = [d for d in datasets if any(r["dataset"] == d for r in ok)]
    piv = _pivot_mean(ok, present_ds, present_methods, key)
    header = ["method"] + present_ds + ["mean"]
    rows = []
    for m in present_methods:
        vals = [piv.get((m, d)) for d in present_ds]
        finite = [v for v in vals if v is not None and not (isinstance(v, float) and _math.isnan(v))]
        mean_v = (sum(finite) / len(finite)) if finite else float("nan")
        rows.append([m] + [_fmt(v) for v in vals] + [_fmt(mean_v)])
    _write_csv(path, header, rows)

def _pivot_mean(ok, datasets, methods, key):
    buckets = defaultdict(list)
    for r in ok: buckets[(r["method"], r["dataset"])].append(r.get(key))
    return {(m, d): _mean(buckets[(m, d)]) for m in methods for d in datasets if (m, d) in buckets}

def _write_markdown_summary(ok, datasets, methods):
    present_methods = [m for m in methods if any(r["method"] == m for r in ok)]
    present_ds = [d for d in datasets if any(r["dataset"] == d for r in ok)]
    lines = ["# Propagation Benchmark — Summary (Rewired Graphs)\n",
             "Cells are the mean over seeds. Original graph references are below.\n"]

    orig_ref = {}
    for r in ok: orig_ref.setdefault(r["dataset"], r)

    metrics = [
        ("bfs_coverage", "BFS Coverage (%)", True),
        ("bfs_steps", "BFS Steps (Depth)", False),
        ("lp_test_acc", "Label Prop Test Accuracy", True),
        ("lp_steps", "Label Prop Steps", False)
    ]

    for key, desc, is_float in metrics:
        lines.append(f"\n## {desc}\n")
        lines.append("| method | " + " | ".join(present_ds) + " |")
        lines.append("|" + "---|" * (len(present_ds) + 1))
        
        orow = ["| original "]
        for d in present_ds:
            r = orig_ref.get(d)
            orow.append(_fmt(r.get(f"orig_{key}")) if r else "")
        lines.append(" | ".join(orow) + " |")
        
        piv = _pivot_mean(ok, present_ds, present_methods, f"rew_{key}")
        for m in present_methods:
            row = [f"| {m}"]
            for d in present_ds:
                row.append(_fmt(piv.get((m, d))))
            lines.append(" | ".join(row) + " |")

    # ---- Clustering section ----
    lines.append("\n---\n\n## Clustering Metrics (KMeans, k = num_classes)\n")
    lines.append("> Only methods that learn their own embedding (glare, idgl, gadc, lpkg)\n"
                 "> get non-NaN values in the **model embedding** tables below. Every\n"
                 "> method populates the **LP-F** tables (the label-propagation\n"
                 "> soft-label matrix, used as a generic embedding).\n")
    clust_metrics = [
        ("clust_emb_all_nmi",  "NMI — model embedding, all nodes"),
        ("clust_emb_all_acc",  "ACC (Hungarian) — model embedding, all nodes"),
        ("clust_emb_all_ari",  "ARI — model embedding, all nodes"),
        ("clust_emb_all_silhouette",     "Silhouette — model embedding, all nodes"),
        ("clust_emb_test_nmi", "NMI — model embedding, test nodes only"),
        ("clust_lp_all_nmi",   "NMI — LP soft-label F, all nodes"),
        ("clust_lp_all_acc",   "ACC — LP soft-label F, all nodes"),
        ("clust_lp_all_ari",   "ARI — LP soft-label F, all nodes"),
        ("clust_lp_test_nmi",  "NMI — LP soft-label F, test nodes only"),
    ]
    for key, desc in clust_metrics:
        lines.append(f"\n### {desc}\n")
        lines.append("| method | " + " | ".join(present_ds) + " |")
        lines.append("|" + "---|" * (len(present_ds) + 1))
        piv = _pivot_mean(ok, present_ds, present_methods, key)
        for m in present_methods:
            row = [f"| {m}"] + [_fmt(piv.get((m, d))) for d in present_ds]
            lines.append(" | ".join(row) + " |")

    with open(PROP_OUT_DIR / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def run_propagation_benchmark(cfg):
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))
    # GLARE18Model does torch.device(cfg.device) internally (verbatim from
    # glare_bestv2.py / glare_benchmark_final.py), so cfg.device must hold a
    # resolved "cuda"/"cpu" string, not "auto".
    cfg.device = str(device)

    print("\n" + "=" * 78)
    print("  PROPAGATION BENCHMARK  (rewiring methods)")
    print("=" * 78)
    print(f"  Datasets : {cfg.datasets}")
    print(f"  Methods  : {cfg.methods}")
    print(f"  Seeds    : {cfg.seeds}")
    print(f"  Device   : {device}")
    print(f"  Results  -> {PROP_OUT_DIR.resolve()}")
    print("=" * 78)

    store = _checkpoint.ResultStore(PROP_RESULTS_JSONL, key_fields=PROP_KEY_FIELDS)
    if store.count():
        print(f"  [resume] {store.count()} completed run(s) found; they are skipped.\n")

    orig_cache = _load_json(PROP_ORIG_JSON)

    total = len(cfg.datasets) * len(cfg.methods) * len(cfg.seeds)
    tracker = _progress.ProgressTracker(total, label="propagation").start()

    for dataset in cfg.datasets:
        print(f"\n{'-'*78}\n  DATASET: {dataset}\n{'-'*78}")
        try:
            data = load_dataset_any(dataset, cfg)
        except Exception as e:
            print(f"  [SKIP dataset] {dataset}: {e}")
            tracker.tick_skipped(len(cfg.methods) * len(cfg.seeds))
            continue

        N, C = data.num_nodes, data.num_classes
        y_np = data.y.detach().cpu().numpy().astype(np.int64)
        print(f"  N={N}  C={C}  edges={data.edge_index.shape[1]}")

        # Compute original measures just once per dataset
        if dataset not in orig_cache:
            orig_bfs = bfs_propagation_metrics(data.edge_index, N, data.train_mask)
            orig_lp_metrics, orig_lp_F = label_propagation_run(
                data.edge_index, N, C, data.y, data.train_mask, data.test_mask
            )
            ogs = graph_stats(data.edge_index, N)

            lp_orig_path = EMBEDDINGS_DIR / f"{dataset}_orig_lp_emb.npy"
            np.save(lp_orig_path, orig_lp_F)

            orig_cache[dataset] = {
                "n_nodes": int(N), "n_classes": int(C),
                "orig_edges": ogs["edges_directed"],
                "orig_avg_degree": ogs["avg_degree"],
                **{f"orig_{k}": v for k, v in orig_bfs.items()},
                **{f"orig_{k}": v for k, v in orig_lp_metrics.items()}
            }
            _save_json(PROP_ORIG_JSON, orig_cache)
            
        oc = orig_cache[dataset]
        print(f"  [orig]  BFS(cov={_fmt(oc['orig_bfs_coverage'])}, steps={oc['orig_bfs_steps']}) "
              f"LP(acc={_fmt(oc['orig_lp_test_acc'])}, steps={oc['orig_lp_steps']})")

        for method in cfg.methods:
            for seed in cfg.seeds:
                if cfg.resume and store.exists(dataset=dataset, method=method, seed=seed):
                    tracker.tick_skipped()
                    continue

                if method == "dhgr" and DHGRModelHandler is None:
                    print("  [DHGR] repo unavailable — skipping.")
                    tracker.tick_done(0.0, note="dhgr-unavailable")
                    continue

                print(f"\n  [{method.upper()}] {dataset} seed={seed}")
                t0 = time.perf_counter()
                try:
                    set_global_seed(seed)
                    
                    # Ensure active mask is single dimensional for proper downstream handling
                    if getattr(data.train_mask, "dim", lambda: 1)() == 2:
                        col = seed % data.train_mask.shape[1]
                        active_train_mask = data.train_mask[:, col]
                        active_test_mask = data.test_mask[:, col]
                    else:
                        active_train_mask = data.train_mask
                        active_test_mask = data.test_mask

                    fn = REWIRE_FNS[method]
                    if method in _TAKES_DATASET_NAME:
                        res = fn(data, cfg, device, seed, dataset_name=dataset)
                    else:
                        res = fn(data, cfg, device, seed)

                    ei = res.edge_index
                    te_bool = _mask_to_bool_np(active_test_mask, N)

                    # Compute Propagation Metrics on Rewired graph
                    rew_bfs = bfs_propagation_metrics(ei, N, active_train_mask)
                    rew_lp_metrics, rew_lp_F = label_propagation_run(
                        ei, N, C, data.y, active_train_mask, active_test_mask
                    )
                    gs = graph_stats(ei, N)
                    dt = time.perf_counter() - t0

                    # Save embeddings (for later re-use / re-clustering)
                    emb_path_str = None
                    if res.embedding is not None:
                        emb_path = EMBEDDINGS_DIR / f"{dataset}_{method}_s{seed}_model_emb.npy"
                        np.save(emb_path, res.embedding.cpu().numpy())
                        emb_path_str = str(emb_path)
                    lp_rew_path = EMBEDDINGS_DIR / f"{dataset}_{method}_s{seed}_rewired_lp_emb.npy"
                    np.save(lp_rew_path, rew_lp_F)

                    # Clustering metrics: model embedding (if any) + LP-F,
                    # each on ALL nodes and TEST nodes only.
                    emb_np = res.embedding.numpy() if res.embedding is not None else None
                    clust_metrics = run_clustering_for_record(
                        emb_np, rew_lp_F, y_np, te_bool, C, seed)

                    rec = {
                        "dataset": dataset, "method": method, "seed": seed, "status": "ok",
                        "n_nodes": int(N), "n_classes": int(C),
                        "structure_changed": res.structure_changed,
                        "has_embedding": res.embedding is not None,
                        "rew_edges": gs["edges_directed"],
                        "rew_self_loops": gs["self_loops"],
                        "rew_avg_degree": gs["avg_degree"],
                        "edges_delta_directed": gs["edges_directed"] - oc["orig_edges"],
                        "extra": res.extra, "rewire_time_s": dt,
                        "emb_saved_path": emb_path_str,
                        "lp_F_saved_path": str(lp_rew_path),
                        **oc,
                        **{f"rew_{k}": v for k, v in rew_bfs.items()},
                        **{f"rew_{k}": v for k, v in rew_lp_metrics.items()},
                        **clust_metrics,
                    }
                    store.append(rec)
                    
                    note = (f"{dataset}/{method}/s{seed} "
                            f"bfs_cov {oc['orig_bfs_coverage']:.2f}->{rew_bfs['bfs_coverage']:.2f} "
                            f"E {oc['orig_edges']}->{gs['edges_directed']} "
                            f"NMI_emb={_fmt(clust_metrics.get('clust_emb_all_nmi'))} "
                            f"NMI_lp={_fmt(clust_metrics.get('clust_lp_all_nmi'))}")
                    print(f"    [ok] rewired BFS_Cov={_fmt(rew_bfs['bfs_coverage'])} "
                          f"BFS_Steps={rew_bfs['bfs_steps']} LP_Acc={_fmt(rew_lp_metrics['lp_test_acc'])} "
                          f"NMI_emb={_fmt(clust_metrics.get('clust_emb_all_nmi'))} "
                          f"ACC_emb={_fmt(clust_metrics.get('clust_emb_all_acc'))} "
                          f"NMI_lp={_fmt(clust_metrics.get('clust_lp_all_nmi'))} "
                          f"edges={gs['edges_directed']} ({dt:.1f}s)")
                    tracker.tick_done(dt, note=note)

                except Exception as e:
                    dt = time.perf_counter() - t0
                    print(f"    [SKIP run] {method}/{dataset}/seed={seed}: {e}")
                    if getattr(cfg, "verbose", False): traceback.print_exc()
                    store.append({
                        "dataset": dataset, "method": method, "seed": seed,
                        "status": "skipped", "reason": str(e), "rewire_time_s": dt,
                    })
                    tracker.tick_done(dt, note="skipped")

            write_all_tables(store.all(), cfg.datasets, cfg.methods)
        write_all_tables(store.all(), cfg.datasets, cfg.methods)

    tracker.finish()
    write_all_tables(store.all(), cfg.datasets, cfg.methods)
    print(f"\n[Done] Propagation tables, summary, and embeddings in {PROP_OUT_DIR.resolve()}")


# =============================================================================
#  SECTION C4 — CLI
# =============================================================================

def parse_args_propagation(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Propagation benchmark for graph-rewiring methods.")

    p.add_argument("--datasets", nargs="+", default=PROP_DEFAULT_DATASETS)
    p.add_argument("--methods", nargs="+", default=PROP_ALL_METHODS, choices=PROP_ALL_METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # Method-Specific Arguments
    p.add_argument("--idgl_hidden", type=int, default=64)
    p.add_argument("--idgl_num_pers", type=int, default=4)
    p.add_argument("--idgl_epsilon", type=float, default=0.5)
    p.add_argument("--idgl_lambda", type=float, default=0.8)
    p.add_argument("--idgl_eta", type=float, default=0.5)
    p.add_argument("--idgl_max_iter", type=int, default=10)
    p.add_argument("--idgl_eps_adj", type=float, default=1e-4)
    p.add_argument("--idgl_epochs", type=int, default=200)
    p.add_argument("--idgl_lr", type=float, default=1e-3)
    p.add_argument("--idgl_dropout", type=float, default=0.5)
    p.add_argument("--idgl_reg_alpha", type=float, default=1.0)
    p.add_argument("--idgl_reg_beta", type=float, default=1.0)
    p.add_argument("--idgl_reg_gamma", type=float, default=0.5)
    p.add_argument("--idgl_num_anchors", type=int, default=None)

    p.add_argument("--gadc_epsilon", type=float, default=1.0)
    p.add_argument("--gadc_lam", type=float, default=1.0)
    p.add_argument("--gadc_K", type=int, default=16)

    p.add_argument("--lpkg_gae_hid", type=int, default=256)
    p.add_argument("--lpkg_gae_lat", type=int, default=128)
    p.add_argument("--lpkg_gae_lr", type=float, default=1e-4)
    p.add_argument("--lpkg_gae_epochs", type=int, default=200)
    p.add_argument("--lpkg_k", type=int, default=5)
    p.add_argument("--lpkg_knn_batch_size", type=int, default=512)

    p.add_argument("--graphite_binarize", type=str, default="topk", choices=["median", "positive", "topk"])
    p.add_argument("--graphite_topk", type=int, default=10)
    p.add_argument("--graphite_max_feat_edges", type=int, default=2_000_000)

    p.add_argument("--dhgr_min_deg", type=int, default=10)
    p.add_argument("--dhgr_roman_min_deg", type=int, default=3)

    p.add_argument("--comfy_budget_add", type=int, default=100)
    p.add_argument("--comfy_budget_delete", type=int, default=100)
    p.add_argument("--fosr_iterations", type=int, default=10)
    p.add_argument("--fosr_initial_power_iters", type=int, default=5)

    p.add_argument("--glare_outer_loops", type=int, default=25)
    p.add_argument("--glare_gnn_epochs", type=int, default=80)
    p.add_argument("--glare_rewire_steps", type=int, default=120)
    p.add_argument("--glare_neg_edge_budget", type=int, default=10000)
    p.add_argument("--glare_lambda_entropy", type=float, default=0.02)
    p.add_argument("--glare_lambda_sparsity", type=float, default=0.01)
    p.add_argument("--glare_lambda_original", type=float, default=0.10)
    p.add_argument("--glare_rewire_lr", type=float, default=5e-3)
    p.add_argument("--glare_gnn_lr", type=float, default=1e-3)
    p.add_argument("--glare_gnn_hidden", type=int, default=64)
    p.add_argument("--glare_gnn_dropout", type=float, default=0.5)
    p.add_argument("--glare_threshold", type=float, default=0.5)
    p.add_argument("--glare_temperature", type=float, default=1.0)
    p.add_argument("--glare18_lambda_prior", type=float, default=0.10)
    p.add_argument("--glare18_lambda_degree", type=float, default=0.05)
    p.add_argument("--glare18_lambda_hom", type=float, default=0.25)
    p.add_argument("--glare18_struct_mix", type=float, default=0.5)
    p.add_argument("--glare18_kappa_prior", type=float, default=2.0)
    p.add_argument("--glare18_density_ratio", type=float, default=1.0)
    p.add_argument("--glare18_density_adapt_strength", type=float, default=1.0)
    p.add_argument("--glare18_dist_M", type=int, default=2)
    p.add_argument("--glare18_dist_alpha", type=float, default=0.3)
    p.add_argument("--glare18_use_learned_knn", type=int, default=1, choices=[0, 1])
    p.add_argument("--glare18_sim_hidden", type=int, default=64)
    p.add_argument("--glare18_sim_lr", type=float, default=1e-3)
    p.add_argument("--glare18_sim_weight_decay", type=float, default=5e-4)
    p.add_argument("--glare18_sim_pretrain_epochs", type=int, default=200)
    p.add_argument("--glare18_sim_finetune_epochs", type=int, default=30)
    p.add_argument("--glare18_sim_batch_k", type=int, default=512)
    p.add_argument("--glare18_sim_max_iter", type=int, default=20)
    p.add_argument("--glare18_sim_M", type=int, default=2)
    p.add_argument("--glare18_knn_K", type=int, default=8)
    p.add_argument("--glare18_sim_aug_drop", type=float, default=0.2)
    p.add_argument("--glare18_sim_nce_temp", type=float, default=0.5)
    p.add_argument("--glare18_knn_epsilon", type=float, default=0.1)
    p.add_argument("--glare18_knn_block", type=int, default=512)
    p.add_argument("--glare18_learned_affinity_weight", type=float, default=0.5)
    p.add_argument("--glare18_affinity_chunk_size", type=int, default=150_000)
    p.add_argument("--glare18_orig_retain_floor", type=float, default=0.3)
    p.add_argument("--glare18_target_edge_ratio", type=float, default=0.7)
    p.add_argument("--label_mask_ratio", type=float, default=1.0)

    args, _unknown = p.parse_known_args(argv)

    if args.smoke_test:
        print("[Smoke-test mode] reduced epochs, single seed")
        args.seeds = args.seeds[:1]
        args.idgl_epochs = min(args.idgl_epochs, 15)
        args.idgl_max_iter = min(args.idgl_max_iter, 2)
        args.lpkg_gae_epochs = min(args.lpkg_gae_epochs, 15)
        args.glare_outer_loops = min(args.glare_outer_loops, 3)
        args.glare_gnn_epochs = min(args.glare_gnn_epochs, 10)
        args.glare_rewire_steps = min(args.glare_rewire_steps, 10)
        args.glare18_sim_pretrain_epochs = min(args.glare18_sim_pretrain_epochs, 10)
        args.glare18_sim_finetune_epochs = min(args.glare18_sim_finetune_epochs, 5)

    return args

if __name__ == "__main__":
    _cfg = parse_args_propagation()
    run_propagation_benchmark(_cfg)