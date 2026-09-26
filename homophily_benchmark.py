"""
homophily_benchmark.py
======================
Single-file homophily / graph-statistics benchmark for graph-rewiring methods.

WHAT THIS DOES
--------------
For each (dataset, method, seed) it REWIRES the graph with the method's own
rewiring / representation code and then reports STANDARD homophily measures and
basic graph statistics — for both the ORIGINAL and the REWIRED graph. It does
NOT train any downstream classifier and reports no accuracy.

Methods (model architectures are reproduced VERBATIM — nothing changed):
    glare, idgl, gadc, lpkg, dhgr, graphite, fosr, comfy

Measures (all standard / published):
    LABEL-BASED
      * edge homophily            (Zhu et al. 2020)
      * node homophily            (Pei et al. 2020, Geom-GCN)
      * adjusted homophily        (Platonov et al. 2023 = assortativity coeff.,
                                    degree-weighted — the correct definition)
      * class-insensitive edge homophily (Lim et al. 2021)
      * label informativeness LI  (Platonov et al. 2023)
    FEATURE-BASED (generalised edge homophily, Jin et al. 2022)
      * on the ORIGINAL features, and
      * on the method's LEARNED EMBEDDING when it produces one
        (GLARE -> Z, GADC -> diffused F, LPkG -> GAE latent, IDGL -> hidden h).

Per-method notes:
    * GADC / GRAPHITE do not rewire the node-node structure (GADC diffuses
      features; GRAPHITE adds bipartite feature nodes) — structural homophily
      equals the original, and this is flagged in every record.
    * IDGL learns a soft dense adjacency; a discrete rewired graph is obtained
      by top-k sparsification (dense case) or cosine-kNN over the learned
      hidden (anchor case, N>2000). Documented measurement choice, not a model
      change.

OUTPUT (folder: homophily_results/)
    results.jsonl                       one JSON record per finished run
    original_measures.json              cached per-dataset original measures
    SUMMARY.md                          pivoted human-readable summary
    tables/original_graph_measures.csv
    tables/table_<measure>.csv          one file per measure (orig/rewired/delta)
    tables/table_feature_homophily.csv
    tables/table_graph_stats.csv
    tables/all_results_flat.csv

RESUME
    Runs are keyed by (dataset, method, seed) and appended durably; interrupting
    and re-running continues exactly where it stopped ("run one, write one").

USAGE
    python homophily_benchmark.py                     # all methods, 1 seed
    python homophily_benchmark.py --smoke_test
    python homophily_benchmark.py --methods glare fosr --datasets Actor
    python homophily_benchmark.py --seeds 0 1 2 --device cuda

REQUIREMENTS
    Place the DHGR/ repo and the common/ package next to this file (as in the
    original project). torch, torch_geometric, scipy, scikit-learn, networkx,
    numpy are required; pandas/matplotlib are pulled in by common/ but the
    homophily tables themselves use only the csv module.
"""

from __future__ import annotations



import argparse
import json
import time
import traceback
import warnings
import copy
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import copy
import networkx as nx

from torch_geometric.utils import to_networkx, from_networkx

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

# ─────────────────────────────────────────────────────────────────────────────
OUT_DIR = Path("others_benchmark_results_homophily")
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = OUT_DIR / "results_faithful.jsonl"
SUMMARY_CSV  = OUT_DIR / "summary_faithful.csv"

SPLITS_DIR = OUT_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def save_split(dataset_name, data, splits_dir=SPLITS_DIR):
    """
    Save the train/val/test masks used for `dataset_name` to disk, so the
    exact split behind any reported result can be inspected or reloaded
    later without re-running the (seeded) split logic.

    NOTE: `load_real_dataset` / `generate_synthetic_dataset` build masks
    with an internally fixed seed (0), independent of the benchmark's
    per-method `seed` loop in `run_faithful_benchmark`. So the split is
    identical for every method/seed run against a given dataset in a
    single execution of this script — there is exactly one split per
    dataset, not one per (dataset, seed). We therefore save one file per
    dataset and skip re-saving if it already exists (so re-running with
    --resume doesn't rewrite it).

    Saves both:
      - splits/<dataset>.npz  (train_mask, val_mask, test_mask as bool arrays;
                                framework-agnostic, easy to inspect with numpy)
      - splits/<dataset>.pt   (same three masks as torch bool tensors, for
                                direct reuse in this codebase)
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
#  SECTION 1 — Dataset loading (identical to benchmark_rewiring_v15.py)
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
    """
    Symmetric-normalised adjacency: Ã = D̃^{-1/2}(A+I)D̃^{-1/2}
    Returns a torch sparse COO tensor.
    """
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
    """Row-normalised adjacency: D^{-1}(A+I). Returns sparse COO tensor."""
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


def save_result(rec):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(rec) + "\n")


def load_results():
    if not RESULTS_FILE.exists():
        return []
    with open(RESULTS_FILE) as f:
        return [json.loads(l) for l in f if l.strip()]


def result_exists(dataset, method, seed):
    return any(
        r.get("dataset") == dataset
        and r.get("method") == method
        and r.get("seed") == seed
        for r in load_results()
    )


def build_summary(results):
    import pandas as pd
    agg = defaultdict(lambda: {"vals": [], "tests": []})
    for r in results:
        agg[(r["dataset"], r["method"])]["vals"].append(r["val"])
        agg[(r["dataset"], r["method"])]["tests"].append(r["test"])
    rows = []
    for (ds, m), v in agg.items():
        rows.append({
            "dataset":    ds,
            "method":     m,
            "val_mean":   np.mean(v["vals"]),
            "val_std":    np.std(v["vals"]),
            "test_mean":  np.mean(v["tests"]),
            "test_std":   np.std(v["tests"]),
            "n_seeds":    len(v["tests"]),
        })
    return pd.DataFrame(rows)


def print_table(df, dataset):
    sub = df[df["dataset"] == dataset].copy()
    if sub.empty:
        return
    print(f"\n  {'method':<35} {'val_mean':>9} {'val_std':>8} {'test_mean':>10} {'test_std':>9}")
    print("  " + "-" * 75)
    for _, row in sub.iterrows():
        print(f"  {row['method']:<35} {row['val_mean']:>9.4f} {row['val_std']:>8.4f} "
              f"{row['test_mean']:>10.4f} {row['test_std']:>9.4f}")


# =============================================================================
#  SECTION 3 — IDGL faithful implementation
#  Paper: Chen et al., NeurIPS 2020, Algorithm 1
#
#  Key design decisions from the paper:
#  • Graph learner: multi-head weighted cosine similarity (Eq. 1)
#  • Epsilon-neighbourhood sparsification → sparse non-negative adjacency
#  • Graph combination (Eq. 3): Ā = λ·L⁰ + (1-λ)·[η·f(Aᵗ) + (1-η)·f(A¹)]
#  • GNN: two-layer GCN — MP(F, Ā) = Ā·F·W  (not SAGEConv)
#  • Prediction loss: cross-entropy on train nodes only
#  • Graph reg loss (Eq. 10–11): smoothness (Dirichlet) + connectivity (log)
#    + sparsity (Frobenius)
#  • Iterative stopping: ||Aᵗ-Aᵗ⁻¹||²_F / ||A¹||²_F < ε_adj
#  • Overall loss: L = L¹_pred + Σ_{t≥2} L^t / (T-1)  [averaged per Alg 1]
#  • Final test evaluation: argmax(Z) on test_mask from best val checkpoint
#  • IDGL-ANCH used for N > 2000 (anchor approx, Eqs 7-9)
# =============================================================================

class _IDGLGraphLearner(nn.Module):
    """Multi-head weighted cosine similarity learner — IDGL Eq. (1)."""
    EPS = 1e-12

    def __init__(self, in_dim, num_pers=4, epsilon=0.5):
        super().__init__()
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_pers, in_dim)))
        self.epsilon = epsilon

    def forward(self, x):
        # x: (N, d)  →  S: (N, N) row-normalised, sparse
        xw  = x.unsqueeze(0) * self.W.unsqueeze(1)          # (P, N, d)
        xn  = F.normalize(xw, p=2, dim=-1)
        S   = torch.matmul(xn, xn.transpose(-1, -2)).mean(0) # (N, N)
        S   = S * (S > self.epsilon).float()                 # ε-neighbourhood
        rs  = S.sum(-1, keepdim=True).clamp(min=self.EPS)
        return S / rs, S   # (normalised, unnormalised)


class _IDGLAnchorLearner(nn.Module):
    """Anchor-based multi-head weighted cosine similarity — IDGL Eq. (2)."""
    EPS = 1e-12

    def __init__(self, in_dim, num_anchors, num_pers=4, epsilon=0.1):
        super().__init__()
        self.W       = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(num_pers, in_dim)))
        self.epsilon = epsilon

    def forward(self, x, x_anc):
        # x: (N,d)  x_anc: (s,d)  →  R: (N,s) normalised
        xw  = x.unsqueeze(0)     * self.W.unsqueeze(1)   # (P,N,d)
        aw  = x_anc.unsqueeze(0) * self.W.unsqueeze(1)   # (P,s,d)
        xn  = F.normalize(xw, p=2, dim=-1)
        an  = F.normalize(aw, p=2, dim=-1)
        R   = torch.bmm(xn, an.transpose(-1, -2)).mean(0) # (N,s)
        R   = R * (R > self.epsilon).float()
        rs  = R.sum(-1, keepdim=True).clamp(min=self.EPS)
        return R / rs, R


def _idgl_graph_reg(A_raw, X, alpha=1.0, beta=1.0, gamma=0.5):
    """
    Graph regularisation — IDGL Eq. (10–11).
      L_G = alpha * Ω(A,X)  +  beta * f(A)
    where:
      Ω(A,X) = (1/n²) tr(Xᵀ L X)        [Dirichlet energy / smoothness]
      f(A)   = -(γ/n) 1ᵀ log(A1)         [connectivity — log barrier]
               + (β_sp/n²) ||A||²_F      [sparsity]
    Using the paper's notation: alpha=α, beta for connectivity, gamma for sparsity.
    """
    N  = X.shape[0]
    # Official add_graph_loss does not symmetrize A — use the raw
    # (possibly asymmetric) learned adjacency directly, matching
    # model_handler.py::add_graph_loss in the official repo.
    A  = A_raw
    deg = A.sum(-1)                        # (N,)
    # Smoothness: (1/n²) tr(X^T L X)  where L = diag(deg) - A  [paper Eq. 10]
    LX  = deg.unsqueeze(-1) * X - torch.mm(A, X)
    l_smooth = alpha * (X * LX).sum() / (N * N)
    # Connectivity: -(γ/n) · Σ log(deg_i) = -γ · mean(log(deg_i))  [paper Eq. 11]
    l_conn = -beta  * torch.log(deg.clamp(min=1e-12)).mean()
    # Sparsity: (β/n²) ||A||²_F  [paper Eq. 11]
    l_spar =  gamma * A.pow(2).sum() / (N * N)
    return l_smooth + l_conn + l_spar


def _idgl_anchor_reg(R_raw, X_anc, alpha=1.0, beta=1.0, gamma=0.5):
    """
    Anchor-graph regularisation — IDGL Section 2.5.
    Applied to unnormalised anchor adjacency B̂ = Rᵀ Δ⁻¹ R  (s×s).
    """
    EPS = 1e-12
    s   = R_raw.size(1)
    Delta_inv = 1.0 / R_raw.sum(dim=1).clamp(min=EPS)   # (N,)
    B_hat = R_raw.t() @ (Delta_inv.unsqueeze(-1) * R_raw) # (s,s)
    deg_b = B_hat.sum(-1)
    L_b   = torch.diag(deg_b) - B_hat
    # Smoothness: (1/2s²) tr(X^T L X)  [paper Eq. 10 applied to anchor graph]
    l_sm  = alpha * torch.trace(X_anc.t() @ (L_b @ X_anc)) / (2.0 * s * s)
    # Connectivity: -(γ/s) Σ log(deg_b_k) = -γ · mean(log(deg_b))  [paper Eq. 11]
    l_conn = -beta  * torch.log(deg_b.clamp(min=EPS)).mean()
    # Sparsity: (β/s²) ||B̂||²_F
    l_spar =  gamma * B_hat.pow(2).sum() / (s * s)
    return l_sm + l_conn + l_spar


def _idgl_anchor_mp(x, R_norm, Lam_inv, Del_inv):
    """Two-step anchor message passing MP₁₂ — IDGL Eqs. (7–9)."""
    f1 = Lam_inv.unsqueeze(-1) * (R_norm.t() @ x)   # anchor aggregation
    return Del_inv.unsqueeze(-1) * (R_norm @ f1)     # node aggregation


def _sample_anchors_idgl(N, s, edge_index, device):
    from torch_geometric.utils import degree as pyg_deg
    deg  = pyg_deg(edge_index[1], num_nodes=N).cpu().numpy().astype(np.float64)
    deg  = np.maximum(deg, 1.0)
    prob = deg / deg.sum()
    idx  = np.random.choice(N, size=s, replace=False, p=prob)
    return torch.tensor(idx, dtype=torch.long, device=device)


def run_idgl(data, cfg, device, seed):
    """
    IDGL / IDGL-ANCH faithful to Algorithm 1 (Chen et al., NeurIPS 2020).

    The GNN inside IDGL IS the classifier — no separate downstream model.
    Returns (val_acc, test_acc).

    Hyper-parameters (paper defaults where stated):
      num_pers=4, epsilon=0.5, graph_skip_conn λ=0.8, eta η=0.5
      max_iter T=10, eps_adj=1e-4
      graph_reg: alpha=1.0 (smoothness), beta=1.0 (connectivity), gamma=0.5 (sparsity)
      GCN hidden=64, dropout=0.5, lr=1e-3, epochs=200, weight_decay=5e-4
    """
    torch.manual_seed(seed); np.random.seed(seed)

    N  = data.num_nodes
    C  = data.num_classes
    X  = data.x.float().to(device)
    y  = data.y.to(device)
    tm = data.train_mask.to(device)
    vm = data.val_mask.to(device)
    te = data.test_mask.to(device)

    # Hyper-parameters (paper defaults)
    hidden       = cfg.idgl_hidden       # 64
    num_pers     = cfg.idgl_num_pers     # 4
    epsilon      = cfg.idgl_epsilon      # 0.5
    lam          = cfg.idgl_lambda       # graph_skip_conn λ=0.8
    eta          = cfg.idgl_eta          # η=0.5
    max_iter     = cfg.idgl_max_iter     # 10
    eps_adj      = cfg.idgl_eps_adj      # 1e-4
    epochs       = cfg.idgl_epochs       # 200
    lr           = cfg.idgl_lr           # 1e-3
    dropout      = cfg.idgl_dropout      # 0.5
    reg_alpha    = cfg.idgl_reg_alpha    # 1.0
    reg_beta     = cfg.idgl_reg_beta     # 1.0
    reg_gamma    = cfg.idgl_reg_gamma    # 0.5
    USE_ANCH     = N > 2000
    s            = min(cfg.idgl_num_anchors or 300, N // 4, N)

    # ── Initial graph L⁰ (row-normalised adjacency + self-loops) ─────────────
    A0_sp  = (_to_scipy(data.edge_index.cpu(), N).astype(np.float32)
               + sp.eye(N, format="csr"))
    deg0   = np.asarray(A0_sp.sum(1)).reshape(-1)
    d_inv  = np.where(deg0 > 0, 1.0 / deg0, 0.0)
    L0_sp  = (sp.diags(d_inv) @ A0_sp).tocoo().astype(np.float32)
    L0_ei  = torch.tensor(np.stack([L0_sp.row, L0_sp.col]), dtype=torch.long,  device=device)
    L0_val = torch.tensor(L0_sp.data, dtype=torch.float32, device=device)
    L0_t   = torch.sparse_coo_tensor(L0_ei, L0_val, (N, N), device=device).coalesce()

    def sparse_mp(x):
        return torch.sparse.mm(L0_t, x)

    print(f"  [IDGL] {'ANCH' if USE_ANCH else 'Full'} N={N}"
          + (f" s={s}" if USE_ANCH else "")
          + f" epochs={epochs}")

    # ── Model components ──────────────────────────────────────────────────────
    W1 = nn.Linear(X.shape[1], hidden, bias=False).to(device)
    W2 = nn.Linear(hidden,     C,      bias=False).to(device)

    if not USE_ANCH:
        # Full IDGL — dense N×N adjacency
        L0_dense = torch.tensor(
            (sp.diags(d_inv) @ A0_sp).toarray(), dtype=torch.float32, device=device)
        gl1 = _IDGLGraphLearner(X.shape[1], num_pers, epsilon).to(device)
        gl2 = _IDGLGraphLearner(hidden,     num_pers, epsilon).to(device)

        def combine(At_n, A1_n):
            # Eq. (3): Ā = λ·L⁰ + (1-λ)·[η·f(Aᵗ) + (1-η)·f(A¹)]
            merged = eta * At_n + (1.0 - eta) * A1_n
            return lam * L0_dense + (1.0 - lam) * merged

        def mp(A, x):
            return torch.mm(A, x)

        opt = Adam(
            list(gl1.parameters()) + list(gl2.parameters()) +
            list(W1.parameters())  + list(W2.parameters()),
            lr=lr, weight_decay=5e-4)

        best_val = -1.0; best_Z = None; best_state = None

        for epoch in range(epochs):
            gl1.train(); gl2.train(); W1.train(); W2.train()
            opt.zero_grad()

            # Iteration 1 — graph from raw features
            A1n, A1r = gl1(X)
            Ab1 = combine(A1n, A1n)
            h   = F.relu(F.dropout(mp(Ab1, W1(X)), dropout, training=True))
            Z   = mp(Ab1, W2(h))
            loss = F.cross_entropy(Z[tm], y[tm]) + _idgl_graph_reg(A1r, X, reg_alpha, reg_beta, reg_gamma)
            prev = A1r.detach()

            # Iterations 2..T — graph from updated embeddings
            iter_losses = []
            for _ in range(max_iter):
                Atn, Atr = gl2(h.detach())
                Abt = combine(Atn, A1n)
                h2  = F.relu(F.dropout(mp(Abt, W1(X)), dropout, training=True))
                Z2  = mp(Abt, W2(h2))
                l_t = F.cross_entropy(Z2[tm], y[tm]) + _idgl_graph_reg(Atr, X, reg_alpha, reg_beta, reg_gamma)
                iter_losses.append(l_t)
                diff  = (Atr.detach() - prev).pow(2).sum()
                # Official stopping criterion (paper Algorithm 1, line 6, and
                # matching model_handler.py::diff in the official repo):
                # for standard IDGL the denominator is fixed at the FIRST
                # iteration's adjacency ||A^1||_F^2, not the current one.
                # (Only IDGL-ANCH uses the current-iteration denominator.)
                denom = A1r.detach().pow(2).sum().clamp(min=1e-12)
                prev  = Atr.detach(); h = h2; Z = Z2
                if (diff / denom).item() < eps_adj:
                    break

            # Overall loss: L = L(1) + Σ_{t=2}^{T} L(t) / (T-1)  [Algorithm 1 line 21]
            # .mean() is equivalent: iter_losses has exactly (T-1) entries.
            if iter_losses:
                loss = loss + torch.stack(iter_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(gl1.parameters()) + list(gl2.parameters()), 1.0)
            opt.step()

            # Bug fix: Z is updated to Z2 inside the inner loop (h = h2; Z = Z2),
            # so by here Z already holds the final inner-iteration output. Correct.
            with torch.no_grad():
                va = (Z.argmax(-1)[vm] == y[vm]).float().mean().item()
                if va > best_val:
                    best_val = va
                    best_Z   = Z.detach().cpu()
                    best_state = {
                        k: v.clone() for k, v in
                        dict(**dict(gl1.named_parameters()),
                             **dict(gl2.named_parameters()),
                             **dict(W1.named_parameters()),
                             **dict(W2.named_parameters())).items()
                    }

        # Test accuracy from best val checkpoint
        if best_Z is None:
            best_Z = Z.detach().cpu()
        _pred = best_Z.argmax(-1)[te.cpu()]
        _true = y[te].cpu()
        test_acc = (_pred == _true).float().mean().item()
        _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, _true)
        return best_val, test_acc, test_f1

    else:
        # IDGL-ANCH — anchor approximation
        anchor_idx = _sample_anchors_idgl(N, s, data.edge_index, device)
        gl1 = _IDGLAnchorLearner(X.shape[1], s, num_pers, epsilon * 0.2).to(device)
        gl2 = _IDGLAnchorLearner(hidden,     s, num_pers, epsilon * 0.2).to(device)

        opt = Adam(
            list(gl1.parameters()) + list(gl2.parameters()) +
            list(W1.parameters())  + list(W2.parameters()),
            lr=lr, weight_decay=5e-4)

        best_val = -1.0; best_Z = None

        for epoch in range(epochs):
            gl1.train(); gl2.train(); W1.train(); W2.train()
            opt.zero_grad()

            x_anc = X[anchor_idx]
            R1n, R1r = gl1(X, x_anc)
            Lam1 = R1r.sum(0).clamp(min=1e-12)   # (s,)
            Del1 = R1r.sum(1).clamp(min=1e-12)   # (N,)

            # Eq. (9): Ā·F = λ·L⁰·F + (1-λ)·MP₁₂(F, R¹)
            # Paper Eq.4: H1 = relu(Ā·X·W1). W1 is linear, so no activation before propagation.
            h0_lin = W1(X)   # linear transform only — no relu here (matches paper GCN Eq.4)
            mp1 = _idgl_anchor_mp(h0_lin, R1n, 1.0/Lam1, 1.0/Del1)
            h   = F.relu(F.dropout(
                lam * sparse_mp(h0_lin) + (1.0 - lam) * mp1, dropout, training=True))
            mp2 = _idgl_anchor_mp(h, R1n, 1.0/Lam1, 1.0/Del1)
            Z   = W2(lam * sparse_mp(h) + (1.0 - lam) * mp2)

            loss = (F.cross_entropy(Z[tm], y[tm])
                    + _idgl_anchor_reg(R1r, x_anc, reg_alpha, reg_beta, reg_gamma))
            prev_R = R1r.detach()

            iter_losses = []
            for _ in range(max_iter):
                x_anc_h = h[anchor_idx].detach()
                Rtn, Rtr = gl2(h.detach(), x_anc_h)
                Lamt = Rtr.sum(0).clamp(min=1e-12)
                Delt = Rtr.sum(1).clamp(min=1e-12)

                def comb_mp(v):
                    return (lam * sparse_mp(v) + (1.0 - lam) * (
                        eta * _idgl_anchor_mp(v, Rtn, 1.0/Lamt, 1.0/Delt) +
                        (1.0 - eta) * _idgl_anchor_mp(v, R1n,  1.0/Lam1, 1.0/Del1)))

                h2  = F.relu(F.dropout(comb_mp(W1(X)), dropout, training=True))
                Z2  = W2(comb_mp(h2))
                l_t = (F.cross_entropy(Z2[tm], y[tm])
                       + _idgl_anchor_reg(Rtr, x_anc_h, reg_alpha, reg_beta, reg_gamma))
                iter_losses.append(l_t)
                diff  = (Rtr.detach() - prev_R).pow(2).sum()
                # IDGL-ANCH correctly uses the current-iteration denominator
                # ||R^t||_F^2, per paper Algorithm 1 line 6 / official repo.
                denom = Rtr.detach().pow(2).sum().clamp(min=1e-12)
                prev_R = Rtr.detach(); h = h2; Z = Z2
                if (diff / denom).item() < eps_adj:
                    break

            # Bug fix: divide by (t-1) same as full IDGL — .mean() is equivalent
            if iter_losses:
                loss = loss + torch.stack(iter_losses).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(gl1.parameters()) + list(gl2.parameters()), 1.0)
            opt.step()

            with torch.no_grad():
                va = (Z.argmax(-1)[vm] == y[vm]).float().mean().item()
                if va > best_val:
                    best_val = va
                    best_Z   = Z.detach().cpu()

        if best_Z is None:
            best_Z = Z.detach().cpu()
        _pred = best_Z.argmax(-1)[te.cpu()]
        _true = y[te].cpu()
        test_acc = (_pred == _true).float().mean().item()
        _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, _true)
        return best_val, test_acc, test_f1


# =============================================================================
#  SECTION 4 — GADC faithful implementation
#  Paper: Liu et al., ICML 2024, Algorithm 1 + Section 3.4 Option (I)
#
#  Key design decisions from the paper:
#  • Option (I) for heterophilic graphs (Section 3.4, Table 14 ablation):
#      Φᵢⱼ = cosine_sim(Xᵢ, Xⱼ)  if (i,j) ∈ E,  else 0
#  • T = Ã - ε·Φ  where Ã = D̃^{-1/2}(A+I)D̃^{-1/2}
#  • S = 1/(λ+1) · Σₖ₌₀ᴷ (λ/(λ+1)·T)ᵏ  (Eq. 15)
#  • F = S·X  (pre-computed ONCE, then frozen)
#  • Downstream classifier: 2-layer MLP with 64 hidden units
#    (paper Table 14 / Appendix E: "2-layer MLP with 64 hidden units")
#  • Decoupled training: aggregation happens once, only MLP parameters update
#  • Paper hyperparameters for heterophilic (Table 14): λ=1, K=16, ε=1.0
# =============================================================================

def run_gadc(data, cfg, device, seed):
    """
    GADC faithful to the original GADC implementation,
    while retaining the benchmark's own train/val/test split
    and benchmark-defined number of epochs.

    Main faithful changes:
      1. Row-normalize features before cosine similarity and diffusion.
      2. Use Dropout -> Linear -> ReLU -> Dropout -> Linear.
      3. Use the original GADC learning rate: 0.01.

    The benchmark's number of epochs is intentionally retained.

    Returns:
        (val_acc, test_acc)
    """

    # ============================================================
    # Random seed
    # ============================================================
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ============================================================
    # Basic information
    # ============================================================
    N = data.num_nodes
    C = data.num_classes

    # Original feature matrix
    X = data.x.float().cpu().numpy().astype(np.float32)

    # GADC hyperparameters
    lam = cfg.gadc_lam       # λ = 1
    K   = cfg.gadc_K         # K = 16
    eps = cfg.gadc_epsilon   # ε = 1.0

    # ============================================================
    # Step 1: Original adjacency
    # ============================================================
    A_sp = _to_scipy(
        data.edge_index.cpu(),
        N
    ).astype(np.float32)

    # Remove any existing self-loops.
    # Cosine similarity is calculated only on the original
    # graph edges.
    A_sp = A_sp - sp.diags(A_sp.diagonal())

    # ============================================================
    # Step 2: Symmetrically normalized adjacency
    # ============================================================
    #
    # A_tilde = A + I
    #
    # A_sym = D^(-1/2) A_tilde D^(-1/2)
    #
    # This follows the normalize_adj() operation in the
    # original GADC repository.
    # ============================================================

    A_tilde = A_sp + sp.eye(
        N,
        format="csr",
        dtype=np.float32
    )

    deg_t = np.asarray(
        A_tilde.sum(1)
    ).reshape(-1)

    d_inv = np.where(
        deg_t > 0,
        deg_t ** -0.5,
        0.0
    )

    D_inv = sp.diags(d_inv)

    A_sym = (
        D_inv @ A_tilde @ D_inv
    ).tocoo().astype(np.float32)

    # ============================================================
    # Step 3: ROW-NORMALIZE FEATURES
    # ============================================================
    #
    # Original GitHub implementation uses:
    #
    # feature_tensor_normalize(features)
    #
    # which performs:
    #
    # X_i <- X_i / sum_j X_ij
    #
    # before GADC diffusion.
    #
    # This is an important difference from the previous
    # benchmark implementation.
    # ============================================================

    X_d = torch.tensor(
        X,
        dtype=torch.float32,
        device=device
    )

    row_sum = torch.sum(
        X_d,
        dim=1,
        keepdim=True
    )

    # Equivalent to the GitHub handling of zero rows:
    # 1 / 0 -> inf -> 0
    row_inv = torch.where(
        row_sum != 0,
        1.0 / row_sum,
        torch.zeros_like(row_sum)
    )

    X_d = row_inv * X_d

    # Convert the normalized features back to NumPy
    # because cosine similarity below is computed with NumPy.
    X = X_d.cpu().numpy()

    # ============================================================
    # Step 4: Φ — cosine similarity on ORIGINAL edges
    # ============================================================
    #
    # For every original edge (i,j):
    #
    # Φ_ij =
    #       X_i^T X_j
    #       -------------
    #       ||X_i|| ||X_j||
    #
    # Importantly, row/col come from the ORIGINAL adjacency,
    # not the normalized adjacency.
    # ============================================================

    A_orig_coo = A_sp.tocoo()

    row = A_orig_coo.row.astype(np.int64)
    col = A_orig_coo.col.astype(np.int64)

    xi = X[row]
    xj = X[col]

    ni = np.linalg.norm(
        xi,
        axis=1
    ).clip(min=1e-12)

    nj = np.linalg.norm(
        xj,
        axis=1
    ).clip(min=1e-12)

    cos = (
        np.sum(xi * xj, axis=1)
        / (ni * nj)
    )

    Phi = sp.coo_matrix(
        (
            eps * cos,
            (row, col)
        ),
        shape=(N, N),
        dtype=np.float32
    ).tocsr()

    # ============================================================
    # Step 5: Modified transition matrix
    # ============================================================
    #
    # T = A_sym - ε Φ
    # ============================================================

    T_coo = (
        A_sym.astype(np.float32).tocsr()
        - Phi
    ).tocoo()

    # ============================================================
    # Step 6: Neumann graph diffusion
    # ============================================================
    #
    # S =
    #
    # 1/(1+λ) *
    # Σ_{k=0}^K
    # [ λ/(1+λ) T ]^k
    #
    # F = S X
    #
    # This is the same iterative diffusion used in the
    # original GADC implementation.
    # ============================================================

    neumann = lam / (1.0 + lam)

    T_torch = torch.sparse_coo_tensor(
        torch.tensor(
            np.stack(
                [T_coo.row, T_coo.col]
            ),
            dtype=torch.long
        ),
        torch.tensor(
            T_coo.data * neumann,
            dtype=torch.float32
        ),
        (N, N)
    ).coalesce().to(device)

    # X_d is already the ROW-NORMALIZED feature matrix.
    F_d = X_d.clone()
    cur = X_d.clone()

    for _ in range(K):

        cur = torch.sparse.mm(
            T_torch,
            cur
        )

        F_d = F_d + cur

    F_d = F_d / (1.0 + lam)

    # GADC diffusion is pre-computed.
    F = F_d.detach()

    print(
        f"  [GADC] Diffusion done. "
        f"F shape: {tuple(F.shape)}"
    )

    # ============================================================
    # Step 7: 2-layer MLP
    # ============================================================
    #
    # Original GitHub MLP:
    #
    # Dropout
    #    ↓
    # Linear
    #    ↓
    # ReLU
    #    ↓
    # Dropout
    #    ↓
    # Linear
    #
    # We therefore place the first dropout BEFORE the first
    # linear layer.
    # ============================================================

    hid = cfg.gadc_mlp_hidden   # 64

    mlp = nn.Sequential(

        # First dropout -- matches GitHub
        nn.Dropout(cfg.gadc_dropout),

        # First linear layer
        nn.Linear(
            F.shape[1],
            hid
        ),

        # Non-linearity
        nn.ReLU(),

        # Second dropout -- matches GitHub
        nn.Dropout(cfg.gadc_dropout),

        # Output layer
        nn.Linear(
            hid,
            C
        ),
    ).to(device)

    # ============================================================
    # Labels and masks
    # ============================================================

    y = data.y.to(device)

    tm = data.train_mask.to(device)
    vm = data.val_mask.to(device)
    te = data.test_mask.to(device)

    # ============================================================
    # Step 8: Optimizer
    # ============================================================
    #
    # Original GitHub:
    #
    # Adam(lr=0.01, weight_decay=5e-4)
    #
    # We retain cfg.gadc_wd so your benchmark can keep its
    # existing weight-decay setting if desired.
    # ============================================================

    opt = Adam(
        mlp.parameters(),
        lr=0.01,
        weight_decay=cfg.gadc_wd
    )

    # ============================================================
    # Step 9: Training
    # ============================================================
    #
    # IMPORTANT:
    # We intentionally keep cfg.gadc_epochs because you said
    # the benchmark's epoch count should NOT be changed.
    # ============================================================

    best_val = -1.0
    best_logits = None

    for epoch in range(cfg.gadc_epochs):

        # -----------------------------
        # Training
        # -----------------------------
        mlp.train()

        opt.zero_grad()

        logits = mlp(F)

        loss = torch.nn.functional.cross_entropy(
            logits[tm],
            y[tm]
        )

        loss.backward()

        opt.step()

        # -----------------------------
        # Validation
        # -----------------------------
        #
        # Kept every 10 epochs because this is part of your
        # current benchmark implementation.
        # -----------------------------

        if (epoch + 1) % 10 == 0:

            mlp.eval()

            with torch.no_grad():

                logits_eval = mlp(F)

                va = (
                    logits_eval[vm].argmax(-1)
                    == y[vm]
                ).float().mean().item()

            if va > best_val:

                best_val = va

                best_logits = (
                    logits_eval
                    .detach()
                    .cpu()
                )

    # ============================================================
    # Fallback if validation checkpoint was never saved
    # ============================================================

    if best_logits is None:

        mlp.eval()

        with torch.no_grad():

            best_logits = (
                mlp(F)
                .detach()
                .cpu()
            )

    # ============================================================
    # Step 10: Test accuracy
    # ============================================================

    _pred = best_logits.argmax(-1)[te.cpu()]
    _true = y[te].cpu()
    test_acc = (_pred == _true).float().mean().item()
    _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, _true)

    return best_val, test_acc, test_f1

# ============================================================================
# LPkG — Faithful implementation for benchmark use
# ============================================================================
#
# Paper:
# Hyun Seok Park, Ha-Myung Park
# "Enhancing Heterophilic Graph Neural Network Performance through
#  Label Propagation in K-Nearest Neighbor Graphs"
# IEEE BigComp 2024
#
# Paper pipeline:
#
#   (A, X)
#      |
#      v
#   Feature-reconstruction GAE
#      |
#      v
#   latent representation X'
#      |
#      v
#   cosine k-NN graph A_hat
#      |
#      v
#   Label Propagation
#      |
#      v
#   Z_prob
#
#   (A, X)
#      |
#      v
#   FSGNN
#      |
#      v
#   Z_pred
#
#   Z_new = (1-beta) Z_pred + beta Z_prob
#
# ============================================================================

import copy
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch_geometric.nn import GCNConv


# ============================================================================
# 1. Reproducibility
# ============================================================================

def _lpkg_set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# 2. Sparse normalized adjacency
# ============================================================================

def _lpkg_normalized_adjacency(
    edge_index,
    num_nodes,
    add_self_loops=True,
    device=None
):
    """
    Construct:

        D^(-1/2) A D^(-1/2)

    If add_self_loops=True:

        A <- A + I

    LPkG requires the symmetrically normalized adjacency with
    self-connections for label propagation.

    FSGNN separately requires:
        A_sym
        A_sym + I
    """

    if device is None:
        device = edge_index.device

    edge_index = edge_index.to(device)

    row = edge_index[0]
    col = edge_index[1]

    if add_self_loops:
        loops = torch.arange(
            num_nodes,
            device=device,
            dtype=torch.long
        )

        row = torch.cat([row, loops])
        col = torch.cat([col, loops])

    values = torch.ones(
        row.numel(),
        dtype=torch.float32,
        device=device
    )

    A = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0),
        values,
        size=(num_nodes, num_nodes),
        device=device
    ).coalesce()

    idx = A.indices()
    val = A.values()

    degree = torch.zeros(
        num_nodes,
        dtype=torch.float32,
        device=device
    )

    degree.scatter_add_(
        0,
        idx[0],
        val
    )

    deg_inv_sqrt = degree.clamp_min(1e-12).pow(-0.5)

    norm_values = (
        deg_inv_sqrt[idx[0]]
        * val
        * deg_inv_sqrt[idx[1]]
    )

    A_norm = torch.sparse_coo_tensor(
        idx,
        norm_values,
        size=(num_nodes, num_nodes),
        device=device
    ).coalesce()

    return A_norm


# ============================================================================
# 3. LPkG GAE
# ============================================================================

class _LPkGGAE(nn.Module):
    """
    GAE used in LPkG.

    Paper Section IV-A:

        Encoder:
            GCN -> ReLU -> GCN

        Decoder:
            FC -> sigmoid -> FC

        Reconstruction:
            sigmoid(X_hat)

        Loss:
            MSE(X, sigmoid(X_hat))
    """

    def __init__(self, in_dim, hid_dim, lat_dim):
        super().__init__()

        self.enc1 = GCNConv(
            in_dim,
            hid_dim
        )

        self.enc2 = GCNConv(
            hid_dim,
            lat_dim
        )

        self.dec1 = nn.Linear(
            lat_dim,
            hid_dim
        )

        self.dec2 = nn.Linear(
            hid_dim,
            in_dim
        )

    def encode(self, x, edge_index):

        h = self.enc1(
            x,
            edge_index
        )

        h = torch.relu(h)

        z = self.enc2(
            h,
            edge_index
        )

        return z

    def decode(self, z):

        h = self.dec1(z)

        h = torch.sigmoid(h)

        x_hat = self.dec2(h)

        return x_hat

    def forward(self, x, edge_index):

        z = self.encode(
            x,
            edge_index
        )

        x_hat = self.decode(z)

        # Paper:
        #
        # l(X, X_hat)
        # =
        # 1/n sum_i (X_i - sigmoid(X_hat_i))^2

        reconstruction = torch.sigmoid(
            x_hat
        )

        loss = F.mse_loss(
            reconstruction,
            x
        )

        return z, loss


# ============================================================================
# 4. Official FSGNN architecture
# ============================================================================

class _LPkGFSGNN(nn.Module):
    """
    FSGNN implementation matching the official FSGNN model.py.

    Official FSGNN:
        fc1: one Linear layer per hop-feature matrix
        att: learnable softmax hop selector
        optional L2 normalization per hop
        concatenation
        ReLU
        dropout
        fc2
        log_softmax

    Official source:
        https://github.com/sunilkmaurya/FSGNN
    """

    def __init__(
        self,
        nfeat,
        nlayers,
        nhidden,
        nclass,
        dropout
    ):
        super().__init__()

        self.fc2 = nn.Linear(
            nhidden * nlayers,
            nclass
        )

        self.dropout = dropout

        self.act_fn = nn.ReLU()

        self.fc1 = nn.ModuleList([
            nn.Linear(
                nfeat,
                int(nhidden)
            )
            for _ in range(nlayers)
        ])

        self.att = nn.Parameter(
            torch.ones(nlayers)
        )

        self.sm = nn.Softmax(
            dim=0
        )

    def forward(
        self,
        list_mat,
        layer_norm
    ):

        mask = self.sm(
            self.att
        )

        list_out = []

        for ind, mat in enumerate(list_mat):

            tmp_out = self.fc1[ind](
                mat
            )

            if layer_norm:

                tmp_out = F.normalize(
                    tmp_out,
                    p=2,
                    dim=1
                )

            tmp_out = torch.mul(
                mask[ind],
                tmp_out
            )

            list_out.append(
                tmp_out
            )

        final_mat = torch.cat(
            list_out,
            dim=1
        )

        out = self.act_fn(
            final_mat
        )

        out = F.dropout(
            out,
            self.dropout,
            training=self.training
        )

        out = self.fc2(
            out
        )

        return F.log_softmax(
            out,
            dim=1
        )


# ============================================================================
# 5. Dataset-specific official FSGNN configurations
# ============================================================================

def _lpkg_fsgnn_config(dataset_name):

    name = str(
        dataset_name
    ).lower()

    # These values are from the official FSGNN
    # run_classification_3_hop.sh.
    #
    # FSGNN's official script uses 3 graph-convolution hops.
    #
    # For Actor, the script contains a typo where the command says
    # "film" although the preceding echo explicitly identifies it as
    # Actor. The Actor configuration is therefore the final no-hop-
    # normalization configuration in that script.

    configs = {

        "cora": dict(
            hidden=64,
            dropout=0.6,
            w_att=0.1,
            w_fc2=0.0001,
            w_fc1=0.001,
            lr_fc=0.01,
            lr_att=0.01,
            layer_norm=True,
        ),

        "citeseer": dict(
            hidden=64,
            dropout=0.5,
            w_att=0.0001,
            w_fc2=0.0,
            w_fc1=0.001,
            lr_fc=0.01,
            lr_att=0.005,
            layer_norm=True,
        ),

        "pubmed": dict(
            hidden=64,
            dropout=0.7,
            w_att=0.01,
            w_fc2=0.0001,
            w_fc1=0.0001,
            lr_fc=0.01,
            lr_att=0.005,
            layer_norm=True,
        ),

        "chameleon": dict(
            hidden=64,
            dropout=0.5,
            w_att=0.1,
            w_fc2=0.0,
            w_fc1=0.0,
            lr_fc=0.005,
            lr_att=0.005,
            layer_norm=True,
        ),

        "wisconsin": dict(
            hidden=64,
            dropout=0.5,
            w_att=0.0001,
            w_fc2=0.0001,
            w_fc1=0.001,
            lr_fc=0.01,
            lr_att=0.01,
            layer_norm=True,
        ),

        "texas": dict(
            hidden=64,
            dropout=0.7,
            w_att=0.001,
            w_fc2=0.0,
            w_fc1=0.001,
            lr_fc=0.01,
            lr_att=0.01,
            layer_norm=True,
        ),

        "cornell": dict(
            hidden=64,
            dropout=0.5,
            w_att=0.0,
            w_fc2=0.001,
            w_fc1=0.001,
            lr_fc=0.01,
            lr_att=0.01,
            layer_norm=True,
        ),

        "squirrel": dict(
            hidden=64,
            dropout=0.7,
            w_att=0.1,
            w_fc2=0.001,
            w_fc1=0.0,
            lr_fc=0.01,
            lr_att=0.04,
            layer_norm=True,
        ),

        "actor": dict(
            hidden=64,
            dropout=0.7,
            w_att=0.01,
            w_fc2=0.001,
            w_fc1=0.0001,
            lr_fc=0.01,
            lr_att=0.01,
            layer_norm=False,
        ),

    #     "Tolokers": dict(
    #         hidden=64,
    #         dropout=0.7,
    #         w_att=0.01,
    #         w_fc2=0.001,
    #         w_fc1=0.0001,
    #         lr_fc=0.01,
    #         lr_att=0.01,
    #         layer_norm=False,
    #     ),

    #     "Amazon-ratings": dict(
    #         hidden=64,
    #         dropout=0.7,
    #         w_att=0.01,
    #         w_fc2=0.001,
    #         w_fc1=0.0001,
    #         lr_fc=0.01,
    #         lr_att=0.01,
    #         layer_norm=False,
    #     ),

    #     "Roman-empire": dict(
    #         hidden=64,
    #         dropout=0.7,
    #         w_att=0.01,
    #         w_fc2=0.001,
    #         w_fc1=0.0001,
    #         lr_fc=0.01,
    #         lr_att=0.01,
    #         layer_norm=False,
    #     ),

    #     "HSBM-MED": dict(
    #         hidden=64,
    #         dropout=0.7,
    #         w_att=0.01,
    #         w_fc2=0.001,
    #         w_fc1=0.0001,
    #         lr_fc=0.01,
    #         lr_att=0.01,
    #         layer_norm=False,
    #     ),

    #     "STRUC-HET": dict(
    #         hidden=64,
    #         dropout=0.7,
    #         w_att=0.01,
    #         w_fc2=0.001,
    #         w_fc1=0.0001,
    #         lr_fc=0.01,
    #         lr_att=0.01,
    #         layer_norm=False,
    #     ),

    #     "FEAT-HET": dict(
    #         hidden=64,
    #         dropout=0.7,
    #         w_att=0.01,
    #         w_fc2=0.001,
    #         w_fc1=0.0001,
    #         lr_fc=0.01,
    #         lr_att=0.01,
    #         layer_norm=False,
    #     ),

    #     "MIXED-SIG": dict(
    #         hidden=64,
    #         dropout=0.7,
    #         w_att=0.01,
    #         w_fc2=0.001,
    #         w_fc1=0.0001,
    #         lr_fc=0.01,
    #         lr_att=0.01,
    #         layer_norm=False,
    #     ),
    }

    if name not in configs:
        print(
            f"[LPkG] No dataset-specific FSGNN configuration "
            f"for '{dataset_name}'. "
            f"Using Actor FSGNN configuration."
        )
        return configs["actor"]
    
    return configs[name]


# ============================================================================
# 6. Build FSGNN feature bank
# ============================================================================

@torch.no_grad()
def _lpkg_build_fsgnn_features(
    x,
    edge_index,
    num_nodes,
    num_hops,
    device
):
    """
    Exactly follows the feature construction in official FSGNN node_class.py.

    Starting with:

        list_mat = [X]

    Then for each hop:

        no_loop_mat = A_sym @ no_loop_mat
        loop_mat    = A_sym_I @ loop_mat

        append no-loop
        append self-loop

    For layer=3:

        X
        A X
        A_hat X
        A^2 X
        A_hat^2 X
        A^3 X
        A_hat^3 X

    Total = 7 feature matrices.
    """

    A_sym = _lpkg_normalized_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        add_self_loops=False,
        device=device
    )

    A_sym_i = _lpkg_normalized_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        add_self_loops=True,
        device=device
    )

    list_mat = [
        x
    ]

    no_loop_mat = x
    loop_mat = x

    for _ in range(
        num_hops
    ):

        no_loop_mat = torch.sparse.mm(
            A_sym,
            no_loop_mat
        )

        loop_mat = torch.sparse.mm(
            A_sym_i,
            loop_mat
        )

        list_mat.append(
            no_loop_mat
        )

        list_mat.append(
            loop_mat
        )

    return list_mat


# ============================================================================
# 7. Build LPkG cosine kNN graph
# ============================================================================

@torch.no_grad()
def _lpkg_build_knn_graph(
    z,
    k,
    batch_size
):
    """
    LPkG Section IV-A.

    cosine similarity:

        sim(i,j)
        =
        z_i^T z_j / (||z_i|| ||z_j||)

    and:

        A_hat[i,j] = 1
        iff j belongs to top-k(i).

    IMPORTANT:
    The paper does not state that this graph is converted with
    to_undirected(). Therefore this implementation keeps exactly
    the top-k directed relations.
    """

    n = z.size(0)

    if k >= n:
        raise ValueError(
            f"k={k} must be smaller than N={n}"
        )

    z_norm = F.normalize(
        z,
        p=2,
        dim=1
    )

    rows = []
    cols = []

    for start in range(
        0,
        n,
        batch_size
    ):

        end = min(
            start + batch_size,
            n
        )

        z_batch = z_norm[
            start:end
        ]

        similarity = z_batch @ z_norm.t()

        # Do not allow the node itself to be selected.
        local = torch.arange(
            end - start,
            device=z.device
        )

        global_idx = torch.arange(
            start,
            end,
            device=z.device
        )

        similarity[
            local,
            global_idx
        ] = -float("inf")

        _, neighbors = torch.topk(
            similarity,
            k=k,
            dim=1,
            largest=True,
            sorted=False
        )

        source = (
            torch.arange(
                start,
                end,
                device=z.device
            )
            .unsqueeze(1)
            .expand(-1, k)
            .reshape(-1)
        )

        destination = neighbors.reshape(
            -1
        )

        rows.append(
            source
        )

        cols.append(
            destination
        )

    return torch.stack(
        [
            torch.cat(rows),
            torch.cat(cols)
        ],
        dim=0
    )


# ============================================================================
# 8. LPkG Label Propagation
# ============================================================================

@torch.no_grad()
def _lpkg_label_propagation(
    knn_edge_index,
    num_nodes,
    num_classes,
    train_mask,
    labels,
    alpha,
    n_iter,
    device
):
    """
    LPkG Algorithm 1.

    Initial label matrix:

        Y^(0)

    Algorithm 1 line 4:

        Y <- A_hat Y^(0)

    Algorithm 1 line 5:

        Y_tilde <- row-normalized Y

    Then:

        Y =
        (I-S)
        [
            alpha A_hat Y_tilde
            +
            (1-alpha) A_hat Y^(0)
        ]
        +
        S Y^(0)

    followed by row normalization.

    The paper uses:
        alpha = 0.99
        T = 30
    """

    # Symmetric normalization with self-loops.
    A_hat = _lpkg_normalized_adjacency(
        edge_index=knn_edge_index,
        num_nodes=num_nodes,
        add_self_loops=True,
        device=device
    )

    # ---------------------------------------------------------------
    # Y^(0)
    # ---------------------------------------------------------------

    Y0 = torch.zeros(
        num_nodes,
        num_classes,
        dtype=torch.float32,
        device=device
    )

    Y0[train_mask] = F.one_hot(
        labels[train_mask],
        num_classes=num_classes
    ).float()

    # S_ii = 1 for labeled nodes.
    S = train_mask.float()

    # ---------------------------------------------------------------
    # Algorithm 1, line 4:
    #
    # Y <- A_hat Y^(0)
    # ---------------------------------------------------------------

    Y = torch.sparse.mm(
        A_hat,
        Y0
    )

    # ---------------------------------------------------------------
    # Algorithm 1, line 5
    # ---------------------------------------------------------------

    Y_tilde = (
        Y
        /
        Y.sum(
            dim=1,
            keepdim=True
        ).clamp_min(1e-12)
    )

    # ---------------------------------------------------------------
    # Remaining iterations.
    #
    # Algorithm initializes Y before entering the loop, therefore
    # the first propagated state is already obtained above.
    # ---------------------------------------------------------------

    for _ in range(
        max(n_iter - 1, 0)
    ):

        propagated = (
            alpha
            * torch.sparse.mm(
                A_hat,
                Y_tilde
            )
            +
            (1.0 - alpha)
            * torch.sparse.mm(
                A_hat,
                Y0
            )
        )

        Y = (
            (1.0 - S).unsqueeze(1)
            * propagated
            +
            S.unsqueeze(1)
            * Y0
        )

        Y_tilde = (
            Y
            /
            Y.sum(
                dim=1,
                keepdim=True
            ).clamp_min(1e-12)
        )

    return Y_tilde


# ============================================================================
# 9. Main LPkG
# ============================================================================

def run_lpkg(
    data,
    cfg,
    device,
    seed,
    dataset_name=None
):
    """
    Run LPkG on one supplied train/validation/test split.

    IMPORTANT:
    The LPkG paper reports mean accuracy over 10 random splits.
    Your benchmark can therefore call this function once for each
    split/seed and aggregate the returned accuracies.
    """

    _lpkg_set_seed(
        seed
    )

    device = torch.device(
        device
    )

    # ----------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------

    X = data.x.float().to(
        device
    )

    y = data.y.long().to(
        device
    )

    edge_index = data.edge_index.long().to(
        device
    )

    N = data.num_nodes
    C = data.num_classes

    train_mask = data.train_mask.to(
        device
    )

    val_mask = data.val_mask.to(
        device
    )

    test_mask = data.test_mask.to(
        device
    )

    # If masks contain multiple splits, use the supplied seed
    # deterministically to select one.
    if train_mask.dim() == 2:

        num_splits = train_mask.size(1)

        split_id = seed % num_splits

        train_mask = train_mask[:, split_id]
        val_mask = val_mask[:, split_id]
        test_mask = test_mask[:, split_id]

    # ----------------------------------------------------------------
    # Dataset name
    # ----------------------------------------------------------------

    # dataset_name = getattr(
    #     cfg,
    #     "dataset_name",
    #     getattr(
    #         data,
    #         "name",
    #         ""
    #     )
    # )
    if dataset_name is None:
        dataset_name = getattr(
            cfg,
            "dataset_name",
            getattr(
                data,
                "name",
                ""
            )
        )
    
    dataset_name = str(dataset_name).strip()


    
    # =================================================================
    # LPkG PAPER PARAMETERS
    # =================================================================

    gae_hid = cfg.lpkg_gae_hid
    gae_lat = cfg.lpkg_gae_lat
    gae_lr = cfg.lpkg_gae_lr
    gae_epochs = cfg.lpkg_gae_epochs

    k = cfg.lpkg_k

    lp_alpha = cfg.lpkg_lp_alpha
    lp_iters = cfg.lpkg_lp_iters

    beta_lr = cfg.lpkg_beta_lr
    beta_init = cfg.lpkg_beta_init

    # =================================================================
    # FSGNN PARAMETERS
    # =================================================================
    dataset_key = str(dataset_name).strip().lower()
    if dataset_key == "squirrel-f":
        dataset_key = "squirrel"
    elif dataset_key == "chameleon-f":
        dataset_key = "chameleon"
    fsgnn = _lpkg_fsgnn_config(
        dataset_name
    )

    # Explicit cfg overrides are allowed, but defaults are the
    # official FSGNN settings.
    fsgnn_hidden = getattr(
        cfg,
        "lpkg_fsgnn_hidden",
        fsgnn["hidden"]
    )

    fsgnn_dropout = getattr(
        cfg,
        "lpkg_fsgnn_dropout",
        fsgnn["dropout"]
    )

    fsgnn_hops = getattr(
        cfg,
        "lpkg_fsgnn_hops",
        3
    )

    fsgnn_layer_norm = getattr(
        cfg,
        "lpkg_fsgnn_layer_norm",
        fsgnn["layer_norm"]
    )

    fsgnn_epochs = getattr(
        cfg,
        "lpkg_fsgnn_epochs",
        1500
    )

    fsgnn_patience = getattr(
        cfg,
        "lpkg_fsgnn_patience",
        100
    )

    fsgnn_lr_fc = getattr(
        cfg,
        "lpkg_fsgnn_lr_fc",
        fsgnn["lr_fc"]
    )

    fsgnn_lr_att = getattr(
        cfg,
        "lpkg_fsgnn_lr_att",
        fsgnn["lr_att"]
    )

    fsgnn_w_fc1 = getattr(
        cfg,
        "lpkg_fsgnn_w_fc1",
        fsgnn["w_fc1"]
    )

    fsgnn_w_fc2 = getattr(
        cfg,
        "lpkg_fsgnn_w_fc2",
        fsgnn["w_fc2"]
    )

    fsgnn_w_att = getattr(
        cfg,
        "lpkg_fsgnn_w_att",
        fsgnn["w_att"]
    )

    knn_batch_size = getattr(
        cfg,
        "lpkg_knn_batch_size",
        512
    )

    # =================================================================
    # PRINT CONFIGURATION
    # =================================================================

    print(
        "\n"
        + "=" * 78
    )

    print(
        "LPkG — faithful paper/FSGNN implementation"
    )

    print(
        "=" * 78
    )

    print(
        f"Dataset             : {dataset_name}"
    )

    print(
        f"Nodes               : {N}"
    )

    print(
        f"Features            : {X.shape[1]}"
    )

    print(
        f"Classes             : {C}"
    )

    print(
        f"GAE dimensions      : {gae_hid}/{gae_lat}"
    )

    print(
        f"GAE learning rate   : {gae_lr}"
    )

    print(
        f"GAE epochs          : {gae_epochs}"
    )

    print(
        f"k                   : {k}"
    )

    print(
        f"LP alpha            : {lp_alpha}"
    )

    print(
        f"LP iterations       : {lp_iters}"
    )

    print(
        f"Beta learning rate  : {beta_lr}"
    )

    print(
        f"Beta initialization : {beta_init}"
    )

    print(
        f"FSGNN hops          : {fsgnn_hops}"
    )

    print(
        f"FSGNN hidden        : {fsgnn_hidden}"
    )

    print(
        f"FSGNN dropout       : {fsgnn_dropout}"
    )

    print(
        f"FSGNN layer norm    : {fsgnn_layer_norm}"
    )

    print(
        f"FSGNN epochs        : {fsgnn_epochs}"
    )

    print(
        f"FSGNN patience      : {fsgnn_patience}"
    )

    print(
        "=" * 78
    )

    # =================================================================
    # STAGE 1 — FEATURE RECONSTRUCTION GAE
    # =================================================================

    gae = _LPkGGAE(
        in_dim=X.shape[1],
        hid_dim=gae_hid,
        lat_dim=gae_lat
    ).to(device)

    gae_optimizer = Adam(
        gae.parameters(),
        lr=gae_lr
    )

    gae.train()

    for epoch in range(
        gae_epochs
    ):

        gae_optimizer.zero_grad()

        Z_lat, loss_gae = gae(
            X,
            edge_index
        )

        loss_gae.backward()

        gae_optimizer.step()

    gae.eval()

    with torch.no_grad():

        Z_lat, final_gae_loss = gae(
            X,
            edge_index
        )

    print(
        f"[LPkG] GAE finished:"
        f" loss={final_gae_loss.item():.6f}"
        f" latent={tuple(Z_lat.shape)}"
    )

    # =================================================================
    # STAGE 2 — COSINE kNN GRAPH
    # =================================================================

    knn_edge_index = _lpkg_build_knn_graph(
        z=Z_lat.detach(),
        k=k,
        batch_size=knn_batch_size
    )

    print(
        f"[LPkG] kNN graph:"
        f" edges={knn_edge_index.shape[1]}"
        f" k={k}"
    )

    # =================================================================
    # STAGE 3 — LABEL PROPAGATION
    # =================================================================

    Z_prob = _lpkg_label_propagation(
        knn_edge_index=knn_edge_index,
        num_nodes=N,
        num_classes=C,
        train_mask=train_mask,
        labels=y,
        alpha=lp_alpha,
        n_iter=lp_iters,
        device=device
    )

    print(
        f"[LPkG] Label propagation finished:"
        f" Z_prob={tuple(Z_prob.shape)}"
    )

    # =================================================================
    # STAGE 4 — FSGNN FEATURE BANK
    # =================================================================

    list_mat = _lpkg_build_fsgnn_features(
        x=X,
        edge_index=edge_index,
        num_nodes=N,
        num_hops=fsgnn_hops,
        device=device
    )

    print(
        f"[LPkG] FSGNN feature matrices:"
        f" {len(list_mat)}"
    )

    # =================================================================
    # STAGE 5 — FSGNN + LEARNABLE BETA
    # =================================================================

    model = _LPkGFSGNN(
        nfeat=X.shape[1],
        nlayers=len(list_mat),
        nhidden=fsgnn_hidden,
        nclass=C,
        dropout=fsgnn_dropout
    ).to(device)

    # -----------------------------------------------------------------
    # Beta
    #
    # LPkG paper defines beta as the learnable weighted-average ratio.
    #
    # IMPORTANT:
    # The paper does not specify:
    #   * beta initialization
    #   * a sigmoid/logit parameterization
    #
    # Therefore beta is kept as a direct learnable scalar here.
    #
    # It is constrained to [0,1] only when constructing Z_new.
    # -----------------------------------------------------------------

    beta = nn.Parameter(
        torch.tensor(
            beta_init,
            dtype=torch.float32,
            device=device
        )
    )

    # -----------------------------------------------------------------
    # Official FSGNN optimizer groups.
    # -----------------------------------------------------------------

    optimizer = Adam(
        [
            {
                "params": model.fc2.parameters(),
                "weight_decay": fsgnn_w_fc2,
                "lr": fsgnn_lr_fc,
            },
            {
                "params": model.fc1.parameters(),
                "weight_decay": fsgnn_w_fc1,
                "lr": fsgnn_lr_fc,
            },
            {
                "params": [model.att],
                "weight_decay": fsgnn_w_att,
                "lr": fsgnn_lr_att,
            },
            {
                "params": [beta],
                "weight_decay": 0.0,
                "lr": beta_lr,
            },
        ]
    )

    # =================================================================
    # TRAINING
    # =================================================================

    best_val_loss = float(
        "inf"
    )

    best_model_state = None
    best_beta = None

    bad_counter = 0

    for epoch in range(
        fsgnn_epochs
    ):

        model.train()

        optimizer.zero_grad()

        # FSGNN output = log probabilities.
        Z_pred_log = model(
            list_mat,
            fsgnn_layer_norm
        )

        Z_pred = torch.exp(
            Z_pred_log
        )

        # The paper describes beta as the weighted-average ratio.
        #
        # Keep it inside [0,1] when forming the mixture.
        beta_used = beta.clamp(
            0.0,
            1.0
        )

        # -------------------------------------------------------------
        # LPkG:
        #
        # Z_new = (1-beta) Z_pred + beta Z_prob
        # -------------------------------------------------------------

        Z_new = (
            (1.0 - beta_used)
            * Z_pred
            +
            beta_used
            * Z_prob
        )

        Z_new = Z_new.clamp_min(
            1e-12
        )

        # Cross entropy / NLL on the training labels.
        loss = -torch.log(
            Z_new[
                train_mask,
                y[train_mask]
            ]
        ).mean()

        loss.backward()

        optimizer.step()

        # =============================================================
        # VALIDATION
        # =============================================================

        model.eval()

        with torch.no_grad():

            Z_val_log = model(
                list_mat,
                fsgnn_layer_norm
            )

            Z_val = torch.exp(
                Z_val_log
            )

            beta_val = beta.clamp(
                0.0,
                1.0
            )

            Z_val_new = (
                (1.0 - beta_val)
                * Z_val
                +
                beta_val
                * Z_prob
            )

            Z_val_new = Z_val_new.clamp_min(
                1e-12
            )

            val_loss = -torch.log(
                Z_val_new[
                    val_mask,
                    y[val_mask]
                ]
            ).mean()

            val_pred = Z_val_new.argmax(
                dim=1
            )

            val_acc = (
                val_pred[val_mask]
                ==
                y[val_mask]
            ).float().mean().item()

        # =============================================================
        # FSGNN OFFICIAL CHECKPOINTING
        #
        # Official FSGNN saves the model when validation loss improves
        # and stops after patience epochs without improvement.
        # =============================================================

        if val_loss.item() < best_val_loss:

            best_val_loss = val_loss.item()

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            best_beta = beta.detach().clone()

            bad_counter = 0

        else:

            bad_counter += 1

        if (
            epoch == 0
            or (epoch + 1) % 50 == 0
        ):

            print(
                f"[LPkG] Epoch {epoch + 1:04d}"
                f" | loss={loss.item():.5f}"
                f" | val_loss={val_loss.item():.5f}"
                f" | val_acc={val_acc:.4f}"
                f" | beta={beta_used.item():.5f}"
            )

        if bad_counter == fsgnn_patience:

            print(
                f"[LPkG] Early stopping at epoch "
                f"{epoch + 1}"
            )

            break

    # =================================================================
    # RESTORE BEST VALIDATION CHECKPOINT
    # =================================================================

    if best_model_state is not None:

        model.load_state_dict(
            best_model_state
        )

        with torch.no_grad():
            beta.copy_(
                best_beta
            )

    # =================================================================
    # FINAL TEST
    # =================================================================

    model.eval()

    with torch.no_grad():

        Z_pred_log = model(
            list_mat,
            fsgnn_layer_norm
        )

        Z_pred = torch.exp(
            Z_pred_log
        )

        beta_final = beta.clamp(
            0.0,
            1.0
        )

        Z_final = (
            (1.0 - beta_final)
            * Z_pred
            +
            beta_final
            * Z_prob
        )

        final_pred = Z_final.argmax(
            dim=1
        )

        val_acc = (
            final_pred[val_mask]
            ==
            y[val_mask]
        ).float().mean().item()

        test_acc = (
            final_pred[test_mask]
            ==
            y[test_mask]
        ).float().mean().item()

    print(
        f"[LPkG] Final:"
        f" beta={beta_final.item():.5f}"
        f" | val={val_acc:.4f}"
        f" | test={test_acc:.4f}"
    )

    _pred = final_pred[test_mask]
    _true = y[test_mask]
    _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, _true)
    return val_acc, test_acc, test_f1




# =============================================================================
# DHGR official implementation
# =============================================================================


from pathlib import Path
import sys

# benchmark file is inside:
#   heterophilic rewiring/results_faithful/
#
# DHGR is inside:
#   heterophilic rewiring/DHGR/

def _locate_dhgr_root():
    """Find the official DHGR repository.

    Search order:
      1. $DHGR_ROOT environment variable
      2. ./DHGR                              (next to this script)
      3. ../DHGR                             (one level up)
      4. ../../DHGR                          (original project layout)
    Returns a Path if GraphLearner.py is found, else None.
    """
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

if DHGR_ROOT is None:
    print("[DHGR] GraphLearner.py not found (set $DHGR_ROOT or place the DHGR "
          "repo next to this script). DHGR runs will be skipped; all other "
          "methods work normally.")
else:
    print(f"[DHGR] Using DHGR repository at: {DHGR_ROOT}")
    if str(DHGR_ROOT) not in sys.path:
        sys.path.insert(0, str(DHGR_ROOT))
    try:
        from GraphLearner import ModelHandler as DHGRModelHandler
        print("[DHGR] GraphLearner imported successfully.")
    except Exception as e:
        DHGRModelHandler = None
        print("[DHGR] GraphLearner was found but importing it failed:")
        print(type(e).__name__, ":", e)


# =============================================================================
# DHGR official downstream GCN
# =============================================================================

class _DHGR_GCN(nn.Module):
    """
    DHGR's GCN classifier.

    Matches the official DHGR GCNNet implementation:
      - 2 layers for Actor
      - ReLU + dropout(0.5) between layers
      - log_softmax output
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        num_layers=2,
    ):
        super().__init__()

        from torch_geometric.nn import GCNConv

        self.convs = nn.ModuleList()

        if num_layers == 1:

            self.convs.append(
                GCNConv(
                    in_dim,
                    num_classes,
                )
            )

        else:

            for layer in range(num_layers):

                if layer == 0:

                    self.convs.append(
                        GCNConv(
                            in_dim,
                            hidden_dim,
                        )
                    )

                elif layer == num_layers - 1:

                    self.convs.append(
                        GCNConv(
                            hidden_dim,
                            num_classes,
                        )
                    )

                else:

                    self.convs.append(
                        GCNConv(
                            hidden_dim,
                            hidden_dim,
                        )
                    )

    def forward(self, x, edge_index):

        for i, conv in enumerate(self.convs):

            x = conv(
                x,
                edge_index,
            )

            if i != len(self.convs) - 1:

                x = F.relu(x)

                x = F.dropout(
                    x,
                    p=0.5,
                    training=self.training,
                )

        return F.log_softmax(
            x,
            dim=1,
        )

# =============================================================================
# DHGR faithful implementation
# Paper:
#   Wendong Bi et al.
#   "Make Heterophilic Graphs Better Fit GNN:
#    A Graph Rewiring Approach"
#
# Official DHGR implementation:
#   https://github.com/wendongbi/DHGR
#
# Actor configuration follows the official run.sh:
#   GCN, 2 layers, hidden 64
#   GNN: 200 epochs, lr=0.01, wd=5e-3
#   Graph learner: 200 + 30 epochs
#   Graph learner lr=0.001, wd=5e-3
#   thres_min_deg=10
#   thres_min_deg_ratio=1.0
#   window=[5000,5000]
#   k=8
#   cat_self=False
#   pruning=True
#   pruning threshold=0.5
#
# IMPORTANT:
#   The benchmark's existing train/val/test masks are retained.
#   We do NOT let DHGR create a new split.
# =============================================================================

def run_dhgr(
    data,
    cfg,
    device,
    seed,
    dataset_name=None,
):
    """
    Run DHGR using the official DHGR graph learner and official
    Actor downstream GCN.

    Returns:
        best_val_acc, test_acc
    """

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -------------------------------------------------------------------------
    # Check official DHGR implementation
    # -------------------------------------------------------------------------

    if DHGRModelHandler is None:
        raise ImportError(
            "Could not import DHGR GraphLearner.py.\n"
            "Make sure the official DHGR repository is located at:\n"
            "    ./DHGR/\n"
            "with:\n"
            "    ./DHGR/GraphLearner.py"
        )

    # -------------------------------------------------------------------------
    # Make a PyG Data object.
    #
    # The official DHGR ModelHandler expects a torch_geometric.data.Data
    # object rather than our SimpleNamespace benchmark object.
    # -------------------------------------------------------------------------

    from torch_geometric.data import Data

    x = data.x.float().to(device)
    y = data.y.long().to(device)

    train_mask = data.train_mask.bool().to(device)
    val_mask = data.val_mask.bool().to(device)
    test_mask = data.test_mask.bool().to(device)

    # Our benchmark normally has one-dimensional masks, but protect
    # against datasets containing multiple split columns.
    if train_mask.dim() == 2:

        # Use a deterministic column based on the benchmark seed.
        split_id = seed % train_mask.shape[1]

        train_mask = train_mask[:, split_id]
        val_mask = val_mask[:, split_id]
        test_mask = test_mask[:, split_id]

    edge_index = data.edge_index.long().to(device)

    pyg_data = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    pyg_data.num_nodes = data.num_nodes

    # -------------------------------------------------------------------------
    # Official Actor DHGR hyperparameters
    # -------------------------------------------------------------------------

    hidden = int(
        getattr(
            cfg,
            "dhgr_hidden",
            64,
        )
    )

    gnn_layers = int(
        getattr(
            cfg,
            "dhgr_layers",
            2,
        )
    )

    gnn_epochs = int(
        getattr(
            cfg,
            "dhgr_epochs",
            200,
        )
    )

    gnn_lr = float(
        getattr(
            cfg,
            "dhgr_lr",
            0.01,
        )
    )

    gnn_wd = float(
        getattr(
            cfg,
            "dhgr_wd",
            5e-3,
        )
    )

    # -------------------------------------------------------------------------
    # DHGR graph learner configuration
    # -------------------------------------------------------------------------

    gl_epochs = int(
        getattr(
            cfg,
            "dhgr_gl_epochs",
            10,
        )
    )

    gl_finetune_epochs = int(
        getattr(
            cfg,
            "dhgr_gl_finetune_epochs",
            30,
        )
    )

    gl_lr = float(
        getattr(
            cfg,
            "dhgr_gl_lr",
            0.001,
        )
    )

    gl_wd = float(
        getattr(
            cfg,
            "dhgr_gl_wd",
            5e-3,
        )
    )

    min_deg = int(
        getattr(
            cfg,
            "dhgr_min_deg",
            10,
        )
    )

    min_deg_ratio = float(
        getattr(
            cfg,
            "dhgr_min_deg_ratio",
            1.0,
        )
    )

    # ── United Roman-empire DHGR variant ─────────────────────────────────────
    # Roman-empire does not run with the usual DHGR candidate thresholds; the
    # dedicated roman-only script used min_deg=3, min_deg_ratio=1.8.  We fold
    # that special case in here so a single run_dhgr covers every dataset.
    if dataset_name == "Roman-empire":
        min_deg = int(getattr(cfg, "dhgr_roman_min_deg", 3))
        min_deg_ratio = float(getattr(cfg, "dhgr_roman_min_deg_ratio", 1.8))
        print(f"  [DHGR] Roman-empire override: "
              f"min_deg={min_deg}, min_deg_ratio={min_deg_ratio}")

    k = int(
        getattr(
            cfg,
            "dhgr_k",
            8,
        )
    )

    thres_pruning = float(
        getattr(
            cfg,
            "dhgr_pruning_threshold",
            0.5,
        )
    )

    moment = int(
        getattr(
            cfg,
            "dhgr_moment",
            1,
        )
    )

    cat_self = bool(
        getattr(
            cfg,
            "dhgr_cat_self",
            False,
        )
    )

    pruning = bool(
        getattr(
            cfg,
            "dhgr_pruning",
            True,
        )
    )

    use_cpu_cache = bool(
        getattr(
            cfg,
            "dhgr_use_cpu_cache",
            False,
        )
    )

    # Official Actor setting.
    window = getattr(
        cfg,
        "dhgr_window",
        [5000, 5000],
    )

    # -------------------------------------------------------------------------
    # Build the official DHGR graph learner
    # -------------------------------------------------------------------------

    print("\n  [DHGR] Building graph learner...")

    graph_handler = DHGRModelHandler(
        in_size=pyg_data.num_features,
        num_classes=int(data.num_classes),

        thres_min_deg=min_deg,
        thres_min_deg_ratio=min_deg_ratio,

        hidden=128,

        device=device,

        save_dir=str(
            getattr(
                cfg,
                "dhgr_save_dir",
                "./dhgr_ckpt/",
            )
        ),

        seed=seed,

        num_epoch=gl_epochs,
        num_epoch_finetune=gl_finetune_epochs,

        window_size=window,

        lr=gl_lr,
        weight_decay=gl_wd,

        shuffle=[False, False],
        drop_last=[False, False],

        moment=moment,

        use_cpu_cache=use_cpu_cache,
    )

    # -------------------------------------------------------------------------
    # Perform DHGR rewiring
    #
    # Official implementation:
    #   1. train similarity encoder
    #   2. generate top-k similarity edges
    #   3. prune original edges
    #   4. merge both graphs
    # -------------------------------------------------------------------------

    print("  [DHGR] Rewiring graph...")

    rewired_data = graph_handler(
        pyg_data,

        k=k,

        epsilon=None,

        embedding_post=True,

        cat_self=cat_self,

        prunning=pruning,

        thres_prunning=thres_pruning,

        load_path=None,

        save_path=None,
    )

    rewired_edge_index = (
        rewired_data.edge_index
        .long()
        .to(device)
    )

    print(
        f"  [DHGR] Original edges : "
        f"{edge_index.shape[1]}"
    )

    print(
        f"  [DHGR] Rewired edges  : "
        f"{rewired_edge_index.shape[1]}"
    )

    # -------------------------------------------------------------------------
    # Downstream GCN
    # -------------------------------------------------------------------------

    model = _DHGR_GCN(
        in_dim=pyg_data.num_features,
        hidden_dim=hidden,
        num_classes=int(data.num_classes),
        num_layers=gnn_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=gnn_lr,
        weight_decay=gnn_wd,
    )

    # -------------------------------------------------------------------------
    # Train using the rewired graph
    # -------------------------------------------------------------------------

    best_val = -float("inf")
    best_state = None
    best_test = 0.0

    print(
        f"  [DHGR] Training GCN for "
        f"{gnn_epochs} epochs..."
    )

    for epoch in range(1, gnn_epochs + 1):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        log_probs = model(
            x,
            rewired_edge_index,
        )

        loss = F.nll_loss(
            log_probs[train_mask],
            y[train_mask],
        )

        loss.backward()

        optimizer.step()

        # -------------------------------------------------------------
        # Validation
        # -------------------------------------------------------------

        model.eval()

        with torch.no_grad():

            eval_log_probs = model(
                x,
                rewired_edge_index,
            )

            val_pred = (
                eval_log_probs[val_mask]
                .argmax(dim=-1)
            )

            val_acc = (
                val_pred
                .eq(y[val_mask])
                .float()
                .mean()
                .item()
            )

            test_pred = (
                eval_log_probs[test_mask]
                .argmax(dim=-1)
            )

            test_acc = (
                test_pred
                .eq(y[test_mask])
                .float()
                .mean()
                .item()
            )

        # -------------------------------------------------------------
        # Best validation checkpoint
        # -------------------------------------------------------------

        if val_acc > best_val:

            best_val = val_acc
            best_test = test_acc

            best_state = copy.deepcopy(
                model.state_dict()
            )

        if (
            epoch == 1
            or epoch % 20 == 0
            or epoch == gnn_epochs
        ):

            print(
                f"    Epoch {epoch:03d} | "
                f"loss={loss.item():.4f} | "
                f"val={val_acc:.4f} | "
                f"test={test_acc:.4f}"
            )

    # -------------------------------------------------------------------------
    # Restore best validation checkpoint
    # -------------------------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    model.eval()

    with torch.no_grad():

        final_log_probs = model(
            x,
            rewired_edge_index,
        )

        final_test_acc = (
            final_log_probs[test_mask]
            .argmax(dim=-1)
            .eq(y[test_mask])
            .float()
            .mean()
            .item()
        )

    _pred = final_log_probs[test_mask].argmax(dim=-1)
    _, final_test_f1 = _metrics.accuracy_and_macro_f1(_pred, y[test_mask])

    print(
        f"\n  [DHGR] best_val={best_val:.4f} "
        f"test={final_test_acc:.4f} f1={final_test_f1:.4f}"
    )

    return (
        best_val,
        final_test_acc,
        final_test_f1,
    )








# =============================================================================
# GRAPHITE — FAITHFUL IMPLEMENTATION
# =============================================================================
#
# Paper:
# "Graph Homophily Booster: Rethinking the Role of Discrete Features
#  on Heterophilic Graphs"
#
# GRAPHITE paper:
#   Graph transformation : Eqs. (3)-(9)
#   Neural architecture  : Eqs. (12)-(16)
#   Training             : Appendix / Training & Evaluation
#
# Paper settings:
#   GNN layers       = 8
#   Hidden dimension = 512
#   Dropout          = 0.2
#   Learning rate    = 3e-5
#   Optimizer        = Adam
#   Training steps   = 1000
#
# IMPORTANT:
#   The paper assumes binary/discrete node features.
#   Therefore this implementation does NOT perform:
#       - top-k binarisation
#       - median binarisation
#       - positive-value binarisation
#       - feature-edge capping
#
# =============================================================================


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.optim import Adam


# =============================================================================
# FUNCTION 1
# GRAPHITE GRAPH TRANSFORMATION
# =============================================================================

def graphite_transform(
    data,
    dataset_name=None,
    device="cpu",
):
    """
    Construct the GRAPHITE transformed graph G*.

    Paper equations:
        Eq. (3) : feature nodes
        Eq. (4) : feature edges
        Eq. (5) : transformed node set
        Eq. (6) : transformed edge set
        Eq. (7) : transformed adjacency
        Eq. (8) : original graph-node features
        Eq. (9) : feature-node features

    Returns
    -------
    X_ext
        Features of original + feature nodes.

    graph_ei
        Original graph edges.

    feat_ei
        Feature edges.

    N
        Number of original graph nodes.

    Fdim
        Number of original features.
    """

    # =========================================================================
    # ORIGINAL FEATURES
    # =========================================================================

    X = data.x.detach().cpu().float()

    N, Fdim = X.shape

    print(
        f"[GRAPHITE] Original graph: "
        f"nodes={N:,}, features={Fdim:,}"
    )


    # =========================================================================
    # CHECK BINARY FEATURES
    # =========================================================================
    #
    # Paper definition:
    #
    #       X in {0,1}^{|V| x |X|}
    #
    # We do not silently alter the feature matrix.
    # =========================================================================

    unique_vals = torch.unique(X)

    is_binary = bool(
        torch.all(
            (unique_vals == 0) |
            (unique_vals == 1)
        )
    )

    if not is_binary:

        raise ValueError(
            "\n"
            "GRAPHITE faithful implementation requires binary/discrete "
            "features as used in the paper.\n\n"
            f"Received feature values outside {{0,1}}.\n"
            f"Number of unique values: {len(unique_vals)}\n\n"
            "No top-k, median, or positive-value binarisation is performed "
            "because those operations are not part of the published "
            "GRAPHITE method."
        )


    X_np = X.numpy().astype(
        np.float32
    )


    # =========================================================================
    # FEATURE NODES
    # =========================================================================
    #
    # Paper Eq. (3):
    #
    #       V_X = {x_k : k in X}
    #
    # Therefore:
    #
    #       number of feature nodes = number of original features
    #
    # =========================================================================

    feature_node_ids = (
        N +
        np.arange(
            Fdim,
            dtype=np.int64
        )
    )


    # =========================================================================
    # FEATURE EDGES
    # =========================================================================
    #
    # Paper Eq. (4):
    #
    #       E_X = {(v_i,x_k) : X[i,k] = 1}
    #
    # For message passing we represent each undirected feature edge in both
    # directions:
    #
    #       v_i -> x_k
    #       x_k -> v_i
    #
    # =========================================================================

    row, col = np.nonzero(
        X_np > 0
    )

    feat_nodes_for_edges = (
        N + col
    ).astype(
        np.int64
    )


    # v_i -> x_k
    forward_src = row.astype(
        np.int64
    )

    forward_dst = feat_nodes_for_edges


    # x_k -> v_i
    backward_src = feat_nodes_for_edges

    backward_dst = row.astype(
        np.int64
    )


    feat_src = np.concatenate(
        [
            forward_src,
            backward_src,
        ]
    )

    feat_dst = np.concatenate(
        [
            forward_dst,
            backward_dst,
        ]
    )


    feat_ei = torch.tensor(
        np.stack(
            [
                feat_src,
                feat_dst,
            ],
            axis=0,
        ),
        dtype=torch.long,
        device=device,
    )


    # =========================================================================
    # FEATURE NODE FEATURES
    # =========================================================================
    #
    # Paper Eq. (9):
    #
    #       X*[x_k,:]
    #
    #       =
    #
    #       1 / number_of_nodes_with_feature_k
    #
    #       * sum X[v_i,:]
    #
    # =========================================================================

    X_t = torch.from_numpy(
        X_np
    )


    # Number of nodes having each feature.
    #
    # Shape:
    #
    #       Fdim x 1
    #
    feature_counts = (
        X_t.sum(
            dim=0,
            keepdim=True
        )
        .t()
    )


    # The paper assumes every feature is used.
    #
    # For numerical safety only, clamp zero denominators.
    feature_counts = feature_counts.clamp_min(
        1.0
    )


    # Equivalent to:
    #
    #       for every feature k:
    #
    #       average X[v_i,:] over X[i,k] = 1
    #
    feature_node_features = (
        X_t.t() @ X_t
    ) / feature_counts


    # =========================================================================
    # ORIGINAL GRAPH-NODE FEATURES
    # =========================================================================
    #
    # Paper Eq. (8):
    #
    #       X*[v_i,:] = X[v_i,:]
    #
    # with the dataset-specific settings stated in the appendix.
    # =========================================================================

    graph_node_features = X_t.clone()


    # Normalize dataset name.
    if dataset_name is None:

        ds = ""

    else:

        ds = (
            str(dataset_name)
            .lower()
            .replace("_", "-")
            .strip()
        )


    # =========================================================================
    # SQUIRREL-F
    # =========================================================================
    #
    # Paper:
    #
    # "we use zeros as the features of graph nodes on Squirrel-F"
    #
    # =========================================================================

    if ds in {
        "squirrel",
        "squirrel-f",
    }:

        graph_node_features = torch.zeros_like(
            graph_node_features
        )


    # =========================================================================
    # CORA / CITESEER
    # =========================================================================
    #
    # Paper:
    #
    # "we normalize the features of graph nodes on Cora and CiteSeer
    #  after computing the features of feature nodes."
    #
    # The supplied paper text does not specify the exact normalization
    # formula. We therefore use row-wise L1 normalization.
    #
    # =========================================================================

    elif ds in {
        "cora",
        "citeseer",
    }:

        row_sum = graph_node_features.sum(
            dim=1,
            keepdim=True
        )

        graph_node_features = (
            graph_node_features /
            row_sum.clamp_min(1e-12)
        )


    # =========================================================================
    # COMBINE FEATURES
    # =========================================================================
    #
    # First:
    #
    #       original graph nodes
    #
    # Then:
    #
    #       feature nodes
    #
    # =========================================================================

    X_ext = torch.cat(
        [
            graph_node_features,
            feature_node_features,
        ],
        dim=0,
    ).to(device)


    # =========================================================================
    # ORIGINAL GRAPH EDGES
    # =========================================================================
    #
    # We keep the original graph edges.
    #
    # Self-loops are NOT added here.
    # They are handled explicitly through w_0 in the GNN.
    #
    # =========================================================================

    edge_index_np = (
        data.edge_index
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
    )


    src = edge_index_np[0]

    dst = edge_index_np[1]


    # Remove explicit self-loops.
    keep = src != dst

    src = src[keep]

    dst = dst[keep]


    # Make the undirected graph explicit in both directions.
    graph_pairs = np.vstack(
        [
            np.stack(
                [
                    src,
                    dst,
                ],
                axis=1,
            ),

            np.stack(
                [
                    dst,
                    src,
                ],
                axis=1,
            ),
        ]
    )


    # Remove duplicate pairs.
    graph_pairs = np.unique(
        graph_pairs,
        axis=0
    )


    graph_ei = torch.tensor(
        graph_pairs.T,
        dtype=torch.long,
        device=device,
    )


    # =========================================================================
    # INFORMATION
    # =========================================================================

    print(
        f"[GRAPHITE] Transformed graph:"
        f"\n    original nodes = {N:,}"
        f"\n    feature nodes  = {Fdim:,}"
        f"\n    total nodes    = {N + Fdim:,}"
        f"\n    graph edges    = {graph_ei.shape[1]:,}"
        f"\n    feature edges  = {feat_ei.shape[1]:,}"
    )


    return (
        X_ext,
        graph_ei,
        feat_ei,
        N,
        Fdim,
    )


# =============================================================================
# FUNCTION 2
# GRAPHITE GNN
# =============================================================================

class GraphiteGNN(nn.Module):
    """
    GRAPHITE neural architecture.

    Paper equations:
        Eq. (12) : graph-node weighted degree
        Eq. (13) : feature-node weighted degree
        Eq. (14) : self-gating
        Eq. (15) : graph-node aggregation
        Eq. (16) : feature-node aggregation

    Important:
        alpha is multiplied LINEARLY.

    We do NOT use sqrt(alpha).

    The paper specifies:
        w_E = 1
        w_X > 0
        w_0 > 0
        tau > 0
    """

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        n_orig,
        n_feat,
        n_layers=8,
        dropout=0.2,
        w_X=0.6,
        w_0=0.3,
        tau=0.1,
    ):

        super().__init__()


        self.n_orig = int(
            n_orig
        )

        self.n_feat = int(
            n_feat
        )

        self.n_layers = int(
            n_layers
        )


        self.dropout = float(
            dropout
        )


        # =========================================================================
        # PAPER FIXES w_E = 1
        # =========================================================================

        self.w_E = 1.0


        self.w_X = float(
            w_X
        )

        self.w_0 = float(
            w_0
        )

        self.tau = float(
            tau
        )


        if self.w_X <= 0:

            raise ValueError(
                "w_X must be > 0"
            )


        if self.w_0 <= 0:

            raise ValueError(
                "w_0 must be > 0"
            )


        if self.tau <= 0:

            raise ValueError(
                "tau must be > 0"
            )


        # =========================================================================
        # INPUT PROJECTION
        # =========================================================================

        self.input_proj = nn.Linear(
            in_dim,
            hidden_dim
        )


        # =========================================================================
        # SELF-GATING PARAMETERS
        # =========================================================================
        #
        # Eq. (14):
        #
        # alpha_{u,u'} =
        #
        # tanh(
        #
        #     (a^T(h_u || h_u') + b) / tau
        #
        # )
        #
        # a has dimension 2m.
        #
        # =========================================================================

        self.gate_a = nn.ParameterList()

        self.gate_b = nn.ParameterList()


        for _ in range(
            self.n_layers
        ):

            a = nn.Parameter(
                torch.empty(
                    2 * hidden_dim
                )
            )

            nn.init.xavier_uniform_(
                a.unsqueeze(0)
            )

            self.gate_a.append(
                a
            )


            b = nn.Parameter(
                torch.zeros(1)
            )

            self.gate_b.append(
                b
            )


        # =========================================================================
        # TWO-LAYER MLP AFTER EVERY GNN AGGREGATION
        # =========================================================================
        #
        # The paper says:
        #
        #   "we add a multi-layer perceptron (MLP) with residual connections
        #    after each GNN aggregation."
        #
        # GELU activation.
        #
        # No LayerNorm is specified, so there is no LayerNorm here.
        # =========================================================================

        self.mlps = nn.ModuleList(
            [

                nn.Sequential(

                    nn.Linear(
                        hidden_dim,
                        hidden_dim
                    ),

                    nn.GELU(),

                    nn.Linear(
                        hidden_dim,
                        hidden_dim
                    ),
                )

                for _ in range(
                    self.n_layers
                )
            ]
        )


        # =========================================================================
        # CLASSIFIER
        # =========================================================================

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )


    # =========================================================================
    # SELF-GATING SCORE
    # =========================================================================

    @staticmethod
    def gate_scores(
        h,
        a,
    ):
        """
        Efficient implementation of:

            a^T(h_u || h_v)

        Split:

            a = [a1 ; a2]

        so:

            a^T(h_u || h_v)
            =
            a1^T h_u + a2^T h_v
        """

        hidden_dim = h.shape[1]


        a1 = a[
            :hidden_dim
        ]


        a2 = a[
            hidden_dim:
        ]


        score_source = h @ a1

        score_target = h @ a2


        return (
            score_source,
            score_target,
        )


    # =========================================================================
    # EDGE AGGREGATION
    # =========================================================================

    def aggregate_edges(
        self,
        h,
        edge_index,
        degree,
        score_source,
        score_target,
        bias,
        edge_type,
    ):

        if edge_index.numel() == 0:

            return torch.zeros_like(
                h
            )


        src = edge_index[0]

        dst = edge_index[1]


        # =========================================================================
        # Eq. (14)
        # =========================================================================

        alpha = torch.tanh(
            (
                score_source[src]
                +
                score_target[dst]
                +
                bias
            )
            /
            self.tau
        )


        # =========================================================================
        # NORMALIZATION
        #
        # sqrt(d_u) sqrt(d_v)
        # =========================================================================

        denominator = (
            torch.sqrt(
                degree[src].clamp_min(
                    1e-12
                )
            )
            *
            torch.sqrt(
                degree[dst].clamp_min(
                    1e-12
                )
            )
        )


        # =========================================================================
        # EDGE WEIGHT
        # =========================================================================
        #
        # IMPORTANT:
        #
        #       alpha
        #
        # is used directly.
        #
        # NOT:
        #
        #       sqrt(alpha)
        #
        # =========================================================================

        if edge_type == "graph":

            coefficient = (
                self.w_E
                *
                alpha
                /
                denominator
            )


        elif edge_type == "feature":

            coefficient = (
                self.w_X
                *
                alpha
                /
                denominator
            )


        else:

            raise ValueError(
                f"Unknown edge type: {edge_type}"
            )


        # =========================================================================
        # MESSAGE AGGREGATION
        # =========================================================================

        output = torch.zeros_like(
            h
        )


        output.scatter_add_(
            0,

            dst.unsqueeze(
                1
            ).expand(
                -1,
                h.shape[1]
            ),

            coefficient.unsqueeze(
                1
            )
            *
            h[src],
        )


        return output


    # =========================================================================
    # FORWARD
    # =========================================================================

    def forward(
        self,
        x,
        graph_ei,
        feat_ei,
    ):

        N_total = x.shape[0]


        # =========================================================================
        # INPUT
        # =========================================================================

        h = self.input_proj(
            x
        )


        # =========================================================================
        # GNN LAYERS
        # =========================================================================

        for layer in range(
            self.n_layers
        ):


            # =====================================================================
            # DEGREE — Eqs. (12)-(13)
            # =====================================================================

            degree = torch.full(
                (
                    N_total,
                ),

                self.w_0,

                dtype=h.dtype,

                device=h.device,
            )


            # ---------------------------------------------------------------------
            # Original graph edges
            # ---------------------------------------------------------------------

            if graph_ei.numel() > 0:

                degree.scatter_add_(
                    0,

                    graph_ei[1],

                    torch.full(
                        (
                            graph_ei.shape[1],
                        ),

                        self.w_E,

                        dtype=h.dtype,

                        device=h.device,
                    ),
                )


            # ---------------------------------------------------------------------
            # Feature edges
            # ---------------------------------------------------------------------

            if feat_ei.numel() > 0:

                degree.scatter_add_(
                    0,

                    feat_ei[1],

                    torch.full(
                        (
                            feat_ei.shape[1],
                        ),

                        self.w_X,

                        dtype=h.dtype,

                        device=h.device,
                    ),
                )


            # =====================================================================
            # SELF-GATING
            # =====================================================================

            score_source, score_target = (
                self.gate_scores(
                    h,
                    self.gate_a[layer]
                )
            )


            bias = self.gate_b[
                layer
            ]


            # =====================================================================
            # SELF-LOOP
            #
            # Eq. (15)/(16):
            #
            #       w_0 alpha_uu / d_u * h_u
            #
            # because:
            #
            #       sqrt(d_u) sqrt(d_u) = d_u
            #
            # =====================================================================

            alpha_self = torch.tanh(
                (
                    score_source
                    +
                    score_target
                    +
                    bias
                )
                /
                self.tau
            )


            self_coefficient = (
                self.w_0
                *
                alpha_self
                /
                degree.clamp_min(
                    1e-12
                )
            )


            h_self = (
                self_coefficient.unsqueeze(
                    1
                )
                *
                h
            )


            # =====================================================================
            # ORIGINAL GRAPH EDGES
            # =====================================================================

            h_graph = self.aggregate_edges(

                h=h,

                edge_index=graph_ei,

                degree=degree,

                score_source=score_source,

                score_target=score_target,

                bias=bias,

                edge_type="graph",
            )


            # =====================================================================
            # FEATURE EDGES
            # =====================================================================

            h_feature = self.aggregate_edges(

                h=h,

                edge_index=feat_ei,

                degree=degree,

                score_source=score_source,

                score_target=score_target,

                bias=bias,

                edge_type="feature",
            )


            # =====================================================================
            # TOTAL AGGREGATION
            # =====================================================================

            h_agg = (
                h_self
                +
                h_graph
                +
                h_feature
            )


            # =====================================================================
            # MLP + RESIDUAL
            # =====================================================================

            h = (
                h_agg
                +
                self.mlps[layer](
                    h_agg
                )
            )


            # =====================================================================
            # DROPOUT
            # =====================================================================

            h = F.dropout(
                h,
                p=self.dropout,
                training=self.training
            )


        # =========================================================================
        # CLASSIFICATION
        #
        # Only ORIGINAL graph nodes are classified.
        # Feature nodes are auxiliary nodes.
        # =========================================================================

        return self.classifier(
            h[
                :self.n_orig
            ]
        )


# =============================================================================
# FUNCTION 3
# RUN / TRAIN GRAPHITE
# =============================================================================

def run_graphite(
    data,
    cfg,
    device="cpu",
    seed=42,
    dataset_name=None,
):
    """
    Train GRAPHITE for the number of optimization steps specified in cfg.

    Returns
    -------
    best_val_acc
    test_acc
    """

    # =========================================================================
    # SEED
    # =========================================================================

    torch.manual_seed(
        seed
    )

    np.random.seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


    # =========================================================================
    # LABELS / MASKS
    # =========================================================================

    y = data.y.to(
        device
    ).long()


    train_mask = (
        data.train_mask
        .to(device)
        .bool()
    )


    val_mask = (
        data.val_mask
        .to(device)
        .bool()
    )


    test_mask = (
        data.test_mask
        .to(device)
        .bool()
    )


    # =========================================================================
    # GRAPHITE TRANSFORMATION
    # =========================================================================

    (
        X_ext,
        graph_ei,
        feat_ei,
        N_orig,
        N_feat,
    ) = graphite_transform(

        data=data,

        dataset_name=dataset_name,

        device=device,
    )


    # =========================================================================
    # MODEL
    # =========================================================================

    model = GraphiteGNN(

        in_dim=X_ext.shape[1],

        hidden_dim=cfg.graphite_hidden,

        num_classes=int(
            data.num_classes
        ),

        n_orig=N_orig,

        n_feat=N_feat,

        n_layers=cfg.graphite_n_layers,

        dropout=cfg.graphite_dropout,

        w_X=cfg.graphite_w_X,

        w_0=cfg.graphite_w_0,

        tau=cfg.graphite_tau,

    ).to(device)


    # =========================================================================
    # ADAM
    # =========================================================================

    optimizer = Adam(

        model.parameters(),

        lr=cfg.graphite_lr,

    )


    # =========================================================================
    # BEST VALIDATION RESULT
    # =========================================================================

    best_val_acc = -float(
        "inf"
    )

    best_state = None


    # =========================================================================
    # TRAINING — PAPER = 1000 STEPS
    # =========================================================================

    for step in range(
        cfg.graphite_steps
    ):

        model.train()


        optimizer.zero_grad(
            set_to_none=True
        )


        # ---------------------------------------------------------------------
        # Forward
        # ---------------------------------------------------------------------

        logits = model(

            X_ext,

            graph_ei,

            feat_ei,

        )


        # ---------------------------------------------------------------------
        # Training loss
        # ---------------------------------------------------------------------

        loss = F.cross_entropy(

            logits[
                train_mask
            ],

            y[
                train_mask
            ],

        )


        # ---------------------------------------------------------------------
        # Backpropagation
        # ---------------------------------------------------------------------

        loss.backward()


        optimizer.step()


        # =====================================================================
        # VALIDATION
        # =====================================================================

        model.eval()


        with torch.no_grad():

            val_logits = model(

                X_ext,

                graph_ei,

                feat_ei,

            )


            val_acc = (

                val_logits[
                    val_mask
                ]
                .argmax(
                    dim=-1
                )
                .eq(
                    y[
                        val_mask
                    ]
                )
                .float()
                .mean()
                .item()

            )


        # ---------------------------------------------------------------------
        # Save best validation model
        # ---------------------------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_state = deepcopy(
                model.state_dict()
            )


    # =========================================================================
    # RESTORE BEST VALIDATION MODEL
    # =========================================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )


    # =========================================================================
    # TEST
    # =========================================================================

    model.eval()


    with torch.no_grad():

        test_logits = model(

            X_ext,

            graph_ei,

            feat_ei,

        )


        test_acc = (

            test_logits[
                test_mask
            ]
            .argmax(
                dim=-1
            )
            .eq(
                y[
                    test_mask
                ]
            )
            .float()
            .mean()
            .item()

        )


    # =========================================================================
    # PRINT
    # =========================================================================

    print(
        "\n"
        "============================================================\n"
        "GRAPHITE RESULT\n"
        "============================================================\n"
        f"Dataset       : {dataset_name}\n"
        f"Layers        : {cfg.graphite_n_layers}\n"
        f"Hidden        : {cfg.graphite_hidden}\n"
        f"w_X           : {cfg.graphite_w_X}\n"
        f"w_0           : {cfg.graphite_w_0}\n"
        f"tau           : {cfg.graphite_tau}\n"
        f"Dropout       : {cfg.graphite_dropout}\n"
        f"Learning rate : {cfg.graphite_lr}\n"
        f"Steps         : {cfg.graphite_steps}\n"
        f"Best Val Acc  : {best_val_acc:.6f}\n"
        f"Test Acc      : {test_acc:.6f}\n"
        "============================================================"
    )


    _pred = test_logits[test_mask].argmax(dim=-1)
    _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, y[test_mask])

    return (
        best_val_acc,
        test_acc,
        test_f1,
    )


# =============================================================================
# SECTION — FoSR FAITHFUL NODE-CLASSIFICATION IMPLEMENTATION
#
# Paper:
#   Karhadkar, Banerjee, Montúfar
#   "FoSR: First-order spectral rewiring for addressing oversquashing in GNNs"
#
# IMPORTANT:
#   The original FoSR paper was primarily evaluated for graph-level tasks.
#   For node classification, the FoSR baseline used in the ComFy ICLR-2025
#   comparison is FoSR preprocessing followed by a GCN.
#
#   This implementation therefore:
#       1. performs the official FoSR rewiring operation;
#       2. keeps exactly the same node set;
#       3. adds edges only;
#       4. trains a GCN on the resulting graph.
#
#   No labels are used by the rewiring operation.
# =============================================================================


# -----------------------------------------------------------------------------
# FoSR low-level implementation
# -----------------------------------------------------------------------------

def _fosr_choose_edge_to_add(x, edge_index, degrees):
    """
    Official FoSR first-order edge-selection rule.

    The candidate edge (u,v) minimizes

        y_u y_v

    where

        y_i = x_i / sqrt(d_i + 1).

    Existing edges and self-loops are excluded.
    """

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

    return (
        smallest % n,
        smallest // n,
    )


def _fosr_compute_degrees(edge_index, num_nodes):
    """
    Degree vector of an undirected edge_index.
    """

    degrees = np.zeros(
        num_nodes,
        dtype=np.float64,
    )

    for e in range(edge_index.shape[1]):
        u = edge_index[0, e]
        degrees[u] += 1.0

    return degrees


def _fosr_add_edge(edge_index, u, v):
    """
    Add both directions of an undirected edge.
    """

    new_edge = np.array(
        [
            [u, v],
            [v, u],
        ],
        dtype=np.int64,
    )

    return np.concatenate(
        [
            edge_index,
            new_edge,
        ],
        axis=1,
    )


def _fosr_adj_matvec(edge_index, x, num_nodes):
    """
    Compute A x without materialising A.
    """

    y = np.zeros(
        num_nodes,
        dtype=np.float64,
    )

    for e in range(edge_index.shape[1]):

        u = edge_index[0, e]
        v = edge_index[1, e]

        y[u] += x[v]

    return y


def _fosr_rewire_once(
    edge_index,
    initial_power_iters=5,
):
    """
    One FoSR edge-addition operation.

    This follows the official FoSR implementation:

        1. initialise a random vector x
        2. project out the degree-weighted trivial eigenvector
        3. perform power iteration
        4. select the best edge using the first-order approximation
        5. add that edge
        6. perform one power update

    Returns:
        new_edge_index
    """

    edge_index = np.asarray(
        edge_index,
        dtype=np.int64,
    ).copy()

    n = int(
        edge_index.max()
    ) + 1

    degrees = _fosr_compute_degrees(
        edge_index,
        n,
    )

    # Same random initialization as the official implementation.
    x = (
        2.0
        * np.random.random(n)
        - 1.0
    )

    # ---------------------------------------------------------------
    # Initial power iteration
    # ---------------------------------------------------------------

    for _ in range(
        initial_power_iters
    ):

        denom = max(
            degrees.sum(),
            1e-12,
        )

        x = (
            x
            - (
                np.dot(
                    x,
                    np.sqrt(degrees),
                )
                * np.sqrt(degrees)
                / denom
            )
        )

        y = (
            x
            + _fosr_adj_matvec(
                edge_index,
                x / np.sqrt(
                    np.maximum(
                        degrees,
                        1e-12,
                    )
                ),
                n,
            )
            /
            np.sqrt(
                np.maximum(
                    degrees,
                    1e-12,
                )
            )
        )

        norm = np.linalg.norm(y)

        if norm < 1e-12:
            break

        x = y / norm

    # ---------------------------------------------------------------
    # First-order spectral edge choice
    # ---------------------------------------------------------------

    u, v = _fosr_choose_edge_to_add(
        x,
        edge_index,
        degrees,
    )

    edge_index = _fosr_add_edge(
        edge_index,
        u,
        v,
    )

    degrees[u] += 1.0
    degrees[v] += 1.0

    # ---------------------------------------------------------------
    # Update spectral vector
    # ---------------------------------------------------------------

    denom = max(
        degrees.sum(),
        1e-12,
    )

    x = (
        x
        - (
            np.dot(
                x,
                np.sqrt(degrees),
            )
            * np.sqrt(degrees)
            / denom
        )
    )

    y = (
        x
        + _fosr_adj_matvec(
            edge_index,
            x / np.sqrt(
                np.maximum(
                    degrees,
                    1e-12,
                )
            ),
            n,
        )
        /
        np.sqrt(
            np.maximum(
                degrees,
                1e-12,
            )
        )
    )

    norm = np.linalg.norm(y)

    if norm > 1e-12:
        x = y / norm

    return edge_index


def _fosr_rewire(
    edge_index,
    num_iterations,
    initial_power_iters=5,
):
    """
    Apply FoSR repeatedly.

    The official ComFy implementation calls the original FoSR routine
    one iteration at a time.
    """

    e = (
        edge_index
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    # Make sure the input is explicitly undirected.
    e = np.concatenate(
        [
            e,
            e[[1, 0], :],
        ],
        axis=1,
    )

    # Remove duplicates.
    e = np.unique(
        e,
        axis=1,
    )

    for _ in range(
        num_iterations
    ):

        e = _fosr_rewire_once(
            e,
            initial_power_iters=initial_power_iters,
        )

    return torch.tensor(
        e,
        dtype=torch.long,
    )


# -----------------------------------------------------------------------------
# FoSR GCN
# -----------------------------------------------------------------------------

class _FoSRGCN(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        num_layers=1,
        dropout=0.0,
    ):

        super().__init__()

        from torch_geometric.nn import GCNConv

        self.dropout = dropout

        self.convs = nn.ModuleList()

        if num_layers <= 1:

            self.convs.append(
                GCNConv(
                    in_dim,
                    num_classes,
                )
            )

        else:

            self.convs.append(
                GCNConv(
                    in_dim,
                    hidden_dim,
                )
            )

            for _ in range(
                num_layers - 2
            ):

                self.convs.append(
                    GCNConv(
                        hidden_dim,
                        hidden_dim,
                    )
                )

            self.convs.append(
                GCNConv(
                    hidden_dim,
                    num_classes,
                )
            )

    def forward(
        self,
        x,
        edge_index,
    ):

        for i, conv in enumerate(
            self.convs
        ):

            x = conv(
                x,
                edge_index,
            )

            if i < len(
                self.convs
            ) - 1:

                x = F.relu(x)

                x = F.dropout(
                    x,
                    p=self.dropout,
                    training=self.training,
                )

        return x


# -----------------------------------------------------------------------------
# FoSR faithful benchmark runner
# -----------------------------------------------------------------------------

def run_fosr(
    data,
    cfg,
    device,
    seed,
):
    """
    Faithful FoSR node-classification benchmark.

    Rewiring:
        FoSR

    Downstream:
        GCN

    Important:
        The rewiring itself is completely label-free.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    N = int(data.num_nodes)
    C = int(data.num_classes)

    X = data.x.float().to(device)
    y = data.y.long().to(device)

    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    # ---------------------------------------------------------------
    # Handle multi-split datasets
    # ---------------------------------------------------------------

    if train_mask.dim() == 2:

        split_id = (
            seed
            % train_mask.size(1)
        )

        train_mask = train_mask[
            :, split_id
        ]

        val_mask = val_mask[
            :, split_id
        ]

        test_mask = test_mask[
            :, split_id
        ]

    # ---------------------------------------------------------------
    # FoSR parameters
    # ---------------------------------------------------------------

    iterations = int(
        getattr(
            cfg,
            "fosr_iterations",
            10,
        )
    )

    initial_power_iters = int(
        getattr(
            cfg,
            "fosr_initial_power_iters",
            5,
        )
    )

    hidden_dim = int(
        getattr(
            cfg,
            "fosr_hidden",
            64,
        )
    )

    num_layers = int(
        getattr(
            cfg,
            "fosr_layers",
            1,
        )
    )

    dropout = float(
        getattr(
            cfg,
            "fosr_dropout",
            0.0,
        )
    )

    lr = float(
        getattr(
            cfg,
            "fosr_lr",
            1e-2,
        )
    )

    weight_decay = float(
        getattr(
            cfg,
            "fosr_wd",
            5e-4,
        )
    )

    epochs = int(
        getattr(
            cfg,
            "fosr_epochs",
            100,
        )
    )

    # ---------------------------------------------------------------
    # Rewire
    # ---------------------------------------------------------------

    print("\n" + "=" * 78)
    print("FoSR — FAITHFUL NODE-CLASSIFICATION RUN")
    print("=" * 78)

    print(
        f"Nodes                 : {N:,}"
    )

    print(
        f"Original edges        : "
        f"{data.edge_index.shape[1] // 2:,}"
    )

    print(
        f"FoSR iterations       : {iterations}"
    )

    print(
        f"Initial power iters  : "
        f"{initial_power_iters}"
    )

    print(
        f"GCN layers            : {num_layers}"
    )

    print(
        f"GCN hidden            : {hidden_dim}"
    )

    print(
        f"Learning rate         : {lr}"
    )

    print(
        f"Epochs                : {epochs}"
    )

    edge_index_fosr = _fosr_rewire(
        data.edge_index,
        num_iterations=iterations,
        initial_power_iters=initial_power_iters,
    ).to(device)

    print(
        f"Rewired edges         : "
        f"{edge_index_fosr.shape[1] // 2:,}"
    )

    # ---------------------------------------------------------------
    # GCN
    # ---------------------------------------------------------------

    model = _FoSRGCN(
        in_dim=X.shape[1],
        hidden_dim=hidden_dim,
        num_classes=C,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = -float("inf")
    best_state = None

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    for epoch in range(
        epochs
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            X,
            edge_index_fosr,
        )

        loss = F.cross_entropy(
            logits[train_mask],
            y[train_mask],
        )

        loss.backward()

        optimizer.step()

        # -----------------------------------------------------------
        # Validation
        # -----------------------------------------------------------

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X,
                edge_index_fosr,
            )

            val_acc = (
                val_logits[val_mask]
                .argmax(dim=-1)
                .eq(y[val_mask])
                .float()
                .mean()
                .item()
            )

        if val_acc > best_val:

            best_val = val_acc

            best_state = copy.deepcopy(
                model.state_dict()
            )

    # ---------------------------------------------------------------
    # Restore best validation checkpoint
    # ---------------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    model.eval()

    with torch.no_grad():

        logits = model(
            X,
            edge_index_fosr,
        )

        test_acc = (
            logits[test_mask]
            .argmax(dim=-1)
            .eq(y[test_mask])
            .float()
            .mean()
            .item()
        )

    _pred = logits[test_mask].argmax(dim=-1)
    _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, y[test_mask])

    print(
        f"[FoSR] best_val={best_val:.4f} "
        f"test={test_acc:.4f} f1={test_f1:.4f}"
    )

    print("=" * 78)

    return best_val, test_acc, test_f1




# =============================================================================
# SECTION — ComFy FAITHFUL IMPLEMENTATION
#
# Paper:
#   Rubio-Madrigal, Jamadandi, Burkholz
#   "GNNs Getting ComFy: Community and Feature Similarity Guided Rewiring"
#   ICLR 2025
#
# Algorithm 6:
#   ComFy — maximizing feature similarity across communities
#
# Faithful steps:
#   1. Louvain community detection
#   2. Pairwise cosine similarity of node features
#   3. Community-pair budgets
#   4. ADD: select non-edges which most improve mean similarity
#   5. DEL: select existing edges whose removal most improves mean similarity
#   6. Train GCN on resulting graph
#
# No labels are used by the rewiring.
# =============================================================================


def _comfy_cosine_similarity(
    X,
):
    """
    Pairwise cosine similarity.

    The official implementation computes:

        X X^T
        ----------------
        ||X_i|| ||X_j||

    We process it in chunks to avoid unnecessary Python loops.
    """

    X = X.float()

    X_norm = F.normalize(
        X,
        p=2,
        dim=1,
    )

    return (
        X_norm
        @
        X_norm.t()
    )


def _comfy_make_communities(
    edge_index,
    num_nodes,
    seed,
):
    """
    Louvain community detection, matching the ComFy implementation.
    """

    edge_cpu = (
        edge_index
        .detach()
        .cpu()
    )

    G = to_networkx(
        type(
            "TmpData",
            (),
            {
                "edge_index": edge_cpu,
                "num_nodes": num_nodes,
            },
        )(),
        to_undirected=True,
    )

    # Ensure every node exists, including isolated nodes.
    G.add_nodes_from(
        range(num_nodes)
    )

    communities = list(
        nx.community.louvain_communities(
            G,
            seed=seed,
        )
    )

    return G, communities


def _comfy_rewire(
    data,
    budget_add,
    budget_delete,
    seed,
):
    """
    Faithful ComFy rewiring.

    This follows Algorithm 6 / the official repository:

        - Louvain communities
        - pairwise cosine feature similarity
        - budget proportional to |C_i| |C_j|
        - rank candidate additions/deletions by resulting
          community-pair mean similarity
    """

    X = (
        data.x
        .detach()
        .cpu()
        .float()
    )

    N = int(
        data.num_nodes
    )

    # ---------------------------------------------------------------
    # Original graph
    # ---------------------------------------------------------------

    G = nx.Graph()

    G.add_nodes_from(
        range(N)
    )

    e = (
        data.edge_index
        .detach()
        .cpu()
        .numpy()
    )

    for u, v in zip(
        e[0],
        e[1],
    ):

        u = int(u)
        v = int(v)

        if u == v:
            continue

        G.add_edge(
            u,
            v,
        )

    # ---------------------------------------------------------------
    # Louvain
    # ---------------------------------------------------------------

    communities = list(
        nx.community.louvain_communities(
            G,
            seed=seed,
        )
    )

    # Ensure all nodes occur in a community.
    assigned = set()

    for c in communities:
        assigned.update(c)

    for node in range(N):

        if node not in assigned:

            communities.append(
                {node}
            )

    cluster_of = {}

    for cid, community in enumerate(
        communities
    ):

        for node in community:

            cluster_of[node] = cid

    M = len(
        communities
    )

    # ---------------------------------------------------------------
    # Cosine similarity
    # ---------------------------------------------------------------

    Xn = F.normalize(
        X,
        p=2,
        dim=1,
    )

    similarity = (
        Xn @ Xn.t()
    ).numpy()

    # ---------------------------------------------------------------
    # Community-pair budgets
    #
    # This follows the official ComFy implementation:
    #
    #   score(i,j) = |Ci| |Cj|
    #
    # and normalizes all community-pair scores.
    # ---------------------------------------------------------------

    pair_scores = {}

    total_score = 0.0

    for i in range(M):

        ni = len(
            communities[i]
        )

        for j in range(
            i,
            M,
        ):

            nj = len(
                communities[j]
            )

            score = (
                ni * nj
            )

            pair_scores[
                (i, j)
            ] = score

            total_score += score

    if total_score <= 0:

        total_score = 1.0

    budgets_add = {}

    budgets_delete = {}

    for pair, score in pair_scores.items():

        frac = (
            score
            /
            total_score
        )

        budgets_add[pair] = int(
            budget_add * frac
        )

        budgets_delete[pair] = int(
            budget_delete * frac
        )

    # ---------------------------------------------------------------
    # Rewiring
    # ---------------------------------------------------------------

    edges_added = set()
    edges_deleted = set()

    for i in range(M):

        C_i = list(
            communities[i]
        )

        set_i = set(
            C_i
        )

        for j in range(
            i,
            M,
        ):

            C_j = list(
                communities[j]
            )

            set_j = set(
                C_j
            )

            pair = (
                i,
                j,
            )

            add_budget = budgets_add[
                pair
            ]

            del_budget = budgets_delete[
                pair
            ]

            if (
                add_budget <= 0
                and del_budget <= 0
            ):
                continue

            # -------------------------------------------------------
            # Existing edges between C_i and C_j
            # -------------------------------------------------------

            existing_edges = []

            for u in C_i:

                for v in C_j:

                    if u == v:
                        continue

                    if G.has_edge(
                        u,
                        v,
                    ):

                        existing_edges.append(
                            (
                                u,
                                v,
                            )
                        )

            # For i == j, the loops above can include each
            # undirected edge twice. Remove duplicates.
            existing_edges = list(
                {
                    tuple(
                        sorted(
                            e
                        )
                    )
                    for e in existing_edges
                }
            )

            # -------------------------------------------------------
            # Current mean similarity
            # -------------------------------------------------------

            if existing_edges:

                current_sim = np.mean(
                    [
                        similarity[u, v]
                        for u, v
                        in existing_edges
                    ]
                )

            else:

                current_sim = 0.0

            num_existing = len(
                existing_edges
            )

            # -------------------------------------------------------
            # Candidate additions
            #
            # rank_add(u,v)
            #
            # = (sim * |E| + sim(u,v))
            #   ---------------------
            #        |E| + 1
            # -------------------------------------------------------

            addition_candidates = []

            if add_budget > 0:

                for u in C_i:

                    for v in C_j:

                        if u == v:
                            continue

                        if G.has_edge(
                            u,
                            v,
                        ):
                            continue

                        candidate_sim = (
                            similarity[
                                u,
                                v,
                            ]
                        )

                        if (
                            candidate_sim
                            <= current_sim
                        ):
                            continue

                        if num_existing > 0:

                            rank = (
                                current_sim
                                * num_existing
                                + candidate_sim
                            ) / (
                                num_existing
                                + 1
                            )

                        else:

                            rank = candidate_sim

                        addition_candidates.append(
                            (
                                rank,
                                u,
                                v,
                            )
                        )

            addition_candidates.sort(
                key=lambda z: z[0]
            )

            for (
                rank,
                u,
                v,
            ) in addition_candidates[
                -add_budget:
            ]:

                key = tuple(
                    sorted(
                        (
                            u,
                            v,
                        )
                    )
                )

                if key in edges_added:
                    continue

                if G.has_edge(
                    u,
                    v,
                ):
                    continue

                if len(
                    edges_added
                ) >= budget_add:
                    break

                G.add_edge(
                    u,
                    v,
                )

                edges_added.add(
                    key
                )

            # -------------------------------------------------------
            # Candidate deletions
            #
            # rank_del(u,v)
            #
            # = (sim * |E| - sim(u,v))
            #   ---------------------
            #        |E| - 1
            # -------------------------------------------------------

            deletion_candidates = []

            if (
                del_budget > 0
                and num_existing > 1
            ):

                for u, v in existing_edges:

                    edge_sim = (
                        similarity[
                            u,
                            v,
                        ]
                    )

                    if (
                        edge_sim
                        >= current_sim
                    ):
                        continue

                    rank = (
                        current_sim
                        * num_existing
                        - edge_sim
                    ) / (
                        num_existing
                        - 1
                    )

                    deletion_candidates.append(
                        (
                            rank,
                            u,
                            v,
                        )
                    )

            deletion_candidates.sort(
                key=lambda z: z[0]
            )

            for (
                rank,
                u,
                v,
            ) in deletion_candidates[
                -del_budget:
            ]:

                key = tuple(
                    sorted(
                        (
                            u,
                            v,
                        )
                    )
                )

                if key in edges_deleted:
                    continue

                if not G.has_edge(
                    u,
                    v,
                ):
                    continue

                if len(
                    edges_deleted
                ) >= budget_delete:
                    break

                G.remove_edge(
                    u,
                    v,
                )

                edges_deleted.add(
                    key
                )

    # ---------------------------------------------------------------
    # Convert back to edge_index
    # ---------------------------------------------------------------

    edges = list(
        G.edges()
    )

    if edges:

        undirected = np.asarray(
            edges,
            dtype=np.int64,
        )

        directed = np.concatenate(
            [
                undirected,
                undirected[:, [1, 0]],
            ],
            axis=0,
        )

        edge_index = torch.tensor(
            directed.T,
            dtype=torch.long,
        )

    else:

        edge_index = torch.empty(
            (
                2,
                0,
            ),
            dtype=torch.long,
        )

    return (
        edge_index,
        len(edges_added),
        len(edges_deleted),
    )


# -----------------------------------------------------------------------------
# ComFy GCN
# -----------------------------------------------------------------------------

class _ComFyGCN(nn.Module):

    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_classes,
        num_layers=1,
        dropout=0.0,
    ):

        super().__init__()

        from torch_geometric.nn import GCNConv

        self.dropout = dropout

        self.convs = nn.ModuleList()

        if num_layers <= 1:

            self.convs.append(
                GCNConv(
                    in_dim,
                    num_classes,
                )
            )

        else:

            self.convs.append(
                GCNConv(
                    in_dim,
                    hidden_dim,
                )
            )

            for _ in range(
                num_layers - 2
            ):

                self.convs.append(
                    GCNConv(
                        hidden_dim,
                        hidden_dim,
                    )
                )

            self.convs.append(
                GCNConv(
                    hidden_dim,
                    num_classes,
                )
            )

    def forward(
        self,
        x,
        edge_index,
    ):

        for i, conv in enumerate(
            self.convs
        ):

            x = conv(
                x,
                edge_index,
            )

            if i < len(
                self.convs
            ) - 1:

                x = F.relu(x)

                x = F.dropout(
                    x,
                    p=self.dropout,
                    training=self.training,
                )

        return x


# -----------------------------------------------------------------------------
# ComFy faithful runner
# -----------------------------------------------------------------------------

def run_comfy(
    data,
    cfg,
    device,
    seed,
):
    """
    Faithful ComFy benchmark runner.

    ComFy rewiring is performed once before GCN training.

    No label information is used in the graph transformation.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    X = data.x.float().to(device)
    y = data.y.long().to(device)

    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    if train_mask.dim() == 2:

        split_id = (
            seed
            % train_mask.size(1)
        )

        train_mask = train_mask[
            :, split_id
        ]

        val_mask = val_mask[
            :, split_id
        ]

        test_mask = test_mask[
            :, split_id
        ]

    # ---------------------------------------------------------------
    # Official ComFy configuration
    #
    # The official implementation defaults:
    #
    # hidden_dimension = 32
    # LR = 0.01
    # dropout = 0
    # weight_decay = 5e-4
    # training = 100 epochs
    # budgets = 100 additions + 100 deletions
    #
    # ---------------------------------------------------------------

    budget_add = int(
        getattr(
            cfg,
            "comfy_budget_add",
            100,
        )
    )

    budget_delete = int(
        getattr(
            cfg,
            "comfy_budget_delete",
            100,
        )
    )

    hidden_dim = int(
        getattr(
            cfg,
            "comfy_hidden",
            32,
        )
    )

    num_layers = int(
        getattr(
            cfg,
            "comfy_layers",
            1,
        )
    )

    dropout = float(
        getattr(
            cfg,
            "comfy_dropout",
            0.0,
        )
    )

    lr = float(
        getattr(
            cfg,
            "comfy_lr",
            0.01,
        )
    )

    weight_decay = float(
        getattr(
            cfg,
            "comfy_wd",
            5e-4,
        )
    )

    epochs = int(
        getattr(
            cfg,
            "comfy_epochs",
            100,
        )
    )

    # ---------------------------------------------------------------
    # Rewire
    # ---------------------------------------------------------------

    print("\n" + "=" * 78)
    print("ComFy — FAITHFUL NODE-CLASSIFICATION RUN")
    print("=" * 78)

    print(
        f"Nodes                : "
        f"{data.num_nodes:,}"
    )

    print(
        f"Original edges       : "
        f"{data.edge_index.shape[1] // 2:,}"
    )

    print(
        f"Add budget            : "
        f"{budget_add}"
    )

    print(
        f"Delete budget         : "
        f"{budget_delete}"
    )

    edge_index_comfy, added, deleted = (
        _comfy_rewire(
            data,
            budget_add=budget_add,
            budget_delete=budget_delete,
            seed=seed,
        )
    )

    edge_index_comfy = (
        edge_index_comfy.to(device)
    )

    print(
        f"Edges actually added  : "
        f"{added}"
    )

    print(
        f"Edges actually deleted: "
        f"{deleted}"
    )

    print(
        f"Final edges           : "
        f"{edge_index_comfy.shape[1] // 2:,}"
    )

    # ---------------------------------------------------------------
    # GCN
    # ---------------------------------------------------------------

    model = _ComFyGCN(
        in_dim=X.shape[1],
        hidden_dim=hidden_dim,
        num_classes=int(
            data.num_classes
        ),
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val = -float("inf")
    best_state = None

    # ---------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------

    for epoch in range(
        epochs
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(
            X,
            edge_index_comfy,
        )

        loss = F.cross_entropy(
            logits[train_mask],
            y[train_mask],
        )

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X,
                edge_index_comfy,
            )

            val_acc = (
                val_logits[val_mask]
                .argmax(dim=-1)
                .eq(y[val_mask])
                .float()
                .mean()
                .item()
            )

        if val_acc > best_val:

            best_val = val_acc

            best_state = copy.deepcopy(
                model.state_dict()
            )

    # ---------------------------------------------------------------
    # Restore best checkpoint
    # ---------------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    model.eval()

    with torch.no_grad():

        logits = model(
            X,
            edge_index_comfy,
        )

        test_acc = (
            logits[test_mask]
            .argmax(dim=-1)
            .eq(y[test_mask])
            .float()
            .mean()
            .item()
        )

    _pred = logits[test_mask].argmax(dim=-1)
    _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, y[test_mask])

    print(
        f"[ComFy] best_val={best_val:.4f} "
        f"test={test_acc:.4f} f1={test_f1:.4f}"
    )

    print("=" * 78)

    return best_val, test_acc, test_f1





# =============================================================================
#  GLARE model block imports
# =============================================================================
#  The GLARE model code that follows (Sections 2-3, verbatim from
#  glare_homophily_benchmark.py) relies on these torch_geometric names being
#  available at module scope.  The base benchmark file above imports `degree`
#  locally inside individual functions, so we add the module-level aliases here.
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, LINKX  # noqa: F401,E402
from torch_geometric.utils import degree as pyg_degree, to_undirected  # noqa: F401,E402
# =============================================================================
#  SECTION 2 — Matched-density subsampling + semi-supervised label masking
#  (VERBATIM from original)
# =============================================================================

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


# =============================================================================
#  SECTION 3 — GLARE representation / rewiring model
#  (VERBATIM from original — no changes to any model code)
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
    """GraphSAGE-style layer with continuous edge weights (hand-rolled sparse matmul)."""
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
        if u == v:
            continue
        a, b = sorted((u, v))
        if (a, b) not in existing:
            existing.add((a, b)); neg_edges.append((a, b))
    if neg_edges:
        cand.append(torch.tensor(neg_edges, dtype=torch.long).T)
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


def rewire_glare(data, cfg, device, seed):
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




# #############################################################################
# #############################################################################
# ##                                                                         ##
# ##   PART C  —  HOMOPHILY / GRAPH-STATISTICS BENCHMARK HARNESS  (NEW)      ##
# ##                                                                         ##
# ##   Everything above this banner is the VERBATIM model + rewiring code    ##
# ##   from benchmark_others.py (IDGL, GADC, LPkG, DHGR, GRAPHITE, FoSR,     ##
# ##   ComFy, data loading, synthetic generators) and from                  ##
# ##   glare_homophily_benchmark.py (GLARE, Sections 2-3).  No model         ##
# ##   architecture has been modified.                                      ##
# ##                                                                         ##
# ##   Part C adds ONLY:                                                     ##
# ##     1. Standard homophily / heterophily measures (label + feature)     ##
# ##     2. rewire_<method>() wrappers that extract the rewired graph and    ##
# ##        the learned embedding (if any) from each method — NO downstream  ##
# ##        classifier is trained.                                          ##
# ##     3. A resumable "run-one / write-one" harness with one CSV table     ##
# ##        per measure plus a graph-statistics table.                       ##
# ##                                                                         ##
# #############################################################################
# #############################################################################

import math as _math
import csv as _csv

# =============================================================================
#  SECTION C1 — Homophily / heterophily measures
# =============================================================================
#
#  All measures operate on a *directed* edge_index of shape [2, E] (an
#  undirected graph is expected to be stored as two opposite directed edges,
#  which is the convention used throughout PyG and both source benchmarks).
#
#  LABEL-BASED MEASURES
#  --------------------
#    edge homophily          h_edge   (Zhu et al., NeurIPS 2020)
#        fraction of edges whose endpoints share a label.  Range [0, 1].
#
#    node homophily          h_node   (Pei et al., ICLR 2020 — Geom-GCN)
#        per-node fraction of same-label neighbours, averaged over nodes with
#        at least one neighbour.  Range [0, 1].
#
#    adjusted homophily      h_adj    (Platonov et al., NeurIPS 2023;
#                                      a.k.a. Newman assortativity coefficient)
#        h_adj = (h_edge - Sum_k p_k^2) / (1 - Sum_k p_k^2)
#        where  p_k = (Sum_{v: y_v=k} d_v) / (2|E|)  is the DEGREE-WEIGHTED
#        class distribution (the probability that a random edge-endpoint has
#        label k).  This is the correct, class-size / class-count invariant
#        measure.  Range (-inf, 1]; 0 == the random-graph baseline.
#        (The previous code used node-count fractions |C_k|/N here, which is
#         NOT the standard adjusted homophily — that has been fixed.)
#
#    class-insensitive edge homophily  h_ci  (Lim et al., NeurIPS 2021)
#        h_ci = (1/(C-1)) * Sum_k max(0, h_k - |C_k|/N),
#        where h_k is the edge homophily restricted to edges leaving class k.
#        Range [0, 1]; corrects for class imbalance differently from h_adj.
#
#    label informativeness   LI       (Platonov et al., NeurIPS 2023)
#        LI = I(y_u ; y_v) / H(y),  with (u,v) the endpoints of a uniformly
#        sampled edge.  Normalised mutual information between endpoint labels.
#        Range [0, 1]; the recommended companion to adjusted homophily because
#        it distinguishes different *kinds* of heterophily.
#
#  FEATURE-BASED MEASURE
#  ---------------------
#    feature homophily       h_feat   (generalised edge homophily,
#                                      Jin et al., NeurIPS 2022)
#        mean cosine similarity of L2-normalised node representations across
#        edges.  Range [-1, 1].  Computed on the ORIGINAL features and, when a
#        method produces one, on the method's LEARNED EMBEDDING.
# =============================================================================


def hom_edge(edge_index, y):
    """Edge homophily (Zhu et al. 2020)."""
    src, dst = edge_index
    if src.numel() == 0:
        return float("nan")
    return float((y[src] == y[dst]).float().mean().item())


def hom_node(edge_index, y, num_nodes):
    """Node homophily (Pei et al. 2020), averaged over nodes with >=1 neighbour."""
    src, dst = edge_index
    if src.numel() == 0:
        return float("nan")
    same = (y[src] == y[dst]).float()
    sum_same = torch.zeros(num_nodes).scatter_add_(0, dst.cpu(), same.cpu())
    deg = torch.zeros(num_nodes).scatter_add_(
        0, dst.cpu(), torch.ones(dst.numel()))
    has_nb = deg > 0
    if not has_nb.any():
        return float("nan")
    return float((sum_same[has_nb] / deg[has_nb]).mean().item())


def _degree_weighted_class_probs(edge_index, y, num_classes):
    """
    p_k = fraction of directed-edge endpoints whose label is k
        = (Sum_{v: y_v=k} deg(v)) / (2|E|).
    Concatenating src and dst counts every node once per incident endpoint,
    which is exactly the degree-weighted distribution used by adjusted
    homophily / the assortativity coefficient.
    """
    src, dst = edge_index
    endpoints = torch.cat([src, dst]).cpu().long()
    if endpoints.numel() == 0:
        return torch.full((num_classes,), float("nan"))
    counts = torch.bincount(y.cpu().long()[endpoints], minlength=num_classes).float()
    total = counts.sum().clamp(min=1.0)
    return counts / total


def hom_adjusted(edge_index, y, num_classes):
    """Adjusted homophily / assortativity coefficient (Platonov et al. 2023)."""
    h = hom_edge(edge_index, y)
    if _math.isnan(h):
        return float("nan")
    p = _degree_weighted_class_probs(edge_index, y, num_classes)
    expected = float((p ** 2).sum().item())
    denom = 1.0 - expected
    if abs(denom) < 1e-12:
        return float("nan")
    return (h - expected) / denom


def hom_class_insensitive(edge_index, y, num_nodes, num_classes):
    """Class-insensitive edge homophily (Lim et al. 2021)."""
    src, dst = edge_index
    if src.numel() == 0 or num_classes < 2:
        return float("nan")
    y = y.cpu().long()
    ys, yd = y[src.cpu()], y[dst.cpu()]
    same = (ys == yd).float()
    # denominator per class: # directed edges leaving a class-k node
    denom_k = torch.zeros(num_classes).scatter_add_(0, ys, torch.ones(ys.numel()))
    num_k = torch.zeros(num_classes).scatter_add_(0, ys, same)
    class_counts = torch.bincount(y, minlength=num_classes).float()
    N = float(num_nodes)
    acc = 0.0
    for k in range(num_classes):
        if denom_k[k] <= 0:
            continue
        h_k = float(num_k[k] / denom_k[k])
        acc += max(0.0, h_k - float(class_counts[k]) / N)
    return acc / (num_classes - 1)


def label_informativeness(edge_index, y, num_classes):
    """
    Label informativeness LI = I(y_u; y_v) / H(y)  (Platonov et al. 2023),
    with (u, v) the endpoints of a uniformly-sampled directed edge.
    """
    src, dst = edge_index
    if src.numel() == 0:
        return float("nan")
    y = y.cpu().long()
    ys, yd = y[src.cpu()], y[dst.cpu()]
    idx = ys * num_classes + yd
    joint = torch.bincount(idx, minlength=num_classes * num_classes).float()
    joint = (joint / joint.sum()).view(num_classes, num_classes)
    p_src = joint.sum(dim=1)
    p_dst = joint.sum(dim=0)
    eps = 1e-12
    outer = p_src.unsqueeze(1) * p_dst.unsqueeze(0)
    mask = joint > 0
    mi = float((joint[mask] * (joint[mask] / (outer[mask] + eps)).log()).sum().item())
    # endpoint marginal entropy (symmetric average of the two marginals)
    p_end = 0.5 * (p_src + p_dst)
    p_end = p_end[p_end > 0]
    H = float(-(p_end * p_end.log()).sum().item())
    if H < 1e-12:
        return float("nan")
    return mi / H


def feature_homophily(edge_index, feat, chunk_size=200_000):
    """
    Generalised edge homophily (Jin et al. 2022): mean cosine similarity of
    L2-normalised node representations across all directed edges.  `feat` may
    be raw features OR a learned embedding.  Returns nan if feat is None / no
    edges.  Range [-1, 1].
    """
    if feat is None:
        return float("nan")
    src, dst = edge_index
    if src.numel() == 0:
        return float("nan")
    src = src.cpu()
    dst = dst.cpu()
    fn = F.normalize(feat.detach().cpu().float(), dim=-1)
    total, E = 0.0, int(src.numel())
    for s in range(0, E, chunk_size):
        e = min(s + chunk_size, E)
        total += float((fn[src[s:e]] * fn[dst[s:e]]).sum(dim=-1).sum().item())
    return total / E


def all_label_measures(edge_index, y, num_nodes, num_classes):
    """Every label-based measure as a JSON-serialisable dict."""
    ei = edge_index.cpu()
    y = y.cpu().long()
    return {
        "h_edge": hom_edge(ei, y),
        "h_node": hom_node(ei, y, num_nodes),
        "h_adj":  hom_adjusted(ei, y, num_classes),
        "h_ci":   hom_class_insensitive(ei, y, num_nodes, num_classes),
        "LI":     label_informativeness(ei, y, num_classes),
    }


def graph_stats(edge_index, num_nodes):
    """Basic structural statistics of a directed edge_index."""
    ei = edge_index.cpu()
    E_dir = int(ei.shape[1])
    # count undirected edges (unordered pairs, ignoring self-loops)
    src, dst = ei
    non_self = src != dst
    a = torch.minimum(src[non_self], dst[non_self])
    b = torch.maximum(src[non_self], dst[non_self])
    keys = a.long() * int(num_nodes) + b.long()
    E_undir = int(torch.unique(keys).numel())
    self_loops = int((~non_self).sum().item())
    return {
        "edges_directed":   E_dir,
        "edges_undirected": E_undir,
        "self_loops":       self_loops,
        "avg_degree":       (E_dir / num_nodes) if num_nodes else float("nan"),
        "density":          (E_dir / (num_nodes * (num_nodes - 1)))
                            if num_nodes > 1 else float("nan"),
    }


# =============================================================================
#  SECTION C2 — Per-method rewiring wrappers
# =============================================================================
#
#  Each wrapper returns a RewireResult:
#     edge_index   : LongTensor [2, E]  — the rewired node-node graph on the
#                    ORIGINAL node set (falls back to the original edges for
#                    methods that transform features rather than structure).
#     embedding    : FloatTensor [N, d] or None — the learned node
#                    representation the method produces, if any.  Feature
#                    homophily is measured on this when present.
#     structure_changed : bool
#     extra        : dict of method-specific statistics (JSON-serialisable).
#
#  The wrappers call each method's OWN rewiring / representation code (the
#  verbatim implementations above); they do NOT train the paper's downstream
#  classifier.  Model architectures are therefore untouched.
# =============================================================================

# The verbatim GLARE routine from Part B is named `rewire_glare`; alias it so
# the wrapper below (also public-facing as rewire_glare) can call it.
rewire_glare_orig = rewire_glare  # noqa: F811  (Part-B definition)


class RewireResult:
    def __init__(self, edge_index, embedding=None, structure_changed=True, extra=None):
        self.edge_index = edge_index.cpu().long()
        self.embedding = None if embedding is None else embedding.detach().cpu().float()
        self.structure_changed = bool(structure_changed)
        self.extra = extra or {}


# ---------------------------------------------------------------------------
def rewire_glare(data, cfg, device, seed):
    """GLARE — learned rewiring + learned representation Z = [h1; h2]."""
    cfg.device = str(device)
    cfg.seed = seed
    cfg.num_classes = data.num_classes
    # Defensive: the verbatim GLARE routine expects 1-D masks. The base loader
    # already returns 1-D masks, but collapse just in case a dataset yields 2-D.
    if getattr(data.train_mask, "dim", lambda: 1)() == 2:
        col = seed % data.train_mask.shape[1]
        data = SimpleNamespace(
            x=data.x, y=data.y, edge_index=data.edge_index,
            train_mask=data.train_mask[:, col],
            val_mask=data.val_mask[:, col],
            test_mask=data.test_mask[:, col],
            num_nodes=data.num_nodes, num_classes=data.num_classes)
    # `rewire_glare_orig` is the verbatim GLARE routine (aliased below to avoid
    # a name clash with this wrapper).
    ei, Z = rewire_glare_orig(data, cfg, device, seed)
    extra = {"embedding_dim": int(Z.shape[1])}
    return RewireResult(ei, embedding=Z, structure_changed=True, extra=extra)


# ---------------------------------------------------------------------------
def rewire_fosr(data, cfg, device, seed):
    """FoSR — spectral edge addition (label-free).  No new embedding."""
    iters = int(getattr(cfg, "fosr_iterations", 10))
    ip = int(getattr(cfg, "fosr_initial_power_iters", 5))
    ei = _fosr_rewire(data.edge_index, num_iterations=iters,
                      initial_power_iters=ip)
    return RewireResult(ei, embedding=None, structure_changed=True,
                        extra={"fosr_iterations": iters})


# ---------------------------------------------------------------------------
def rewire_comfy(data, cfg, device, seed):
    """ComFy — community-guided edge add/delete (label-free).  No embedding."""
    ba = int(getattr(cfg, "comfy_budget_add", 100))
    bd = int(getattr(cfg, "comfy_budget_delete", 100))
    ei, added, deleted = _comfy_rewire(data, budget_add=ba,
                                       budget_delete=bd, seed=seed)
    return RewireResult(ei, embedding=None, structure_changed=True,
                        extra={"edges_added": int(added),
                               "edges_deleted": int(deleted)})


# ---------------------------------------------------------------------------
def rewire_dhgr(data, cfg, device, seed, dataset_name=None):
    """DHGR — official graph learner (top-k similarity + pruning + merge).

    The downstream DHGR GCN consumes the ORIGINAL features, so no learned
    embedding is reported for feature homophily.
    """
    if DHGRModelHandler is None:
        raise ImportError("DHGR repo not available (set $DHGR_ROOT or place "
                          "the DHGR/ folder next to this script).")
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

    pyg_data = Data(x=x, y=y, edge_index=edge_index,
                    train_mask=tm, val_mask=vm, test_mask=te)
    pyg_data.num_nodes = data.num_nodes

    min_deg = int(getattr(cfg, "dhgr_min_deg", 10))
    min_deg_ratio = float(getattr(cfg, "dhgr_min_deg_ratio", 1.0))
    if dataset_name == "Roman-empire":
        min_deg = int(getattr(cfg, "dhgr_roman_min_deg", 3))
        min_deg_ratio = float(getattr(cfg, "dhgr_roman_min_deg_ratio", 1.8))

    graph_handler = DHGRModelHandler(
        in_size=pyg_data.num_features,
        num_classes=int(data.num_classes),
        thres_min_deg=min_deg,
        thres_min_deg_ratio=min_deg_ratio,
        hidden=128,
        device=device,
        save_dir=str(getattr(cfg, "dhgr_save_dir", "./dhgr_ckpt/")),
        seed=seed,
        num_epoch=int(getattr(cfg, "dhgr_gl_epochs", 10)),
        num_epoch_finetune=int(getattr(cfg, "dhgr_gl_finetune_epochs", 30)),
        window_size=getattr(cfg, "dhgr_window", [5000, 5000]),
        lr=float(getattr(cfg, "dhgr_gl_lr", 0.001)),
        weight_decay=float(getattr(cfg, "dhgr_gl_wd", 5e-3)),
        shuffle=[False, False],
        drop_last=[False, False],
        moment=int(getattr(cfg, "dhgr_moment", 1)),
        use_cpu_cache=bool(getattr(cfg, "dhgr_use_cpu_cache", False)),
    )
    rewired = graph_handler(
        pyg_data,
        k=int(getattr(cfg, "dhgr_k", 8)),
        epsilon=None,
        embedding_post=True,
        cat_self=bool(getattr(cfg, "dhgr_cat_self", False)),
        prunning=bool(getattr(cfg, "dhgr_pruning", True)),
        thres_prunning=float(getattr(cfg, "dhgr_pruning_threshold", 0.5)),
        load_path=None,
        save_path=None,
    )
    ei = rewired.edge_index.long()
    return RewireResult(ei, embedding=None, structure_changed=True,
                        extra={"dhgr_min_deg": min_deg,
                               "dhgr_min_deg_ratio": min_deg_ratio})


# ---------------------------------------------------------------------------
def rewire_gadc(data, cfg, device, seed):
    """GADC — adversarial graph diffusion.

    GADC does not rewire the edge SET; it builds a modified transition matrix
    on the ORIGINAL edges and diffuses features to produce a new node
    representation F = S X.  Structure is therefore unchanged; the learned
    embedding is the diffused feature matrix F.

    The diffusion below is the exact, training-free computation from
    run_gadc().
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


# ---------------------------------------------------------------------------
def rewire_lpkg(data, cfg, device, seed, dataset_name=None):
    """LPkG — GAE feature-reconstruction latent Z, then a cosine kNN graph.

    Rewired structure = the cosine kNN graph over the GAE latent.
    Learned embedding = the GAE latent Z_lat.
    (Stages 1-2 of run_lpkg, verbatim; the label-propagation / FSGNN
    classifier stages are not run.)
    """
    _lpkg_set_seed(seed)
    device = torch.device(device)

    X = data.x.float().to(device)
    edge_index = data.edge_index.long().to(device)

    gae = _LPkGGAE(in_dim=X.shape[1], hid_dim=cfg.lpkg_gae_hid,
                   lat_dim=cfg.lpkg_gae_lat).to(device)
    opt = Adam(gae.parameters(), lr=cfg.lpkg_gae_lr)
    gae.train()
    for _ in range(cfg.lpkg_gae_epochs):
        opt.zero_grad()
        _z, loss = gae(X, edge_index)
        loss.backward()
        opt.step()
    gae.eval()
    with torch.no_grad():
        Z_lat, _ = gae(X, edge_index)

    knn_bs = int(getattr(cfg, "lpkg_knn_batch_size", 512))
    k = int(cfg.lpkg_k)
    knn_ei = _lpkg_build_knn_graph(Z_lat, k=k, batch_size=knn_bs)

    return RewireResult(knn_ei, embedding=Z_lat, structure_changed=True,
                        extra={"lpkg_k": k,
                               "embedding_dim": int(Z_lat.shape[1])})


# ---------------------------------------------------------------------------
def rewire_graphite(data, cfg, device, seed, dataset_name=None):
    """GRAPHITE — augments the graph with FEATURE nodes (bipartite).

    GRAPHITE does not rewire the node-node structure; it appends feature nodes
    and node<->feature edges.  We therefore report the original node-node graph
    for label homophily and record the feature-augmentation statistics.  No new
    node embedding is produced for the original nodes.
    """
    X_ext, graph_ei, feat_ei, N_orig, N_feat = graphite_transform(
        data=data, dataset_name=dataset_name, device=device)
    extra = {
        "note": "feature-node augmentation (node-node structure unchanged)",
        "n_feature_nodes": int(N_feat),
        "feature_edges": int(feat_ei.shape[1]),
        "node_node_edges": int(graph_ei.shape[1]),
    }
    # graph_ei is the original node-node graph re-expressed on the original
    # node indices [0, N_orig).  Use it directly for structural homophily.
    ei = graph_ei[:, (graph_ei[0] < N_orig) & (graph_ei[1] < N_orig)].cpu()
    return RewireResult(ei, embedding=None, structure_changed=False,
                        extra=extra)


# ---------------------------------------------------------------------------
def _sparsify_topk_from_dense(A_dense, k, num_nodes):
    """Top-k (by weight) out-neighbours per row of a dense adjacency, self
    loops removed.  Returns a directed edge_index."""
    A = A_dense.clone()
    A.fill_diagonal_(float("-inf"))
    k = max(1, min(int(k), num_nodes - 1))
    _, nbr = torch.topk(A, k=k, dim=1, largest=True, sorted=False)
    src = torch.arange(num_nodes, device=A.device).unsqueeze(1).expand(-1, k).reshape(-1)
    dst = nbr.reshape(-1)
    # drop any -inf picks (rows with < k finite entries)
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
        sim[local, torch.arange(s, e)] = float("-inf")     # no self loops
        _, nbr = torch.topk(sim, k=k, dim=1, largest=True, sorted=False)
        src = torch.arange(s, e).unsqueeze(1).expand(-1, k).reshape(-1)
        rows.append(src)
        cols.append(nbr.reshape(-1))
    return torch.stack([torch.cat(rows), torch.cat(cols)]).cpu()


def rewire_idgl(data, cfg, device, seed):
    """IDGL — iterative deep graph learning (Chen et al., NeurIPS 2020).

    IDGL learns a *soft, dense* adjacency jointly with its GNN; there is no
    natural discrete edge set.  We run IDGL's own graph learner + GNN (the
    verbatim modules `_IDGLGraphLearner` / `_IDGLAnchorLearner` and the paper's
    combine / message-passing / regularisation), and at the best-validation
    checkpoint we extract:

      * the learned node representation h  -> reported as the embedding, and
      * a DISCRETE rewired graph obtained by keeping, per node, its top-k
        strongest learned neighbours (k = round(original average out-degree)).

    Full IDGL (N<=2000): top-k is taken from the dense combined adjacency A-bar.
    IDGL-ANCH  (N>2000): the combined adjacency is never materialised (it would
    be N x N); the discrete graph is a cosine kNN over the learned h, which is
    the representation IDGL actually propagates.  Both choices are documented
    measurement decisions, not changes to the model.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N = data.num_nodes
    C = data.num_classes
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
    s = min(cfg.idgl_num_anchors or 300, N // 4, N)

    # target degree for sparsifying the learned graph
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
        L0_dense = torch.tensor(
            (sp.diags(d_inv) @ A0_sp).toarray(), dtype=torch.float32, device=device)
        gl1 = _IDGLGraphLearner(X.shape[1], num_pers, epsilon).to(device)
        gl2 = _IDGLGraphLearner(hidden, num_pers, epsilon).to(device)

        def combine(At_n, A1_n):
            merged = eta * At_n + (1.0 - eta) * A1_n
            return lam * L0_dense + (1.0 - lam) * merged

        def mp(A, x):
            return torch.mm(A, x)

        opt = Adam(list(gl1.parameters()) + list(gl2.parameters())
                   + list(W1.parameters()) + list(W2.parameters()),
                   lr=lr, weight_decay=5e-4)

        best_val = -1.0
        best_Ab = None
        best_h = None

        for epoch in range(epochs):
            gl1.train(); gl2.train(); W1.train(); W2.train()
            opt.zero_grad()

            A1n, A1r = gl1(X)
            Ab1 = combine(A1n, A1n)
            h = F.relu(F.dropout(mp(Ab1, W1(X)), dropout, training=True))
            Z = mp(Ab1, W2(h))
            loss = F.cross_entropy(Z[tm], y[tm]) + _idgl_graph_reg(A1r, X, reg_alpha, reg_beta, reg_gamma)
            prev = A1r.detach()
            cur_Ab = Ab1

            iter_losses = []
            for _ in range(max_iter):
                Atn, Atr = gl2(h.detach())
                Abt = combine(Atn, A1n)
                h2 = F.relu(F.dropout(mp(Abt, W1(X)), dropout, training=True))
                Z2 = mp(Abt, W2(h2))
                l_t = F.cross_entropy(Z2[tm], y[tm]) + _idgl_graph_reg(Atr, X, reg_alpha, reg_beta, reg_gamma)
                iter_losses.append(l_t)
                diff = (Atr.detach() - prev).pow(2).sum()
                denom = A1r.detach().pow(2).sum().clamp(min=1e-12)
                prev = Atr.detach(); h = h2; Z = Z2; cur_Ab = Abt
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
                            extra={"idgl_mode": "full",
                                   "topk_per_node": k_target,
                                   "best_val": float(best_val),
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

        for epoch in range(epochs):
            gl1.train(); gl2.train(); W1.train(); W2.train()
            opt.zero_grad()

            x_anc = X[anchor_idx]
            R1n, R1r = gl1(X, x_anc)
            Lam1 = R1r.sum(0).clamp(min=1e-12)
            Del1 = R1r.sum(1).clamp(min=1e-12)

            h0_lin = W1(X)
            mp1 = _idgl_anchor_mp(h0_lin, R1n, 1.0 / Lam1, 1.0 / Del1)
            h = F.relu(F.dropout(
                lam * sparse_mp(h0_lin) + (1.0 - lam) * mp1, dropout, training=True))
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

                def comb_mp(v):
                    return (lam * sparse_mp(v) + (1.0 - lam) * (
                        eta * _idgl_anchor_mp(v, Rtn, 1.0 / Lamt, 1.0 / Delt)
                        + (1.0 - eta) * _idgl_anchor_mp(v, R1n, 1.0 / Lam1, 1.0 / Del1)))

                h2 = F.relu(F.dropout(comb_mp(W1(X)), dropout, training=True))
                Z2 = W2(comb_mp(h2))
                l_t = (F.cross_entropy(Z2[tm], y[tm])
                       + _idgl_anchor_reg(Rtr, x_anc_h, reg_alpha, reg_beta, reg_gamma))
                iter_losses.append(l_t)
                diff = (Rtr.detach() - prev_R).pow(2).sum()
                denom = Rtr.detach().pow(2).sum().clamp(min=1e-12)
                prev_R = Rtr.detach(); h = h2; Z = Z2
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
                            extra={"idgl_mode": "anchor",
                                   "topk_per_node": k_target,
                                   "best_val": float(best_val),
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

HOM_OUT_DIR = Path("homophily_results")
HOM_OUT_DIR.mkdir(parents=True, exist_ok=True)
HOM_RESULTS_JSONL = HOM_OUT_DIR / "results.jsonl"
HOM_TABLES_DIR = HOM_OUT_DIR / "tables"
HOM_TABLES_DIR.mkdir(parents=True, exist_ok=True)
HOM_ORIG_JSON = HOM_OUT_DIR / "original_measures.json"

HOM_KEY_FIELDS = ["dataset", "method", "seed"]

# datasets that are generated on the fly rather than downloaded
HOM_SYNTHETIC = {"HSBM-MED", "STRUC-HET", "FEAT-HET", "MIXED-SIG"}

HOM_ALL_METHODS = [ "idgl","dhgr","comfy","fosr","lpkg","gadc",
                   ]
HOM_DEFAULT_DATASETS = ["Amazon-ratings", "Roman-empire", "Actor", "Chameleon-F", "Squirrel-F", "Tolokers"]

# the label measures shared by original and rewired graphs
LABEL_MEASURES = [
    ("h_edge", "Edge homophily (Zhu 2020)"), 
    ("h_node", "Node homophily (Pei 2020)"),
    ("h_adj",  "Adjusted homophily (Platonov 2023)"),
    ("h_ci",   "Class-insensitive edge homophily (Lim 2021)"),
    ("LI",     "Label informativeness (Platonov 2023)"),
]


def _load_json(path):
    if Path(path).exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_json(path, obj):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    _os.replace(tmp, str(path))


def _fmt(v, nd=4):
    if v is None:
        return ""
    try:
        if isinstance(v, float) and _math.isnan(v):
            return "nan"
    except Exception:
        pass
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def load_dataset_any(name, cfg):
    """Load a real dataset or generate a synthetic one, matching the base file."""
    if name in HOM_SYNTHETIC:
        return generate_synthetic_dataset(name, seed=0)
    return load_real_dataset(name, root=cfg.data_root)


# --------------------------------------------------------------------------- #
#  Table writers (pure-csv; no pandas dependency).  All tables are rebuilt
#  from the full record list after every run, so they are always consistent
#  with results.jsonl ("run one, write one").
# --------------------------------------------------------------------------- #

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

    # ---- original graph measures (one row per dataset) --------------------
    seen = {}
    for r in ok:
        seen.setdefault(r["dataset"], r)
    orig_rows = []
    for ds in datasets:
        if ds not in seen:
            continue
        r = seen[ds]
        orig_rows.append([
            ds, r.get("n_nodes"), r.get("n_classes"), r.get("orig_edges"),
            _fmt(r.get("orig_avg_degree")),
            _fmt(r.get("orig_h_edge")), _fmt(r.get("orig_h_node")),
            _fmt(r.get("orig_h_adj")), _fmt(r.get("orig_h_ci")),
            _fmt(r.get("orig_LI")), _fmt(r.get("orig_h_feat")),
        ])
    _write_csv(HOM_TABLES_DIR / "original_graph_measures.csv",
               ["dataset", "n_nodes", "n_classes", "edges_directed",
                "avg_degree", "h_edge", "h_node", "h_adj", "h_ci", "LI",
                "h_feature_origX"], orig_rows)

    # ---- one CSV per label measure: original vs rewired vs delta ----------
    for key, _desc in LABEL_MEASURES:
        rows = []
        for r in ok:
            o = r.get(f"orig_{key}")
            w = r.get(f"rew_{key}")
            d = (w - o) if (isinstance(o, (int, float)) and isinstance(w, (int, float))
                            and not _math.isnan(o) and not _math.isnan(w)) else float("nan")
            rows.append([r["dataset"], r["method"], r["seed"],
                         _fmt(o), _fmt(w), _fmt(d)])
        _write_csv(HOM_TABLES_DIR / f"table_{key}.csv",
                   ["dataset", "method", "seed", "original", "rewired", "delta"],
                   rows)

    # ---- feature homophily table -----------------------------------------
    frows = []
    for r in ok:
        frows.append([
            r["dataset"], r["method"], r["seed"],
            _fmt(r.get("orig_h_feat")),
            _fmt(r.get("rew_h_feat_origX")),
            _fmt(r.get("rew_h_feat_emb")),
            r.get("has_embedding"),
        ])
    _write_csv(HOM_TABLES_DIR / "table_feature_homophily.csv",
               ["dataset", "method", "seed",
                "orig_feat_origX", "rew_feat_origX", "rew_feat_embedding",
                "has_embedding"], frows)

    # ---- graph statistics table ------------------------------------------
    grows = []
    for r in ok:
        grows.append([
            r["dataset"], r["method"], r["seed"],
            r.get("orig_edges"), r.get("rew_edges"),
            r.get("edges_delta_directed"),
            r.get("rew_edges_undirected"), r.get("rew_self_loops"),
            _fmt(r.get("orig_avg_degree")), _fmt(r.get("rew_avg_degree")),
            _fmt(r.get("rew_density"), 6),
            r.get("structure_changed"),
            json.dumps(r.get("extra", {})),
        ])
    _write_csv(HOM_TABLES_DIR / "table_graph_stats.csv",
               ["dataset", "method", "seed", "orig_edges", "rew_edges",
                "edges_delta", "rew_edges_undirected", "rew_self_loops",
                "orig_avg_degree", "rew_avg_degree", "rew_density",
                "structure_changed", "extra"], grows)

    # ---- pivoted markdown summary (rewired mean over seeds) ----------------
    _write_markdown_summary(ok, datasets, methods)

    # ---- also dump the full flat table -----------------------------------
    _write_full_flat(ok, datasets)


def _pivot_mean(ok, datasets, methods, key):
    """method x dataset -> mean rewired value over seeds."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in ok:
        buckets[(r["method"], r["dataset"])].append(r.get(key))
    return {(m, d): _mean(buckets[(m, d)]) for m in methods for d in datasets
            if (m, d) in buckets}


def _write_markdown_summary(ok, datasets, methods):
    present_methods = [m for m in methods if any(r["method"] == m for r in ok)]
    present_ds = [d for d in datasets if any(r["dataset"] == d for r in ok)]
    lines = ["# Homophily benchmark — summary (rewired graphs)\n",
             "Cells are the mean over seeds of the measure on the **rewired** "
             "graph. The `original` row is the same for every method.\n"]

    # original per-dataset reference (label measures + feature on original X)
    orig_ref = {}
    for r in ok:
        orig_ref.setdefault(r["dataset"], r)

    for key, desc in LABEL_MEASURES + [("h_feat_pair", "Feature homophily")]:
        lines.append(f"\n## {desc}\n")
        header = "| method | " + " | ".join(present_ds) + " |"
        sep = "|" + "---|" * (len(present_ds) + 1)
        lines.append(header)
        lines.append(sep)
        if key == "h_feat_pair":
            # original feature homophily row (original X on original edges)
            orow = ["| original (feat, origX) "]
            for d in present_ds:
                r = orig_ref.get(d)
                orow.append(_fmt(r.get("orig_h_feat")) if r else "")
            lines.append(" | ".join(orow) + " |")
            # per method: rewired feature homophily on the embedding if present,
            # else on original features
            piv_emb = _pivot_mean(ok, present_ds, present_methods, "rew_h_feat_emb")
            piv_x = _pivot_mean(ok, present_ds, present_methods, "rew_h_feat_origX")
            for m in present_methods:
                row = [f"| {m} (emb/origX)"]
                for d in present_ds:
                    ve = piv_emb.get((m, d))
                    vx = piv_x.get((m, d))
                    if ve is not None and not _math.isnan(ve):
                        row.append(_fmt(ve) + " (emb)")
                    else:
                        row.append(_fmt(vx) + " (origX)")
                lines.append(" | ".join(row) + " |")
        else:
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

    # edge-count table
    lines.append("\n## Rewired edge count (directed, mean over seeds)\n")
    header = "| method | " + " | ".join(present_ds) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(present_ds) + 1))
    orow = ["| original "]
    for d in present_ds:
        r = orig_ref.get(d)
        orow.append(str(r.get("orig_edges")) if r else "")
    lines.append(" | ".join(orow) + " |")
    piv = _pivot_mean(ok, present_ds, present_methods, "rew_edges")
    for m in present_methods:
        row = [f"| {m}"]
        for d in present_ds:
            v = piv.get((m, d))
            row.append(f"{v:.0f}" if v is not None and not _math.isnan(v) else "")
        lines.append(" | ".join(row) + " |")

    with open(HOM_OUT_DIR / "SUMMARY.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def _write_full_flat(ok, datasets):
    cols = ["dataset", "method", "seed", "status",
            "n_nodes", "n_classes", "structure_changed", "has_embedding",
            "orig_edges", "rew_edges", "edges_delta_directed",
            "orig_avg_degree", "rew_avg_degree", "rew_density",
            "orig_h_edge", "rew_h_edge",
            "orig_h_node", "rew_h_node",
            "orig_h_adj", "rew_h_adj",
            "orig_h_ci", "rew_h_ci",
            "orig_LI", "rew_LI",
            "orig_h_feat", "rew_h_feat_origX", "rew_h_feat_emb",
            "rewire_time_s"]
    rows = []
    for r in ok:
        rows.append([_fmt(r.get(c)) if isinstance(r.get(c), float) else r.get(c)
                     for c in cols])
    _write_csv(HOM_TABLES_DIR / "all_results_flat.csv", cols, rows)


# --------------------------------------------------------------------------- #
def run_homophily_benchmark(cfg):
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))

    print("\n" + "=" * 78)
    print("  HOMOPHILY / GRAPH-STATISTICS BENCHMARK  (rewiring methods)")
    print("=" * 78)
    print(f"  Datasets : {cfg.datasets}")
    print(f"  Methods  : {cfg.methods}")
    print(f"  Seeds    : {cfg.seeds}")
    print(f"  Device   : {device}")
    print(f"  Results  -> {HOM_OUT_DIR.resolve()}")
    print("=" * 78)

    store = _checkpoint.ResultStore(HOM_RESULTS_JSONL, key_fields=HOM_KEY_FIELDS)
    if store.count():
        print(f"  [resume] {store.count()} completed run(s) found; they are skipped.\n")

    orig_cache = _load_json(HOM_ORIG_JSON)

    total = len(cfg.datasets) * len(cfg.methods) * len(cfg.seeds)
    tracker = _progress.ProgressTracker(total, label="homophily").start()

    for dataset in cfg.datasets:
        print(f"\n{'-'*78}\n  DATASET: {dataset}\n{'-'*78}")
        try:
            data = load_dataset_any(dataset, cfg)
        except Exception as e:
            print(f"  [SKIP dataset] {dataset}: {e}")
            tracker.tick_skipped(len(cfg.methods) * len(cfg.seeds))
            continue

        N, C = data.num_nodes, data.num_classes
        print(f"  N={N}  C={C}  edges={data.edge_index.shape[1]}  "
              f"train={int(data.train_mask.sum())} "
              f"val={int(data.val_mask.sum())} "
              f"test={int(data.test_mask.sum())}")

        # ---- original graph measures (computed once, cached) --------------
        if dataset not in orig_cache:
            olm = all_label_measures(data.edge_index, data.y, N, C)
            ogs = graph_stats(data.edge_index, N)
            ofeat = feature_homophily(data.edge_index, data.x)
            orig_cache[dataset] = {
                "n_nodes": int(N), "n_classes": int(C),
                "orig_edges": ogs["edges_directed"],
                "orig_edges_undirected": ogs["edges_undirected"],
                "orig_avg_degree": ogs["avg_degree"],
                "orig_h_edge": olm["h_edge"], "orig_h_node": olm["h_node"],
                "orig_h_adj": olm["h_adj"], "orig_h_ci": olm["h_ci"],
                "orig_LI": olm["LI"], "orig_h_feat": ofeat,
            }
            _save_json(HOM_ORIG_JSON, orig_cache)
        oc = orig_cache[dataset]
        print(f"  [orig]  h_edge={_fmt(oc['orig_h_edge'])} "
              f"h_node={_fmt(oc['orig_h_node'])} "
              f"h_adj={_fmt(oc['orig_h_adj'])} "
              f"h_ci={_fmt(oc['orig_h_ci'])} "
              f"LI={_fmt(oc['orig_LI'])} "
              f"h_feat={_fmt(oc['orig_h_feat'])}")

        for method in cfg.methods:
            for seed in cfg.seeds:
                if cfg.resume and store.exists(dataset=dataset, method=method, seed=seed):
                    tracker.tick_skipped()
                    continue

                if method == "dhgr" and DHGRModelHandler is None:
                    print("  [DHGR] repo unavailable — skipping (not recorded; "
                          "will retry on resume). Set $DHGR_ROOT to enable.")
                    tracker.tick_done(0.0, note="dhgr-unavailable")
                    continue

                print(f"\n  [{method.upper()}] {dataset} seed={seed}")
                t0 = time.perf_counter()
                try:
                    set_global_seed(seed)
                    fn = REWIRE_FNS[method]
                    if method in _TAKES_DATASET_NAME:
                        res = fn(data, cfg, device, seed, dataset_name=dataset)
                    else:
                        res = fn(data, cfg, device, seed)

                    ei = res.edge_index
                    lm = all_label_measures(ei, data.y, N, C)
                    gs = graph_stats(ei, N)
                    feat_x = feature_homophily(ei, data.x)
                    feat_e = (feature_homophily(ei, res.embedding)
                              if res.embedding is not None else float("nan"))
                    dt = time.perf_counter() - t0

                    rec = {
                        "dataset": dataset, "method": method, "seed": seed,
                        "status": "ok",
                        "n_nodes": int(N), "n_classes": int(C),
                        "structure_changed": res.structure_changed,
                        "has_embedding": res.embedding is not None,
                        # original (cached)
                        "orig_edges": oc["orig_edges"],
                        "orig_avg_degree": oc["orig_avg_degree"],
                        "orig_h_edge": oc["orig_h_edge"],
                        "orig_h_node": oc["orig_h_node"],
                        "orig_h_adj": oc["orig_h_adj"],
                        "orig_h_ci": oc["orig_h_ci"],
                        "orig_LI": oc["orig_LI"],
                        "orig_h_feat": oc["orig_h_feat"],
                        # rewired
                        "rew_h_edge": lm["h_edge"], "rew_h_node": lm["h_node"],
                        "rew_h_adj": lm["h_adj"], "rew_h_ci": lm["h_ci"],
                        "rew_LI": lm["LI"],
                        "rew_h_feat_origX": feat_x,
                        "rew_h_feat_emb": feat_e,
                        "rew_edges": gs["edges_directed"],
                        "rew_edges_undirected": gs["edges_undirected"],
                        "rew_self_loops": gs["self_loops"],
                        "rew_avg_degree": gs["avg_degree"],
                        "rew_density": gs["density"],
                        "edges_delta_directed": gs["edges_directed"] - oc["orig_edges"],
                        "extra": res.extra,
                        "rewire_time_s": dt,
                    }
                    store.append(rec)
                    note = (f"{dataset}/{method}/s{seed} "
                            f"h_edge {oc['orig_h_edge']:.3f}->{lm['h_edge']:.3f} "
                            f"E {oc['orig_edges']}->{gs['edges_directed']}")
                    print(f"    [ok] rewired h_edge={_fmt(lm['h_edge'])} "
                          f"h_adj={_fmt(lm['h_adj'])} LI={_fmt(lm['LI'])} "
                          f"feat_emb={_fmt(feat_e)} edges={gs['edges_directed']} "
                          f"({dt:.1f}s)")
                    tracker.tick_done(dt, note=note)

                except Exception as e:
                    dt = time.perf_counter() - t0
                    print(f"    [SKIP run] {method}/{dataset}/seed={seed}: {e}")
                    if getattr(cfg, "verbose", False):
                        traceback.print_exc()
                    # record a 'skipped' marker so a permanent incompatibility
                    # (e.g. GRAPHITE needs binary features) is not retried
                    # forever; delete the line from results.jsonl to retry.
                    store.append({
                        "dataset": dataset, "method": method, "seed": seed,
                        "status": "skipped", "reason": str(e),
                        "rewire_time_s": dt,
                    })
                    tracker.tick_done(dt, note="skipped")

            # rebuild tables after each method finishes (run-one / write-one)
            write_all_tables(store.all(), cfg.datasets, cfg.methods)

        write_all_tables(store.all(), cfg.datasets, cfg.methods)

    tracker.finish()
    write_all_tables(store.all(), cfg.datasets, cfg.methods)
    print(f"\n[Done] Homophily tables, summary and raw records in "
          f"{HOM_OUT_DIR.resolve()}")


# =============================================================================
#  SECTION C4 — CLI
# =============================================================================

def parse_args_homophily(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        description="Homophily / graph-statistics benchmark for graph-rewiring "
                    "methods (IDGL, GADC, LPkG, DHGR, GRAPHITE, FoSR, ComFy, "
                    "GLARE). Computes standard homophily measures on the "
                    "original and rewired graphs; no downstream classification.")

    p.add_argument("--datasets", nargs="+", default=HOM_DEFAULT_DATASETS)
    p.add_argument("--methods", nargs="+", default=HOM_ALL_METHODS,
                   choices=HOM_ALL_METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="Default is a single seed [0].")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip (dataset, method, seed) runs already recorded "
                        "(default on).")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # ---- IDGL ----
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

    # ---- GADC ----
    p.add_argument("--gadc_epsilon", type=float, default=1.0)
    p.add_argument("--gadc_lam", type=float, default=1.0)
    p.add_argument("--gadc_K", type=int, default=16)

    # ---- LPkG ----
    p.add_argument("--lpkg_gae_hid", type=int, default=256)
    p.add_argument("--lpkg_gae_lat", type=int, default=128)
    p.add_argument("--lpkg_gae_lr", type=float, default=1e-4)
    p.add_argument("--lpkg_gae_epochs", type=int, default=200)
    p.add_argument("--lpkg_k", type=int, default=5)
    p.add_argument("--lpkg_knn_batch_size", type=int, default=512)

    # ---- GRAPHITE ----
    p.add_argument("--graphite_binarize", type=str, default="topk",
                   choices=["median", "positive", "topk"])
    p.add_argument("--graphite_topk", type=int, default=10)
    p.add_argument("--graphite_max_feat_edges", type=int, default=2_000_000)

    # ---- GLARE (defaults unchanged from the GLARE benchmark) ----
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
    _cfg = parse_args_homophily()
    run_homophily_benchmark(_cfg)
