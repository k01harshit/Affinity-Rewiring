"""
benchmark_faithful.py
=====================
Faithful paper-exact implementations of four graph rewiring / classification
methods evaluated on the same datasets and seeds as benchmark_rewiring_v15.py.

Each method uses ONLY the classifier proposed in its paper:
  • IDGL      — two-layer GCN (W1/W2 linear + message-passing), joint loss
                 Chen et al., NeurIPS 2020, Algorithm 1
  • GADC      — pre-computed diffusion F=S·X, then 2-layer MLP only
                 Liu et al., ICML 2024, Algorithm 1 + Section 3.4 (Option I)
  • LPkG      — GAE (feature-recon MSE) → kNN graph → LP → blend with GNN
                 Park & Park, BigComp 2024, Algorithm 1
  • GRAPHITE  — graph transformation + custom FAGCN-style self-gating GNN
                 Qiu et al., arXiv 2025, Equations 12–16

Shares with benchmark_rewiring_v15.py:
  • Identical dataset loading functions (copy-pasted, no import dependency)
  • Identical seeds list [42, 0, 1] by default
  • Identical train/val/test masks (same loading logic, same fixed splits)

Output:
  faithful_results/results_faithful.jsonl   — per-seed records
  faithful_results/summary_faithful.csv     — mean ± std per (dataset, method)

Usage:
  python benchmark_faithful.py
  python benchmark_faithful.py --datasets Actor Squirrel-F --methods idgl gadc
  python benchmark_faithful.py --smoke_test
  python benchmark_faithful.py --seeds 42 0 1 --device cuda
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
OUT_DIR = Path("others_benchmark_results")
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
#  SECTION 7 — Unified benchmark loop
# =============================================================================
#
# Runs every "other" rewiring/classification method, each using its own
# paper-proposed classifier, over seeds {0, 1, 2}.  For every (dataset, method,
# seed) we record accuracy, macro-F1 and runtime.  The DHGR runner now covers
# Roman-empire too (it internally switches to the min_deg=3 / min_deg_ratio=1.8
# thresholds for that dataset), so the previously-separate roman-only script is
# no longer needed.
#
# Results are appended to a JSONL file after every run, so an interrupted sweep
# resumes exactly where it stopped.  Per-dataset tables and figures are written
# at the end and refreshed after each dataset.

ALL_DATASETS = [
    "Amazon-ratings", "Roman-empire",
    "Actor", "Chameleon-F", "Squirrel-F", "Tolokers",
    "HSBM-MED", "STRUC-HET", "FEAT-HET", "MIXED-SIG",
]

FAITHFUL_METHODS = [
    "idgl",
    "dhgr",
    "comfy",
    "fosr",
    "lpkg",
    "gadc",
    "graphite",
]

SYNTHETIC = {"HSBM-MED", "STRUC-HET", "FEAT-HET", "MIXED-SIG"}

RESULTS_JSONL = OUT_DIR / "results.jsonl"
TABLES_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "plots"
KEY_FIELDS = ["dataset", "method", "seed"]
GROUP_FIELDS = ["method"]


def _dispatch(method, data, cfg, device, seed, dataset_name):
    """Call the right run_* function; every one now returns (val, test, f1)."""
    if method == "idgl":
        return run_idgl(data, cfg, device, seed)
    if method == "gadc":
        return run_gadc(data, cfg, device, seed)
    if method == "lpkg":
        return run_lpkg(data, cfg, device, seed, dataset_name=dataset_name)
    if method == "graphite":
        return run_graphite(data, cfg, device, seed, dataset_name=dataset_name)
    if method == "fosr":
        return run_fosr(data, cfg, device, seed)
    if method == "comfy":
        return run_comfy(data, cfg, device, seed)
    if method == "dhgr":
        return run_dhgr(data, cfg, device, seed, dataset_name=dataset_name)
    raise ValueError(f"Unknown method: {method}")


def _refresh_reports(store, datasets):
    records = store.all()
    summary, avg = _reporting.write_all_csvs(
        records, GROUP_FIELDS, TABLES_DIR,
        dataset_order=datasets, make_average=True,
    )
    if summary.empty:
        return
    print("\n" + "=" * 72)
    print("  OTHER-METHODS BENCHMARK — SUMMARY SO FAR")
    print("=" * 72)
    _reporting.print_dataset_tables(summary, GROUP_FIELDS, datasets)
    _reporting.print_average_table(avg, GROUP_FIELDS)
    _plotting.generate_all_figures(summary, avg, GROUP_FIELDS, FIG_DIR,
                                   dataset_order=datasets)


def run_faithful_benchmark(cfg):
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))

    print("\n" + "=" * 72)
    print("  Faithful Paper Implementations Benchmark")
    print("  Each method uses ONLY its paper-proposed classifier")
    print("=" * 72)
    print(f"  Datasets : {cfg.datasets}")
    print(f"  Methods  : {cfg.methods}")
    print(f"  Seeds    : {cfg.seeds}")
    print(f"  Device   : {device}")
    print(f"  Results  -> {OUT_DIR.resolve()}")
    print("=" * 72)

    store = _checkpoint.ResultStore(RESULTS_JSONL, key_fields=KEY_FIELDS)
    if store.count():
        print(f"  [resume] found {store.count()} completed run(s); "
              f"they will be skipped.\n")

    total_runs = len(cfg.datasets) * len(cfg.methods) * len(cfg.seeds)
    tracker = _progress.ProgressTracker(total_runs, label="others").start()

    for dataset in cfg.datasets:
        print(f"\n{'─'*72}\n  DATASET: {dataset}\n{'─'*72}")

        try:
            data = (generate_synthetic_dataset(dataset, seed=0)
                    if dataset in SYNTHETIC
                    else load_real_dataset(dataset, root=cfg.data_root))
        except Exception as e:
            print(f"  [SKIP dataset] {dataset}: {e}")
            tracker.tick_skipped(len(cfg.methods) * len(cfg.seeds))
            continue

        print(f"  N={data.num_nodes}  C={data.num_classes}  "
              f"train={int(data.train_mask.sum())}  "
              f"val={int(data.val_mask.sum())}  "
              f"test={int(data.test_mask.sum())}")

        # Persist the exact split for this dataset (same logic as the CVGAE
        # script, so the two benchmarks share identical train/val/test masks).
        try:
            save_split(dataset, data)
        except Exception as e:
            print(f"  [split] could not save split for {dataset}: {e}")

        for method in cfg.methods:
            for seed in cfg.seeds:
                if cfg.resume and store.exists(
                    dataset=dataset, method=method, seed=seed,
                ):
                    tracker.tick_skipped()
                    continue

                if method == "dhgr" and DHGRModelHandler is None:
                    print("  [DHGR] repo unavailable — skipping (set $DHGR_ROOT "
                          "to enable). Nothing recorded; will retry on resume.")
                    tracker.tick_done(0.0, note="dhgr-unavailable")
                    continue

                print(f"\n  [{method.upper()}] dataset={dataset} seed={seed}")
                t0 = time.perf_counter()
                try:
                    set_global_seed(seed)
                    val, test, f1 = _dispatch(
                        method, data, cfg, device, seed, dataset)

                    elapsed = time.perf_counter() - t0
                    rec = {
                        "dataset": dataset,
                        "method": method,
                        "seed": seed,
                        "acc": float(test),
                        "f1": float(f1),
                        "val_acc": float(val),
                        "total_time_s": elapsed,
                        "elapsed_s": elapsed,
                    }
                    store.append(rec)
                    note = (f"{dataset}/{method}/s{seed} "
                            f"acc={test:.4f} f1={f1:.4f}")
                    tracker.tick_done(elapsed, note=note)

                except Exception as e:
                    print(f"  [SKIP run] {method}/{dataset}/seed={seed}: {e}")
                    if cfg.verbose:
                        traceback.print_exc()
                    tracker.tick_done(time.perf_counter() - t0, note="FAILED")

        _refresh_reports(store, cfg.datasets)

    tracker.finish()
    _refresh_reports(store, cfg.datasets)
    print(f"\n[Done] Results, tables and plots in {OUT_DIR.resolve()}")


# =============================================================================
#  SECTION 8 — CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Faithful paper implementations benchmark")

    p.add_argument("--datasets",  nargs="+", default=ALL_DATASETS)
    p.add_argument("--methods",   nargs="+", default=FAITHFUL_METHODS,
                   help="Subset of: idgl gadc lpkg graphite")
    p.add_argument("--seeds",     nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data_root", default="./data")
    p.add_argument("--device",    default="auto")
    p.add_argument("--resume",    action="store_true", default=True,
                   help="Skip runs already in the results file (default on).")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--smoke_test",action="store_true")
    p.add_argument("--verbose",   action="store_true")

    # ── IDGL hyper-parameters (paper defaults) ────────────────────────────────
    p.add_argument("--idgl_hidden",       type=int,   default=64)
    p.add_argument("--idgl_num_pers",     type=int,   default=4)
    p.add_argument("--idgl_epsilon",      type=float, default=0.5)
    p.add_argument("--idgl_lambda",       type=float, default=0.8,
                   help="graph_skip_conn λ (Eq. 3)")
    p.add_argument("--idgl_eta",          type=float, default=0.5,
                   help="η weighting between Aᵗ and A¹ (Eq. 3)")
    p.add_argument("--idgl_max_iter",     type=int,   default=10,
                   help="Maximum inner iterations T")
    p.add_argument("--idgl_eps_adj",      type=float, default=1e-4,
                   help="Stopping criterion ε_adj")
    p.add_argument("--idgl_epochs",       type=int,   default=200)
    p.add_argument("--idgl_lr",           type=float, default=1e-3)
    p.add_argument("--idgl_dropout",      type=float, default=0.5)
    p.add_argument("--idgl_reg_alpha",    type=float, default=1.0,
                   help="Smoothness weight α in graph regularisation")
    p.add_argument("--idgl_reg_beta",     type=float, default=1.0,
                   help="Connectivity weight in graph regularisation")
    p.add_argument("--idgl_reg_gamma",    type=float, default=0.5,
                   help="Sparsity weight γ in graph regularisation")
    p.add_argument("--idgl_patience",    type=int, default=100,
                   help="Patience")
    p.add_argument(
        "--idgl_wd",
        type=float,
        default=5e-4,
        help="IDGL GCN weight decay (paper/default: 5e-4)",
    )

    p.add_argument("--idgl_num_anchors",  type=int,   default=None,
                   help="Anchors for IDGL-ANCH (None → auto min(300,N/4))")

    # ── GADC hyper-parameters (paper Table 14 / Appendix E) ──────────────────
    p.add_argument("--gadc_epsilon",     type=float, default=1.0,
                   help="ε for adversarial transition (paper Table 14: ε=1.0)")
    p.add_argument("--gadc_lam",         type=float, default=1.0,
                   help="λ diffusion coefficient (paper: λ=1)")
    p.add_argument("--gadc_K",           type=int,   default=16,
                   help="Number of diffusion steps K (paper: K=16)")
    p.add_argument("--gadc_mlp_hidden",  type=int,   default=64,
                   help="MLP hidden dim (paper Appendix E: 64 units)")
    p.add_argument("--gadc_dropout",     type=float, default=0.5)
    p.add_argument("--gadc_lr",          type=float, default=0.02,
                   help="MLP learning rate (paper Appendix Table 8: lr=0.02)")
    p.add_argument("--gadc_wd",          type=float, default=1e-5,
                   help="Weight decay (paper Appendix Table 8: 1e-5)")
    p.add_argument("--gadc_epochs",      type=int,   default=100,
                   help="MLP training epochs (paper Table 8: 100 epochs)")

    # ── LPkG hyper-parameters (paper Table V / Section V-E) ──────────────────
    p.add_argument("--lpkg_gae_hid",     type=int,   default=256,
                   help="GAE hidden dim (paper: 256/128 best)")
    p.add_argument("--lpkg_gae_lat",     type=int,   default=128,
                   help="GAE latent dim (paper: 256/128 best)")
    p.add_argument("--lpkg_gae_lr",      type=float, default=1e-4,
                   help="GAE learning rate (paper Section V-E: 0.0001)")
    p.add_argument("--lpkg_gae_epochs",  type=int,   default=200)
    p.add_argument("--lpkg_k",           type=int,   default=5,
                   help="kNN k for supplementary graph (paper Fig 7: k=5 default)")
    p.add_argument("--lpkg_lp_alpha",    type=float, default=0.99,
                   help="LP weight α (paper Section V-E: fixed at 0.99)")
    p.add_argument("--lpkg_lp_iters",    type=int,   default=30,
                   help="LP iterations (paper Section V-E: converges at 30)")
    p.add_argument("--lpkg_gnn_hid",     type=int,   default=64)
    p.add_argument("--lpkg_gnn_dropout", type=float, default=0.5)
    p.add_argument("--lpkg_gnn_lr",      type=float, default=1e-3)
    p.add_argument(
        "--lpkg_beta_init",
        type=float,
        default=0.5,
        help="Initial value of learnable LPkG beta",
    )
    p.add_argument("--lpkg_beta_lr",     type=float, default=1e-3,
                   help="Learning rate for β blend parameter (paper Table V: 0.0001-0.005)")
    p.add_argument("--lpkg_gnn_epochs",  type=int,   default=300)

    # ── GRAPHITE hyper-parameters (paper Appendix Training & Evaluation) ──────
    p.add_argument("--graphite_binarize", type=str,   default="topk",
                   choices=["median", "positive", "topk"],
                   help="Binarisation mode. 'topk' (default) bounds edge count for "
                        "continuous features.  Paper uses native binary features where "
                        "'median' is equivalent; 'topk' is the correct analogue for "
                        "continuous feature datasets.")
    p.add_argument("--graphite_topk",     type=int,   default=10,
                   help="Active features per node for topk binarisation. "
                        "Controls feature-edge density: total feat edges ≈ N × topk × 2.")
    p.add_argument("--graphite_max_feat_edges", type=int, default=2_000_000,
                   help="Safety cap on total feature edges. If exceeded, topk is halved "
                        "until the count fits. Prevents OOM on large datasets.")
    p.add_argument("--graphite_hidden",   type=int,   default=512,
                   help="GNN hidden dim (paper: 512)")
    p.add_argument("--graphite_n_layers", type=int,   default=8,
                   help="Number of GNN layers (paper: 8)")
    p.add_argument("--graphite_dropout",  type=float, default=0.2,
                   help="Dropout rate (paper: 0.2)")
    p.add_argument("--graphite_w_E",      type=float, default=1.0,
                   help="Graph-edge weight w_E (paper: 1.0 reference)")
    p.add_argument("--graphite_w_X",      type=float, default=0.6,
                   help="Feature-edge weight w_X (paper tuned: {0.01,0.1,0.6,8})")
    p.add_argument("--graphite_w_0",      type=float, default=0.3,
                   help="Self-loop weight w_0 (paper tuned: {0.1,0.2,0.3,0.5,1,8})")
    p.add_argument("--graphite_tau",      type=float, default=0.1,
                   help="Temperature τ for self-gating (paper tuned: {0.01,0.1,1})")
    p.add_argument("--graphite_lr",       type=float, default=3e-5,
                   help="Learning rate (paper: 0.00003)")
    p.add_argument("--graphite_steps",    type=int,   default=1000,
                   help="Training steps (paper: 1000)")

    args = p.parse_args()

    if args.smoke_test:
        print("[Smoke-test mode]")
        args.seeds            = args.seeds[:1]
        args.idgl_epochs      = min(args.idgl_epochs,      20)
        args.gadc_epochs      = min(args.gadc_epochs,      20)
        args.lpkg_gae_epochs  = min(args.lpkg_gae_epochs,  20)
        args.lpkg_gnn_epochs  = min(args.lpkg_gnn_epochs,  20)
        args.graphite_steps   = min(args.graphite_steps,   50)
        args.idgl_max_iter    = min(args.idgl_max_iter,     2)

    return args


if __name__ == "__main__":
    cfg = parse_args()
    run_faithful_benchmark(cfg)
