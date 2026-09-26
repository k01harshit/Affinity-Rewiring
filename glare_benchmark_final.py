"""
glare_benchmark.py
====================
GLARE (rewired graph + learned-Z representation) benchmark, built on the
same harness as eureka_benchmark.py -- real-dataset loading, matched-density
subsampling, semi-supervised label masking, downstream classifiers,
resumable JSONL results, progress tracking, tables and plots are all in the
same style and (where model-agnostic) the same code as eureka_benchmark.py.

Only the representation/rewiring model changed, from EUREKA to GLARE, and it
is reproduced VERBATIM from glare_bestv2.py: the candidate-edge sampling,
the learned similarity (NT-Xent contrastive) encoder, the distribution
affinity signal, the GLARE-18 rewiring module (EM loop over a GraphSAGE-style
GNN and a per-edge soft/hard adjacency), the downstream classifiers (with
their input->logits residual connections) and the classifier training loop
(label smoothing, grad clipping, val-every-5 model selection) are all copied
unchanged from glare_bestv2.py. Nothing about how GLARE computes its
representation, rewires the graph, or trains/evaluates classifiers has been
modified here -- only the benchmarking harness around it (data loading,
resumable results, tables/plots, CLI) matches eureka_benchmark.py.

--------------------------------------------------------------------------------
WHAT THIS FILE EVALUATES
--------------------------------------------------------------------------------
Exactly ONE graph/feature configuration -- "glare" (glare_rewired_learnedZ) --
run with all 5 downstream classifiers (gcn, gat, sage, h2gcn, linkx), over
seeds {0, 1, 2}, on all 6 real datasets:

    Actor, Squirrel-F, Chameleon-F, Roman-empire, Amazon-ratings, Tolokers

  "glare"  ==  GLARE-18-rewired graph + [ L2-normalize(original features)
               | L2-normalize(learned Z18 representation) ]

The "original" baseline and any other GLARE variants (e.g. glare_rewired
without learned-Z) are intentionally NOT run in this file.

Semi-supervised label visibility is controlled by --label_mask_ratio, exactly
as in eureka_benchmark.py:
  * --label_mask_ratio 1.0   (default) -- run this first (full supervision)
  * --label_mask_ratio 0.0               -- run this second (label-blind)

For every (dataset, method, classifier, seed) we record accuracy, macro-F1 and
runtime (rewiring + classifier training). Results are appended to a JSONL
file after every single run, so an interrupted sweep resumes exactly where it
left off. Per-dataset tables, an "average over datasets per classifier" table
and all figures are produced at the end (and refreshed after every dataset).

Usage examples
--------------
  python glare_benchmark.py --label_mask_ratio 1.0
  python glare_benchmark.py --label_mask_ratio 0.0 --device cuda
  python glare_benchmark.py --resume --device cuda
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
import traceback
import warnings
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore")

try:
    import torch_geometric  # noqa: F401
except ImportError:
    # Colab-ready, copied from glare_bestv2.py: installs torch_geometric on
    # first run so this file works as a single self-contained script.
    import sys as _sys_install
    subprocess.run([_sys_install.executable, "-m", "pip", "install", "-q", "torch_geometric"], check=True)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, LINKX
from torch_geometric.utils import degree as pyg_degree, to_undirected

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -- Unified-benchmark additions ---------------------------------------------
import os as _os
import random as _random
import sys as _sys

# Make the shared ``common`` package importable no matter where this script is
# launched from (it lives next to this file, same as eureka_benchmark.py).
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))

from common import progress as _progress        # noqa: E402
from common import checkpoint as _checkpoint    # noqa: E402
from common import reporting as _reporting      # noqa: E402
from common import plotting as _plotting        # noqa: E402


def set_global_seed(seed: int):
    """Seed every RNG we can reach and put cuDNN in deterministic mode.
    (Harness-level seeding, identical in spirit to eureka_benchmark.py's
    set_global_seed and glare_bestv2.py's set_seed.)"""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    _os.environ["PYTHONHASHSEED"] = str(seed)

# -----------------------------------------------------------------------------
OUT_DIR   = Path("glare_benchmark_results")
PLOTS_DIR = OUT_DIR / "plots"
CKPT_DIR  = OUT_DIR / "checkpoints"
for _d in [OUT_DIR, PLOTS_DIR, CKPT_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = OUT_DIR / "results.jsonl"
SUMMARY_CSV  = OUT_DIR / "summary.csv"


# =============================================================================
#  SECTION 1 -- Data loading  (fixed for Actor / Squirrel-F / Chameleon-F)
#  Copied verbatim from eureka_benchmark.py -- same harness, real datasets
#  only (this file does not run the synthetic datasets).
# =============================================================================

def _safe_extract_masks(d):
    """
    Robustly extract train/val/test masks from a PyG Data object.

    Handles three mask formats found across PyG versions:
      1. 2-D boolean mask  (N, K)  — take column 0
      2. 1-D boolean mask  (N,)
      3. 1-D integer index tensor — convert to boolean
    """
    def _to_bool(m, N):
        if m is None:
            return torch.zeros(N, dtype=torch.bool)
        if m.dtype == torch.bool:
            if m.dim() == 2:
                return m[:, 0]
            return m
        # integer indices
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
    # ------------------------------------------------------------------
    # Actor  (older PyG: Actor dataset; newer PyG: same but masks differ)
    # ------------------------------------------------------------------
    if name == "Actor":
        from torch_geometric.datasets import Actor as _Actor
        ds = _Actor(root=root)
        d  = ds[0]
        tm, vm, te = _safe_extract_masks(d)
        # Actor sometimes has no split at all in very old versions — fallback
        if int(tm.sum()) == 0:
            N   = d.num_nodes
            rng = torch.Generator(); rng.manual_seed(0)
            perm = torch.randperm(N, generator=rng)
            tm = torch.zeros(N, dtype=torch.bool); tm[perm[:int(0.6*N)]] = True
            vm = torch.zeros(N, dtype=torch.bool); vm[perm[int(0.6*N):int(0.8*N)]] = True
            te = torch.zeros(N, dtype=torch.bool); te[perm[int(0.8*N):]] = True
        return _make_ns(d, tm, vm, te)

    # ------------------------------------------------------------------
    # Squirrel-F / Chameleon-F  (filtered Wikipedia networks)
    # Newer PyG renamed the `geom_gcn_preprocess` flag and changed how
    # masks are stored.  We try multiple strategies.
    # ------------------------------------------------------------------
    _wiki_map = {
        "Squirrel-F":  "squirrel",
        "Chameleon-F": "chameleon",
    }
    if name in _wiki_map:
        wiki_name = _wiki_map[name]
        from torch_geometric.datasets import WikipediaNetwork as _Wiki

        # Try three strategies in order; each fully wrapped so that errors
        # in BOTH the constructor AND ds[0] are caught cleanly.
        # Strategy 1: geom_gcn_preprocess=True  (produces per-split masks)
        # Strategy 2: geom_gcn_preprocess=False (raw graph, we build split)
        # Strategy 3: no keyword               (very old PyG)
        _wiki_d = None
        for _kwargs in [
            {"geom_gcn_preprocess": True},
            {"geom_gcn_preprocess": False},
            {},
        ]:
            try:
                _ds = _Wiki(root=root, name=wiki_name, **_kwargs)
                _d  = _ds[0]
                # Validate that y exists and is non-empty before accepting
                if _d.y is None or _d.y.numel() == 0:
                    continue
                _tm, _vm, _te = _safe_extract_masks(_d)
                if int(_tm.sum()) > 0:
                    return _make_ns(_d, _tm, _vm, _te)
                # Masks empty but graph valid — keep as fallback for manual split
                _wiki_d = _d
            except Exception:
                continue

        if _wiki_d is None:
            raise RuntimeError(
                f"Cannot load {name}: all WikipediaNetwork strategies failed. "
                "Try: pip install torch_geometric --upgrade"
            )

        # Graph loaded but all splits were empty — create a standard 60/20/20 split
        N   = _wiki_d.num_nodes
        rng = torch.Generator(); rng.manual_seed(0)
        perm = torch.randperm(N, generator=rng)
        tm = torch.zeros(N, dtype=torch.bool); tm[perm[:int(0.6*N)]] = True
        vm = torch.zeros(N, dtype=torch.bool); vm[perm[int(0.6*N):int(0.8*N)]] = True
        te = torch.zeros(N, dtype=torch.bool); te[perm[int(0.8*N):]] = True
        return _make_ns(_wiki_d, tm, vm, te)

    # ------------------------------------------------------------------
    # HeterophilousGraphDataset (Roman-empire, Amazon-ratings, Tolokers …)
    # These Yandex .npz files contain pickled objects, so np.load must be
    # called with allow_pickle=True.  Older PyG versions don't pass that
    # flag, so we monkey-patch np.load around the call and always restore
    # the original afterwards.
    # ------------------------------------------------------------------
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
        _np_inner.load = _orig_np_load   # always restore, even on error

    tm, vm, te = _safe_extract_masks(d)
    return _make_ns(d, tm, vm, te)


# =============================================================================
#  SECTION 2 -- Matched-density subsampling + semi-supervised label masking
#  Copied verbatim from eureka_benchmark.py. These are also the exact
#  functions glare_bestv2.py's own rewire_glare() calls internally.
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
#  SECTION 3 -- GLARE representation / rewiring model
#  Copied verbatim from glare_bestv2.py -- no changes: candidate-edge
#  sampling, the learned-similarity (NT-Xent contrastive) encoder, the
#  distribution-affinity signal, the GLARE-18 module (EM loop over a
#  GraphSAGE-style GNN + per-edge soft/hard adjacency), and rewire_glare()
#  (which trains GLARE-18, extracts the learned Z18 representation, and
#  density-matches the rewired graph) are all reproduced exactly as in
#  glare_bestv2.py.
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
    """# FIX: a real GraphSAGE-style layer (self features concatenated with a
    weighted mean of neighbors), hand-rolled with a sparse matmul so it
    natively supports the continuous edge weights the EM loop needs (PyG's
    SAGEConv does not reliably support edge_weight). Concatenating self
    features instead of GCN's symmetric-normalized mixing is cheaper per
    step and less prone to smearing dissimilar (heterophilic) neighbors
    together than the GCNConv this replaced -- despite the class being
    named "GLARESAGEModel", the original implementation used GCNConv."""
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
        self.bn1 = nn.BatchNorm1d(hid_c)   # FIX: stabilizes the EM-loop training (rewire <-> classify)
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
    """# FIX: continuous coverage-based confidence (was a hard alpha threshold
    that stayed False almost everywhere) + a hard, unambiguous anchor for
    edges directly between two currently-visible labeled nodes, so
    label_mask_ratio actually moves this signal."""
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
        soft_trust = (coverage[sc] * coverage[dc]).clamp(0, 1)          # FIX: continuous
        blended = soft_trust * ls + (1.0 - soft_trust) * fs
        # FIX: asymmetric anchor. Only PULL same-class visible-label pairs
        # together (+1); different-class visible-label pairs are left to the
        # normal blended signal instead of being pushed to -1. On
        # heterophilic graphs, cross-class candidate edges are informative,
        # not noise -- punishing them undoes exactly what H2GCN/LINKX-style
        # architectures rely on. This still ties directly to
        # label_mask_ratio (the +1 pairs vanish as visible labels shrink).
        same_class_visible = train_mask[sc] & train_mask[dc] & (y[sc] == y[dc])
        affinity[s:e] = torch.where(same_class_visible, torch.ones_like(blended), blended)
        trust[s:e] = torch.where(same_class_visible, torch.ones_like(soft_trust), soft_trust)
    return affinity, trust


class _SimLearner(nn.Module):
    """# FIX: was a weak 2-layer MLP (Linear-ReLU-Linear). Now 3 linear layers
    with BatchNorm + PReLU on each, plus a residual (skip) connection from a
    linear projection of the input straight to the output -- gives the
    contrastive pretraining signal below (NT-Xent) a much higher-capacity,
    better-conditioned encoder to push through."""
    def __init__(self, in_dim, hid):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid); self.bn1 = nn.BatchNorm1d(hid); self.act1 = nn.PReLU()
        self.fc2 = nn.Linear(hid, hid); self.bn2 = nn.BatchNorm1d(hid); self.act2 = nn.PReLU()
        self.fc3 = nn.Linear(hid, hid); self.bn3 = nn.BatchNorm1d(hid)
        self.act_out = nn.PReLU()
        self.skip = nn.Linear(in_dim, hid)   # projects input to hid-dim for the residual add

    def forward(self, X):
        h = self.act1(self.bn1(self.fc1(X)))
        h = self.act2(self.bn2(self.fc2(h)))
        h = self.bn3(self.fc3(h))
        h = h + self.skip(X)                 # residual: input -> output
        return self.act_out(h)


def _feature_dropout(x, p):
    """Zeros out a random subset of feature dimensions per call -- a cheap,
    graph-agnostic augmentation used to build the two views for NT-Xent."""
    if p <= 0.0:
        return x
    keep = (torch.rand_like(x) > p).float()
    return x * keep


def _nt_xent_loss(z1, z2, temperature):
    """# FIX: NT-Xent contrastive loss (SimCLR-style) over two augmented
    views of the same batch of nodes, replacing the old MSE-on-random-pairs
    pretraining objective. Each anchor's positive is its own other view;
    everything else in the 2N batch (including the other N-1 nodes' both
    views) is a negative."""
    n = z1.shape[0]
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)                       # (2n, d)
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

    mask_y = train_mask.float()   # FIX: finetune directly on visible labeled nodes, not a coverage threshold

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
    # FIX: NT-Xent contrastive pretraining on feature-dropout-augmented views
    # (was MSE regression toward a fixed feature-similarity target on random
    # pairs -- slow to converge and only ever as informative as that fixed
    # target). Two independently-dropped-out views of the same batch of raw
    # node features are pulled together / pushed apart from everything else.
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
            h = F.relu(gnn.bn1(gnn.conv1(x, ei_soft, edge_weight=ew_soft)))  # FIX: through bn1, matches forward()
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
        # FIX: was only the 1-hop hidden (64-dim). Now cat([1-hop, 2-hop]) = 128-dim:
        # h1 is the usual conv1 hidden rep; h2 is h1 propagated one more hop
        # (simple degree-normalized mean aggregation over the same rewired
        # graph) so Z18 carries both local and 2-hop neighborhood signal.
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


# =============================================================================
#  SECTION 4 -- Downstream classifiers
#  Copied verbatim from glare_bestv2.py, including the input->logits
#  residual/skip connections (NodeClassifier, H2GCNClassifier) and the
#  _ResidualWrapper added around PyG's LINKX.
# =============================================================================

class NodeClassifier(nn.Module):
    def __init__(self, backbone, in_dim, hid_dim, num_classes, dropout=0.5):
        super().__init__(); self.dropout = dropout
        if backbone == "gcn":
            self.conv1 = GCNConv(in_dim, hid_dim); self.conv2 = GCNConv(hid_dim, num_classes)
        elif backbone == "gat":
            heads = 4
            self.conv1 = GATConv(in_dim, hid_dim // heads, heads=heads, dropout=dropout)
            self.conv2 = GATConv(hid_dim, num_classes, heads=1, concat=False, dropout=dropout)
        elif backbone == "sage":
            self.conv1 = SAGEConv(in_dim, hid_dim); self.conv2 = SAGEConv(hid_dim, num_classes)
        else:
            raise ValueError(backbone)
        self.bn = nn.BatchNorm1d(hid_dim)
        self.skip = nn.Linear(in_dim, num_classes)   # FIX: residual path, input -> logits

    def forward(self, x, edge_index):
        x_in = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.bn(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index) + self.skip(x_in)


class H2GCNClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, num_classes, dropout=0.5):
        super().__init__(); self.dropout = dropout
        self.fc_ego = nn.Linear(in_dim, hid_dim); self.fc_h1 = nn.Linear(in_dim, hid_dim)
        self.fc_h2 = nn.Linear(in_dim, hid_dim); self.cls = nn.Linear(3 * hid_dim, num_classes)
        self.bn = nn.BatchNorm1d(3 * hid_dim)
        self.skip = nn.Linear(in_dim, num_classes)   # FIX: residual path, input -> logits

    def _agg(self, x, edge_index, N):
        src, dst = edge_index
        deg = torch.zeros(N, device=x.device).scatter_add_(0, dst, torch.ones(len(dst), device=x.device)).clamp(min=1)
        out = torch.zeros_like(x)
        out.scatter_add_(0, dst.unsqueeze(-1).expand(-1, x.shape[1]), x[src])
        return out / deg.unsqueeze(-1)

    def forward(self, x, edge_index):
        N = x.shape[0]
        h = torch.cat([F.relu(self.fc_ego(x)),
                       F.relu(self.fc_h1(self._agg(x, edge_index, N))),
                       F.relu(self.fc_h2(self._agg(self._agg(x, edge_index, N), edge_index, N)))], dim=-1)
        return self.cls(F.dropout(self.bn(h), p=self.dropout, training=self.training)) + self.skip(x)


class _ResidualWrapper(nn.Module):
    """# FIX: wraps a classifier that has no input->logits skip of its own
    (PyG's built-in LINKX) and adds one, so every downstream classifier gets
    the same residual path as NodeClassifier / H2GCNClassifier."""
    def __init__(self, base, in_dim, num_classes):
        super().__init__()
        self.base = base
        self.skip = nn.Linear(in_dim, num_classes)

    def forward(self, x, edge_index):
        return self.base(x, edge_index) + self.skip(x)


def build_classifier(backbone, in_dim, hid_dim, num_classes, num_nodes=None):
    if backbone == "h2gcn":
        return H2GCNClassifier(in_dim, hid_dim, num_classes)
    if backbone == "linkx":
        base = LINKX(num_nodes=num_nodes, in_channels=in_dim, hidden_channels=hid_dim,
                     out_channels=num_classes, num_layers=2, num_edge_layers=1,
                     num_node_layers=1, dropout=0.5)
        return _ResidualWrapper(base, in_dim, num_classes)   # FIX: residual path, input -> logits
    return NodeClassifier(backbone, in_dim, hid_dim, num_classes)


# =============================================================================
#  SECTION 5 -- Train / evaluate a downstream classifier
#  Copied verbatim from glare_bestv2.py: label smoothing (0.1), grad-norm
#  clipping (5.0), and validation-checked-every-5-epochs model selection.
# =============================================================================

def _macro_f1(pred, true, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = ((pred == c) & (true == c)).sum().item()
        fp = ((pred == c) & (true != c)).sum().item()
        fn = ((pred != c) & (true == c)).sum().item()
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp > 0 else 0.0
        rec = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def train_and_eval(model, X, edge_index, data, cfg, device):
    model = model.to(device); X_d = X.to(device); ei_d = edge_index.to(device)
    y, tm, vm, te = data.y.to(device), data.train_mask.to(device), data.val_mask.to(device), data.test_mask.to(device)
    opt = Adam(model.parameters(), lr=cfg.clf_lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.clf_epochs, eta_min=1e-5)
    best_val, best_state = 0.0, None
    for ep in range(cfg.clf_epochs):
        model.train(); opt.zero_grad()
        # FIX: label smoothing (was plain CE) to curb overconfidence on noisy/rewired graphs
        loss = F.cross_entropy(model(X_d, ei_d)[tm], y[tm], label_smoothing=cfg.clf_label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clf_grad_clip)  # FIX: grad clipping
        opt.step(); sched.step()
        if (ep + 1) % cfg.clf_val_every == 0:   # FIX: was every 10 epochs, now every 5
            model.eval()
            with torch.no_grad():
                pred = model(X_d, ei_d).argmax(-1)
                vacc = (pred[vm] == y[vm]).float().mean().item()
            if vacc > best_val:
                best_val = vacc; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_d, ei_d).argmax(-1)
        tacc = (pred[te] == y[te]).float().mean().item()
        tf1 = _macro_f1(pred[te], y[te], data.num_classes)
    return best_val, tacc, tf1


# =============================================================================
#  SECTION 6 -- Resumable disk cache around rewire_glare()
#  (harness addition, same spirit as eureka_benchmark.py's rewire_eureka()
#  wrapper around run_eureka(); rewire_glare() itself, in Section 3 above,
#  is completely untouched.)
# =============================================================================

def rewire_glare_resumable(data, cfg, device, seed):
    """Checkpoint GLARE's (rewired edge_index, learned Z18) to disk, keyed by
    dataset / seed / label_mask_ratio, so a resumed sweep does not retrain
    GLARE-18 for a (dataset, seed, ratio) combo it has already computed."""
    ratio_tag = f"_ratio{cfg.label_mask_ratio:.3f}"
    ckpt = CKPT_DIR / f"{cfg.dataset}_glare_seed{seed}{ratio_tag}.pt"
    if ckpt.exists():
        print(f"    [Resume] cached GLARE embeddings seed={seed}")
        blob = torch.load(ckpt, map_location="cpu")
        return blob["ei_matched"], blob["Z"]
    ei_matched, Z = rewire_glare(data, cfg, device, seed)
    torch.save({"ei_matched": ei_matched, "Z": Z}, ckpt)
    print(f"    [Saved] GLARE -> {ckpt}")
    return ei_matched, Z


# =============================================================================
#  SECTION 7 -- Results persistence
#  (Unused helpers kept for structural parity with eureka_benchmark.py --
#  the actual persistence/resume mechanism is common.checkpoint.ResultStore,
#  used in run_benchmark() below.)
# =============================================================================

def save_result(rec):
    with open(RESULTS_FILE, "a") as f: f.write(json.dumps(rec) + "\n")

def load_results():
    if not RESULTS_FILE.exists(): return []
    with open(RESULTS_FILE) as f: return [json.loads(l) for l in f if l.strip()]

def result_exists(dataset, method, backbone, seed):
    return any(r.get("dataset") == dataset and r.get("method") == method
               and r.get("backbone") == backbone and r.get("seed") == seed
               for r in load_results())


# =============================================================================
#  SECTION 8 -- Unified benchmark loop  (glare_rewired_learnedZ ONLY)
# =============================================================================
#
# This file evaluates exactly ONE graph/feature configuration, run with all
# 5 downstream classifiers, over seeds {0, 1, 2}, on all 6 real datasets:
#
#   "glare"  ==  GLARE-18-rewired graph + [ normalize(X_raw) | normalize(Z18) ]
#
# "original" and any other GLARE variant are intentionally excluded (per
# request: only glare_rewired_learnedZ is run here).
#
# For every (dataset, method, classifier, seed) we record accuracy, macro-F1
# and runtime (rewiring + classifier training). Results are appended to a
# JSONL file after every single run, so an interrupted sweep resumes exactly
# where it left off. Per-dataset tables, an "average over datasets per
# classifier" table and all figures are produced at the end (and refreshed
# after every dataset) -- identical output structure to eureka_benchmark.py.

ALL_DATASETS = [
    "Actor", "Squirrel-F", "Chameleon-F",
    "Roman-empire", "Amazon-ratings", "Tolokers",
]

# All 5 downstream classifiers are kept (each is a standard node classifier).
CLASSIFIERS = ["gcn", "gat", "sage", "h2gcn", "linkx"]

# Only one configuration is evaluated in this file: glare_rewired_learnedZ,
# labeled "glare" in the results (all other methods -- "original", plain
# glare_rewired without learned-Z, etc. -- are removed per request).
METHODS = ["glare"]

# Output layout (OUT_DIR / CKPT_DIR were created at the top of the file).
RESULTS_JSONL = OUT_DIR / "results.jsonl"
TABLES_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "plots"
KEY_FIELDS = ["dataset", "method", "classifier", "seed"]
GROUP_FIELDS = ["method", "classifier"]


def _refresh_reports(store, datasets):
    """(Re)write summary CSVs, print tables and regenerate figures."""
    records = store.all()
    summary, avg = _reporting.write_all_csvs(
        records, GROUP_FIELDS, TABLES_DIR,
        dataset_order=datasets, make_average=True,
    )
    if summary.empty:
        return
    print("\n" + "=" * 72)
    print("  GLARE BENCHMARK -- SUMMARY SO FAR")
    print("=" * 72)
    _reporting.print_dataset_tables(summary, GROUP_FIELDS, datasets)
    _reporting.print_average_table(avg, GROUP_FIELDS)
    _plotting.generate_all_figures(summary, avg, GROUP_FIELDS, FIG_DIR,
                                   dataset_order=datasets)


def run_benchmark(cfg):
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))
    # GLARE18Model does torch.device(cfg.device) internally (glare_bestv2.py,
    # unchanged), so cfg.device must hold a resolved "cuda"/"cpu" string,
    # not "auto".
    cfg.device = str(device)

    print("\n" + "=" * 72)
    print("  GLARE Rewiring Benchmark  (glare_rewired_learnedZ only)")
    print("=" * 72)
    print(f"  Datasets    : {cfg.datasets}")
    print(f"  Methods     : {METHODS}")
    print(f"  Classifiers : {cfg.classifiers}")
    print(f"  Seeds       : {cfg.seeds}")
    print(f"  label_mask_ratio : {cfg.label_mask_ratio}")
    print(f"  Device      : {device}")
    print(f"  Results     -> {OUT_DIR.resolve()}")
    print("=" * 72)

    store = _checkpoint.ResultStore(RESULTS_JSONL, key_fields=KEY_FIELDS)
    if store.count():
        print(f"  [resume] found {store.count()} completed run(s); "
              f"they will be skipped.\n")

    total_runs = (len(cfg.datasets) * len(METHODS)
                  * len(cfg.classifiers) * len(cfg.seeds))
    tracker = _progress.ProgressTracker(total_runs, label="glare").start()

    for dataset in cfg.datasets:
        cfg.dataset = dataset
        print(f"\n{'-'*72}\n  DATASET: {dataset}\n{'-'*72}")

        try:
            data = load_real_dataset(dataset, root=cfg.data_root)
        except Exception as e:
            print(f"  [SKIP dataset] {dataset}: {e}")
            tracker.tick_skipped(len(METHODS) * len(cfg.classifiers) * len(cfg.seeds))
            continue

        X_raw = data.x.cpu()
        N, C = data.num_nodes, data.num_classes
        print(f"  N={N}  C={C}  edges={data.edge_index.shape[1]}  "
              f"train={int(data.train_mask.sum())}  "
              f"val={int(data.val_mask.sum())}  "
              f"test={int(data.test_mask.sum())}")

        # Lazily-computed, per-seed GLARE rewiring (cached to disk by
        # rewire_glare_resumable, so resume is cheap).
        glare_cache = {}   # seed -> (edge_index, Z18, rewire_time_s)

        def get_glare(seed):
            if seed in glare_cache:
                return glare_cache[seed]
            set_global_seed(seed)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            ei, Z = rewire_glare_resumable(data, cfg, device, seed)
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            rt = time.perf_counter() - t0
            glare_cache[seed] = (ei.cpu(), Z.cpu(), rt)
            return glare_cache[seed]

        for method in METHODS:
            if method not in cfg.methods:
                # method disabled on the CLI -- account for its runs and skip
                tracker.tick_skipped(len(cfg.classifiers) * len(cfg.seeds))
                continue
            for clf_name in cfg.classifiers:
                for seed in cfg.seeds:
                    if cfg.resume and store.exists(
                        dataset=dataset, method=method,
                        classifier=clf_name, seed=seed,
                    ):
                        tracker.tick_skipped()
                        continue

                    t_run = time.perf_counter()
                    try:
                        set_global_seed(seed)

                        # method == "glare"  (glare_rewired_learnedZ, the only
                        # configuration run in this file)
                        ei_use, Z, rewire_time = get_glare(seed)
                        # FIX (from glare_bestv2.py, unchanged): L2-normalize
                        # raw features and learned Z separately before
                        # concatenating -- Z (post-BN GNN hidden states) and
                        # X_raw (raw, possibly unnormalized node features)
                        # live on different scales, so a naive concat lets
                        # whichever has larger norm dominate the downstream
                        # classifier's first layer.
                        X_raw_n = F.normalize(X_raw.to(Z.dtype), dim=-1)
                        Z_n = F.normalize(Z, dim=-1)
                        X_use = torch.cat([X_raw_n, Z_n], dim=1)

                        # Reseed right before classifier init for reproducibility.
                        set_global_seed(seed)
                        clf = build_classifier(
                            clf_name, X_use.shape[1], cfg.clf_hid, C,
                            num_nodes=N,
                        )

                        t_clf0 = time.perf_counter()
                        val_acc, test_acc, test_f1 = train_and_eval(
                            clf, X_use, ei_use, data, cfg, device,
                        )
                        if torch.cuda.is_available() and device.type == "cuda":
                            torch.cuda.synchronize()
                        clf_time = time.perf_counter() - t_clf0

                        total_time = rewire_time + clf_time
                        rec = {
                            "dataset": dataset,
                            "method": method,
                            "classifier": clf_name,
                            "seed": seed,
                            "acc": test_acc,
                            "f1": test_f1,
                            "val_acc": val_acc,
                            "rewiring_time_s": rewire_time,
                            "train_time_s": clf_time,
                            "total_time_s": total_time,
                        }
                        store.append(rec)
                        note = (f"{dataset}/{method}/{clf_name}/s{seed} "
                                f"acc={test_acc:.4f} f1={test_f1:.4f}")
                        tracker.tick_done(time.perf_counter() - t_run, note=note)

                    except Exception as e:
                        print(f"    [SKIP run] {dataset}/{method}/{clf_name}"
                              f"/seed={seed}: {e}")
                        if cfg.verbose:
                            traceback.print_exc()
                        # Count it so the ETA does not stall; it will be retried
                        # on a future resume because nothing was written.
                        tracker.tick_done(time.perf_counter() - t_run,
                                          note="FAILED")

        # Refresh reports/plots after each dataset so partial progress is visible.
        _refresh_reports(store, cfg.datasets)

    tracker.finish()
    _refresh_reports(store, cfg.datasets)
    print(f"\n[Done] Results, tables and plots in {OUT_DIR.resolve()}")


# =============================================================================
#  SECTION 9 -- CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="GLARE (glare_rewired_learnedZ only) benchmark")

    p.add_argument("--datasets",   nargs="+", default=ALL_DATASETS)
    p.add_argument("--methods",    nargs="+", default=METHODS,
                   choices=METHODS,
                   help="Only 'glare' (glare_rewired_learnedZ) is supported in this file.")
    p.add_argument("--classifiers", nargs="+", default=CLASSIFIERS,
                   choices=CLASSIFIERS)
    p.add_argument("--seeds",      nargs="+", type=int, default=[0, 1, 2])

    # GLARE-18 hyperparameters (defaults copied unchanged from glare_bestv2.py's CFG)
    p.add_argument("--glare_outer_loops",       type=int,   default=25)
    p.add_argument("--glare_gnn_epochs",        type=int,   default=80)
    p.add_argument("--glare_rewire_steps",      type=int,   default=120)
    p.add_argument("--glare_neg_edge_budget",   type=int,   default=10000)
    p.add_argument("--glare_lambda_entropy",    type=float, default=0.02)
    p.add_argument("--glare_lambda_sparsity",   type=float, default=0.01)
    p.add_argument("--glare_lambda_original",   type=float, default=0.10)
    p.add_argument("--glare_rewire_lr",         type=float, default=5e-3)
    p.add_argument("--glare_gnn_lr",            type=float, default=1e-3)
    p.add_argument("--glare_gnn_hidden",        type=int,   default=64)
    p.add_argument("--glare_gnn_dropout",       type=float, default=0.5)
    p.add_argument("--glare_threshold",         type=float, default=0.5)
    p.add_argument("--glare_temperature",       type=float, default=1.0)
    p.add_argument("--glare18_lambda_prior",    type=float, default=0.10)
    p.add_argument("--glare18_lambda_degree",   type=float, default=0.05)
    p.add_argument("--glare18_lambda_hom",      type=float, default=0.25)
    p.add_argument("--glare18_struct_mix",      type=float, default=0.5)
    p.add_argument("--glare18_kappa_prior",     type=float, default=2.0)
    p.add_argument("--glare18_density_ratio",   type=float, default=1.0)
    p.add_argument("--glare18_density_adapt_strength", type=float, default=1.0)
    p.add_argument("--glare18_dist_M",          type=int,   default=2)
    p.add_argument("--glare18_dist_alpha",      type=float, default=0.3)
    p.add_argument("--glare18_use_learned_knn", type=int,   default=1, choices=[0, 1])
    p.add_argument("--glare18_sim_hidden",      type=int,   default=64)
    p.add_argument("--glare18_sim_lr",          type=float, default=1e-3)
    p.add_argument("--glare18_sim_weight_decay", type=float, default=5e-4)
    p.add_argument("--glare18_sim_pretrain_epochs", type=int, default=200)
    p.add_argument("--glare18_sim_finetune_epochs", type=int, default=30)
    p.add_argument("--glare18_sim_batch_k",     type=int,   default=512)
    p.add_argument("--glare18_sim_max_iter",    type=int,   default=20)
    p.add_argument("--glare18_sim_M",           type=int,   default=2)
    p.add_argument("--glare18_knn_K",           type=int,   default=8)
    p.add_argument("--glare18_sim_aug_drop",    type=float, default=0.2)
    p.add_argument("--glare18_sim_nce_temp",    type=float, default=0.5)
    p.add_argument("--glare18_knn_epsilon",     type=float, default=0.1)
    p.add_argument("--glare18_knn_block",       type=int,   default=512)
    p.add_argument("--glare18_learned_affinity_weight", type=float, default=0.5)
    p.add_argument("--glare18_affinity_chunk_size", type=int, default=150_000)
    p.add_argument("--glare18_orig_retain_floor", type=float, default=0.3)
    p.add_argument("--glare18_target_edge_ratio", type=float, default=0.7)

    p.add_argument("--label_mask_ratio",   type=float, default=1.0, metavar="R",
                   help="Fraction of training labels visible to GLARE "
                        "(1.0 = full supervision -- run this first; "
                        "0.0 = fully label-blind -- run this second).")

    # Downstream classifier hyper-parameters (defaults unchanged from glare_bestv2.py)
    p.add_argument("--clf_epochs",          type=int,   default=300)
    p.add_argument("--clf_hid",             type=int,   default=128)
    p.add_argument("--clf_lr",              type=float, default=5e-3)
    p.add_argument("--clf_label_smoothing", type=float, default=0.1)
    p.add_argument("--clf_grad_clip",       type=float, default=5.0)
    p.add_argument("--clf_val_every",       type=int,   default=5)

    # Runtime
    p.add_argument("--data_root",  default="./data")
    p.add_argument("--device",     default="auto")
    p.add_argument("--resume",     action="store_true", default=True,
                   help="Skip runs already present in results.jsonl (default on).")
    p.add_argument("--no_resume",  dest="resume", action="store_false")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--verbose",    action="store_true")

    args = p.parse_args()

    if args.smoke_test:
        print("[Smoke-test mode]")
        args.seeds                       = args.seeds[:1]
        args.glare_outer_loops           = min(args.glare_outer_loops, 3)
        args.glare_gnn_epochs            = min(args.glare_gnn_epochs, 10)
        args.glare_rewire_steps          = min(args.glare_rewire_steps, 10)
        args.glare18_sim_pretrain_epochs = min(args.glare18_sim_pretrain_epochs, 10)
        args.glare18_sim_finetune_epochs = min(args.glare18_sim_finetune_epochs, 5)
        args.clf_epochs                  = min(args.clf_epochs, 30)

    if not (0.0 <= args.label_mask_ratio <= 1.0):
        p.error("--label_mask_ratio must be in [0.0, 1.0]")

    return args


if __name__ == "__main__":
    cfg = parse_args()
    run_benchmark(cfg)
