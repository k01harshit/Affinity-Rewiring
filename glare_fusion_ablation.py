"""
glare_fusion_ablation.py
========================
2x2 ablation separating GLARE's two outputs -- the rewired TOPOLOGY and the
fused FEATURES X_hat = [l2(X) | l2(Z)] -- so each can be credited separately.

                         features = l2(X)         features = X_hat
    original graph       orig_X                   orig_Xhat
    GLARE-rewired graph  rew_X                    rew_Xhat   (= the paper's GLARE)

Optional controls (off by default, enable with --configs):
    orig_Xrand / rew_Xrand : [l2(X) | l2(R)] with R ~ N(0, I), same shape as Z.
                             Rules out "the gain is just extra input width".
    --with_mlp             : a graph-free MLP on l2(X) and on X_hat. If the MLP
                             on X_hat already matches GLARE, the graph is not
                             where the gain comes from.

Everything GLARE-related is imported unchanged from glare_benchmark_final.py:
data loading, the rewiring (rewire_glare_resumable, which reuses the SAME
cached checkpoints as the main benchmark), classifiers, and train_and_eval.
Only the (graph, features) pair fed to the classifier changes between cells.

Notes on interpretation
-----------------------
* All "X" cells use l2-normalised X, not raw X, so the ONLY difference between
  a *_X cell and its *_Xhat partner is the appended Z block. (orig_X can
  therefore differ slightly from the paper's "Original" row in Table 4.)
* Z is always the Z GLARE produced, i.e. computed on the rewired graph. So
  orig_Xhat means "rewiring-derived features, original message passing".
  That is exactly the confound in question: if orig_Xhat ~ rew_Xhat, the
  gain travels through the features, not the topology.
* Run it for both --label_mask_ratio 1.0 and 0.0. At 0.0 the inner GNN is
  never trained, so Z is a random-weight projection; comparing the two
  ratios tells you how much of the X_hat effect is label information.

Usage (run from the SAME directory as the main benchmark so the cached
GLARE rewirings in glare_benchmark_results/checkpoints are reused):

  python glare_fusion_ablation.py --device cuda
  python glare_fusion_ablation.py --device cuda --label_mask_ratio 0.0
  python glare_fusion_ablation.py --configs orig_X orig_Xhat rew_X rew_Xhat \
         orig_Xrand rew_Xrand --with_mlp
  python glare_fusion_ablation.py --smoke_test --datasets Chameleon-F
  python glare_fusion_ablation.py --summarize_only     # rebuild tables only

Any GLARE / classifier hyperparameter flag accepted by glare_benchmark_final.py
(e.g. --glare_outer_loops, --clf_epochs, --seeds, --classifiers) is passed
through to it unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

ALL_CONFIGS = ["orig_X", "orig_Xhat", "rew_X", "rew_Xhat", "orig_Xrand", "rew_Xrand"]
DEFAULT_CONFIGS = ["orig_X", "orig_Xhat", "rew_X", "rew_Xhat"]
MLP_NAME = "mlp_nograph"
KEY_FIELDS = ("dataset", "ratio", "config", "classifier", "seed")


# =============================================================================
#  Results store (plain JSONL, resumable)
# =============================================================================

class JsonlStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._keys = set()
        for r in self.all():
            self._keys.add(self._key(r))

    @staticmethod
    def _key(r):
        return tuple(str(r[k]) for k in KEY_FIELDS)

    def all(self):
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(l) for l in f if l.strip()]

    def exists(self, **kw):
        return tuple(str(kw[k]) for k in KEY_FIELDS) in self._keys

    def append(self, rec):
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        self._keys.add(self._key(rec))


# =============================================================================
#  Summaries (pure pandas; no torch needed)
# =============================================================================

def summarize(records, out_dir: Path, dataset_order=None):
    """Write per-classifier, 2x2 and effect-decomposition tables."""
    if not records:
        print("[summary] no records yet")
        return None
    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = [d for d in (dataset_order or []) if d in set(df.dataset)] or sorted(df.dataset.unique())
    lines = ["# GLARE feature-fusion vs topology ablation", ""]

    for ratio, dfr in df.groupby("ratio"):
        tag = f"ratio{float(ratio):.3f}"
        lines += [f"## label_mask_ratio = {ratio}", ""]

        # --- 1. Per (config, classifier, dataset): mean +- std over seeds ----
        per_clf = (dfr.groupby(["config", "classifier", "dataset"])["acc"]
                   .agg(["mean", "std", "count"]).reset_index())
        per_clf.to_csv(out_dir / f"per_classifier_{tag}.csv", index=False)

        # --- 2. 2x2 cells averaged over GNN classifiers (MLP excluded) -------
        gnn = dfr[dfr.classifier != MLP_NAME]
        # seed-mean per classifier first, then mean/std across classifiers
        clf_means = gnn.groupby(["config", "classifier", "dataset"])["acc"].mean().reset_index()
        cell = (clf_means.groupby(["config", "dataset"])["acc"]
                .agg(["mean", "std"]).reset_index())
        wide = cell.pivot(index="config", columns="dataset", values="mean")
        wide = wide.reindex(columns=[d for d in datasets if d in wide.columns])
        wide["Mean"] = wide.mean(axis=1)
        order = [c for c in ALL_CONFIGS if c in wide.index]
        wide = wide.loc[order]
        spread = cell.pivot(index="config", columns="dataset", values="std").reindex(
            index=order, columns=[d for d in datasets if d in wide.columns])

        if (dfr.classifier == MLP_NAME).any():
            mlp = (dfr[dfr.classifier == MLP_NAME]
                   .groupby(["config", "dataset"])["acc"].mean().unstack())
            mlp = mlp.reindex(columns=[d for d in datasets if d in mlp.columns])
            mlp["Mean"] = mlp.mean(axis=1)
            mlp.index = [f"MLP (no graph) on {'X' if c == 'orig_X' else 'X_hat'}"
                         for c in mlp.index]
        else:
            mlp = None

        wide.to_csv(out_dir / f"cells_2x2_{tag}.csv")
        spread.to_csv(out_dir / f"cells_2x2_spread_across_classifiers_{tag}.csv")
        lines += ["### Test accuracy, mean over GNN classifiers and seeds", "",
                  _md(wide), ""]
        lines += ["### Std. dev. across classifiers (classifier-choice gap)", "",
                  _md(spread), ""]
        if mlp is not None:
            mlp.to_csv(out_dir / f"mlp_nograph_{tag}.csv")
            lines += ["### Graph-free reference", "", _md(mlp), ""]

        # --- 3. Effect decomposition (accuracy points) -----------------------
        need = {"orig_X", "orig_Xhat", "rew_X", "rew_Xhat"}
        if need.issubset(wide.index):
            w = wide * 100.0
            dec = pd.DataFrame({
                "topology (rew_X - orig_X)":        w.loc["rew_X"] - w.loc["orig_X"],
                "features (orig_Xhat - orig_X)":    w.loc["orig_Xhat"] - w.loc["orig_X"],
                "interaction":                      (w.loc["rew_Xhat"] - w.loc["rew_X"])
                                                    - (w.loc["orig_Xhat"] - w.loc["orig_X"]),
                "total (rew_Xhat - orig_X)":        w.loc["rew_Xhat"] - w.loc["orig_X"],
                "topology given X_hat (rew_Xhat - orig_Xhat)":
                                                    w.loc["rew_Xhat"] - w.loc["orig_Xhat"],
            }).T
            if "rew_Xrand" in w.index:
                dec.loc["Z vs random block on rewired graph (rew_Xhat - rew_Xrand)"] = (
                    w.loc["rew_Xhat"] - w.loc["rew_Xrand"])
            if "orig_Xrand" in w.index:
                dec.loc["Z vs random block on original graph (orig_Xhat - orig_Xrand)"] = (
                    w.loc["orig_Xhat"] - w.loc["orig_Xrand"])
            dec.to_csv(out_dir / f"effect_decomposition_{tag}.csv")
            lines += ["### Effect decomposition (accuracy points)", "", _md(dec, fmt="{:+.1f}"), "",
                      "Main effects are measured from orig_X; interaction = how much the "
                      "features' gain changes when the graph is rewired.", ""]

            # per-classifier decomposition: is the topology effect uniform?
            pc = clf_means.pivot_table(index=["classifier", "dataset"], columns="config",
                                       values="acc") * 100.0
            if need.issubset(pc.columns):
                pcd = pd.DataFrame({
                    "topology": pc["rew_X"] - pc["orig_X"],
                    "features": pc["orig_Xhat"] - pc["orig_X"],
                    "total": pc["rew_Xhat"] - pc["orig_X"],
                }).reset_index()
                pcd.to_csv(out_dir / f"effect_per_classifier_{tag}.csv", index=False)
                topo_pos = int((pcd["topology"] > 0).sum())
                lines += [f"Topology alone (rew_X vs orig_X) helps in {topo_pos}/{len(pcd)} "
                          "classifier-dataset cells.", ""]

    report = "\n".join(lines)
    (out_dir / "ablation_report.md").write_text(report)
    print("\n" + report)
    return report


def _md(df, fmt="{:.3f}"):
    cols = [str(c) for c in df.columns]
    out = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = ["" if pd.isna(v) else fmt.format(v) for v in row.values]
        out.append(f"| {idx} | " + " | ".join(cells) + " |")
    return "\n".join(out)


# =============================================================================
#  Benchmark loop
# =============================================================================

def build_cfg(argv_rest):
    """Parse all GLARE/classifier flags with the ORIGINAL parser, unchanged."""
    import glare_benchmark_final as gb
    saved = sys.argv
    try:
        sys.argv = [saved[0]] + list(argv_rest)
        cfg = gb.parse_args()
    finally:
        sys.argv = saved
    return gb, cfg


def run(args, argv_rest):
    gb, cfg = build_cfg(argv_rest)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class MLPNoGraph(nn.Module):
        """Graph-free reference with the same shape as the harness classifiers
        (2 layers, BN, dropout 0.5, input->logits skip). Ignores edge_index."""
        def __init__(self, in_dim, hid, n_cls, dropout=0.5):
            super().__init__()
            self.fc1, self.fc2 = nn.Linear(in_dim, hid), nn.Linear(hid, n_cls)
            self.bn, self.skip, self.dropout = nn.BatchNorm1d(hid), nn.Linear(in_dim, n_cls), dropout

        def forward(self, x, edge_index=None):
            h = F.dropout(x, self.dropout, self.training)
            h = F.elu(self.bn(self.fc1(h)))
            h = F.dropout(h, self.dropout, self.training)
            return self.fc2(h) + self.skip(x)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if cfg.device == "auto" else torch.device(cfg.device))
    cfg.device = str(device)   # GLARE18Model needs a resolved device string
    ratio = float(cfg.label_mask_ratio)

    out_dir = Path(args.out_dir)
    store = JsonlStore(out_dir / "results.jsonl")
    print("=" * 72)
    print("  GLARE 2x2 feature-fusion ablation")
    print(f"  datasets={cfg.datasets}  configs={args.configs}")
    print(f"  classifiers={cfg.classifiers}{' + MLP' if args.with_mlp else ''}  "
          f"seeds={cfg.seeds}  label_mask_ratio={ratio}  device={device}")
    print(f"  GLARE cache: {gb.CKPT_DIR.resolve()}")
    print("=" * 72)

    for dataset in cfg.datasets:
        cfg.dataset = dataset
        try:
            data = gb.load_real_dataset(dataset, root=cfg.data_root)
        except Exception as e:
            print(f"[SKIP dataset] {dataset}: {e}")
            continue
        N, C = data.num_nodes, data.num_classes
        X_n = F.normalize(data.x.float().cpu(), dim=-1)
        print(f"\n--- {dataset}: N={N} C={C} edges={data.edge_index.shape[1]}")

        for seed in cfg.seeds:
            # Which runs are still missing for this seed?
            jobs = []
            for conf in args.configs:
                clfs = list(cfg.classifiers)
                if args.with_mlp and conf in ("orig_X", "orig_Xhat"):
                    clfs.append(MLP_NAME)
                for clf_name in clfs:
                    if not store.exists(dataset=dataset, ratio=ratio, config=conf,
                                        classifier=clf_name, seed=seed):
                        jobs.append((conf, clf_name))
            if not jobs:
                print(f"  seed {seed}: all runs cached")
                continue

            # GLARE outputs (loaded from the main benchmark's cache if present).
            need_glare = any(c.startswith("rew_") or c.endswith("Xhat") or c.endswith("Xrand")
                             for c, _ in jobs)
            ei_rew = Z = None
            if need_glare:
                gb.set_global_seed(seed)
                t0 = time.perf_counter()
                ei_rew, Z = gb.rewire_glare_resumable(data, cfg, device, seed)
                ei_rew, Z = ei_rew.cpu(), Z.float().cpu()
                print(f"  seed {seed}: GLARE ready in {time.perf_counter() - t0:.1f}s "
                      f"(rewired edges={ei_rew.shape[1]}, Z dim={Z.shape[1]})")

            graphs = {"orig": data.edge_index.cpu(), "rew": ei_rew}
            feats = {"X": X_n}
            if Z is not None:
                feats["Xhat"] = torch.cat([X_n, F.normalize(Z, dim=-1)], dim=1)
                g = torch.Generator().manual_seed(10_000 + seed)
                R = torch.randn(Z.shape, generator=g)
                feats["Xrand"] = torch.cat([X_n, F.normalize(R, dim=-1)], dim=1)

            for conf, clf_name in jobs:
                graph_key, feat_key = conf.split("_", 1)
                ei, X_use = graphs[graph_key], feats[feat_key]
                t_run = time.perf_counter()
                try:
                    gb.set_global_seed(seed)
                    if clf_name == MLP_NAME:
                        clf = MLPNoGraph(X_use.shape[1], cfg.clf_hid, C)
                    else:
                        clf = gb.build_classifier(clf_name, X_use.shape[1], cfg.clf_hid, C,
                                                  num_nodes=N)
                    val_acc, test_acc, test_f1 = gb.train_and_eval(clf, X_use, ei, data, cfg, device)
                except Exception as e:
                    print(f"    [FAILED] {conf}/{clf_name}/s{seed}: {e}")
                    if cfg.verbose:
                        traceback.print_exc()
                    continue
                store.append({
                    "dataset": dataset, "ratio": ratio, "config": conf,
                    "graph": graph_key, "features": feat_key,
                    "classifier": clf_name, "seed": seed,
                    "acc": test_acc, "f1": test_f1, "val_acc": val_acc,
                    "num_edges": int(ei.shape[1]), "in_dim": int(X_use.shape[1]),
                    "train_time_s": time.perf_counter() - t_run,
                })
                print(f"    {conf:<11} {clf_name:<12} s{seed}  acc={test_acc:.4f}  f1={test_f1:.4f}")

        summarize(store.all(), out_dir, dataset_order=cfg.datasets)

    summarize(store.all(), out_dir, dataset_order=cfg.datasets)
    print(f"\n[Done] results and tables in {out_dir.resolve()}")


def main():
    p = argparse.ArgumentParser(
        description="2x2 topology-vs-features ablation for GLARE. Unrecognised flags "
                    "are forwarded to glare_benchmark_final.parse_args().")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS, choices=ALL_CONFIGS)
    p.add_argument("--with_mlp", action="store_true",
                   help="also run a graph-free MLP on l2(X) and on X_hat")
    p.add_argument("--out_dir", default="glare_fusion_ablation_results")
    p.add_argument("--summarize_only", action="store_true",
                   help="rebuild tables from existing results.jsonl without training")
    args, rest = p.parse_known_args()

    if args.summarize_only:
        store = JsonlStore(Path(args.out_dir) / "results.jsonl")
        summarize(store.all(), Path(args.out_dir))
        return
    run(args, rest)


if __name__ == "__main__":
    main()
