"""
tolokers_auc_benchmark.py
=========================
Unified Tolokers-only benchmark that records ROC-AUC (plus accuracy and
macro-F1) for every method in the paper:

  * original graph  : the 5 downstream classifiers on the input graph, raw X
  * glare           : GLARE (label_mask_ratio = 1.0, validation selection)
  * glare_u         : GLARE-U (fully unsupervised: no labels, last iterate)
  * idgl dhgr comfy fosr lpkg gadc : the six baselines, each with its own
                      paper classifier (graphite available via --methods)

Methodology is NOT changed
--------------------------
This script does not re-implement anything. It loads the two original files,
unchanged on disk,

    glare_benchmark_final.py   (GLARE + the 5-classifier training harness)
    benchmark_others.py        (the six baselines)

and applies a few tiny text patches IN MEMORY before executing them:

  1. One "hook" line next to the final test evaluation of every method. It
     only copies the test-set scores (logits, log-probabilities or LP
     probabilities, whatever the method already computes for its argmax) so
     that ROC-AUC can be computed from them. Training, model selection and
     the accuracy / macro-F1 computation are untouched.
  2. For glare_u only: the same fully-unsupervised switch that was used for
     the GLARE-U results in the paper (labels replaced by a dummy vector,
     empty train/val masks, last outer iterate kept, separate cache tag).
     With cfg.unsupervised = False these lines do nothing, so GLARE itself
     runs exactly as before.

Every patch is checked: if the expected text is not found the exact number
of times, the script stops with an error instead of running something else.

ROC-AUC
-------
Tolokers is binary. AUC is computed with sklearn from the positive-class
probability softmax(scores)[:, 1]. For binary scores this equals ranking by
s1 - s0, which is the same decision score the method's own argmax uses, so it
is valid for logits, log-probabilities and probabilities alike. Each record
also stores the fraction of test nodes predicted positive, which exposes
runs that collapse to a single class (as GADC appears to do).

Usage (run from the folder that contains both original files, the `common`
package, and the glare_benchmark_results/checkpoints cache, so cached GLARE
and GLARE-U graphs are reused instead of recomputed):

  python tolokers_auc_benchmark.py --device cuda
  python tolokers_auc_benchmark.py --methods gadc --verbose
  python tolokers_auc_benchmark.py --smoke_test
  python tolokers_auc_benchmark.py --check_patches     # verify patches only
  python tolokers_auc_benchmark.py --summarize_only

Outputs (in --out_dir, default tolokers_auc_results/):
  results.jsonl        one line per (method, classifier, seed), resumable
  summary.csv          mean and std over seeds of acc, f1, auc per row
  summary_methods.csv  per method (GLARE rows averaged over classifiers)
  summary.md           the same tables in Markdown
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = "Tolokers"
GLARE_FILE = HERE / "glare_benchmark_final.py"
OTHERS_FILE = HERE / "benchmark_others.py"

GLARE_METHODS = ["original", "glare", "glare_u"]
BASELINES = ["idgl", "dhgr", "comfy", "fosr", "lpkg", "gadc"]
OPTIONAL = ["graphite"]
CLASSIFIERS = ["gcn", "gat", "sage", "h2gcn", "linkx"]
KEY = ("method", "classifier", "seed")

# =============================================================================
#  Score hook shared by both patched modules
# =============================================================================

_AUC_STATE = {"last": None}


def _AUC_HOOK(scores, y_true):
    """Called once at the final test evaluation of a method. Stores a copy
    of the test scores; it does not modify anything the method uses."""
    _AUC_STATE["last"] = (scores.detach().float().cpu().clone(),
                          y_true.detach().cpu().clone())


# =============================================================================
#  In-memory patches (old_text, new_text, expected_count)
# =============================================================================

def _glare_patches():
    P = []
    # --- 1. score hook in the downstream-classifier harness ----------------
    old = ("    with torch.no_grad():\n"
           "        pred = model(X_d, ei_d).argmax(-1)\n"
           "        tacc = (pred[te] == y[te]).float().mean().item()")
    new = ("    with torch.no_grad():\n"
           "        _auc_scores = model(X_d, ei_d)\n"
           "        pred = _auc_scores.argmax(-1)\n"
           "        _AUC_HOOK(_auc_scores[te], y[te])\n"
           "        tacc = (pred[te] == y[te]).float().mean().item()")
    P.append((old, new, 1))

    # --- 2. GLARE-U switch (inactive unless cfg.unsupervised is True) ------
    old = ("        gnn.eval(); glare.eval()\n"
           "        with torch.no_grad():\n"
           "            logits = glare.forward_gnn(gnn, hard=True)\n"
           "            val_acc = (logits[val_mask].argmax(-1) == y[val_mask]).float().mean().item()\n"
           "        if val_acc > best_val:\n"
           "            best_val = val_acc\n"
           "            best_state = {\"gnn\": copy.deepcopy(gnn.state_dict()), \"theta\": glare.theta.detach().clone()}\n")
    new = ("        if cfg.glare_select == \"val\":\n"
           "            gnn.eval(); glare.eval()\n"
           "            with torch.no_grad():\n"
           "                logits = glare.forward_gnn(gnn, hard=True)\n"
           "                val_acc = (logits[val_mask].argmax(-1) == y[val_mask]).float().mean().item()\n"
           "            if val_acc > best_val:\n"
           "                best_val = val_acc\n"
           "                best_state = {\"gnn\": copy.deepcopy(gnn.state_dict()), \"theta\": glare.theta.detach().clone()}\n")
    P.append((old, new, 1))

    old = "    return ei, Z18, best_val\n"
    new = ("    if cfg.glare_select != \"val\":\n"
           "        best_val = float(\"nan\")\n"
           "    return ei, Z18, best_val\n")
    P.append((old, new, 1))

    old = ("def rewire_glare(data, cfg, device, seed):\n"
           "    masked_train = apply_label_mask(data.train_mask, cfg.label_mask_ratio, seed=seed)\n"
           "    data_glare = SimpleNamespace(x=data.x, y=data.y, edge_index=data.edge_index,\n"
           "                                  train_mask=masked_train, val_mask=data.val_mask,\n"
           "                                  test_mask=data.test_mask, num_nodes=data.num_nodes,\n"
           "                                  num_classes=data.num_classes)")
    new = ("def rewire_glare(data, cfg, device, seed):\n"
           "    masked_train = apply_label_mask(data.train_mask, cfg.label_mask_ratio, seed=seed)\n"
           "    y_glare, val_glare, test_glare = data.y, data.val_mask, data.test_mask\n"
           "    if cfg.unsupervised:\n"
           "        y_glare = torch.zeros_like(data.y)\n"
           "        masked_train = torch.zeros_like(data.train_mask)\n"
           "        val_glare = torch.zeros_like(data.val_mask)\n"
           "        test_glare = torch.zeros_like(data.test_mask)\n"
           "        assert int(masked_train.sum()) == 0 and int(val_glare.sum()) == 0\n"
           "        assert cfg.glare_select == \"final\"\n"
           "    data_glare = SimpleNamespace(x=data.x, y=y_glare, edge_index=data.edge_index,\n"
           "                                  train_mask=masked_train, val_mask=val_glare,\n"
           "                                  test_mask=test_glare, num_nodes=data.num_nodes,\n"
           "                                  num_classes=data.num_classes)")
    P.append((old, new, 1))

    old = "    ratio_tag = f\"_ratio{cfg.label_mask_ratio:.3f}\"\n"
    new = ("    ratio_tag = f\"_ratio{cfg.label_mask_ratio:.3f}\"\n"
           "    if cfg.unsupervised:\n"
           "        ratio_tag += \"_unsup\"\n")
    P.append((old, new, 1))
    return P


def _others_patches():
    P = []
    # IDGL (two variants, identical final block)
    old = ("        _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, _true)\n"
           "        return best_val, test_acc, test_f1")
    P.append((old, "        _AUC_HOOK(best_Z[te.cpu()], _true)\n" + old, 2))
    # GADC
    old = ("    _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, _true)\n"
           "\n"
           "    return best_val, test_acc, test_f1")
    P.append((old, "    _AUC_HOOK(best_logits[te.cpu()], _true)\n" + old, 1))
    # LPkG (final blended probabilities)
    old = ("    _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, _true)\n"
           "    return val_acc, test_acc, test_f1")
    P.append((old, "    _AUC_HOOK(Z_final[test_mask], _true)\n" + old, 1))
    # DHGR (log-probabilities)
    old = "    _, final_test_f1 = _metrics.accuracy_and_macro_f1(_pred, y[test_mask])"
    P.append((old, "    _AUC_HOOK(final_log_probs[test_mask], y[test_mask])\n" + old, 1))
    # FoSR and ComFy (identical final block)
    old = ("    _pred = logits[test_mask].argmax(dim=-1)\n"
           "    _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, y[test_mask])")
    P.append((old, old + "\n    _AUC_HOOK(logits[test_mask], y[test_mask])", 2))
    # GRAPHITE (optional)
    old = ("    _pred = test_logits[test_mask].argmax(dim=-1)\n"
           "    _, test_f1 = _metrics.accuracy_and_macro_f1(_pred, y[test_mask])")
    P.append((old, old + "\n    _AUC_HOOK(test_logits[test_mask], y[test_mask])", 1))
    return P


def _apply(src, patches, name):
    for i, (old, new, count) in enumerate(patches):
        n = src.count(old)
        if n != count:
            raise RuntimeError(
                f"[patch] {name} patch #{i + 1}: expected {count} match(es), "
                f"found {n}. The source file differs from the version this "
                f"script was written for; nothing was run.")
        src = src.replace(old, new)
    return src


def _load_module(path, modname, patches):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (put this script next to it)")
    src = _apply(path.read_text(encoding="utf-8"), patches, path.name)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    mod = types.ModuleType(modname)
    mod.__file__ = str(path)
    mod.__dict__["_AUC_HOOK"] = _AUC_HOOK
    sys.modules[modname] = mod
    exec(compile(src, str(path), "exec"), mod.__dict__)
    return mod


def check_patches():
    for path, patches in [(GLARE_FILE, _glare_patches()),
                          (OTHERS_FILE, _others_patches())]:
        _apply(path.read_text(encoding="utf-8"), patches, path.name)
        print(f"[ok] {path.name}: all {len(patches)} patch sites found")
    src = OTHERS_FILE.read_text(encoding="utf-8")
    n_eval = src.count("accuracy_and_macro_f1(")
    print(f"[ok] benchmark_others.py has {n_eval} final evaluation calls; "
          f"all are covered by a hook")


# =============================================================================
#  Results store and summaries
# =============================================================================

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys = {tuple(str(r[k]) for k in KEY) for r in self.all()}

    def all(self):
        if not self.path.exists():
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def exists(self, **kw):
        return tuple(str(kw[k]) for k in KEY) in self.keys

    def append(self, rec):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.keys.add(tuple(str(rec[k]) for k in KEY))


def summarize(out_dir):
    import pandas as pd
    out_dir = Path(out_dir)
    recs = Store(out_dir / "results.jsonl").all()
    if not recs:
        print("[summary] no results yet")
        return
    df = pd.DataFrame(recs)
    order = {m: i for i, m in enumerate(GLARE_METHODS + BASELINES + OPTIONAL)}
    g = (df.groupby(["method", "classifier"])
           .agg(acc_mean=("acc", "mean"), acc_std=("acc", "std"),
                f1_mean=("f1", "mean"), f1_std=("f1", "std"),
                auc_mean=("auc", "mean"), auc_std=("auc", "std"),
                pred_pos_frac=("pred_pos_frac", "mean"),
                n_seeds=("seed", "nunique"))
           .reset_index())
    g["_o"] = g["method"].map(order)
    g = g.sort_values(["_o", "classifier"]).drop(columns="_o")
    g.to_csv(out_dir / "summary.csv", index=False)

    m = (df.groupby(["method", "classifier"])[["acc", "f1", "auc"]].mean()
           .groupby("method").agg(["mean", "std"]))
    m.columns = [f"{a}_{b}_over_classifiers" for a, b in m.columns]
    m = m.reset_index()
    m["_o"] = m["method"].map(order)
    m = m.sort_values("_o").drop(columns="_o")
    m.to_csv(out_dir / "summary_methods.csv", index=False)

    pos = df["test_pos_rate"].iloc[0]
    lines = [f"# Tolokers: accuracy, macro-F1 and ROC-AUC", "",
             f"Test positive-class rate: {pos:.3f}. A constant majority-class "
             f"predictor gets accuracy {max(pos, 1 - pos):.3f} and ROC-AUC 0.500.", "",
             "| Method | Classifier | Accuracy | Macro-F1 | ROC-AUC | Pred. positive | Seeds |",
             "|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        sd = lambda v: "" if pd.isna(v) else f" ± {v:.3f}"
        lines.append(f"| {r.method} | {r.classifier} | {r.acc_mean:.3f}{sd(r.acc_std)} | "
                     f"{r.f1_mean:.3f}{sd(r.f1_std)} | {r.auc_mean:.3f}{sd(r.auc_std)} | "
                     f"{r.pred_pos_frac:.3f} | {r.n_seeds} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


# =============================================================================
#  Benchmark
# =============================================================================

def _module_cfg(module, extra_argv):
    saved = sys.argv
    try:
        sys.argv = [saved[0]] + extra_argv
        return module.parse_args()
    finally:
        sys.argv = saved


def run(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    out_dir = Path(args.out_dir)
    store = Store(out_dir / "results.jsonl")
    if not args.resume and store.path.exists():
        store.path.unlink(); store = Store(out_dir / "results.jsonl")

    common_argv = ["--device", str(device), "--data_root", args.data_root]
    if args.smoke_test:
        common_argv.append("--smoke_test")

    want_glare = any(m in args.methods for m in GLARE_METHODS)
    want_base = any(m in args.methods for m in BASELINES + OPTIONAL)

    gb = ob = None
    if want_glare:
        gb = _load_module(GLARE_FILE, "glare_benchmark_auc", _glare_patches())
        if args.smoke_test:   # never let a reduced smoke run poison the real cache
            gb.CKPT_DIR = out_dir / "smoke_checkpoints"
            gb.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        gcfg = _module_cfg(gb, common_argv)
        gcfg.device = str(device)          # GLARE needs a resolved device string
        gcfg.dataset = DATASET
        gcfg.label_mask_ratio = 1.0
        gcfg.unsupervised, gcfg.glare_select = False, "val"
        ucfg = copy.deepcopy(gcfg)
        ucfg.label_mask_ratio = 0.0
        ucfg.unsupervised, ucfg.glare_select = True, "final"
        gdata = gb.load_real_dataset(DATASET, root=args.data_root)
    if want_base:
        ob = _load_module(OTHERS_FILE, "benchmark_others_auc", _others_patches())
        ocfg = _module_cfg(ob, common_argv)
        odata = ob.load_real_dataset(DATASET, root=args.data_root)

    data = gdata if want_glare else odata
    if want_glare and want_base:
        same = all(torch.equal(getattr(gdata, k).cpu(), getattr(odata, k).cpu())
                   for k in ["train_mask", "val_mask", "test_mask", "y"])
        print(f"  [check] both loaders give identical Tolokers splits: {same}")
        if not same:
            print("  [WARNING] splits differ between the two loaders; each method "
                  "is still evaluated with its own original loader")

    te = data.test_mask.cpu()
    test_pos_rate = float(data.y.cpu()[te].float().mean())
    seeds = args.seeds[:1] if args.smoke_test else args.seeds
    print("=" * 72)
    print(f"  Tolokers ROC-AUC benchmark   device={device}   seeds={seeds}")
    print(f"  methods={args.methods}   classifiers={args.classifiers}")
    print(f"  test nodes={int(te.sum())}   positive rate={test_pos_rate:.3f}")
    print("=" * 72)

    def finish(method, clf, seed, val, acc, f1, t0, extra=None):
        last = _AUC_STATE["last"]
        if last is None:
            raise RuntimeError("score hook was not called")
        scores, y_true = last
        prob_pos = torch.softmax(scores, dim=-1)[:, 1].numpy()
        yt = y_true.numpy()
        auc = float(roc_auc_score(yt, prob_pos))
        pred_pos = float((scores.argmax(-1) == 1).float().mean())
        rec = {"dataset": DATASET, "method": method, "classifier": clf,
               "seed": seed, "acc": float(acc), "f1": float(f1), "auc": auc,
               "val_acc": float(val), "pred_pos_frac": pred_pos,
               "test_pos_rate": test_pos_rate, "n_test": int(len(yt)),
               "total_time_s": time.perf_counter() - t0}
        if extra:
            rec.update(extra)
        store.append(rec)
        print(f"    {method:<9} {clf:<6} s{seed}  acc={acc:.4f}  f1={f1:.4f}  "
              f"auc={auc:.4f}  pred_pos={pred_pos:.3f}")

    # ---------------- original graph / GLARE / GLARE-U --------------------
    if want_glare:
        X_raw = gdata.x.cpu()
        N, C = gdata.num_nodes, gdata.num_classes
        cache = {}

        def get_graph(method, seed):
            key = (method, seed)
            if key not in cache:
                cfg = gcfg if method == "glare" else ucfg
                gb.set_global_seed(seed)
                t0 = time.perf_counter()
                ei, Z = gb.rewire_glare_resumable(gdata, cfg, device, seed)
                cache[key] = (ei.cpu(), Z.cpu(), time.perf_counter() - t0)
            return cache[key]

        for method in [m for m in GLARE_METHODS if m in args.methods]:
            for clf_name in args.classifiers:
                for seed in seeds:
                    if store.exists(method=method, classifier=clf_name, seed=seed):
                        continue
                    t0 = time.perf_counter()
                    try:
                        gb.set_global_seed(seed)
                        extra = {}
                        if method == "original":
                            ei_use, X_use = gdata.edge_index.cpu(), X_raw.float()
                        else:
                            ei_use, Z, rt = get_graph(method, seed)
                            X_raw_n = F.normalize(X_raw.to(Z.dtype), dim=-1)
                            X_use = torch.cat([X_raw_n, F.normalize(Z, dim=-1)], dim=1)
                            extra["rewiring_time_s"] = rt
                        gb.set_global_seed(seed)
                        clf = gb.build_classifier(clf_name, X_use.shape[1],
                                                  gcfg.clf_hid, C, num_nodes=N)
                        _AUC_STATE["last"] = None
                        val, acc, f1 = gb.train_and_eval(clf, X_use, ei_use,
                                                         gdata, gcfg, device)
                        finish(method, clf_name, seed, val, acc, f1, t0, extra)
                    except Exception as e:
                        print(f"    [FAILED] {method}/{clf_name}/s{seed}: {e}")
                        if args.verbose:
                            traceback.print_exc()

    # ---------------- baselines -------------------------------------------
    if want_base:
        for method in [m for m in BASELINES + OPTIONAL if m in args.methods]:
            for seed in seeds:
                if store.exists(method=method, classifier="paper", seed=seed):
                    continue
                if method == "dhgr" and getattr(ob, "DHGRModelHandler", None) is None:
                    print("  [DHGR] repo unavailable, skipping (set $DHGR_ROOT)")
                    break
                print(f"\n  [{method.upper()}] seed={seed}")
                t0 = time.perf_counter()
                try:
                    ob.set_global_seed(seed)
                    _AUC_STATE["last"] = None
                    val, acc, f1 = ob._dispatch(method, odata, ocfg, device,
                                                seed, DATASET)
                    finish(method, "paper", seed, val, acc, f1, t0)
                except Exception as e:
                    print(f"    [FAILED] {method}/s{seed}: {e}")
                    if args.verbose:
                        traceback.print_exc()

    summarize(out_dir)
    print(f"\n[Done] results in {out_dir.resolve()}")


def main():
    p = argparse.ArgumentParser(description="Tolokers-only ROC-AUC benchmark")
    p.add_argument("--methods", nargs="+", default=GLARE_METHODS + BASELINES,
                   choices=GLARE_METHODS + BASELINES + OPTIONAL)
    p.add_argument("--classifiers", nargs="+", default=CLASSIFIERS, choices=CLASSIFIERS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--device", default="auto")
    p.add_argument("--data_root", default="./data")
    p.add_argument("--out_dir", default="tolokers_auc_results")
    p.add_argument("--no_resume", dest="resume", action="store_false")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--check_patches", action="store_true",
                   help="only verify that all patch sites are found, then exit")
    p.add_argument("--summarize_only", action="store_true")
    args = p.parse_args()

    if args.check_patches:
        check_patches(); return
    if args.summarize_only:
        summarize(args.out_dir); return
    if args.smoke_test and args.out_dir == "tolokers_auc_results":
        args.out_dir = "tolokers_auc_smoke"
    run(args)


if __name__ == "__main__":
    main()
