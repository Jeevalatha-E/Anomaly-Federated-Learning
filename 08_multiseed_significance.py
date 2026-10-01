"""
08_multiseed_significance.py  --  repeated runs + statistical significance
==========================================================================
Addresses the reviewer remark that results came from a single seed. Each
aggregator is trained and evaluated across SEEDS independent runs (the data
split is fixed by 01_preprocess.py; only the training randomness -- weight
init, client sampling, DP noise, secure-aggregation masks, and the RF head --
varies with the seed). We report mean +/- std for AUC / F1 / EER and a paired
t-test between FedProx and FedAvg on the per-seed AUC.

Outputs:
  * multiseed_results.json    (per-seed values + mean/std + paired test)
  * figures/multiseed_auc.png (mean per-user AUC with std error bars)

Run AFTER 01_preprocess.py.
"""

import os, json, numpy as np, torch
from sklearn.metrics import f1_score
from common import CONFIG, set_seed, ensure_dirs, clone_config, load_processed, plot_bars
from fl_core import federated_train
from eval_core import evaluate_per_user

SEEDS       = [0, 1, 2, 3, 4]
AGGREGATORS = ["fedavg", "fedprox", "median", "krum"]

ensure_dirs()
device = "cuda" if torch.cuda.is_available() else "cpu"
client_data, enroll_by_user, geval_by_user, feat, users = load_processed()
val_normal = np.vstack(list(client_data.values()))
print(f"[multiseed] seeds={SEEDS} | aggregators={AGGREGATORS}")


def run_once(agg, seed):
    set_seed(seed)
    cfg = clone_config(); cfg.AGGREGATOR = agg; cfg.SEED = seed
    model, _ = federated_train(client_data, feat, cfg, val_normal=val_normal,
                               device=device, verbose=False)
    r = evaluate_per_user(model, enroll_by_user, geval_by_user, cfg,
                          device=device, seed=seed)
    f1 = f1_score(r["labels"], r["preds"], zero_division=0)
    return dict(auc=float(r["mean_user_auc"]), f1=float(f1), eer=float(r["eer"]))


results = {}
for agg in AGGREGATORS:
    runs = [run_once(agg, s) for s in SEEDS]
    arr = {k: np.array([r[k] for r in runs]) for k in ("auc", "f1", "eer")}
    results[agg] = dict(
        per_seed=runs,
        auc_mean=float(arr["auc"].mean()), auc_std=float(arr["auc"].std(ddof=1)),
        f1_mean=float(arr["f1"].mean()),   f1_std=float(arr["f1"].std(ddof=1)),
        eer_mean=float(arr["eer"].mean()), eer_std=float(arr["eer"].std(ddof=1)))
    print(f"  {agg:8s} | AUC {results[agg]['auc_mean']:.4f}+/-{results[agg]['auc_std']:.4f} "
          f"| F1 {results[agg]['f1_mean']:.4f}+/-{results[agg]['f1_std']:.4f} "
          f"| EER {results[agg]['eer_mean']:.4f}+/-{results[agg]['eer_std']:.4f}")

# paired significance: FedProx vs FedAvg on per-seed AUC
paired = {}
try:
    from scipy import stats
    a = np.array([r["auc"] for r in results["fedavg"]["per_seed"]])
    b = np.array([r["auc"] for r in results["fedprox"]["per_seed"]])
    t, p = stats.ttest_rel(b, a)
    paired = dict(comparison="fedprox_vs_fedavg_auc",
                  mean_diff=float(b.mean() - a.mean()),
                  t_stat=float(t), p_value=float(p), n=len(SEEDS))
    print(f"\n[paired t-test] FedProx-FedAvg AUC: mean diff={b.mean()-a.mean():+.4f}, "
          f"t={t:.3f}, p={p:.4f}")
except Exception as e:
    paired = dict(error=str(e))
    print("[paired t-test] scipy unavailable:", e)

with open(os.path.join(CONFIG.ARTIFACT_DIR, "multiseed_results.json"), "w") as fp:
    json.dump(dict(seeds=SEEDS, results=results, paired_test=paired), fp, indent=2)

plot_bars(AGGREGATORS,
          [results[a]["auc_mean"] for a in AGGREGATORS],
          "Mean per-user AUC",
          f"Aggregation strategies (mean +/- std over {len(SEEDS)} seeds)",
          os.path.join(CONFIG.FIG_DIR, "multiseed_auc.png"),
          err=[results[a]["auc_std"] for a in AGGREGATORS])

print("\nSaved multiseed_results.json, figures/multiseed_auc.png")
print("Report AUC/F1/EER as mean +/- std; cite the paired-test p-value in the text.")
