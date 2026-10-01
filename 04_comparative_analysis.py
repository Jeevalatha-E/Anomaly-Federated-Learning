"""
04_comparative_analysis.py  --  Section 6: Comparative Analysis
==============================================================
Retrains the federated pipeline under different choices and compares detection
quality (mean per-user AUC, F1, EER):

  A. Aggregation strategy:  FedAvg vs FedProx vs Median vs Krum
  B. Anomaly model:         AE vs VAE vs LSTM-AE

Charts:
  * figures/compare_aggregators.png
  * figures/compare_models.png
  * comparative_results.json

Run AFTER 01_preprocess.py (does not need 02/03).
"""

import os, json, numpy as np, torch
from sklearn.metrics import f1_score
from common import CONFIG, set_seed, ensure_dirs, plot_bars, clone_config, load_processed
from fl_core import federated_train
from eval_core import evaluate_per_user

ensure_dirs()
device = "cuda" if torch.cuda.is_available() else "cpu"
client_data, enroll_by_user, geval_by_user, feat, users = load_processed()


def run(cfg):
    set_seed()
    model, _ = federated_train(client_data, feat, cfg,
                               val_normal=np.vstack(list(client_data.values())),
                               device=device, verbose=False)
    r = evaluate_per_user(model, enroll_by_user, geval_by_user, cfg, device=device)
    f1 = f1_score(r["labels"], r["preds"], zero_division=0)
    return dict(auc=float(r["mean_user_auc"]), f1=float(f1), eer=float(r["eer"]))


results = {"aggregators": {}, "models": {}}

print("== Comparative: aggregation strategies ==")
for agg in ["fedavg", "fedprox", "median", "krum"]:
    cfg = clone_config(); cfg.AGGREGATOR = agg
    results["aggregators"][agg] = run(cfg)
    print(f"  {agg:8s} -> {results['aggregators'][agg]}")

print("== Comparative: anomaly models ==")
for mdl in ["ae", "vae", "lstm_ae"]:
    cfg = clone_config(); cfg.MODEL = mdl; cfg.AGGREGATOR = "fedavg"
    results["models"][mdl] = run(cfg)
    print(f"  {mdl:8s} -> {results['models'][mdl]}")

with open(os.path.join(CONFIG.ARTIFACT_DIR, "comparative_results.json"), "w") as fp:
    json.dump(results, fp, indent=2)

aggs = list(results["aggregators"])
plot_bars(aggs, [results["aggregators"][a]["auc"] for a in aggs],
          "Mean per-user AUC", "Aggregation strategy comparison",
          os.path.join(CONFIG.FIG_DIR, "compare_aggregators.png"))
mdls = list(results["models"])
plot_bars(mdls, [results["models"][m]["auc"] for m in mdls],
          "Mean per-user AUC", "Anomaly model comparison",
          os.path.join(CONFIG.FIG_DIR, "compare_models.png"))

print("\nSaved compare_aggregators.png, compare_models.png, comparative_results.json")
print("Next: run 05_ablation_privacy_utility.py")
