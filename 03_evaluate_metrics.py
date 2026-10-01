"""
03_evaluate_metrics.py  --  Section 6: Evaluation & Metrics
===========================================================
Rebuilds the federated model, scores every user's genuine (own) windows against
their profile and other users' windows as impostors (eval_core), and turns the
pooled results into every requested metric and chart:

  * Confusion matrix              -> figures/confusion_matrix.png
  * ROC curve + AUC               -> figures/roc_curve.png
  * Anomaly-score distribution    -> figures/score_distribution.png
  * Accuracy / Precision / Recall / F1 / TPR / FPR / AUC / EER  -> metrics.json
                                     + bar chart figures/metric_bars.png

We report both the POOLED AUC (all users together) and the MEAN PER-USER AUC
(the standard behavioural-biometrics verification metric). Predictions use each
user's own operating threshold (CONFIG.GENUINE_PCT percentile of their genuine
spread), which is the zero-trust, per-user calibration.

Run AFTER 02_federated_train.py.
"""

import os, json, numpy as np, torch
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_auc_score, accuracy_score)
from common import (CONFIG, ensure_dirs, load_processed,
                    plot_confusion, plot_roc, plot_score_hist, plot_bars)
from models import build_model
from eval_core import evaluate_per_user

ensure_dirs()
device = "cuda" if torch.cuda.is_available() else "cpu"

_, enroll_by_user, geval_by_user, feat, users = load_processed()

# rebuild the exact architecture the checkpoint was trained with
meta_path = os.path.join(CONFIG.ARTIFACT_DIR, "model_meta.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
model_name = meta.get("model", CONFIG.MODEL)
latent_dim = int(meta.get("latent_dim", CONFIG.LATENT_DIM))
model = build_model(model_name, feat, latent_dim).to(device)
model.load_state_dict(torch.load(os.path.join(CONFIG.ARTIFACT_DIR, "global_model.pt"),
                                 map_location=device))
model.eval()

res = evaluate_per_user(model, enroll_by_user, geval_by_user, CONFIG, device=device)
y_true, scores, y_pred = res["labels"], res["scores"], res["preds"]

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)     # = TPR
f1   = f1_score(y_true, y_pred, zero_division=0)
auc  = roc_auc_score(y_true, scores)
eer  = res["eer"]

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
tpr = tp / (tp + fn) if (tp + fn) else 0.0
fpr = fp / (fp + tn) if (fp + tn) else 0.0

metrics = dict(accuracy=acc, precision=prec, recall_TPR=rec, f1=f1,
               roc_auc_pooled=auc, mean_user_auc=res["mean_user_auc"],
               eer=eer, fpr=fpr, threshold_eer=res["eer_threshold"],
               n_users=res["n_users"],
               confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
               model=model_name, aggregator=CONFIG.AGGREGATOR,
               dp=CONFIG.DP_ENABLED, secagg=CONFIG.SECAGG_ENABLED)

with open(os.path.join(CONFIG.ARTIFACT_DIR, "metrics.json"), "w") as fp:
    json.dump(metrics, fp, indent=2)

# ---- charts ----
plot_confusion(cm, ["Genuine", "Impostor"],
               f"Confusion Matrix ({model_name}, {CONFIG.AGGREGATOR})",
               os.path.join(CONFIG.FIG_DIR, "confusion_matrix.png"))
plot_roc(y_true, scores, auc, f"ROC Curve  (AUC={auc:.3f})",
         os.path.join(CONFIG.FIG_DIR, "roc_curve.png"))
plot_score_hist(res["genuine_scores"], res["impostor_scores"], None,
                os.path.join(CONFIG.FIG_DIR, "score_distribution.png"))
plot_bars(["Accuracy", "Precision", "Recall", "F1", "AUC", "1-EER"],
          [acc, prec, rec, f1, auc, 1 - eer],
          "Score", "Detection metrics",
          os.path.join(CONFIG.FIG_DIR, "metric_bars.png"))

print(json.dumps(metrics, indent=2))
print("\nSaved confusion_matrix.png, roc_curve.png, score_distribution.png, "
      "metric_bars.png, metrics.json")
print("Section 6 (core) complete. Next: run 04_comparative_analysis.py")
