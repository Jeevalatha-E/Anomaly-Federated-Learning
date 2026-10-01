"""
02_federated_train.py  --  Sections 2,3,4: FL setup, anomaly model, optimisation
================================================================================
Trains the anomaly model FEDERATEDLY across the non-IID clients (metaverse
platforms) using the configured aggregator (FedAvg/FedProx/Median/Krum),
Differential Privacy and Secure Aggregation. The trained model is used as a
shared ENCODER; detection itself is per-user (one-class) and lives in eval_core.

Outputs (to CONFIG.ARTIFACT_DIR):
  * global_model.pt   trained global model (shared encoder)
  * model_meta.json   architecture/feature dims, so later stages rebuild it
  * scores.npz        per-user genuine/impostor distance scores + loss history
  * figures/loss_curve.png

Run AFTER 01_preprocess.py.
"""

import os, json, numpy as np, torch
from common import CONFIG, set_seed, ensure_dirs, plot_curves, load_processed
from fl_core import federated_train
from eval_core import evaluate_per_user

set_seed(); ensure_dirs()
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] {device} | model={CONFIG.MODEL} | aggregator={CONFIG.AGGREGATOR} "
      f"| DP={CONFIG.DP_ENABLED} | SecAgg={CONFIG.SECAGG_ENABLED}")

client_data, enroll_by_user, geval_by_user, feature_dim, users = load_processed()
print(f"[data] clients={sorted(client_data)} | enrolled users={len(users)}")

# ---- federated training on the clients' enrolment windows ----
model, history = federated_train(client_data, feature_dim, CONFIG,
                                 val_normal=np.vstack(list(client_data.values())),
                                 device=device, verbose=True)

# ---- per-user (one-class) evaluation via the shared encoder ----
res = evaluate_per_user(model, enroll_by_user, geval_by_user, CONFIG, device=device)
print(f"[eval] users scored={res['n_users']} | mean per-user AUC={res['mean_user_auc']:.4f} "
      f"| EER={res['eer']:.4f}")

# ---- persist ----
torch.save(model.state_dict(), os.path.join(CONFIG.ARTIFACT_DIR, "global_model.pt"))
with open(os.path.join(CONFIG.ARTIFACT_DIR, "model_meta.json"), "w") as fp:
    json.dump({"model": CONFIG.MODEL, "feature_dim": feature_dim,
               "latent_dim": CONFIG.LATENT_DIM}, fp, indent=2)
np.savez(os.path.join(CONFIG.ARTIFACT_DIR, "scores.npz"),
         genuine_scores=res["genuine_scores"], impostor_scores=res["impostor_scores"],
         rounds=np.array(list(history.keys())),
         val_recon=np.array(list(history.values())))
plot_curves(history, "Validation reconstruction error",
            os.path.join(CONFIG.FIG_DIR, "loss_curve.png"))

print("Saved global_model.pt, model_meta.json, scores.npz, figures/loss_curve.png")
print("Section 2-4 complete. Next: run 03_evaluate_metrics.py")
