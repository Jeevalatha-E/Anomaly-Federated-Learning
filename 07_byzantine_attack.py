"""
07_byzantine_attack.py  --  Section 6.3: Byzantine Attack Robustness (Table X)
==============================================================================
Fills the missing Byzantine-attack results. One of the N clients is designated
malicious and performs a model-update SIGN-FLIP attack (Delta -> -lambda*Delta,
the paper's primary attack). For each aggregator we train CLEAN and UNDER ATTACK
and report both detection metrics and the direct model-space effect:

  * AUC / F1 / EER      (detection, via eval_core)   clean vs attacked
  * final val recon.    (model-space robustness)     clean vs attacked

FedAvg/FedProx are non-robust references; Coordinate-Median and Krum are the
Byzantine-resilient aggregators. The model-space (val-recon) column is the most
direct robustness signal, because the hybrid representation used for detection
can mask encoder corruption.

Outputs:
  * byzantine_results.json
  * figures/byzantine_attack.png     (attacked AUC + AUC drop per aggregator)
  * figures/byzantine_valrecon.png   (clean vs attacked val reconstruction error)

Config: CONFIG.KRUM_BYZANTINE malicious clients assumed by Krum; the attack uses
ATTACK_SCALE (lambda). Run AFTER 01_preprocess.py.
"""

import os, json, numpy as np, torch
from sklearn.metrics import f1_score
from common import CONFIG, set_seed, ensure_dirs, clone_config, load_processed, _mpl
from fl_core import federated_train
from eval_core import evaluate_per_user

ATTACK        = "sign_flip"   # paper's primary Byzantine attack
ATTACK_SCALE  = 5.0           # lambda; >=1 makes the malicious update adversarial
N_MALICIOUS   = 1             # one malicious client among the N

ensure_dirs()
device = "cuda" if torch.cuda.is_available() else "cpu"
client_data, enroll_by_user, geval_by_user, feat, users = load_processed()
val_normal = np.vstack(list(client_data.values()))
malicious = sorted(client_data.keys())[:N_MALICIOUS]
print(f"[byzantine] clients={sorted(client_data)} | malicious={malicious} "
      f"| attack={ATTACK} lambda={ATTACK_SCALE}")


def run(agg, attacked):
    set_seed()
    cfg = clone_config(); cfg.AGGREGATOR = agg
    model, hist = federated_train(
        client_data, feat, cfg, val_normal=val_normal, device=device, verbose=False,
        malicious_clients=(malicious if attacked else None),
        attack=(ATTACK if attacked else None), attack_scale=ATTACK_SCALE)
    r = evaluate_per_user(model, enroll_by_user, geval_by_user, cfg, device=device)
    f1 = f1_score(r["labels"], r["preds"], zero_division=0)
    return dict(auc=float(r["mean_user_auc"]), f1=float(f1), eer=float(r["eer"]),
                val_recon=float(hist[max(hist)]))


rows = {}
for agg in ["fedavg", "fedprox", "median", "krum"]:
    clean = run(agg, attacked=False)
    att   = run(agg, attacked=True)
    rows[agg] = dict(
        clean=clean, attacked=att,
        auc_drop=round(clean["auc"] - att["auc"], 4),
        eer_increase=round(att["eer"] - clean["eer"], 4),
        valrecon_increase=round(att["val_recon"] - clean["val_recon"], 4),
        robust=(agg in ("median", "krum")))
    print(f"  {agg:8s} | AUC {clean['auc']:.4f}->{att['auc']:.4f} "
          f"(drop {rows[agg]['auc_drop']:+.4f}) | "
          f"EER {clean['eer']:.4f}->{att['eer']:.4f} | "
          f"val_recon {clean['val_recon']:.4f}->{att['val_recon']:.4f}")

with open(os.path.join(CONFIG.ARTIFACT_DIR, "byzantine_results.json"), "w") as fp:
    json.dump(dict(attack=ATTACK, attack_scale=ATTACK_SCALE, malicious=malicious,
                   results=rows), fp, indent=2)

# ---- chart 1: attacked AUC + drop ----
aggs = list(rows)
plt = _mpl()
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(aggs)); w = 0.38
ax.bar(x - w/2, [rows[a]["clean"]["auc"] for a in aggs], w, label="Clean AUC")
ax.bar(x + w/2, [rows[a]["attacked"]["auc"] for a in aggs], w, label="Attacked AUC")
ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in aggs])
ax.set_ylabel("Mean per-user AUC"); ax.set_ylim(0, 1.02)
ax.set_title(f"Byzantine robustness: clean vs sign-flip attack (lambda={ATTACK_SCALE})")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(CONFIG.FIG_DIR, "byzantine_attack.png"), dpi=140)
plt.close(fig)

# ---- chart 2: model-space effect (val reconstruction error) ----
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x - w/2, [rows[a]["clean"]["val_recon"] for a in aggs], w, label="Clean")
ax.bar(x + w/2, [rows[a]["attacked"]["val_recon"] for a in aggs], w, label="Under attack")
ax.set_xticks(x); ax.set_xticklabels([a.upper() for a in aggs])
ax.set_ylabel("Final validation reconstruction error")
ax.set_title("Model-space effect of the Byzantine client (lower = more robust)")
ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(CONFIG.FIG_DIR, "byzantine_valrecon.png"), dpi=140)
plt.close(fig)

print("\nSaved byzantine_results.json, figures/byzantine_attack.png, "
      "figures/byzantine_valrecon.png")
print("Use byzantine_results.json to fill Table X (clean vs attacked AUC/F1/EER + degradation).")
