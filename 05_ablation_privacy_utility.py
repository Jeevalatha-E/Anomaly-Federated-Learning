"""
05_ablation_privacy_utility.py  --  Section 6: Ablation + Privacy-Utility
========================================================================
  A. INCREMENTAL ABLATION -- start from plain FedAvg and add ONE component at a
     time, so each row isolates the marginal effect of that component:
        (1) FedAvg (plain)          no DP, no SecAgg, no robust agg
        (2) + Differential Privacy  add DP
        (3) + Secure Aggregation    add SecAgg   (lossless -> AUC ~ unchanged)
        (4) + Robust Agg (Median)   swap aggregator to coordinate-median
        (5) + Robust Agg (Krum)     swap aggregator to Krum
     -> figures/ablation.png , ablation_results.json

  B. PRIVACY-UTILITY -- sweep the DP noise multiplier, convert to a REAL
     (epsilon, delta)-DP budget with an RDP accountant (privacy_accounting;
     Opacus if installed, analytic RDP otherwise), and plot detection AUC vs
     epsilon. The grid includes noise levels that reach the epsilon < 10 regime
     usually expected for meaningful DP, so the true trade-off is visible.
     -> figures/privacy_utility.png , privacy_utility.json

Detection quality is the mean per-user AUC (eval_core). Run AFTER 01_preprocess.py.
"""

import os, json, numpy as np, torch
from sklearn.metrics import f1_score
from common import CONFIG, set_seed, ensure_dirs, plot_bars, plot_privacy_utility, \
                   clone_config, load_processed
from privacy_accounting import compute_epsilon, privacy_report
from fl_core import federated_train
from eval_core import evaluate_per_user

ensure_dirs()
device = "cuda" if torch.cuda.is_available() else "cpu"
client_data, enroll_by_user, geval_by_user, feat, users = load_processed()
val_normal = np.vstack(list(client_data.values()))


def run(cfg):
    set_seed()
    model, _ = federated_train(client_data, feat, cfg, val_normal=val_normal,
                               device=device, verbose=False)
    r = evaluate_per_user(model, enroll_by_user, geval_by_user, cfg, device=device)
    f1 = f1_score(r["labels"], r["preds"], zero_division=0)
    return dict(auc=float(r["mean_user_auc"]), f1=float(f1), eer=float(r["eer"]))


# ---------------------------------------------------------- A. INCREMENTAL ABLATION
print("== Incremental ablation ==")
ablation, order = {}, [
    ("FedAvg (plain)",        dict(AGGREGATOR="fedavg", DP_ENABLED=False, SECAGG_ENABLED=False)),
    ("+ Differential Privacy",dict(AGGREGATOR="fedavg", DP_ENABLED=True,  SECAGG_ENABLED=False)),
    ("+ Secure Aggregation",  dict(AGGREGATOR="fedavg", DP_ENABLED=True,  SECAGG_ENABLED=True)),
    ("+ Robust Agg (Median)", dict(AGGREGATOR="median", DP_ENABLED=True,  SECAGG_ENABLED=True)),
    ("+ Robust Agg (Krum)",   dict(AGGREGATOR="krum",   DP_ENABLED=True,  SECAGG_ENABLED=True)),
]
for name, over in order:
    cfg = clone_config()
    for k, v in over.items():
        setattr(cfg, k, v)
    ablation[name] = run(cfg)
    print(f"  {name:24s} -> {ablation[name]}")

with open(os.path.join(CONFIG.ARTIFACT_DIR, "ablation_results.json"), "w") as fp:
    json.dump(ablation, fp, indent=2)
names = [n for n, _ in order]
plot_bars(names, [ablation[n]["auc"] for n in names],
          "Mean per-user AUC", "Incremental ablation",
          os.path.join(CONFIG.FIG_DIR, "ablation.png"))

# -------------------------------------------------- B. PRIVACY-UTILITY SWEEP
print("\n== Privacy-utility trade-off (real epsilon via RDP) ==")
q = float(CONFIG.CLIENT_FRACTION)          # client sampling rate for the accountant
delta = 1e-5
noise_grid = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0]   # reaches the eps<10 regime
pu = []
for nm in noise_grid:
    cfg = clone_config()
    cfg.AGGREGATOR = "fedavg"; cfg.SECAGG_ENABLED = True
    cfg.DP_ENABLED = (nm > 0); cfg.DP_NOISE_MULT = nm
    r = run(cfg)
    eps = compute_epsilon(nm, CONFIG.ROUNDS, sample_rate=q, delta=delta) if nm > 0 \
          else float("inf")
    rep = privacy_report(nm, CONFIG.ROUNDS, sample_rate=q, delta=delta, clip=CONFIG.DP_CLIP)
    pu.append(dict(noise_mult=nm, epsilon=(None if np.isinf(eps) else round(eps, 3)),
                   auc=r["auc"], report=rep))
    print(f"  noise={nm:<4} eps={('inf' if np.isinf(eps) else round(eps,2)):>9} "
          f"AUC={r['auc']:.3f}  ({rep['accountant']})")

with open(os.path.join(CONFIG.ARTIFACT_DIR, "privacy_utility.json"), "w") as fp:
    json.dump(dict(delta=delta, sample_rate=q, rounds=CONFIG.ROUNDS, clip=CONFIG.DP_CLIP,
                   sweep=pu), fp, indent=2)

finite = sorted([(p["epsilon"], p["auc"]) for p in pu if p["epsilon"] is not None])
if finite:
    xs, ys = zip(*finite)
    plot_privacy_utility(list(xs), list(ys),
                         os.path.join(CONFIG.FIG_DIR, "privacy_utility.png"))

print("\nSaved ablation.png, privacy_utility.png, ablation_results.json, privacy_utility.json")
print("privacy_utility.json records z, C, q, T, delta and the accountant used -> "
      "fully reproducible epsilon.")
print("Next: run 06_realtime_zero_trust.py")
