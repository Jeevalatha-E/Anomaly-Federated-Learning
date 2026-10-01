"""
06_realtime_zero_trust.py  --  Section 5: Real-Time Response (Zero-Trust)
========================================================================
Turns per-user anomaly scores into graduated zero-trust decisions. Every session
is untrusted until verified; the k-NN distance to the enrolled user's profile
drives a trust gate with per-user thresholds:

    score <= t_low                 -> ALLOW                (matches the user)
    t_low < score <= t_high        -> STEP_UP_AUTH         (multi-factor prompt)
    score >  t_high                -> ISOLATE_AND_ALERT    (restrict + notify SOC)

Thresholds are calibrated per user from their enrolment spread
(CONFIG.GENUINE_PCT and a stricter high percentile). We also measure per-session
decision LATENCY (embed + kNN distance + gate).

Outputs:
  * figures/response_actions.png
  * figures/latency_hist.png
  * realtime_report.json

Run AFTER 02_federated_train.py.
"""

import os, json, time, numpy as np, torch
from common import CONFIG, ensure_dirs, load_processed
from models import build_model
from eval_core import fit_detector, score, threshold_for

ensure_dirs()
device = "cuda" if torch.cuda.is_available() else "cpu"
_, enroll_by_user, geval_by_user, feat, users = load_processed()

meta_path = os.path.join(CONFIG.ARTIFACT_DIR, "model_meta.json")
meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
model_name = meta.get("model", CONFIG.MODEL)
latent_dim = int(meta.get("latent_dim", CONFIG.LATENT_DIM))
model = build_model(model_name, feat, latent_dim).to(device)
model.load_state_dict(torch.load(os.path.join(CONFIG.ARTIFACT_DIR, "global_model.pt"),
                                 map_location=device))
model.eval()

detector = fit_detector(model, enroll_by_user, CONFIG, device)
eval_users = [u for u in detector["users"] if u in geval_by_user and len(geval_by_user[u])]
rng = np.random.default_rng(CONFIG.SEED)

ACTIONS = ["ALLOW", "STEP_UP_AUTH", "ISOLATE_AND_ALERT"]
gen_counts = [0, 0, 0]; imp_counts = [0, 0, 0]; latencies = []


def decide(s, t_low, t_high):
    if s <= t_low:  return 0
    if s <= t_high: return 1
    return 2


for u in eval_users:
    cal = detector["cal"].get(u)
    t_low  = float(np.percentile(cal, CONFIG.GENUINE_PCT))
    t_high = float(np.percentile(cal, min(99, CONFIG.GENUINE_PCT + 7)))

    pool = [geval_by_user[o] for o in eval_users if o != u and len(geval_by_user[o])]
    pool = np.vstack(pool) if pool else np.empty((0, feat))
    k = min(CONFIG.IMPOSTOR_PER_USER, len(pool))
    imp = pool[rng.choice(len(pool), size=k, replace=False)] if k else np.empty((0, feat))

    for X, counts in [(geval_by_user[u], gen_counts), (imp, imp_counts)]:
        for i in range(len(X)):
            t0 = time.perf_counter()
            sc = float(score(detector, model, X[i:i+1], u, device)[0])
            a = decide(sc, t_low, t_high)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            counts[a] += 1

all_lat = np.array(latencies)
gen_total = sum(gen_counts); imp_total = sum(imp_counts)
report = dict(
    users=len(eval_users),
    genuine=dict(total=int(gen_total), **{ACTIONS[i]: int(gen_counts[i]) for i in range(3)}),
    impostor=dict(total=int(imp_total), **{ACTIONS[i]: int(imp_counts[i]) for i in range(3)}),
    impostor_challenge_rate=float((imp_counts[1] + imp_counts[2]) / max(1, imp_total)),
    genuine_friction_rate=float((gen_counts[1] + gen_counts[2]) / max(1, gen_total)),
    latency_ms=dict(mean=float(all_lat.mean()), p50=float(np.percentile(all_lat, 50)),
                    p95=float(np.percentile(all_lat, 95)), p99=float(np.percentile(all_lat, 99))),
)
with open(os.path.join(CONFIG.ARTIFACT_DIR, "realtime_report.json"), "w") as fp:
    json.dump(report, fp, indent=2)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x = np.arange(3); w = 0.38
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.bar(x - w/2, gen_counts, w, label="Genuine")
ax.bar(x + w/2, imp_counts, w, label="Impostor")
ax.set_xticks(x); ax.set_xticklabels(ACTIONS, rotation=10)
ax.set_ylabel("Sessions"); ax.set_title("Zero-trust response actions")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(CONFIG.FIG_DIR, "response_actions.png"), dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.hist(all_lat, bins=40)
ax.axvline(report["latency_ms"]["p95"], color="red", ls="--",
           label=f"p95={report['latency_ms']['p95']:.2f} ms")
ax.set_xlabel("Per-session decision latency (ms)"); ax.set_ylabel("Count")
ax.set_title("Real-time detection latency"); ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(CONFIG.FIG_DIR, "latency_hist.png"), dpi=140); plt.close(fig)

print(json.dumps(report, indent=2))
print("\nSaved response_actions.png, latency_hist.png, realtime_report.json")
print("All sections complete.")
