"""
eval_core.py  --  per-user verification scoring shared by all evaluators
========================================================================
Behavioural biometrics is a VERIFICATION problem: "is this session the enrolled
user, or an impostor?".  Key findings that drive the design (measured on the real
BB-MAS data via diagnose.py):

  * The 61-d features carry strong identity: a classifier reaches ~86% user-ID on
    117 users (chance 0.85%).
  * A one-class DISTANCE scorer (k-NN / centroid) only extracts a fraction of that
    (AUC ~0.68) because it weighs every feature equally and ignores what actually
    separates users. A DISCRIMINATIVE classifier score recovers most of it
    (AUC ~0.9x in matched simulations).

So detection works like this:
  1. The federated model provides a representation R(x) = [ latent_embedding(x) ||
     features(x) ]  (keeps the federated model in the loop; the aggregator/DP
     choices in 04/05 change this representation and hence the scores).
  2. A discriminative classifier is trained on the enrolled users' representations.
  3. The anomaly score for a session claiming identity u is  s(x,u) = 1 - P(u | x).
     Genuine sessions score low; impostors (other users) score high.

Every downstream script (03/04/05/06) uses these helpers, so the detection logic
is identical everywhere.
"""

import numpy as np
from models import embed
from common import CONFIG, compute_eer


def represent(model, X, device="cpu"):
    """Federated representation: [latent embedding || normalised features]."""
    if len(X) == 0:
        return np.empty((0, 0))
    return np.concatenate([embed(model, X, device), X], axis=1)


def _make_clf(cfg):
    kind = getattr(cfg, "DETECTOR", "rf").lower()
    if kind == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=300, n_jobs=-1)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=getattr(cfg, "RF_TREES", 200),
                                  n_jobs=-1, random_state=cfg.SEED)


def fit_detector(model, enroll_by_user, cfg=CONFIG, device="cpu", seed=None):
    """
    Train a discriminative verification head on the enrolled users' representations.
    A slice of each user's enrolment windows is held out to CALIBRATE per-user
    operating thresholds without leakage. Returns a detector dict.
    """
    seed = cfg.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    users = [u for u, E in enroll_by_user.items() if len(E) >= 4]

    Xtr, ytr, cal_windows = [], [], {}
    for u in users:
        E = enroll_by_user[u]
        idx = rng.permutation(len(E))
        cut = max(2, int(0.85 * len(E)))
        Xtr.append(represent(model, E[idx[:cut]], device)); ytr += [u] * cut
        cal_windows[u] = E[idx[cut:]] if len(E) - cut > 0 else E[idx[:1]]

    Xtr = np.vstack(Xtr); ytr = np.array(ytr)
    clf = _make_clf(cfg).fit(Xtr, ytr)
    cidx = {c: i for i, c in enumerate(clf.classes_)}

    # per-user calibration scores  s = 1 - P(u)
    cal = {}
    for u in users:
        if u in cidx and len(cal_windows[u]):
            p = clf.predict_proba(represent(model, cal_windows[u], device))[:, cidx[u]]
            cal[u] = 1.0 - p
    return dict(clf=clf, cidx=cidx, cal=cal, users=[u for u in users if u in cidx])


def score(detector, model, X, user, device="cpu"):
    """Anomaly score s(x,user) = 1 - P(user | x) for every row of X."""
    if len(X) == 0:
        return np.empty(0)
    if user not in detector["cidx"]:
        return np.full(len(X), np.nan)
    p = detector["clf"].predict_proba(represent(model, X, device))[:, detector["cidx"][user]]
    return 1.0 - p


def threshold_for(detector, user, cfg=CONFIG):
    c = detector["cal"].get(user)
    return float(np.percentile(c, cfg.GENUINE_PCT)) if c is not None and len(c) else np.inf


def evaluate_per_user(model, enroll_by_user, geval_by_user, cfg=CONFIG,
                      device="cpu", seed=None):
    """
    Score every user's genuine (own held-out) windows against the verification head,
    plus an equal-opportunity sample of OTHER users' windows as impostors. Returns
    pooled arrays (confusion/ROC over the whole system) and the mean per-user AUC.
    """
    seed = cfg.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    det = fit_detector(model, enroll_by_user, cfg, device, seed)
    users = [u for u in det["users"] if u in geval_by_user and len(geval_by_user[u])]

    all_scores, all_labels, all_preds, per_user_auc = [], [], [], []
    from sklearn.metrics import roc_auc_score

    for u in users:
        thr = threshold_for(det, u, cfg)
        g_d = score(det, model, geval_by_user[u], u, device)

        pool = [geval_by_user[o] for o in users
                if o != u and len(geval_by_user[o])]
        if not pool:
            continue
        pool = np.vstack(pool)
        k = min(cfg.IMPOSTOR_PER_USER, len(pool))
        i_d = score(det, model, pool[rng.choice(len(pool), size=k, replace=False)], u, device)

        y = np.r_[np.zeros(len(g_d)), np.ones(len(i_d))]
        s = np.r_[g_d, i_d]
        if len(np.unique(y)) == 2 and len(s) > 1:
            per_user_auc.append(roc_auc_score(y, s))
        all_scores.append(s); all_labels.append(y)
        all_preds.append((s > thr).astype(int))

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(int)
    preds  = np.concatenate(all_preds).astype(int)
    eer, eer_thr = compute_eer(labels, scores)

    return dict(
        scores=scores, labels=labels, preds=preds,
        genuine_scores=scores[labels == 0], impostor_scores=scores[labels == 1],
        mean_user_auc=float(np.mean(per_user_auc)) if per_user_auc else float("nan"),
        per_user_auc=per_user_auc,
        eer=float(eer), eer_threshold=float(eer_thr),
        n_users=len(users),
    )
