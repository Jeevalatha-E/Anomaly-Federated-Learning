"""
common.py
=========
Shared utilities for the Metaverse Federated Anomaly-Detection project (Objective 3).

Everything that more than one section needs lives here:
  * global CONFIG + reproducible seeding
  * BB-MAS dataset loading (robust to the real IEEE-DataPort folder layout)
  * keystroke-dynamics feature extraction (hold times + digraph latencies -> window features)
  * genuine/impostor labelling for anomaly detection
  * federated (non-IID) client partitioning
  * metric + plotting helpers (confusion matrix, ROC, EER, score distribution)

This module is imported by every 0X_*.py section script.
Run 01_preprocess.py FIRST; it caches processed arrays to ARTIFACT_DIR.
"""

import os
import re
import glob
import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG  -- tweak here, every script reads these values
# ----------------------------------------------------------------------------
class CONFIG:
    SEED              = 42

    # Where the extracted BB-MAS dataset lives after you unzip it in Colab.
    # 01_preprocess.py extracts your Drive zip into this folder.
    DATA_ROOT         = "/content/BBMAS"

    # Where processed arrays / models / figures are written.
    ARTIFACT_DIR      = "/content/artifacts"
    FIG_DIR           = "/content/artifacts/figures"

    # --- keystroke feature extraction ---
    # Larger windows make the per-key/per-digraph averages stable. With English
    # text most keys are rare, so small windows sample each key only 0-2 times and
    # the features are dominated by noise -> big windows are the main accuracy lever.
    WINDOW            = 200     # keystrokes per session window
    STRIDE            = 80      # sliding-window stride
    MIN_EVENTS        = 200     # skip a (user, device) stream with fewer raw events
    MAX_USERS         = None    # set to an int (e.g. 40) for a fast trial run; None = all

    # --- multimodal feature extraction (Option B: keystroke + motion + touch) ---
    USE_MOTION        = True    # accelerometer + gyroscope (gait / device motion)
    USE_TOUCH         = True    # swipe / touch gestures
    MOTION_WINDOW     = 200     # inertial samples per motion window
    MOTION_STRIDE     = 100
    TOUCH_WINDOW      = 40      # touch/swipe points per window
    TOUCH_STRIDE      = 20

    # --- federated setup ---
    N_CLIENTS         = 5       # metaverse platforms (users split disjointly => non-IID)
    ROUNDS            = 30      # federated communication rounds
    LOCAL_EPOCHS      = 2
    CLIENT_FRACTION   = 1.0     # fraction of clients sampled per round
    BATCH_SIZE        = 64
    LR                = 1e-3

    # --- differential privacy (per-round gradient perturbation) ---
    DP_ENABLED        = True
    DP_CLIP           = 1.0     # L2 clip norm on the client update
    DP_NOISE_MULT     = 0.05    # sigma multiplier (higher = more private, less useful)

    # --- secure aggregation (simulated pairwise masking) ---
    SECAGG_ENABLED    = True

    # --- robust aggregation: one of {"fedavg","fedprox","median","krum"} ---
    AGGREGATOR        = "fedavg"
    FEDPROX_MU        = 0.01
    KRUM_BYZANTINE    = 1       # assumed # of malicious clients for Krum

    # --- anomaly model: one of {"ae","vae","lstm_ae"} ---
    MODEL             = "ae"
    LATENT_DIM        = 32

    # fraction of each user's windows held out for genuine evaluation
    TEST_FRACTION     = 0.30
    # --- per-user (one-class) anomaly detection ---
    MIN_WINDOWS_PER_USER = 12   # a user needs at least this many windows to enrol+test
    IMPOSTOR_PER_USER    = 40   # impostor windows sampled from OTHER users per target user
    GENUINE_PCT          = 90   # per-user operating threshold = this percentile of genuine dists
    KNN_K                = 10    # (legacy) k for distance scorers
    DETECTOR             = "rf"  # verification head: "rf" (RandomForest) or "logreg"
    RF_TREES             = 200   # trees when DETECTOR="rf"


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed=CONFIG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dirs():
    os.makedirs(CONFIG.ARTIFACT_DIR, exist_ok=True)
    os.makedirs(CONFIG.FIG_DIR, exist_ok=True)


def clone_config():
    """
    Return an INDEPENDENT, mutable copy of CONFIG as a SimpleNamespace.
    Needed because CONFIG is a class object: copy.copy(CONFIG) would return the
    same object and mutating it would corrupt global state across experiments.
    """
    import types
    fields = {k: getattr(CONFIG, k) for k in vars(CONFIG)
              if not k.startswith("__") and not callable(getattr(CONFIG, k))}
    return types.SimpleNamespace(**fields)


# ----------------------------------------------------------------------------
# BB-MAS loading helpers
# ----------------------------------------------------------------------------
# The real IEEE-DataPort BB-MAS zip stores one folder per user (named by the
# integer user id), and inside it several CSVs per device. Keystroke files carry
# a key column, a press/release direction column, and a millisecond timestamp.
# Column spelling varies slightly between releases, so we normalise defensively.

_KEY_ALIASES       = ["key", "keyname", "keycode", "char", "keys"]
_DIR_ALIASES       = ["direction", "dir", "press", "action", "event", "keyevent", "state"]
_TIME_ALIASES      = ["time", "timestamp", "epochtime", "epoch", "time(ms)", "ts", "systemtime"]

_DOWN_TOKENS       = {"0", "down", "keydown", "press", "pressed", "d", "true", "1down"}
_UP_TOKENS         = {"1", "up", "keyup", "release", "released", "u", "false", "1up"}


def _norm(col):
    return re.sub(r"[^a-z0-9]", "", str(col).strip().lower())


def _pick(colmap, aliases):
    """Return the original column whose normalised name matches an alias."""
    for a in aliases:
        if a in colmap:
            return colmap[a]
    return None


def _user_id_from_path(path):
    """Extract integer user id from a folder like .../<root>/37/37_Desktop_....csv"""
    parts = os.path.normpath(path).split(os.sep)
    for p in reversed(parts):
        if re.fullmatch(r"\d+", p):
            return int(p)
    m = re.match(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def _device_from_name(path):
    name = os.path.basename(path).lower()
    for dev in ("desktop", "phone", "tablet"):
        if dev in name:
            return dev
    return "unknown"


def find_keystroke_csvs(data_root):
    """
    Recursively locate keystroke CSVs. A file qualifies if its header exposes a
    key-like, a direction-like and a time-like column. Returns list of dicts.
    """
    all_csv = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
    found = []
    for f in all_csv:
        low = os.path.basename(f).lower()
        # cheap filename filter first (keeps it fast on a 1.5 GB tree)
        if not any(k in low for k in ("key", "tap", "type", "keyboard", "keystroke")):
            # still peek at header in case naming is unusual
            pass
        try:
            head = pd.read_csv(f, nrows=0)
        except Exception:
            continue
        colmap = {_norm(c): c for c in head.columns}
        kcol = _pick(colmap, _KEY_ALIASES)
        dcol = _pick(colmap, _DIR_ALIASES)
        tcol = _pick(colmap, _TIME_ALIASES)
        if kcol and dcol and tcol:
            uid = _user_id_from_path(f)
            if uid is None:
                continue
            found.append(dict(path=f, user=uid, device=_device_from_name(f),
                              kcol=kcol, dcol=dcol, tcol=tcol))
    return found


def _dir_to_binary(series):
    """Map a raw direction column to 0=down, 1=up. Robust to text/number encodings."""
    s = series.astype(str).str.strip().str.lower()
    out = np.full(len(s), -1, dtype=int)
    out[s.isin(_DOWN_TOKENS)] = 0
    out[s.isin(_UP_TOKENS)]   = 1
    # numeric fallback: smallest value == down
    if (out == -1).all():
        num = pd.to_numeric(series, errors="coerce")
        med = np.nanmedian(num.values)
        out = (num.values > med).astype(int)
    return out


def _parse_time_to_ms(series):
    """
    Return a float Series of timestamps in MILLISECONDS.

    BB-MAS stores the time column either as a numeric epoch/ms value OR as a
    datetime string like '2017-05-01 13:19:14.762'. We only ever use time
    DIFFERENCES (hold times and inter-key latencies), so any constant offset is
    irrelevant -- we just need a consistent monotonic millisecond scale.
    """
    t_num = pd.to_numeric(series, errors="coerce")
    # if most values parsed as numbers, treat the column as already numeric (ms)
    if t_num.notna().sum() >= max(3, int(0.5 * len(series))):
        return t_num.astype(float)
    # otherwise parse as datetime and convert to milliseconds.
    # NOTE: use timedelta.total_seconds() rather than .astype('int64'); pandas
    # can parse datetimes at second/ms/us resolution and astype('int64') would
    # then return seconds/us instead of the nanoseconds we assumed -> corrupt ms.
    t_dt = pd.to_datetime(series.astype("object"), format="mixed", errors="coerce")
    base = t_dt.min()
    ms = (t_dt - base).dt.total_seconds() * 1000.0   # float ms since first event
    return ms.astype(float)


def _norm_key(k):
    """Canonicalise a raw key token (case-insensitive; unify the space bar)."""
    s = str(k).strip().lower()
    if s in (" ", "space", "spacebar", "key.space", "space_bar"):
        return "space"
    return s


# Fixed vocabularies so every window maps to the SAME fixed-length vector.
# Per-key hold times and per-digraph flight times are what actually encode a
# typist's identity (coarse whole-window averages do not -> AUC ~0.5).
KEY_VOCAB = list("abcdefghijklmnopqrstuvwxyz") + ["space"]          # 27
DIGRAPH_VOCAB = [                                                    # 30 common pairs
    ("t","h"),("h","e"),("i","n"),("e","r"),("a","n"),("r","e"),("o","n"),
    ("a","t"),("e","n"),("n","d"),("t","i"),("e","s"),("o","r"),("t","e"),
    ("o","f"),("e","d"),("i","s"),("i","t"),("a","l"),("a","r"),("s","t"),
    ("t","o"),("n","t"),("n","g"),("s","e"),("h","a"),("a","s"),("o","u"),
    ("i","o"),("l","e"),
]
KEY_DIM    = len(KEY_VOCAB) + len(DIGRAPH_VOCAB) + 4                # 61  keystroke
MOTION_DIM = 28                                                     # 14 accel + 14 gyro
TOUCH_DIM  = 13                                                     # swipe / touch
FEATURE_DIM = KEY_DIM + MOTION_DIM + TOUCH_DIM                      # 102 multimodal (=3*34)

KEY_FEATURE_NAMES = (
    [f"hold_{k}" for k in KEY_VOCAB] +
    [f"flight_{a}{b}" for (a, b) in DIGRAPH_VOCAB] +
    ["hold_mean", "hold_std", "flight_mean", "flight_std"]
)
_MOT_AX = ["x", "y", "z"]
_MOT_STAT = ["mean", "std", "energy", "spec_entropy"]
MOTION_FEATURE_NAMES = (
    [f"accel_{ax}_{st}" for ax in _MOT_AX for st in _MOT_STAT] + ["accel_mag_mean", "accel_mag_std"] +
    [f"gyro_{ax}_{st}"  for ax in _MOT_AX for st in _MOT_STAT] + ["gyro_mag_mean",  "gyro_mag_std"]
)
TOUCH_FEATURE_NAMES = [
    "swipe_path_len", "swipe_displacement", "swipe_vel_mean", "swipe_vel_std",
    "swipe_vel_max", "swipe_dx_mean", "swipe_dy_mean", "swipe_duration",
    "swipe_absdx_mean", "swipe_absdy_mean", "swipe_n_points", "swipe_curvature",
    "swipe_straightness",
]
FEATURE_NAMES = KEY_FEATURE_NAMES + MOTION_FEATURE_NAMES + TOUCH_FEATURE_NAMES


def extract_keystroke_features(df, kcol, dcol, tcol,
                               window=CONFIG.WINDOW, stride=CONFIG.STRIDE):
    """
    Turn one raw keystroke stream into per-window feature vectors that capture
    the typist's identity:

      * per-key mean HOLD time (key-down -> key-up) for each key in KEY_VOCAB
      * per-digraph mean FLIGHT time (down -> next down) for each DIGRAPH_VOCAB pair
      * four global summary stats (mean/std of hold and flight)

    Entries absent from a given window are filled with that window's global mean,
    so the vector stays dense and missing slots are identity-neutral. This is the
    representation that lifts detection AUC well above chance.
    """
    d = df[[kcol, dcol, tcol]].copy()
    d.columns = ["key", "dir", "t"]
    d["t"]   = _parse_time_to_ms(d["t"])
    d["dir"] = _dir_to_binary(d["dir"])
    d = d.dropna(subset=["t"]).sort_values("t").reset_index(drop=True)

    keys = d["key"].map(_norm_key).to_numpy()
    dirs = d["dir"].to_numpy()
    tms  = d["t"].to_numpy(dtype=float)

    # Build the ordered key-DOWN sequence, attaching each down's hold time.
    seq_key, seq_down, seq_hold = [], [], []
    open_downs = {}                       # key -> FIFO list of unmatched down indices
    for i in range(len(keys)):
        k = keys[i]
        if dirs[i] == 0:                  # key-down
            seq_key.append(k); seq_down.append(tms[i]); seq_hold.append(np.nan)
            open_downs.setdefault(k, []).append(len(seq_key) - 1)
        else:                             # key-up -> close earliest matching down
            lst = open_downs.get(k)
            if lst:
                j = lst.pop(0)
                h = tms[i] - seq_down[j]
                if 0 < h < 5000:
                    seq_hold[j] = h

    n = len(seq_key)
    if n < window:
        return np.empty((0, FEATURE_DIM))

    seq_key  = np.asarray(seq_key)
    seq_hold = np.asarray(seq_hold, dtype=float)
    seq_down = np.asarray(seq_down, dtype=float)
    flight   = np.diff(seq_down)          # flight[i] pairs (seq_key[i], seq_key[i+1])

    key_index = {k: i for i, k in enumerate(KEY_VOCAB)}
    dig_index = {p: i for i, p in enumerate(DIGRAPH_VOCAB)}

    feats = []
    for start in range(0, n - window + 1, stride):
        end = start + window
        wk  = seq_key[start:end]
        wh  = seq_hold[start:end]
        wf  = flight[start:end - 1]
        wpair = list(zip(seq_key[start:end - 1], seq_key[start + 1:end]))

        gk = np.nanmean(wh) if np.isfinite(wh).any() else 0.0
        good_f = wf[(wf >= 0) & (wf < 5000)]
        gd = good_f.mean() if len(good_f) else 0.0

        keyvec = np.full(len(KEY_VOCAB), np.nan)
        for kk in key_index:
            vals = wh[(wk == kk)]
            vals = vals[np.isfinite(vals)]
            if len(vals):
                keyvec[key_index[kk]] = vals.mean()

        digvec = np.full(len(DIGRAPH_VOCAB), np.nan)
        acc = {}
        for (a, b), f in zip(wpair, wf):
            if (a, b) in dig_index and 0 <= f < 5000:
                acc.setdefault((a, b), []).append(f)
        for p, vals in acc.items():
            digvec[dig_index[p]] = np.mean(vals)

        keyvec = np.where(np.isnan(keyvec), gk, keyvec)   # neutral fill
        digvec = np.where(np.isnan(digvec), gd, digvec)
        gstats = [gk,
                  np.nanstd(wh) if np.isfinite(wh).any() else 0.0,
                  gd,
                  good_f.std() if len(good_f) else 0.0]
        feats.append(np.concatenate([keyvec, digvec, gstats]))

    return np.asarray(feats, dtype=float) if feats else np.empty((0, FEATURE_DIM))


# ----------------------------------------------------------------------------
# Non-IID federated partitioning
# ----------------------------------------------------------------------------
def partition_users_to_clients(user_ids, n_clients, seed=CONFIG.SEED):
    """Disjoint user groups -> non-IID clients (each 'metaverse platform')."""
    rng = np.random.default_rng(seed)
    users = list(user_ids)
    rng.shuffle(users)
    return {c: users[c::n_clients] for c in range(n_clients)}


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
def compute_eer(y_true, scores):
    """Equal Error Rate + the threshold at which FPR == FNR."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    return float(eer), float(thr[idx])


def threshold_from_genuine(train_scores, percentile=95):
    """Operating threshold = a high percentile of genuine (normal) scores."""
    return float(np.percentile(train_scores, percentile))


def approx_epsilon(noise_mult, rounds, delta=1e-5, sample_rate=1.0):
    """
    (epsilon, delta)-DP for `rounds` compositions of the Gaussian mechanism at
    noise multiplier z = noise_mult and client sampling rate `sample_rate`.
    Delegates to privacy_accounting.compute_epsilon (Opacus RDP accountant if
    installed; documented analytic RDP fallback otherwise) so that the number
    reported in the paper is consistent with fl_core's DP mechanism
    (std = z * C / m) and fully reproducible. Returns +inf when noise_mult <= 0.
    """
    try:
        from privacy_accounting import compute_epsilon
        return compute_epsilon(noise_mult, rounds, sample_rate=sample_rate, delta=delta)
    except Exception:
        if noise_mult <= 0:
            return float("inf")
        z = float(noise_mult); best = float("inf")
        for alpha in np.arange(1.01, 64, 0.25):
            rdp = rounds * alpha / (2.0 * z * z)
            best = min(best, rdp + np.log(1.0 / delta) / (alpha - 1.0))
        return float(best)


# ----------------------------------------------------------------------------
# Load the per-user cache written by 01_preprocess.py
# ----------------------------------------------------------------------------
def load_processed(path=None):
    """
    Returns (client_data, enroll_by_user, geval_by_user, feature_dim, users).
      client_data     : {client_id -> enrolment windows}  (federated training)
      enroll_by_user  : {user_id   -> enrolment windows}  (build the profile)
      geval_by_user   : {user_id   -> held-out windows}   (genuine evaluation)
    """
    import numpy as _np
    if path is None:
        path = os.path.join(CONFIG.ARTIFACT_DIR, "processed.npz")
    data = _np.load(path)
    feat = int(data["feature_dim"][0])
    users = [int(u) for u in data["users"]]
    client_data = {int(k.split("_")[1]): data[k]
                   for k in data.files if k.startswith("client_")}
    enroll_by_user = {int(k.split("_")[1]): data[k]
                      for k in data.files if k.startswith("enroll_")}
    geval_by_user  = {int(k.split("_")[1]): data[k]
                      for k in data.files if k.startswith("geval_")}
    return client_data, enroll_by_user, geval_by_user, feat, users


# ----------------------------------------------------------------------------
# Plotting helpers  (matplotlib only, one chart per figure, saved to FIG_DIR)
# ----------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(cm, labels, title, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_roc(y_true, scores, auc, title, path):
    from sklearn.metrics import roc_curve
    plt = _mpl()
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_curves(history, title, path):
    """history: dict round->value (e.g. loss). Single line chart."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(list(history.keys()), list(history.values()), marker="o", ms=3)
    ax.set_xlabel("Communication round"); ax.set_ylabel(title)
    ax.set_title(title); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_score_hist(genuine, impostor, thr, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(genuine,  bins=40, alpha=0.6, label="Genuine (normal)", density=True)
    ax.hist(impostor, bins=40, alpha=0.6, label="Impostor (anomaly)", density=True)
    if thr is not None:
        ax.axvline(thr, color="red", ls="--", label=f"threshold={thr:.3f}")
    ax.set_xlabel("Anomaly score (distance to user profile)")
    ax.set_ylabel("Density"); ax.set_title("Anomaly-score distribution")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_bars(labels, values, ylabel, title, path, err=None):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(labels)), values, yerr=err, capsize=4)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel(ylabel); ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_privacy_utility(eps, auc, path):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(eps, auc, marker="o")
    ax.set_xlabel("Privacy budget  epsilon  (higher = weaker privacy)")
    ax.set_ylabel("Detection AUC"); ax.set_title("Privacy-Utility trade-off")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


# ============================================================================
# MULTIMODAL EXTENSION (Option B): motion (accel/gyro) + touch/swipe features
# ----------------------------------------------------------------------------
# BB-MAS stores, per user, inertial CSVs (accelerometer / gyroscope, each with
# x/y/z + a time column) and swipe/touch CSVs (x/y coordinates + time, sometimes
# pressure). Column spelling varies, so detection is alias- and fallback-based,
# exactly like the keystroke loader. Modalities are fused at the feature level:
# per-window motion and touch feature vectors are computed per user and aligned
# to that user's keystroke windows by index (documented simplification, since
# the three streams are not guaranteed to share one absolute clock).
# ============================================================================

_ACC_TOKENS   = ("accel", "acc", "linearaccel", "useracceleration")
_GYR_TOKENS   = ("gyro", "gyroscope", "angular", "rotationrate")
_MOTION_FILE_TOKENS = _ACC_TOKENS + _GYR_TOKENS + ("motion", "imu", "inertial")
_TOUCH_FILE_TOKENS  = ("swipe", "touch", "scroll", "gesture", "tap", "stroke")

_X_ALIASES = ["x", "xvalue", "xaxis", "xcoordinate", "xcoord", "accelx", "gyrox",
              "xvalues", "xaxisvalue", "px", "positionx"]
_Y_ALIASES = ["y", "yvalue", "yaxis", "ycoordinate", "ycoord", "accely", "gyroy",
              "yvalues", "yaxisvalue", "py", "positiony"]
_Z_ALIASES = ["z", "zvalue", "zaxis", "zcoordinate", "zcoord", "accelz", "gyroz",
              "zvalues", "zaxisvalue"]
_PRESSURE_ALIASES = ["pressure", "p", "force", "size", "touchmajor"]


def _numeric_non_time_cols(head, tcol):
    """Original column names that look numeric and are not the time column."""
    out = []
    for c in head.columns:
        if c == tcol:
            continue
        try:
            pd.to_numeric(head[c].head(5), errors="raise")
            out.append(c)
        except Exception:
            pass
    return out


def find_motion_csvs(data_root):
    """Locate accelerometer/gyroscope CSVs. Returns dicts with axis+time columns."""
    found = []
    for f in glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True):
        low = os.path.basename(f).lower()
        name_hit = any(t in low for t in _MOTION_FILE_TOKENS)
        try:
            head = pd.read_csv(f, nrows=6)
        except Exception:
            continue
        if any(t in low for t in _TOUCH_FILE_TOKENS):     # swipe/touch handled elsewhere
            continue
        colmap = {_norm(c): c for c in head.columns}
        tcol = _pick(colmap, _TIME_ALIASES)
        xcol = _pick(colmap, _X_ALIASES); ycol = _pick(colmap, _Y_ALIASES); zcol = _pick(colmap, _Z_ALIASES)
        if tcol is None:
            continue
        # Accept only genuine inertial streams: a motion-named file, OR a file that
        # exposes all THREE axes (x,y,z). This prevents 2-axis touch/swipe files
        # from being mistaken for motion.
        if not name_hit and not (xcol and ycol and zcol):
            continue
        if not (xcol and ycol):
            nums = _numeric_non_time_cols(head, tcol)
            if len(nums) < 3:
                continue
            xcol, ycol, zcol = nums[0], nums[1], nums[2]
        uid = _user_id_from_path(f)
        if uid is None:
            continue
        kind = "gyro" if any(t in low for t in _GYR_TOKENS) else "accel"
        found.append(dict(path=f, user=uid, kind=kind,
                          xcol=xcol, ycol=ycol, zcol=zcol, tcol=tcol))
    return found


def find_touch_csvs(data_root):
    """Locate swipe/touch CSVs. Returns dicts with x/y(/pressure)+time columns."""
    found = []
    for f in glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True):
        low = os.path.basename(f).lower()
        name_hit = any(t in low for t in _TOUCH_FILE_TOKENS)
        if not name_hit:
            continue
        try:
            head = pd.read_csv(f, nrows=6)
        except Exception:
            continue
        colmap = {_norm(c): c for c in head.columns}
        tcol = _pick(colmap, _TIME_ALIASES)
        xcol = _pick(colmap, _X_ALIASES); ycol = _pick(colmap, _Y_ALIASES)
        pcol = _pick(colmap, _PRESSURE_ALIASES)
        if tcol is None:
            continue
        if not (xcol and ycol):
            nums = _numeric_non_time_cols(head, tcol)
            if len(nums) < 2:
                continue
            xcol, ycol = nums[0], nums[1]
        uid = _user_id_from_path(f)
        if uid is None:
            continue
        found.append(dict(path=f, user=uid, xcol=xcol, ycol=ycol,
                          pcol=pcol, tcol=tcol))
    return found


def _spectral_entropy(x):
    """Normalised power-spectral entropy of a 1-D window (0..1)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if len(x) < 4 or np.allclose(x, 0):
        return 0.0
    ps = np.abs(np.fft.rfft(x)) ** 2
    tot = ps.sum()
    if tot <= 0:
        return 0.0
    p = ps / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(p) + 1e-12))


def extract_motion_features(df, xcol, ycol, zcol, tcol,
                            window=CONFIG.MOTION_WINDOW, stride=CONFIG.MOTION_STRIDE):
    """
    Per-window inertial features for ONE motion file (accel OR gyro):
      for each of 3 axes: mean, std, energy, spectral-entropy  (12)
      + magnitude mean, magnitude std                          (2)   -> 14 dims.
    Missing z is zero-filled. Returns (n_windows, 14).
    """
    cols = [c for c in (xcol, ycol, zcol) if c is not None]
    d = df[[*cols, tcol]].copy()
    d[tcol] = _parse_time_to_ms(d[tcol])
    d = d.dropna().sort_values(tcol)
    A = d[cols].to_numpy(dtype=float)
    if A.shape[0] < window:
        return np.empty((0, 14))
    if A.shape[1] < 3:                                   # pad to 3 axes
        A = np.hstack([A, np.zeros((A.shape[0], 3 - A.shape[1]))])
    A = A[:, :3]
    mag = np.sqrt((A ** 2).sum(axis=1))
    feats = []
    for start in range(0, A.shape[0] - window + 1, stride):
        w = A[start:start + window]; m = mag[start:start + window]
        row = []
        for ax in range(3):
            col = w[:, ax]
            row += [col.mean(), col.std(), (col ** 2).mean(), _spectral_entropy(col)]
        row += [m.mean(), m.std()]
        feats.append(row)
    return np.asarray(feats, dtype=float) if feats else np.empty((0, 14))


def extract_touch_features(df, xcol, ycol, tcol, pcol=None,
                           window=CONFIG.TOUCH_WINDOW, stride=CONFIG.TOUCH_STRIDE):
    """
    Per-window swipe/touch features (13 dims): path length, displacement,
    velocity mean/std/max, dx/dy mean, duration, |dx|/|dy| mean, n points,
    curvature (mean turn angle), straightness (displacement/path length).
    Returns (n_windows, 13).
    """
    cols = [xcol, ycol, tcol] + ([pcol] if pcol else [])
    d = df[cols].copy()
    d.columns = ["x", "y", "t"] + (["p"] if pcol else [])
    d["t"] = _parse_time_to_ms(d["t"])
    d = d.dropna(subset=["x", "y", "t"]).sort_values("t").reset_index(drop=True)
    x = d["x"].to_numpy(float); y = d["y"].to_numpy(float); t = d["t"].to_numpy(float)
    if len(x) < window:
        return np.empty((0, 13))
    feats = []
    for start in range(0, len(x) - window + 1, stride):
        e = start + window
        wx, wy, wt = x[start:e], y[start:e], t[start:e]
        dx = np.diff(wx); dy = np.diff(wy); dt = np.diff(wt)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        vel = dist / (dt + 1e-6)
        path_len = dist.sum()
        disp = np.hypot(wx[-1] - wx[0], wy[-1] - wy[0])
        ang = np.arctan2(dy, dx)
        curvature = np.abs(np.diff(ang)).mean() if len(ang) > 1 else 0.0
        feats.append([
            path_len, disp, vel.mean(), vel.std(), vel.max() if len(vel) else 0.0,
            dx.mean(), dy.mean(), wt[-1] - wt[0],
            np.abs(dx).mean(), np.abs(dy).mean(), float(window), curvature,
            disp / (path_len + 1e-6),
        ])
    return np.asarray(feats, dtype=float) if feats else np.empty((0, 13))


def assemble_motion(accel_W, gyro_W, accel_mean, gyro_mean):
    """Pair a user's accel(14) and gyro(14) windows by index -> (n, 28). Mean-fill a
    missing sub-modality. Returns None if the user has neither."""
    na = 0 if accel_W is None else len(accel_W)
    ng = 0 if gyro_W is None else len(gyro_W)
    if na == 0 and ng == 0:
        return None
    n = max(na, ng)
    A = accel_W[np.arange(n) % na] if na else np.tile(accel_mean, (n, 1))
    G = gyro_W[np.arange(n) % ng] if ng else np.tile(gyro_mean, (n, 1))
    return np.concatenate([A, G], axis=1)


def fuse_user(key_W, motion_W, touch_W, motion_mean, touch_mean, cfg=CONFIG):
    """
    Feature-level fusion anchored on keystroke windows. Motion(28)/touch(13)
    windows are aligned to the n keystroke windows by index (tiled if fewer),
    or mean-filled if the user lacks that modality. Returns (n, FEATURE_DIM).
    """
    n = len(key_W)
    blocks = [key_W]
    if cfg.USE_MOTION:
        if motion_W is not None and len(motion_W):
            blocks.append(motion_W[np.arange(n) % len(motion_W)])
        else:
            blocks.append(np.tile(motion_mean, (n, 1)))
    if cfg.USE_TOUCH:
        if touch_W is not None and len(touch_W):
            blocks.append(touch_W[np.arange(n) % len(touch_W)])
        else:
            blocks.append(np.tile(touch_mean, (n, 1)))
    return np.concatenate(blocks, axis=1)
