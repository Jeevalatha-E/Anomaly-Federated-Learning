"""
01_preprocess.py  --  Section 1: MULTIMODAL Data Collection & Preprocessing
===========================================================================
Option B: genuine multimodal behavioral biometrics.

Extracts and FUSES three modalities from SU-AIS BB-MAS:
  * keystroke dynamics   -> 61 features (per-key hold + per-digraph flight + stats)
  * inertial motion      -> 28 features (accelerometer 14 + gyroscope 14:
                            per-axis mean/std/energy/spectral-entropy + magnitude)
  * touch / swipe         -> 13 features (path length, velocity, curvature, ...)
Fused window = [keystroke(61) | motion(28) | touch(13)] = 102 dims.

Fusion is feature-level and anchored on keystroke windows: each user's motion and
touch per-window features are computed independently and aligned to that user's
keystroke windows by index (tiled where fewer); a modality a user lacks is filled
with the global mean of that modality (identity-neutral after normalisation).
Set CONFIG.USE_MOTION / USE_TOUCH = False to drop a modality.

>>> EDIT `ZIP_PATH` to point at your BB-MAS zip in Google Drive. <<<  Run FIRST.
"""

import os, zipfile, numpy as np, pandas as pd
from collections import defaultdict
from common import (CONFIG, set_seed, ensure_dirs,
                    find_keystroke_csvs, extract_keystroke_features,
                    find_motion_csvs, find_touch_csvs,
                    extract_motion_features, extract_touch_features,
                    assemble_motion, fuse_user,
                    partition_users_to_clients,
                    FEATURE_DIM, KEY_DIM, MOTION_DIM, TOUCH_DIM)

ZIP_PATH = "/content/drive/MyDrive/BB-MAS_Dataset.zip"   # <-- change to your file

set_seed(); ensure_dirs()


def mount_and_extract():
    already = os.path.isdir(CONFIG.DATA_ROOT) and \
              any(f.endswith(".csv") for _, _, fs in os.walk(CONFIG.DATA_ROOT) for f in fs)
    if already:
        print(f"[extract] {CONFIG.DATA_ROOT} already populated, skipping."); return
    try:
        from google.colab import drive
        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")
    except Exception as e:
        print("[extract] not in Colab or Drive already mounted:", e)
    if not os.path.isfile(ZIP_PATH):
        raise FileNotFoundError(f"Zip not found at {ZIP_PATH}. Set ZIP_PATH.")
    os.makedirs(CONFIG.DATA_ROOT, exist_ok=True)
    print(f"[extract] unzipping {ZIP_PATH} -> {CONFIG.DATA_ROOT}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(CONFIG.DATA_ROOT)
    print("[extract] done.")

mount_and_extract()

# ----------------------------------------------------------------------------
# 1. Locate every modality's CSVs
# ----------------------------------------------------------------------------
key_files   = find_keystroke_csvs(CONFIG.DATA_ROOT)
mot_files   = find_motion_csvs(CONFIG.DATA_ROOT)   if CONFIG.USE_MOTION else []
tou_files   = find_touch_csvs(CONFIG.DATA_ROOT)    if CONFIG.USE_TOUCH  else []
print(f"[scan] keystroke CSVs: {len(key_files)} | motion CSVs: {len(mot_files)} "
      f"({sum(f['kind']=='accel' for f in mot_files)} accel / "
      f"{sum(f['kind']=='gyro' for f in mot_files)} gyro) | touch CSVs: {len(tou_files)}")
if not key_files:
    raise RuntimeError("No keystroke CSVs detected; check CONFIG.DATA_ROOT layout.")
if CONFIG.USE_MOTION and not mot_files:
    print("[warn] USE_MOTION=True but no motion CSVs found -> motion block will be "
          "mean-filled. Check filenames/columns or set USE_MOTION=False.")
if CONFIG.USE_TOUCH and not tou_files:
    print("[warn] USE_TOUCH=True but no touch CSVs found -> touch block will be "
          "mean-filled. Check filenames/columns or set USE_TOUCH=False.")

# ----------------------------------------------------------------------------
# 2. PASS 1 -- per-user per-modality window arrays
# ----------------------------------------------------------------------------
key_by_user   = defaultdict(list)
accel_by_user = defaultdict(list)
gyro_by_user  = defaultdict(list)
touch_by_user = defaultdict(list)

for f in key_files:
    try: df = pd.read_csv(f["path"])
    except Exception: continue
    W = extract_keystroke_features(df, f["kcol"], f["dcol"], f["tcol"])
    if len(W): key_by_user[f["user"]].append(W)

for f in mot_files:
    try: df = pd.read_csv(f["path"])
    except Exception: continue
    W = extract_motion_features(df, f["xcol"], f["ycol"], f["zcol"], f["tcol"])
    if len(W):
        (accel_by_user if f["kind"] == "accel" else gyro_by_user)[f["user"]].append(W)

for f in tou_files:
    try: df = pd.read_csv(f["path"])
    except Exception: continue
    W = extract_touch_features(df, f["xcol"], f["ycol"], f["tcol"], f["pcol"])
    if len(W): touch_by_user[f["user"]].append(W)

key_by_user   = {u: np.vstack(v) for u, v in key_by_user.items() if v}
accel_by_user = {u: np.vstack(v) for u, v in accel_by_user.items() if v}
gyro_by_user  = {u: np.vstack(v) for u, v in gyro_by_user.items() if v}
touch_by_user = {u: np.vstack(v) for u, v in touch_by_user.items() if v}

# global modality means for identity-neutral fill of missing modalities
def _mean(d, dim):
    return np.vstack(list(d.values())).mean(0) if d else np.zeros(dim)
accel_mean = _mean(accel_by_user, 14)
gyro_mean  = _mean(gyro_by_user, 14)
motion_mean = np.concatenate([accel_mean, gyro_mean])   # 28
touch_mean  = _mean(touch_by_user, TOUCH_DIM)           # 13
cov_m = sum(u in accel_by_user or u in gyro_by_user for u in key_by_user)
cov_t = sum(u in touch_by_user for u in key_by_user)
print(f"[coverage] users with keystroke: {len(key_by_user)} | "
      f"with motion: {cov_m} | with touch: {cov_t}")

# ----------------------------------------------------------------------------
# 3. PASS 2 -- fuse to [key | motion | touch] = 102 dims per window, per user
# ----------------------------------------------------------------------------
user_X = {}
for u, kW in key_by_user.items():
    motW = assemble_motion(accel_by_user.get(u), gyro_by_user.get(u),
                           accel_mean, gyro_mean) if CONFIG.USE_MOTION else None
    touW = touch_by_user.get(u) if CONFIG.USE_TOUCH else None
    fused = fuse_user(kW, motW, touW, motion_mean, touch_mean, CONFIG)
    assert fused.shape[1] == FEATURE_DIM, (fused.shape, FEATURE_DIM)
    user_X[u] = fused

if CONFIG.MAX_USERS:
    user_X = {u: user_X[u] for u in sorted(user_X)[:CONFIG.MAX_USERS]}
user_X = {u: X for u, X in user_X.items() if len(X) >= CONFIG.MIN_WINDOWS_PER_USER}
users = sorted(user_X)
print(f"[features] fused dim={FEATURE_DIM} (key {KEY_DIM}+motion {MOTION_DIM}+touch {TOUCH_DIM}) "
      f"| usable users: {len(users)} | total windows: {sum(len(v) for v in user_X.values())}")
if len(users) < 3:
    raise RuntimeError("Too few users with usable windows; check dataset path/format.")

# ----------------------------------------------------------------------------
# 4. Per-user enrol/eval split (unchanged verification protocol)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(CONFIG.SEED)
enroll_by_user, geval_by_user = {}, {}
for u in users:
    X = user_X[u].copy(); rng.shuffle(X)
    k = max(1, int((1 - CONFIG.TEST_FRACTION) * len(X)))
    enroll_by_user[u] = X[:k]
    geval_by_user[u]  = X[k:] if len(X) - k > 0 else X[-1:]

# ----------------------------------------------------------------------------
# 5. Normalise with enrolment statistics
# ----------------------------------------------------------------------------
enroll_all = np.vstack([enroll_by_user[u] for u in users])
mu = enroll_all.mean(0); sd = enroll_all.std(0) + 1e-8
norm = lambda A: (A - mu) / sd
enroll_by_user = {u: norm(v) for u, v in enroll_by_user.items()}
geval_by_user  = {u: norm(v) for u, v in geval_by_user.items()}

# ----------------------------------------------------------------------------
# 6. Non-IID federated clients
# ----------------------------------------------------------------------------
parts = partition_users_to_clients(users, CONFIG.N_CLIENTS, CONFIG.SEED)
client_arrays = {}
for c, us in parts.items():
    us = [u for u in us if u in enroll_by_user]
    if us:
        client_arrays[c] = np.vstack([enroll_by_user[u] for u in us])
client_arrays = {c: v for c, v in client_arrays.items() if len(v) >= CONFIG.BATCH_SIZE}
print(f"[federated] clients with usable data: {sorted(client_arrays)}")
for c, v in client_arrays.items():
    print(f"   client {c}: {v.shape[0]} windows")

# ----------------------------------------------------------------------------
# 7. Cache
# ----------------------------------------------------------------------------
save = dict(mu=mu, sd=sd, feature_dim=np.array([FEATURE_DIM]), users=np.array(users))
for c, v in client_arrays.items(): save[f"client_{c}"] = v
for u in users:
    save[f"enroll_{u}"] = enroll_by_user[u]
    save[f"geval_{u}"]  = geval_by_user[u]
out = os.path.join(CONFIG.ARTIFACT_DIR, "processed.npz")
np.savez_compressed(out, **save)
print(f"\n[cache] wrote {out}")
print(f"[cache] {len(users)} users | multimodal feature_dim={FEATURE_DIM}")
print("Section 1 complete. Next: run 02_federated_train.py")
