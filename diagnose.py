"""
diagnose.py -- pinpoint what limits detection accuracy on the REAL data.
Run AFTER 01_preprocess.py.  Prints a short report; copy the whole thing back.
"""
import os, numpy as np, pandas as pd
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from common import CONFIG, find_keystroke_csvs, extract_keystroke_features, load_processed

print("="*70); print("DIAGNOSTIC REPORT"); print("="*70)

_, enroll_by_user, geval_by_user, feat, users = load_processed()
counts = np.array([len(enroll_by_user[u]) + len(geval_by_user[u]) for u in users])
print(f"\n[1] users={len(users)} feat_dim={feat} | windows/user "
      f"min={counts.min()} med={int(np.median(counts))} max={counts.max()} "
      f"| <20: {(counts<20).sum()}")

# [2] SUPERVISED CEILING: do the features carry identity at all?
from sklearn.ensemble import RandomForestClassifier
Xtr=np.vstack([enroll_by_user[u] for u in users]); ytr=np.concatenate([[u]*len(enroll_by_user[u]) for u in users])
Xte=np.vstack([geval_by_user[u] for u in users]); yte=np.concatenate([[u]*len(geval_by_user[u]) for u in users])
clf=RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0).fit(Xtr,ytr)
proba=clf.predict_proba(Xte); classes=list(clf.classes_); order=np.argsort(-proba,axis=1)
top1=(clf.predict(Xte)==yte).mean()
top5=np.mean([yte[i] in [classes[j] for j in order[i,:5]] for i in range(len(yte))])
print(f"\n[2] SUPERVISED user-ID ceiling: top-1={top1*100:.1f}% top-5={top5*100:.1f}% "
      f"(chance={100/len(users):.2f}%)")
print(f"    HIGH -> features are fine, scorer is the limit. LOW -> features too weak.")

# helper: per-user kNN verification AUC given {group_key: windows}, target=user
def knn_auc(win_by_key, key_user, seed=0, k=10, imp=120):
    rng=np.random.default_rng(seed); keys=list(win_by_key)
    enr={}; gev={}
    for kk in keys:
        X=win_by_key[kk].copy(); rng.shuffle(X); c=int(0.7*len(X))
        if c<2 or len(X)-c<1: continue
        enr[kk]=X[:c]; gev[kk]=X[c:]
    allf=np.vstack([enr[kk] for kk in enr]); mu=allf.mean(0); sd=allf.std(0)+1e-8
    aucs=[]
    for kk in enr:
        ref=(enr[kk]-mu)/sd; kn=min(k,len(ref)-1) or 1
        nn=NearestNeighbors(n_neighbors=kn).fit(ref)
        g=(gev[kk]-mu)/sd
        # impostors = other-USER groups
        pool=np.vstack([(gev[o]-mu)/sd for o in gev if key_user(o)!=key_user(kk)])
        ii=pool[rng.choice(len(pool),min(imp,len(pool)),replace=False)]
        sg=nn.kneighbors(g)[0].mean(1); si=nn.kneighbors(ii)[0].mean(1)
        y=np.r_[np.zeros(len(sg)),np.ones(len(si))]; aucs.append(roc_auc_score(y,np.r_[sg,si]))
    return float(np.mean(aucs)), len(enr)

# [3] POOLED (per-user) vs PER-DEVICE kNN verification AUC
print(f"\n[3] Re-reading CSVs to test pooled vs per-device profiling...")
files=find_keystroke_csvs(CONFIG.DATA_ROOT)
def dev_of(p):
    b=os.path.basename(p).lower()
    for t in ["desktop","phone","tablet","hand","watch"]:
        if t in b: return t
    return "other"
byU=defaultdict(list); byUD=defaultdict(list)
for f in files:
    try: df=pd.read_csv(f["path"])
    except Exception: continue
    W=extract_keystroke_features(df,f["kcol"],f["dcol"],f["tcol"])
    if len(W):
        byU[f["user"]].append(W); byUD[(f["user"],dev_of(f["path"]))].append(W)
byU={u:np.vstack(v) for u,v in byU.items() if len(np.vstack(v))>=6}
byUD={k:np.vstack(v) for k,v in byUD.items() if len(np.vstack(v))>=6}
a_pool,n1=knn_auc(byU,   key_user=lambda k:k)
a_dev ,n2=knn_auc(byUD,  key_user=lambda k:k[0])
print(f"    POOLED  per-user profile     AUC={a_pool:.3f}  ({n1} users)")
print(f"    PER-DEVICE profile+match     AUC={a_dev:.3f}  ({n2} user-device groups)")
print(f"    devices: {sorted(set(k[1] for k in byUD))}")
print(f"    -> if PER-DEVICE >> POOLED, device mixing is the problem.")
print("\n"+"="*70+"\nCopy this whole report back.")
