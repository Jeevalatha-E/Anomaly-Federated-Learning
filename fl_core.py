"""
fl_core.py  (updated: Byzantine attack hook + consistent DP-FedAvg noise)
=========================================================================
DP noise is added AFTER aggregation for every aggregator, with
    std = DP_NOISE_MULT * DP_CLIP / m           (m = # participating clients)
so that z = DP_NOISE_MULT is the exact NOISE MULTIPLIER accounted for in
privacy_accounting.compute_epsilon(z, rounds, sample_rate).

SecAgg masks the weighted contributions (cancel exactly in the FedAvg sum);
Median/Krum are non-linear so SecAgg is skipped for them. Designated malicious
clients corrupt their update (sign-flip / gaussian / scale) for the Byzantine
robustness experiment (07_byzantine_attack.py).
"""

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import build_model


def flatten_state(state):
    return torch.cat([v.detach().reshape(-1).float() for v in state.values()])


def unflatten_to(state_template, vector):
    out, i = {}, 0
    for k, v in state_template.items():
        n = v.numel()
        out[k] = vector[i:i + n].reshape(v.shape).to(v.dtype)
        i += n
    return out


def local_train(global_state, X, cfg, device="cpu"):
    """Train locally, return the (optionally clipped) update vector and #samples."""
    model = build_model(cfg.MODEL, X.shape[1], cfg.LATENT_DIM).to(device)
    model.load_state_dict(global_state)
    model.train()

    global_vec = flatten_state(global_state).to(device)
    global_params = [p.clone().detach() for p in
                     unflatten_to(global_state, global_vec).values()]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32)),
                        batch_size=cfg.BATCH_SIZE, shuffle=True)

    for _ in range(cfg.LOCAL_EPOCHS):
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = model.loss(xb)
            if cfg.AGGREGATOR == "fedprox":
                prox = sum(((p - g.to(device)) ** 2).sum()
                           for p, g in zip(model.parameters(), global_params))
                loss = loss + 0.5 * cfg.FEDPROX_MU * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

    update = flatten_state(model.state_dict()).to(device) - global_vec

    if cfg.DP_ENABLED:                              # per-client L2 clip
        norm = update.norm() + 1e-12
        update = update * min(1.0, cfg.DP_CLIP / norm)

    return update.detach().cpu(), len(X)


def apply_byzantine_attack(update, attack, scale, seed):
    """
    Corrupt one client's update vector to simulate a Byzantine participant.
      * "sign_flip" : Delta -> -scale * Delta   (paper's primary attack, lambda=scale)
      * "gaussian"  : Delta -> random dir, matched norm * scale
      * "scale"     : Delta ->  scale * Delta
    """
    if attack in (None, "none"):
        return update
    if attack == "sign_flip":
        return -float(scale) * update
    if attack == "gaussian":
        g = torch.as_tensor(
            np.random.default_rng(seed).standard_normal(update.numel()),
            dtype=update.dtype)
        return g / (g.norm() + 1e-12) * update.norm() * float(scale)
    if attack == "scale":
        return float(scale) * update
    raise ValueError(f"unknown attack '{attack}'")


def _pairwise_masks(n, dim, seed):
    rng = np.random.default_rng(seed)
    masks = [torch.zeros(dim) for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            m = torch.as_tensor(rng.standard_normal(dim), dtype=torch.float32)
            masks[i] += m
            masks[j] -= m
    return masks


def _fedavg_secure(updates, sizes, cfg):
    w = torch.tensor(sizes, dtype=torch.float32)
    w = w / w.sum()
    contribs = [w[i] * updates[i] for i in range(len(updates))]
    if cfg.SECAGG_ENABLED:
        masks = _pairwise_masks(len(contribs), contribs[0].numel(), cfg.SEED)
        contribs = [c + m for c, m in zip(contribs, masks)]
    return torch.stack(contribs).sum(0)            # masks cancel -> exact weighted mean


def _median(updates):
    return torch.stack(updates).median(dim=0).values


def _krum(updates, f=1):
    stacked = torch.stack(updates)
    n = len(updates)
    k = max(1, n - f - 2)
    dists = torch.cdist(stacked, stacked) ** 2
    scores = [torch.sort(dists[i]).values[1:k + 1].sum() for i in range(n)]
    return stacked[int(torch.argmin(torch.tensor(scores)))]


def aggregate(updates, sizes, cfg):
    """Combine client updates, then add DP noise uniformly (all aggregators)."""
    if cfg.AGGREGATOR in ("fedavg", "fedprox"):
        agg = _fedavg_secure(updates, sizes, cfg)
    elif cfg.AGGREGATOR == "median":
        agg = _median(updates)
    elif cfg.AGGREGATOR == "krum":
        agg = _krum(updates, f=cfg.KRUM_BYZANTINE)
    else:
        raise ValueError(cfg.AGGREGATOR)

    if cfg.DP_ENABLED:
        # Gaussian mechanism, noise multiplier z = DP_NOISE_MULT on an average
        # whose per-client L2 sensitivity is C/m  ->  std = z * C / m.
        m = max(1, len(updates))
        std = cfg.DP_NOISE_MULT * cfg.DP_CLIP / m
        agg = agg + torch.randn_like(agg) * std
    return agg


def federated_train(client_data, input_dim, cfg, val_normal=None,
                    device="cpu", verbose=True,
                    malicious_clients=None, attack=None, attack_scale=1.0):
    """
    client_data: dict client_id -> np.ndarray of NORMAL training windows.
    malicious_clients: iterable of client ids that act as Byzantine participants.
    attack: None | "sign_flip" | "gaussian" | "scale".
    Returns (global_model, history: round -> mean val reconstruction error).
    """
    malicious = set(malicious_clients or [])
    global_model = build_model(cfg.MODEL, input_dim, cfg.LATENT_DIM).to(device)
    global_state = copy.deepcopy(global_model.state_dict())
    history = {}
    rng = np.random.default_rng(cfg.SEED)
    clients = list(client_data.keys())

    for rnd in range(1, cfg.ROUNDS + 1):
        m = max(1, int(round(cfg.CLIENT_FRACTION * len(clients))))
        chosen = list(rng.choice(clients, size=m, replace=False))

        updates, sizes = [], []
        for c in chosen:
            upd, n = local_train(global_state, client_data[c], cfg, device)
            if c in malicious:
                upd = apply_byzantine_attack(upd, attack, attack_scale,
                                             seed=cfg.SEED + rnd)
            updates.append(upd)
            sizes.append(n)

        agg = aggregate(updates, sizes, cfg).to(device)
        new_vec = flatten_state(global_state).to(device) + agg
        global_state = unflatten_to(global_state, new_vec)
        global_model.load_state_dict(global_state)

        if val_normal is not None:
            global_model.eval()
            with torch.no_grad():
                v = torch.as_tensor(val_normal, dtype=torch.float32).to(device)
                history[rnd] = float(global_model.score(v).mean().cpu())
        if verbose and (rnd == 1 or rnd % 5 == 0):
            print(f"  round {rnd:>3}/{cfg.ROUNDS}  "
                  f"val_recon={history.get(rnd, float('nan')):.5f}")

    return global_model, history
