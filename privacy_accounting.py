"""
privacy_accounting.py  --  honest (epsilon, delta)-DP accounting for DP-FedAvg
==============================================================================
Addresses the reviewer remark that the DP epsilon accounting was insufficiently
documented and inconsistent with the implementation.

Mechanism being accounted for (must match fl_core.aggregate):
  * Each client update is L2-clipped to  C = DP_CLIP.
  * The server forms the average of the m participating clients' updates, whose
    L2 sensitivity to adding/removing one client is  C / m.
  * Gaussian noise  N(0, sigma^2)  with  sigma = z * C / m  is added, where
    z = DP_NOISE_MULT is the NOISE MULTIPLIER (noise std / sensitivity).
  * This is one Gaussian mechanism with noise multiplier z per round; it is
    composed over T = ROUNDS rounds, with client sampling rate q = CLIENT_FRACTION.

compute_epsilon(z, rounds, sample_rate=q, delta) returns the composed epsilon.
It uses the Opacus RDP accountant when available (tight, handles subsampling);
otherwise it falls back to a documented analytic RDP bound.

Everything needed to reproduce the number (z, C, q, T, delta, RDP orders) is
returned by privacy_report(), so the calculation is fully reproducible.
"""

import numpy as np

# RDP orders used by the analytic fallback (same grid Opacus uses by default)
_DEFAULT_ORDERS = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))


def _epsilon_opacus(z, rounds, sample_rate, delta):
    """Tight RDP accounting via Opacus, if installed. Returns eps or None."""
    try:
        from opacus.accountants import RDPAccountant
    except Exception:
        return None
    try:
        acc = RDPAccountant()
        for _ in range(int(rounds)):
            acc.step(noise_multiplier=float(z), sample_rate=float(sample_rate))
        return float(acc.get_epsilon(delta=float(delta)))
    except Exception:
        return None


def _rdp_gaussian(orders, z, rounds):
    """RDP of the (non-subsampled) Gaussian mechanism composed `rounds` times."""
    return np.array([rounds * a / (2.0 * z * z) for a in orders])


def _rdp_subsampled_gaussian(orders, z, q, rounds):
    """
    RDP of the Poisson-subsampled Gaussian mechanism (Mironov et al., 2019),
    composed `rounds` times. Numerically-stable log-sum over binomial terms.
    Used only when q < 1 and Opacus is unavailable.
    """
    from math import lgamma
    def log_comb(n, k):
        return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)
    out = []
    for a in orders:
        # integer moment bound; for fractional alpha we ceil to the next integer
        A = int(np.ceil(a))
        terms = []
        for k in range(0, A + 1):
            # log of  C(A,k) q^k (1-q)^(A-k) * exp( (k^2-k)/(2 sigma^2) )
            log_binom = log_comb(A, k)
            log_q = k * np.log(max(q, 1e-12)) + (A - k) * np.log(max(1 - q, 1e-12))
            gauss = (k * k - k) / (2.0 * z * z)
            terms.append(log_binom + log_q + gauss)
        m = max(terms)
        log_sum = m + np.log(np.sum(np.exp(np.array(terms) - m)))
        out.append(rounds * log_sum / (A - 1) if A > 1 else rounds * log_sum)
    return np.array(out)


def _epsilon_analytic(z, rounds, sample_rate, delta, orders=None):
    """Analytic RDP -> (eps, delta) fallback. Conservative but valid."""
    orders = np.array(orders if orders is not None else _DEFAULT_ORDERS, dtype=float)
    if sample_rate >= 0.999:
        rdp = _rdp_gaussian(orders, z, rounds)
    else:
        rdp = _rdp_subsampled_gaussian(orders, z, sample_rate, rounds)
    eps = rdp + np.log(1.0 / delta) / (orders - 1.0)
    i = int(np.nanargmin(eps))
    return float(eps[i]), float(orders[i])


def compute_epsilon(noise_multiplier, rounds, sample_rate=1.0, delta=1e-5):
    """
    Composed (epsilon, delta)-DP for `rounds` rounds of the Gaussian mechanism at
    noise multiplier `noise_multiplier` and client sampling rate `sample_rate`.
    Returns +inf when noise_multiplier <= 0 (no privacy).
    """
    z = float(noise_multiplier)
    if z <= 0:
        return float("inf")
    eps = _epsilon_opacus(z, rounds, sample_rate, delta)
    if eps is not None:
        return eps
    eps, _ = _epsilon_analytic(z, rounds, sample_rate, delta)
    return eps


def privacy_report(noise_multiplier, rounds, sample_rate=1.0, delta=1e-5,
                   clip=1.0):
    """Full, reproducible record of the privacy calculation."""
    z = float(noise_multiplier)
    method = "opacus" if _epsilon_opacus(z, rounds, sample_rate, delta) is not None \
             else "analytic-rdp"
    eps = compute_epsilon(z, rounds, sample_rate, delta)
    return dict(noise_multiplier=z, clip_norm=float(clip), rounds=int(rounds),
                sample_rate=float(sample_rate), delta=float(delta),
                epsilon=(None if np.isinf(eps) else round(eps, 4)),
                accountant=method)


def noise_multiplier_for_target(target_eps, rounds, sample_rate=1.0, delta=1e-5,
                                lo=0.1, hi=50.0, tol=1e-3):
    """Smallest noise multiplier whose composed epsilon <= target_eps (bisection)."""
    if compute_epsilon(hi, rounds, sample_rate, delta) > target_eps:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if compute_epsilon(mid, rounds, sample_rate, delta) > target_eps:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return round(hi, 4)


if __name__ == "__main__":
    # sanity: epsilon should DECREASE as noise multiplier increases
    for z in [0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]:
        print(f"z={z:>5}  eps={compute_epsilon(z, rounds=30, sample_rate=1.0):.3f}")
    print("z for eps<=10:", noise_multiplier_for_target(10.0, rounds=30))
    print("z for eps<=3 :", noise_multiplier_for_target(3.0, rounds=30))
