"""Learning buyers: the shared Thompson Sampling core with P / C / S switches.

One call simulates a batch of runs that share a seed (and hence the base environment
and the common random numbers). Runs in a batch may differ in cell, hyperparameters
and in each buyer's (P, C, S) configuration, so self-play, deviation and robustness
runs all go through the same code. A run's result does not depend on which other runs
share its batch (checked in test_checks.py).
"""
import time

import numpy as np
from scipy.special import betaincinv

from env import (K, N, NB, P_MIN, PRICES, T, budgets, capacities, crn_uniforms,
                 draw_base, success_prob)

EPS = 1e-9


def make_spec(rho, tau, P, C, S, eta, L, gamma=1.0, delay=1, prior=4.0, meta=None, **extra):
    """A run: cell, hyperparameters and per-buyer switches (scalars broadcast to all K)."""
    as_k = lambda x: np.broadcast_to(np.asarray(x, dtype=bool), (K,)).copy()
    return dict(rho=rho, tau=tau, gamma=gamma, delay=int(delay), L=int(L), eta=float(eta),
                prior=float(prior), P=as_k(P), C=as_k(C), S=as_k(S), meta={**(meta or {}), **extra})


def bucket(n):
    """b(n) = min(floor(n / 5), 9): loads 0-4, 5-9, ..., 45-50."""
    return np.minimum(n // 5, NB - 1)


class Recorder:
    """Per-round bookkeeping shared by the learners and the baselines."""

    def __init__(self, R, horizon, V, B0):
        self.R, self.T = R, horizon
        self.V, self.B0 = V, B0
        self.U_exp = np.zeros((R, K))   # sum_t V q_a(t) - p_a : expected surplus given actions
        self.U_real = np.zeros((R, K))  # sum_t V X(t) - p_a   : realised surplus
        self.spend = np.zeros((R, K))
        self.attempts = np.zeros((R, K), dtype=np.int32)
        self.successes = np.zeros((R, K), dtype=np.int32)
        self.vol_abstain = np.zeros((R, K), dtype=np.int32)  # abstained while able to pay
        self.exhaust = np.full((R, K), horizon, dtype=np.int32)
        self.hist = np.zeros((R, horizon, N), dtype=np.int16)
        self.hhi_sum = np.zeros(R)
        self.hhi_cnt = np.zeros(R)

    def step(self, t, a, n, qc, u_x_t, B):
        """a: (R,K) actions in 0..N; n: (R,N) loads; qc: (R,N) congested success probs;
        B: (R,K) budgets at the start of the round. Returns (X, price paid, provider idx, took)."""
        took = a > 0
        ai = np.maximum(a - 1, 0)
        q_ka = np.take_along_axis(qc, ai, axis=1) * took
        price = PRICES[ai] * took
        X = u_x_t[None, :] < q_ka
        able = B >= P_MIN - EPS
        self.U_exp += self.V * q_ka - price
        self.U_real += self.V * X - price
        self.spend += price
        self.attempts += took
        self.successes += X
        self.vol_abstain += (~took) & able
        self.hist[:, t] = n
        tot = n.sum(axis=1)
        share = n / np.maximum(tot, 1)[:, None]
        self.hhi_sum += np.where(tot > 0, (share ** 2).sum(axis=1), 0.0)
        self.hhi_cnt += tot > 0
        newly = (B - price < P_MIN - EPS) & able
        self.exhaust[newly] = t + 1
        return X, price, ai, took

    def results(self, specs, runtime):
        h = self.hist.astype(float)
        mean = h.mean(axis=1)
        std = h.std(axis=1)
        cv = np.where(mean > 0, std / np.where(mean > 0, mean, 1), np.nan)
        out = []
        for r, spec in enumerate(specs):
            row = dict(spec.get("meta", {}))
            row.update(
                W_exp=float(self.U_exp[r].sum()),
                W_real=float(self.U_real[r].sum()),
                spend=float(self.spend[r].sum()),
                attempts=int(self.attempts[r].sum()),
                successes=int(self.successes[r].sum()),
                load_cv=float(np.nanmean(cv[r])) if np.isfinite(cv[r]).any() else float("nan"),
                hhi=float(self.hhi_sum[r] / max(self.hhi_cnt[r], 1)),
                abstain_rate=float(1 - self.attempts[r].sum() / (K * self.T)),
                vol_abstain_rate=float(self.vol_abstain[r].sum() / (K * self.T)),
                exhaust_frac=float((self.exhaust[r] < self.T).mean()),
                exhaust_median=float(np.median(self.exhaust[r])),
                overspend=float(np.max(self.spend[r] - self.B0[r])),
                runtime=runtime / len(specs),
            )
            out.append((row, self.U_exp[r].copy(), self.U_real[r].copy()))
        return out


def simulate_learners(seed, specs, horizon=T, trace=False):
    """Simulate a batch of TS-family runs on one seed. Returns [(row, U_exp_k, U_real_k)]."""
    t0 = time.perf_counter()
    base = draw_base(seed)
    u_x, u_ts = crn_uniforms(seed, horizon)
    R = len(specs)
    rr = np.arange(R)
    ridx, kidx = rr[:, None], np.arange(K)[None, :]

    qmax = np.broadcast_to(base.qmax, (R, N))
    c = np.stack([capacities(base, s["rho"]) for s in specs])
    B = np.stack([budgets(base, s["tau"], horizon) for s in specs])
    V = base.V[None, :]
    gamma = np.array([s["gamma"] for s in specs])[:, None]
    delay = np.array([s["delay"] for s in specs])
    L = np.array([s["L"] for s in specs])
    eta = np.array([s["eta"] for s in specs])[:, None]
    prior = np.array([s["prior"] for s in specs])
    P = np.stack([s["P"] for s in specs])
    C = np.stack([s["C"] for s in specs])
    S = np.stack([s["S"] for s in specs])

    # Informed prior: alpha = m p_i / 4, beta = m (4 - p_i) / 4 with m pseudo-observations,
    # identical for every bucket; a contextual learner indexes buckets, others use bucket 0.
    a0 = prior[:, None] * PRICES[None, :] / 4.0
    b0 = prior[:, None] * (4.0 - PRICES[None, :]) / 4.0
    alpha = np.repeat(np.repeat(a0[:, None, :, None], K, axis=1), NB, axis=3)
    beta = np.repeat(np.repeat(b0[:, None, :, None], K, axis=1), NB, axis=3)

    Lmax = int(L.max())
    ring_p = np.full((R, K, Lmax), -1, dtype=np.int8)  # provider observed in that round (-1: none)
    ring_b = np.zeros((R, K, Lmax), dtype=np.int8)
    ring_x = np.zeros((R, K, Lmax), dtype=np.int8)
    phi = np.zeros((R, K))
    rec = Recorder(R, horizon, V, B.copy())
    tr = {k: [] for k in ("nlag", "bucket_used", "a", "phi", "X", "evicted")} if trace else None

    for t in range(horizon):
        # 1. context: the lagged public load n(t - d), never the current one
        lag = t - delay
        nlag = np.where((lag >= 0)[:, None], rec.hist[rr, np.maximum(lag, 0)], 0)
        bk = bucket(nlag)[:, None, :] * C[:, :, None]               # (R,K,N); 0 if C = 0

        # 2. sliding window: drop the observation made in round t - L (S = 1 only)
        slot = t % L
        sp = ring_p[ridx, kidx, slot[:, None]]
        ev = (sp >= 0) & S
        if ev.any():
            r_, k_ = np.nonzero(ev)
            i_ = sp[ev].astype(np.intp)
            b_ = ring_b[r_, k_, slot[r_]].astype(np.intp)
            x_ = ring_x[r_, k_, slot[r_]]
            alpha[r_, k_, i_, b_] -= x_
            beta[r_, k_, i_, b_] -= 1 - x_

        # 3. Thompson samples from the active posterior, via the inverse CDF on u_ts
        ap = np.take_along_axis(alpha, bk[..., None], axis=3)[..., 0]
        bp = np.take_along_axis(beta, bk[..., None], axis=3)[..., 0]
        able = B >= P_MIN - EPS
        qs = np.zeros((R, K, N))
        r_, k_ = np.nonzero(able)
        if r_.size:
            qs[r_, k_] = betaincinv(ap[r_, k_], bp[r_, k_], u_ts[t, k_])

        # 4. paced (P = 1) or plain (P = 0) greedy choice over the feasible set, abstain = 0
        feas = PRICES[None, None, :] <= B[:, :, None] + EPS
        score = V[..., None] * qs - PRICES[None, None, :] * (1.0 + phi[..., None])
        score = np.where(feas, score, -np.inf)
        a = np.concatenate([np.zeros((R, K, 1)), score], axis=2).argmax(axis=2)  # ties -> lower index

        # 5. joint action fixes the load; outcomes from Eq. (3)
        n = (a[..., None] == np.arange(1, N + 1)).sum(axis=1)
        qc = success_prob(qmax, c, n, gamma)
        X, price, ai, took = rec.step(t, a, n, qc, u_x[t], B)

        # 6. pacing multiplier uses B(t) at the start of the round; P = 0 keeps phi = 0
        phi = np.where(P, np.maximum(0.0, phi + eta * (price - B / (horizon - t))), 0.0)
        B = B - price

        # 7. each buyer updates only its own posterior, in the bucket it saw at decision time
        b_used = np.take_along_axis(bk, ai[..., None], axis=2)[..., 0]
        r_, k_ = np.nonzero(took)
        if r_.size:
            xi = X[took].astype(float)
            alpha[r_, k_, ai[took], b_used[took]] += xi
            beta[r_, k_, ai[took], b_used[took]] += 1.0 - xi
        ring_p[ridx, kidx, slot[:, None]] = np.where(took, ai, -1)
        ring_b[ridx, kidx, slot[:, None]] = b_used
        ring_x[ridx, kidx, slot[:, None]] = X

        if trace:
            tr["nlag"].append(nlag.copy())
            tr["bucket_used"].append(np.where(took, b_used, -1))
            tr["a"].append(a.copy())
            tr["phi"].append(phi.copy())
            tr["X"].append(X.copy())
            tr["evicted"].append(ev.copy())

    out = rec.results(specs, time.perf_counter() - t0)
    if trace:
        tr = {k: np.stack(v) for k, v in tr.items()}
        tr.update(alpha=alpha, beta=beta, hist=rec.hist, a0=a0, b0=b0, B_end=B, spend=rec.spend)
        return out, tr
    return out
