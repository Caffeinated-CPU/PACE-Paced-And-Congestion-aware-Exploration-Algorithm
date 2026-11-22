"""Baselines and benchmarks: random, centralized greedy, LP upper bound, LP-guided policy."""
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from agents import EPS, Recorder
from env import K, N, P_MIN, PRICES, T, budgets, capacities, crn_uniforms, draw_base, success_prob


def lp_bound(qmax, c, V, B0, horizon=T):
    """Eq. (6): full-horizon fluid LP. Returns (W_LP, z) with z of shape (K, N).

    Variables z[k, i] (index k*N + i) count the rounds in which buyer k uses provider i
    at full quality. Valid upper bound on expected surplus for gamma >= 1 (Proposition 1).
    """
    coef = qmax[None, :] * V[:, None] - PRICES[None, :]
    rows, cols, vals = [], [], []
    for i in range(N):                                   # capacity: sum_k z_ki <= c_i T
        rows += [i] * K; cols += [k * N + i for k in range(K)]; vals += [1.0] * K
    for k in range(K):                                   # actions: sum_i z_ki <= T
        rows += [N + k] * N; cols += [k * N + i for i in range(N)]; vals += [1.0] * N
    for k in range(K):                                   # budget: sum_i p_i z_ki <= B_k(0)
        rows += [N + K + k] * N; cols += [k * N + i for i in range(N)]; vals += list(PRICES)
    A = coo_matrix((vals, (rows, cols)), shape=(N + 2 * K, K * N)).tocsr()
    b = np.concatenate([c * horizon, np.full(K, float(horizon)), B0])
    res = linprog(-coef.ravel(), A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return -res.fun, res.x.reshape(K, N)


def _setup(seed, specs, horizon):
    base = draw_base(seed)
    u_x, u_ts = crn_uniforms(seed, horizon)
    R = len(specs)
    qmax = np.broadcast_to(base.qmax, (R, N))
    c = np.stack([capacities(base, s["rho"]) for s in specs])
    B = np.stack([budgets(base, s["tau"], horizon) for s in specs])
    gamma = np.array([s.get("gamma", 1.0) for s in specs])[:, None]
    return base, u_x, u_ts, R, qmax, c, B, gamma


def simulate_baselines(seed, specs, horizon=T):
    """specs: dicts with rho, tau, policy in {'random', 'lp_guided'} and meta.

    'random' samples uniformly from A_k(t) = {0} U {i : p_i <= B_k(t)}.
    'lp_guided' picks provider i with probability z*_ki / T (abstaining otherwise) and
    abstains if the pick is not affordable. Both use u_ts[t, k, 0] as their uniform.
    """
    t0 = time.perf_counter()
    base, u_x, u_ts, R, qmax, c, B, gamma = _setup(seed, specs, horizon)
    V = base.V[None, :]
    rec = Recorder(R, horizon, V, B.copy())
    is_lp = np.array([s["policy"] == "lp_guided" for s in specs])
    cum = np.zeros((R, K, N))
    for r, s in enumerate(specs):
        if s["policy"] == "lp_guided":
            _, z = lp_bound(base.qmax, c[r], base.V, B[r], horizon)
            cum[r] = np.cumsum(np.clip(z, 0, None) / horizon, axis=1)
    for t in range(horizon):
        u = u_ts[t, :, 0][None, :]
        n_feas = (PRICES[None, None, :] <= B[:, :, None] + EPS).sum(axis=2)   # prices sorted
        a_rand = np.minimum(np.floor(u * (n_feas + 1)).astype(int), n_feas)
        pick = (u[..., None] >= cum).sum(axis=2) + 1                          # 1..N+1
        pick = np.where(pick > N, 0, pick)
        a_lp = np.where(pick <= n_feas, pick, 0)
        a = np.where(is_lp[:, None], a_lp, a_rand)
        n = (a[..., None] == np.arange(1, N + 1)).sum(axis=1)
        qc = success_prob(qmax, c, n, gamma)
        _, price, _, _ = rec.step(t, a, n, qc, u_x[t], B)
        B = B - price
    return rec.results(specs, time.perf_counter() - t0)


def simulate_greedy(seed, specs, horizon=T):
    """Centralized greedy heuristic with full information.

    Each round, budget-feasible buyers are processed in descending order of V_k; each is
    assigned to the affordable provider with the largest marginal immediate expected
    surplus q_i(n+1)(SV_i + V_k) - q_i(n) SV_i - p_i, or abstains if that is <= 0.
    """
    t0 = time.perf_counter()
    base, u_x, _, R, qmax, c, B, gamma = _setup(seed, specs, horizon)
    V = base.V
    order = np.argsort(-V)
    rec = Recorder(R, horizon, V[None, :], B.copy())
    rr = np.arange(R)
    for t in range(horizon):
        n = np.zeros((R, N))
        sv = np.zeros((R, N))
        a = np.zeros((R, K), dtype=int)
        for k in order:
            q_now = np.where(n > 0, success_prob(qmax, c, n, gamma), 0.0)
            q_new = success_prob(qmax, c, n + 1, gamma)
            gain = q_new * (sv + V[k]) - q_now * sv - PRICES[None, :]
            gain = np.where(PRICES[None, :] <= B[:, k:k + 1] + EPS, gain, -np.inf)
            best = gain.argmax(axis=1)
            ok = gain[rr, best] > 0
            a[ok, k] = best[ok] + 1
            n[rr[ok], best[ok]] += 1
            sv[rr[ok], best[ok]] += V[k]
        n = n.astype(int)
        qc = success_prob(qmax, c, n, gamma)
        _, price, _, _ = rec.step(t, a, n, qc, u_x[t], B)
        B = B - price
    return rec.results(specs, time.perf_counter() - t0)
