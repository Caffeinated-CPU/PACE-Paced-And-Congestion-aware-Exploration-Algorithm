"""Environment: base draws, per-cell rescaling, common random numbers, congestion rule.

Everything random about an environment is a function of the seed index alone, so the
same base environment (and the same uniforms) appear in every cell, variant and
deviation run.
"""
from dataclasses import dataclass

import numpy as np

N, K, T = 5, 50, 5000              # providers, buyers, rounds
PRICES = np.linspace(1.0, 3.0, N)  # sorted by q_max, so p_1 < ... < p_N
P_BAR = PRICES.mean()              # 2.0
P_MIN = PRICES[0]
NB = 10                            # load buckets for the contextual learner
ROOT = 41                          # group number; root of every random stream

RHO = {"Low": 0.5, "Medium": 1.0, "High": 2.0}          # contention  K / sum(c)
TAU = {"Slack": 0.5, "Moderate": 1.0, "Tight": 2.0}     # budget tightness  T p_bar / mean B(0)
CELLS = [(r, t) for r in RHO for t in TAU]               # cell m = 3*rho_index + tau_index

VALIDATION_SEEDS = list(range(0, 10))
HOLDOUT_SEEDS = list(range(10, 50))


@dataclass(frozen=True)
class BaseEnv:
    seed: int
    qmax: np.ndarray        # (N,) sorted ascending
    c_base: np.ndarray      # (N,) capacity base draws, carried with their provider
    V: np.ndarray           # (K,) private values
    budget_mult: np.ndarray  # (K,) Uniform(0.8, 1.2)


def draw_base(seed: int) -> BaseEnv:
    rng = np.random.default_rng([ROOT, seed, 0])
    qmax = rng.uniform(0.5, 1.0, N)
    c = rng.uniform(5.0, 20.0, N)
    order = np.argsort(qmax)
    V = rng.uniform(4.0, 10.0, K)
    mult = rng.uniform(0.8, 1.2, K)
    return BaseEnv(seed, qmax[order], c[order], V, mult)


def capacities(base: BaseEnv, rho: float) -> np.ndarray:
    """Rescale the base capacities so that K / sum(c) = rho."""
    return base.c_base * (K / rho) / base.c_base.sum()


def budgets(base: BaseEnv, tau: float, horizon: int = T) -> np.ndarray:
    """B_k(0) = Uniform(0.8, 1.2) * (T p_bar / tau)."""
    return base.budget_mult * horizon * P_BAR / tau


def crn_uniforms(seed: int, horizon: int = T):
    """Pre-drawn uniforms indexed by (round, buyer[, provider]).

    u_x[t, k]     decides the success draw: X = 1 iff u_x < q.
    u_ts[t, k, i] is pushed through the Beta inverse CDF for the Thompson sample of
                  provider i (and u_ts[t, k, 0] drives the randomised baselines).
    Because the arrays do not depend on any policy, switching one buyer's strategy
    changes nobody's random draws.
    """
    rng = np.random.default_rng([ROOT, seed, 1])
    u_x = rng.random((horizon, K))
    u_ts = rng.random((horizon, K, N))
    return u_x, u_ts


def success_prob(qmax, c, n, gamma):
    """Eq. (3): q_i(t) = q_max * min(1, (c_i / n_i)^gamma), and q_max when n_i = 0."""
    n_safe = np.maximum(n, 1)
    return np.where(n > 0, qmax * np.minimum(1.0, (c / n_safe) ** gamma), qmax)
