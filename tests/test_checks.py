"""Implementation checks (proposal Section 4.5), run before any holdout run.

    python tests/test_checks.py        (or: pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agents import bucket, make_spec, simulate_learners
from baselines import lp_bound, simulate_baselines, simulate_greedy
from env import CELLS, K, PRICES, RHO, TAU, budgets, capacities, draw_base

SEED = 10
H = 600  # short horizon keeps the checks fast; the logic does not depend on T


def variants(rho, tau, eta=0.05, L=50, **kw):
    return [make_spec(rho, tau, (v >> 2) & 1, (v >> 1) & 1, v & 1, eta, L,
                      variant=f"{(v >> 2) & 1}{(v >> 1) & 1}{v & 1}", **kw) for v in range(8)]


def test_context_uses_lagged_load():
    for d in (1, 2, 5):
        specs = [make_spec(2.0, 2.0, 1, 1, 1, 0.05, 50, delay=d),
                 make_spec(2.0, 2.0, 0, 0, 0, 0.05, 50, delay=d)]
        _, tr = simulate_learners(SEED, specs, horizon=H, trace=True)
        hist = tr["hist"]
        for t in range(H):
            lagged = hist[:, t - d] if t >= d else np.zeros_like(hist[:, 0])
            assert np.array_equal(tr["nlag"][t], lagged), "context must be n(t-d), never n(t)"
            a = tr["a"][t]
            took = a[0] > 0
            expect = bucket(lagged[0, a[0][took] - 1])
            assert np.array_equal(tr["bucket_used"][t][0][took], expect)
        assert set(np.unique(tr["bucket_used"][:, 1])) <= {-1, 0}, "C = 0 must use one posterior"


def test_no_overspend():
    specs = [s for m in CELLS for s in variants(RHO[m[0]], TAU[m[1]])]
    out, tr = simulate_learners(SEED, specs, horizon=H, trace=True)
    assert max(row["overspend"] for row, _, _ in out) <= 1e-9
    assert tr["B_end"].min() >= -1e-9
    base = [dict(rho=RHO[m[0]], tau=TAU[m[1]], policy=p, meta={}) for m in CELLS
            for p in ("random", "lp_guided")]
    assert max(r["overspend"] for r, _, _ in simulate_baselines(SEED, base, horizon=H)) <= 1e-9
    assert max(r["overspend"] for r, _, _ in simulate_greedy(SEED, base[::2], horizon=H)) <= 1e-9


def test_no_run_beats_lp_bound():
    b = draw_base(SEED)
    for rn, tn in CELLS:
        W_lp, _ = lp_bound(b.qmax, capacities(b, RHO[rn]), b.V, budgets(b, TAU[tn], H), H)
        rows = [r for r, _, _ in simulate_learners(SEED, variants(RHO[rn], TAU[tn]), horizon=H)]
        bs = [dict(rho=RHO[rn], tau=TAU[tn], policy=p, meta={}) for p in ("random", "lp_guided")]
        rows += [r for r, _, _ in simulate_baselines(SEED, bs, horizon=H)]
        rows += [r for r, _, _ in simulate_greedy(SEED, bs[:1], horizon=H)]
        assert all(r["W_exp"] <= W_lp + 1e-6 for r in rows), (rn, tn)


def test_switch_isolation_and_window():
    L = 40
    specs = [make_spec(1.0, 1.0, 0, 1, 1, 0.05, L), make_spec(1.0, 1.0, 1, 0, 0, 0.05, L)]
    _, tr = simulate_learners(SEED, specs, horizon=H, trace=True)
    assert np.all(tr["phi"][:, 0] == 0), "P = 0 must keep phi = 0"
    assert not tr["evicted"][:, 1].any(), "S = 0 must never evict"
    a, X, bu = tr["a"], tr["X"], tr["bucket_used"]
    for r, window in ((0, range(H - L, H)), (1, range(H))):
        alpha = np.repeat(tr["a0"][r][None, :, None], K, axis=0).repeat(10, axis=2)
        beta = np.repeat(tr["b0"][r][None, :, None], K, axis=0).repeat(10, axis=2)
        for t in window:
            for k in np.nonzero(a[t, r] > 0)[0]:
                i, b = a[t, r, k] - 1, bu[t, r, k]
                alpha[k, i, b] += X[t, r, k]
                beta[k, i, b] += 1 - X[t, r, k]
        assert np.allclose(alpha, tr["alpha"][r]) and np.allclose(beta, tr["beta"][r])


def test_same_prior_for_all_variants():
    _, tr = simulate_learners(SEED, variants(1.0, 1.0), horizon=5, trace=True)
    assert np.allclose(tr["a0"], tr["a0"][0]) and np.allclose(tr["b0"], tr["b0"][0])
    assert np.allclose(tr["a0"][0] / (tr["a0"][0] + tr["b0"][0]), PRICES / 4)


def test_crn_batch_invariance():
    """A run gives identical results alone or inside any batch (CRN across variants/cells)."""
    dev = np.zeros(K, bool); dev[7] = True
    alone = simulate_learners(SEED, [make_spec(2.0, 2.0, 0, 0, 0, 0.05, 50)], horizon=H)
    mixed = simulate_learners(SEED, variants(0.5, 0.5) + [
        make_spec(2.0, 2.0, dev, dev, dev, 0.05, 50),
        make_spec(2.0, 2.0, 0, 0, 0, 0.05, 50)], horizon=H)
    assert np.array_equal(alone[0][1], mixed[-1][1])


def test_negative_control_pacing_inert_when_slack():
    rows = [r for r, _, _ in simulate_learners(SEED, variants(1.0, 0.5), horizon=H)]
    by = {r["variant"]: r["W_exp"] for r in rows}
    assert all(by[f"0{cs}"] == by[f"1{cs}"] for cs in ("00", "01", "10", "11"))


def test_learners_beat_random():
    """The plan's sanity check was 'vanilla beats random everywhere'. It does not hold in this
    model, for two reasons reported in the paper: under Tight budgets unpaced TS buys the
    expensive providers and goes broke early, and under High contention every vanilla buyer
    attempts every round, while random's abstentions relieve congestion. What must hold:
    vanilla beats random with no budget pressure and little congestion (Low/Slack), and
    PACE beats random whenever budgets are Tight."""
    for rn, tn in CELLS:
        lrn = simulate_learners(SEED, [make_spec(RHO[rn], TAU[tn], 0, 0, 0, 0.05, 200),
                                       make_spec(RHO[rn], TAU[tn], 1, 1, 1, 0.05, 200)], horizon=H)
        rnd = simulate_baselines(SEED, [dict(rho=RHO[rn], tau=TAU[tn], policy="random", meta={})],
                                 horizon=H)
        if (rn, tn) == ("Low", "Slack"):
            assert lrn[0][0]["W_exp"] > rnd[0][0]["W_exp"], (rn, tn)
        if tn == "Tight":
            assert lrn[1][0]["W_exp"] > rnd[0][0]["W_exp"], (rn, tn)


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t = time.perf_counter()
            fn()
            print(f"PASS {name}  ({time.perf_counter() - t:.1f}s)")
