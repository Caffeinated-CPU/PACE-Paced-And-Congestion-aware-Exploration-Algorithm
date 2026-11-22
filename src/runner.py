"""Experiment runner. Each run writes one row to results/runs.csv (plus each buyer's surplus
to results/buyers.csv); rows are appended whenever a batch of up to CHUNK runs finishes, so a
crash loses at most the batches in progress and finished runs are skipped on restart.

    python src/runner.py lp          LP upper bound for every (seed, cell)
    python src/runner.py tune        grid search of eta and L on the 10 validation seeds
    python src/runner.py main        factorial, deviation, baselines, robustness, shares
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from agents import make_spec, simulate_learners
from baselines import lp_bound, simulate_baselines, simulate_greedy
from env import (CELLS, HOLDOUT_SEEDS, K, RHO, TAU, VALIDATION_SEEDS, budgets, capacities,
                 draw_base)

OUT = Path(__file__).resolve().parent.parent / "results"
WORKERS = int(os.environ.get("PACE_WORKERS", max(1, (os.cpu_count() or 2) - 1)))
CHUNK = 40

# The first grid (eta <= 0.3, L <= 1600) peaked at its corner, so it was extended before any
# holdout run. L stays below T: a window of length T would never evict anything.
ETA_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
L_GRID = [100, 200, 400, 800, 1600, 3200]
DEV_QUANTILES = [10, 30, 50, 70, 90]
SHARES = [0.25, 0.5, 0.75]
ROBUST_CELLS = [("High", "Tight"), ("Low", "Slack")]
ROBUST_SEEDS = HOLDOUT_SEEDS[:20]

META = ["run_id", "study", "seed", "cell", "rho_name", "tau_name", "rho", "tau", "variant",
        "population", "dev_q", "dev_buyer", "share", "setting", "gamma", "delay", "L", "eta",
        "prior"]
METRICS = ["W_exp", "W_real", "spend", "attempts", "successes", "load_cv", "hhi",
           "abstain_rate", "vol_abstain_rate", "exhaust_frac", "exhaust_median", "overspend",
           "runtime"]


def vname(P, C, S):
    return f"{int(P)}{int(C)}{int(S)}"


def meta(study, seed, rn, tn, variant, population="self", setting="primary", gamma=1.0,
         delay=1, L=None, eta=None, prior=4.0, dev_q="", dev_buyer="", share=""):
    rid = "|".join(str(x) for x in (study, seed, rn, tn, variant, population, dev_q, share,
                                    setting, L, eta))
    return dict(run_id=rid, study=study, seed=seed, cell=f"{rn}/{tn}", rho_name=rn, tau_name=tn,
                rho=RHO[rn], tau=TAU[tn], variant=variant, population=population, dev_q=dev_q,
                dev_buyer=dev_buyer, share=share, setting=setting, gamma=gamma, delay=delay,
                L=L, eta=eta, prior=prior)


def learner(study, seed, rn, tn, P, C, S, eta, L, variant=None, **kw):
    """A TS-family run; mixed populations pass per-buyer P/C/S arrays and a variant label."""
    m = meta(study, seed, rn, tn, variant or vname(P, C, S), eta=eta, L=L, **kw)
    return make_spec(RHO[rn], TAU[tn], P, C, S, eta, L, gamma=kw.get("gamma", 1.0),
                     delay=kw.get("delay", 1), prior=kw.get("prior", 4.0), meta=m)


def deviant_buyers(seed):
    """Buyers at the 10/30/50/70/90th percentile of V_k (one deviant per run)."""
    order = np.argsort(draw_base(seed).V)
    return {q: int(order[int(round(q / 100 * (K - 1)))]) for q in DEV_QUANTILES}


def share_mask(seed, f):
    """PACE buyers spread evenly over the value ranks; floor(f K) of them."""
    rank = np.empty(K, int)
    rank[np.argsort(draw_base(seed).V)] = np.arange(K)
    return np.floor((rank + 1) * f) > np.floor(rank * f)


# ---------------------------------------------------------------- task definitions
def tuning_tasks():
    tasks = []
    for s in VALIDATION_SEEDS:
        specs = [learner("tune", s, rn, tn, 1, 1, 1, eta, L, setting=f"eta={eta},L={L}")
                 for eta in ETA_GRID for L in L_GRID for rn, tn in CELLS]
        tasks += [("learn", s, specs[i:i + CHUNK]) for i in range(0, len(specs), CHUNK)]
    return tasks


def main_tasks(eta, L):
    tasks = []
    for s in HOLDOUT_SEEDS:
        fac = [learner("factorial", s, rn, tn, P, C, S, eta, L)
               for rn, tn in CELLS for P in (0, 1) for C in (0, 1) for S in (0, 1)]
        dev = []
        for q, k in deviant_buyers(s).items():
            one = np.zeros(K, bool)
            one[k] = True
            for rn, tn in CELLS:
                dev.append(learner("deviation", s, rn, tn, one, one, one, eta, L, variant="dev",
                                   population="invasion", dev_q=q, dev_buyer=k))
                dev.append(learner("deviation", s, rn, tn, ~one, ~one, ~one, eta, L,
                                   variant="dev", population="stability", dev_q=q, dev_buyer=k))
        shares = []
        for f in SHARES:
            m = share_mask(s, f)
            shares += [learner("shares", s, rn, tn, m, m, m, eta, L, variant="mix",
                               population="share", share=f) for rn, tn in CELLS]
        rob = []
        if s in ROBUST_SEEDS:
            settings = {"delay=2": dict(delay=2), "delay=5": dict(delay=5),
                        "L=50": dict(L=50), "L=800": dict(L=800),
                        "eta/2": dict(eta=eta / 2), "2eta": dict(eta=2 * eta),
                        "gamma=0.5": dict(gamma=0.5), "gamma=2": dict(gamma=2.0),
                        "prior=1": dict(prior=1.0), "prior=20": dict(prior=20.0)}
            for name, ov in settings.items():
                for rn, tn in ROBUST_CELLS:
                    for v in (0, 1):
                        kw = dict(eta=eta, L=L, setting=name, population="self")
                        kw.update(ov)
                        e, l = kw.pop("eta"), kw.pop("L")
                        rob.append(learner("robust", s, rn, tn, v, v, v, e, l, **kw))
        specs = fac + dev + shares + rob
        tasks += [("learn", s, specs[i:i + CHUNK]) for i in range(0, len(specs), CHUNK)]
        base = [dict(rho=RHO[rn], tau=TAU[tn], policy=p,
                     meta=meta("baseline", s, rn, tn, p)) for rn, tn in CELLS
                for p in ("random", "lp_guided")]
        tasks.append(("base", s, base))
        tasks.append(("greedy", s, [dict(rho=RHO[rn], tau=TAU[tn],
                                         meta=meta("baseline", s, rn, tn, "greedy"))
                                    for rn, tn in CELLS]))
    return tasks


# ---------------------------------------------------------------- execution
def run_task(task):
    kind, seed, specs = task
    fn = {"learn": simulate_learners, "base": simulate_baselines, "greedy": simulate_greedy}[kind]
    return fn(seed, specs)


def done_ids():
    f = OUT / "runs.csv"
    if not f.exists():
        return set()
    with open(f, newline="") as fh:
        return {r["run_id"] for r in csv.DictReader(fh)}


def execute(tasks):
    OUT.mkdir(exist_ok=True)
    done = done_ids()
    todo = []
    for kind, seed, specs in tasks:
        specs = [s for s in specs if s["meta"]["run_id"] not in done]
        if specs:
            todo.append((kind, seed, specs))
    n_runs = sum(len(t[2]) for t in todo)
    print(f"{n_runs} runs in {len(todo)} tasks on {WORKERS} workers", flush=True)
    runs_f, buy_f = OUT / "runs.csv", OUT / "buyers.csv"
    new_r, new_b = not runs_f.exists(), not buy_f.exists()
    t0, finished = time.time(), 0
    with open(runs_f, "a", newline="") as fr, open(buy_f, "a", newline="") as fb, \
            ProcessPoolExecutor(WORKERS) as pool:
        wr = csv.DictWriter(fr, META + METRICS, extrasaction="ignore")
        wb = csv.writer(fb)
        if new_r:
            wr.writeheader()
        if new_b:
            wb.writerow(["run_id", "kind"] + [f"U{k}" for k in range(K)])
        futs = [pool.submit(run_task, t) for t in sorted(todo, key=lambda t: -len(t[2]))]
        for fut in as_completed(futs):
            for row, u_exp, u_real in fut.result():
                wr.writerow(row)
                wb.writerow([row["run_id"], "exp"] + [f"{x:.4f}" for x in u_exp])
                wb.writerow([row["run_id"], "real"] + [f"{x:.4f}" for x in u_real])
                finished += 1
            fr.flush(); fb.flush()
            el = time.time() - t0
            print(f"  {finished}/{n_runs} runs  {el / 60:.1f} min elapsed, "
                  f"~{el / finished * (n_runs - finished) / 60:.1f} min left", flush=True)


def compute_lp():
    OUT.mkdir(exist_ok=True)
    rows = []
    for s in VALIDATION_SEEDS + HOLDOUT_SEEDS:
        b = draw_base(s)
        for rn, tn in CELLS:
            t = time.perf_counter()
            W, _ = lp_bound(b.qmax, capacities(b, RHO[rn]), b.V, budgets(b, TAU[tn]))
            rows.append(dict(seed=s, cell=f"{rn}/{tn}", W_LP=W, solve_ms=1e3 * (time.perf_counter() - t)))
    with open(OUT / "lp.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ms = [r["solve_ms"] for r in rows]
    print(f"{len(rows)} LPs, median {np.median(ms):.1f} ms, max {max(ms):.1f} ms")


def choose_hyperparameters():
    import pandas as pd
    runs = pd.read_csv(OUT / "runs.csv")
    lp = pd.read_csv(OUT / "lp.csv")
    tune = runs[runs.study == "tune"].merge(lp, on=["seed", "cell"])
    tune["pct"] = tune.W_exp / tune.W_LP
    score = tune.groupby(["eta", "L"]).pct.mean().unstack()
    eta, L = score.stack().idxmax()
    choice = dict(eta=float(eta), L=int(L), score=float(score.loc[eta, L]))
    (OUT / "tuning_choice.json").write_text(json.dumps(choice, indent=2))
    print((100 * score).round(2).to_string())
    print("chosen:", choice)
    return choice


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "lp":
        compute_lp()
    elif cmd == "tune":
        execute(tuning_tasks())
        choose_hyperparameters()
    elif cmd == "main":
        ch = json.loads((OUT / "tuning_choice.json").read_text())
        execute(main_tasks(ch["eta"], ch["L"]))
