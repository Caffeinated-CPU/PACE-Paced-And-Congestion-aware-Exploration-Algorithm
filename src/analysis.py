"""Statistics and figures from the raw per-run data (results/*.csv); nothing is re-simulated.

    python src/analysis.py   -> results/summary.json, figures/*.{png,svg,pdf}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from env import K, RHO, TAU, VALIDATION_SEEDS
from runner import ROBUST_CELLS, SHARES, share_mask

import os
ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("PACE_RESULTS", ROOT / "results"))
FIG = Path(os.environ.get("PACE_FIGURES", ROOT / "figures"))
CELLS = [f"{r}/{t}" for r in RHO for t in TAU]
ALPHA_BONF = 0.05 / 9
NBOOT = 10_000
RNG = np.random.default_rng(2026)

VARIANTS = ["000", "100", "010", "001", "110", "101", "011", "111"]
VNAME = {"000": "Vanilla", "100": "P", "010": "C", "001": "S", "110": "P+C", "101": "P+S",
         "011": "C+S", "111": "PACE", "random": "Random", "greedy": "Greedy (full info)",
         "lp_guided": "LP-guided (full info)"}


# ---------------------------------------------------------------- loading
def load():
    runs = pd.read_csv(RES / "runs.csv", dtype={"variant": str, "run_id": str})
    lp = pd.read_csv(RES / "lp.csv")
    runs = runs.merge(lp[["seed", "cell", "W_LP"]], on=["seed", "cell"], how="left")
    runs["pct"] = runs.W_exp / runs.W_LP
    buy = pd.read_csv(RES / "buyers.csv", dtype={"run_id": str})
    U = {kind: g.set_index("run_id")[[f"U{k}" for k in range(K)]]
         for kind, g in buy.groupby("kind")}
    return runs, lp, U


def wide(df, col="W_exp"):
    """seed x cell x variant array from the factorial runs."""
    p = df.pivot_table(index="seed", columns=["cell", "variant"], values=col)
    return np.stack([[p[(c, v)].to_numpy() for v in VARIANTS] for c in CELLS], axis=0)  # (9,8,S)


# ---------------------------------------------------------------- bootstrap helpers
def boot_weights(n_seeds, nboot=NBOOT):
    """Seed-level bootstrap as multinomial weights (resampling whole seeds keeps CRN pairs)."""
    idx = RNG.integers(0, n_seeds, size=(nboot, n_seeds))
    w = np.zeros((nboot, n_seeds))
    np.add.at(w, (np.arange(nboot)[:, None], idx), 1.0)
    return w / n_seeds


def boot_mean(x, w):
    """x: (..., S) seed-level values. Returns mean, 95% CI, one-sided p for mean > 0."""
    m = x.mean(axis=-1)
    bs = x @ w.T if x.ndim > 1 else w @ x
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=-1)
    p = (bs <= 0).mean(axis=-1)
    return m, lo, hi, p


# ---------------------------------------------------------------- H1-H3: factorial ANOVA
def anova(Wc):
    """Wc: (8 variants, S seeds) for one cell. OLS with effect-coded P, C, S, all interactions
    and seed fixed effects (Eq. 7). Returns {term: (theta, se, lo, hi, p_one_sided)}."""
    S = Wc.shape[1]
    codes = np.array([[int(ch) * 2 - 1 for ch in v] for v in VARIANTS], float)  # (8,3)
    Pc, Cc, Sc = codes.T
    terms = {"P": Pc, "C": Cc, "S": Sc, "PC": Pc * Cc, "PS": Pc * Sc, "CS": Cc * Sc,
             "PCS": Pc * Cc * Sc}
    rows, y = [], []
    for v in range(8):
        for s in range(S):
            seed_d = np.zeros(S - 1)
            if s > 0:
                seed_d[s - 1] = 1
            rows.append(np.concatenate([[1.0], [t[v] for t in terms.values()], seed_d]))
            y.append(Wc[v, s])
    X, y = np.array(rows), np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    cov = resid @ resid / dof * np.linalg.inv(X.T @ X)
    tq = stats.t.ppf(0.975, dof)
    out = {}
    for j, name in enumerate(terms, start=1):
        th, se = beta[j], np.sqrt(cov[j, j])
        out[name] = dict(theta=th, se=se, lo=th - tq * se, hi=th + tq * se,
                         p=float(stats.t.sf(th / se, dof)), dof=dof)
    return out


# ---------------------------------------------------------------- H4: environmental regression
def lambda_fit(cell_means):
    """cell_means: (..., 9). Balanced design with seed FE -> OLS on cell means."""
    rt = np.log2([RHO[c.split("/")[0]] for c in CELLS])
    tt = np.log2([TAU[c.split("/")[1]] for c in CELLS])
    Z = np.column_stack([np.ones(9), rt, tt, rt * tt])
    return cell_means @ np.linalg.pinv(Z).T          # (..., 4): lambda0..lambda3


def h4(dW, w):
    """dW: (9, S). Returns point estimates and bootstrap summaries for lambda."""
    lam = lambda_fit(dW.mean(axis=1))
    bs = lambda_fit((dW @ w.T).T)                      # (nboot, 4)
    out = {}
    for j in range(4):
        out[f"lambda{j}"] = dict(est=float(lam[j]), se=float(bs[:, j].std(ddof=1)),
                                 lo=float(np.percentile(bs[:, j], 2.5)),
                                 hi=float(np.percentile(bs[:, j], 97.5)),
                                 p=float((bs[:, j] <= 0).mean()))
    return out


# ---------------------------------------------------------------- H5: incentives
def deviation_gains(runs, U):
    fac = runs[runs.study == "factorial"].set_index(["seed", "cell", "variant"]).run_id
    dev = runs[runs.study == "deviation"].copy()
    dev["dev_buyer"] = dev.dev_buyer.astype(int)
    out = []
    for kind in ("exp", "real"):
        Uk = U[kind]
        for r in dev.itertuples():
            col = f"U{r.dev_buyer}"
            u_dev = Uk.at[r.run_id, col]
            if r.population == "invasion":
                ref = Uk.at[fac[(r.seed, r.cell, "000")], col]
                G = u_dev - ref                               # U(PACE; Van) - U(Van; Van)
            else:
                ref = Uk.at[fac[(r.seed, r.cell, "111")], col]
                G = ref - u_dev                               # U(PACE; PACE) - U(Van; PACE)
            out.append(dict(kind=kind, seed=r.seed, cell=r.cell, population=r.population,
                            dev_q=int(r.dev_q), buyer=r.dev_buyer, G=G, U_ref=ref))
    return pd.DataFrame(out)


def classify(dw_sig_pos, g_sig_pos):
    if dw_sig_pos:
        return "Aligned" if g_sig_pos else "Social dilemma"
    return "Tragedy" if g_sig_pos else "Not worth adopting"


# ---------------------------------------------------------------- main analysis
def main():
    FIG.mkdir(exist_ok=True)
    runs, lp, U = load()
    summ = {}

    # tuning
    tune = runs[runs.study == "tune"].copy()
    tune[["eta", "L"]] = tune[["eta", "L"]].astype(float)
    tgrid = tune.groupby(["eta", "L"]).pct.mean().unstack()
    choice = json.loads((RES / "tuning_choice.json").read_text())
    summ["tuning"] = dict(choice=choice, grid={f"{e}": {f"{int(l)}": float(tgrid.loc[e, l])
                                                        for l in tgrid.columns}
                                               for e in tgrid.index},
                          n_runs=int(len(tune)), seeds=VALIDATION_SEEDS)
    seed_scores = tune.groupby(["eta", "L", "seed"]).pct.mean()          # mean over cells per seed
    summ["tuning"]["max_se"] = float(seed_scores.groupby(["eta", "L"]).sem().max())
    eta_s, L_s = choice["eta"], choice["L"]

    fac = runs[runs.study == "factorial"]
    seeds = np.sort(fac.seed.unique())
    S = len(seeds)
    w = boot_weights(S)
    W = wide(fac)                                    # (9,8,S)
    Wr = wide(fac, "W_real")
    WLP = lp.pivot(index="seed", columns="cell", values="W_LP").loc[seeds, CELLS].to_numpy().T
    summ["n_seeds"] = int(S)
    summ["W_LP_mean"] = {c: float(WLP[m].mean()) for m, c in enumerate(CELLS)}

    # H1-H3
    h13 = {}
    for kind, arr in (("exp", W), ("real", Wr)):
        h13[kind] = {}
        for m, c in enumerate(CELLS):
            res = anova(arr[m])
            for t, d in res.items():
                d["pct_lp"] = d["theta"] / WLP[m].mean()
                d["reject"] = bool(d["p"] < ALPHA_BONF)
            h13[kind][c] = res
    summ["h13"] = h13

    # self-play effect and H4
    dW = W[:, 7, :] - W[:, 0, :]                      # PACE - vanilla, (9,S)
    dW_pct = dW / WLP
    rel = dW / W[:, 0, :]
    m_, lo_, hi_, p_ = boot_mean(dW, w)
    mp, lop, hip, pp = boot_mean(dW_pct, w)
    mneg = boot_mean(-dW, w)[3]
    summ["dW"] = {c: dict(mean=float(m_[i]), lo=float(lo_[i]), hi=float(hi_[i]),
                          p_pos=float(p_[i]), p_neg=float(mneg[i]),
                          pct=float(mp[i]), pct_lo=float(lop[i]), pct_hi=float(hip[i]),
                          rel=float(rel[i].mean()), sig_pos=bool(p_[i] < ALPHA_BONF),
                          sig_neg=bool(mneg[i] < ALPHA_BONF))
                  for i, c in enumerate(CELLS)}
    summ["h4"] = dict(abs=h4(dW, w), rel=h4(rel, w), pct_lp=h4(dW_pct, w),
                      real=h4(Wr[:, 7] - Wr[:, 0], w))

    # benchmarks: % of W_LP
    bench = runs[runs.study.isin(["factorial", "baseline"])]
    tab = bench.groupby(["variant", "cell"]).pct.agg(["mean", "std", "count"])
    summ["pct_lp"] = {v: {c: dict(mean=float(tab.loc[(v, c), "mean"]),
                                  se=float(tab.loc[(v, c), "std"] / np.sqrt(tab.loc[(v, c), "count"])))
                          for c in CELLS} for v in VARIANTS + ["random", "greedy", "lp_guided"]}
    piv = bench.pivot_table(index="seed", columns=["variant", "cell"], values="W_exp")
    summ["van_vs_random"] = {c: int((piv[("000", c)] > piv[("random", c)]).sum()) for c in CELLS}
    summ["pace_vs_random"] = {c: int((piv[("111", c)] > piv[("random", c)]).sum()) for c in CELLS}
    gamma1 = runs[(runs.gamma == 1.0) & runs.W_LP.notna()]
    summ["lp_check"] = dict(n_runs=int(len(gamma1)), max_pct_exp=float(gamma1.pct.max()),
                            max_pct_real=float((gamma1.W_real / gamma1.W_LP).max()),
                            violations=int((gamma1.W_exp > gamma1.W_LP + 1e-6).sum()),
                            max_overspend=float(runs.overspend.max()),
                            n_at_bound=int((gamma1.pct > 0.9999).sum()),
                            at_bound_policies=sorted(gamma1[gamma1.pct > 0.9999].variant.unique()))
    summ["lp_solve_ms"] = dict(median=float(lp.solve_ms.median()), max=float(lp.solve_ms.max()))

    # diagnostics
    fac = fac.assign(spend_per_attempt=fac.spend / fac.attempts.clip(lower=1))
    diag = fac.groupby(["variant", "cell"])[["load_cv", "hhi", "abstain_rate", "vol_abstain_rate",
                                             "exhaust_frac", "exhaust_median",
                                             "spend_per_attempt"]].mean()
    summ["diagnostics"] = {v: {c: diag.loc[(v, c)].to_dict() for c in CELLS} for v in VARIANTS}

    # H5
    G = deviation_gains(runs, U)
    h5 = {}
    for kind in ("exp", "real"):
        h5[kind] = {}
        for pop in ("invasion", "stability"):
            g = G[(G.kind == kind) & (G.population == pop)]
            seed_mean = g.pivot_table(index="seed", columns="cell", values="G").loc[seeds, CELLS]
            ref_mean = g.groupby("cell").U_ref.mean()
            gm, glo, ghi, gp = boot_mean(seed_mean.to_numpy().T, w)
            gneg = boot_mean(-seed_mean.to_numpy().T, w)[3]
            byq = g.groupby(["cell", "dev_q"]).G.mean()
            h5[kind][pop] = {c: dict(mean=float(gm[i]), lo=float(glo[i]), hi=float(ghi[i]),
                                     p=float(gp[i]), p_neg=float(gneg[i]),
                                     sig_pos=bool(gp[i] < ALPHA_BONF),
                                     sig_neg=bool(gneg[i] < ALPHA_BONF),
                                     rel=float(gm[i] / ref_mean[c]),
                                     by_q={int(q): float(byq[(c, q)]) for q in (10, 30, 50, 70, 90)})
                             for i, c in enumerate(CELLS)}
    summ["h5"] = h5
    summ["h5_class"] = {c: dict(category=classify(summ["dW"][c]["sig_pos"],
                                                   h5["exp"]["invasion"][c]["sig_pos"]),
                                stable=h5["exp"]["stability"][c]["sig_pos"]) for c in CELLS}

    # shares extension: each buyer's gain relative to its own payoff in the all-vanilla market,
    # so the two groups (different buyers) are compared on a paired basis
    sh = runs[runs.study == "shares"]
    fac_idx = fac.set_index(["seed", "cell", "variant"])
    Ue = U["exp"]
    base = lambda s, c, v: Ue.loc[fac_idx.at[(s, c, v), "run_id"]].to_numpy()
    curves = {c: {"pace": {}, "van": {}, "W": {}} for c in CELLS}   # values are (seed, value)
    for r in sh.itertuples():
        f = float(r.share)
        mask = share_mask(r.seed, f)
        d = Ue.loc[r.run_id].to_numpy() - base(r.seed, r.cell, "000")
        for key, val in (("pace", d[mask].mean()), ("van", d[~mask].mean()), ("W", r.pct)):
            curves[r.cell][key].setdefault(f, []).append((r.seed, val))
    for c in CELLS:
        curves[c]["van"][0.0] = [(s, 0.0) for s in seeds]
        curves[c]["pace"][1.0] = [(s, (base(s, c, "111") - base(s, c, "000")).mean()) for s in seeds]
        for v, f in (("000", 0.0), ("111", 1.0)):
            x = fac_idx.xs((c, v), level=("cell", "variant"))
            curves[c]["W"][f] = list(zip(x.index, x.pct))
        inv = G[(G.kind == "exp") & (G.population == "invasion") & (G.cell == c)]
        stab = G[(G.kind == "exp") & (G.population == "stability") & (G.cell == c)]
        curves[c]["pace"][1 / K] = list(zip(inv.seed, inv.G))   # lone PACE buyers: G(Van)
        curves[c]["van"][1 - 1 / K] = [(s, u_ref - g - base(s, c, "000")[k]) for s, k, u_ref, g in
                                       zip(stab.seed, stab.buyer, stab.U_ref, stab.G)]

    def seed_ci(pairs):
        """Mean and 95% half-width from seed-level means (deviants of one seed averaged first)."""
        m = pd.DataFrame(pairs, columns=["seed", "v"]).groupby("seed").v.mean().to_numpy()
        hw = stats.t.ppf(0.975, len(m) - 1) * m.std(ddof=1) / np.sqrt(len(m)) if m.std() > 0 else 0.0
        return float(m.mean()), float(hw)

    summ["shares"] = {c: {k: {f"{f:.2f}": seed_ci(v)[0] for f, v in sorted(d.items())}
                          for k, d in curves[c].items()} for c in CELLS}
    summ["shares_hw"] = {c: {k: {f"{f:.2f}": seed_ci(v)[1] for f, v in sorted(d.items())}
                             for k, d in curves[c].items()} for c in CELLS}
    dse = fac.groupby(["variant", "cell"])[["hhi", "load_cv", "vol_abstain_rate", "exhaust_frac"]].sem()
    shown = dse.loc[["000", "100", "010", "111"]]
    summ["diag_max_se"] = {m: float(shown[m].max()) for m in dse.columns}
    summ["diag_max_se_at"] = {m: list(shown[m].idxmax()) for m in dse.columns}

    # robustness
    rob = runs[runs.study == "robust"]
    robust = {}
    for (setting, cell), g in rob.groupby(["setting", "cell"]):
        p = g.pivot_table(index="seed", columns="variant", values="W_exp")
        d = (p["111"] - p["000"]).to_numpy()
        ww = boot_weights(len(d), 4000)
        m, lo, hi, pv = boot_mean(d, ww)
        robust.setdefault(setting, {})[cell] = dict(mean=float(m), lo=float(lo), hi=float(hi),
                                                    rel=float((d / p["000"].to_numpy()).mean()),
                                                    p_pos=float(pv), n=int(len(d)))
    rseeds = np.sort(rob.seed.unique())
    for rn, tn in ROBUST_CELLS:
        c = f"{rn}/{tn}"
        m_i = CELLS.index(c)
        sel = np.isin(seeds, rseeds)
        d = dW[m_i, sel]
        ww = boot_weights(len(d), 4000)
        m, lo, hi, pv = boot_mean(d, ww)
        robust.setdefault("primary", {})[c] = dict(mean=float(m), lo=float(lo), hi=float(hi),
                                                   rel=float(rel[m_i, sel].mean()),
                                                   p_pos=float(pv), n=int(len(d)))
    summ["robust"] = robust

    # agreement between expected-given-actions and realised surplus
    agree = sum(h13["exp"][c][t]["reject"] == h13["real"][c][t]["reject"]
                for c in CELLS for t in ("P", "C", "S"))
    summ["exp_vs_real_agreement"] = dict(h13_decisions_agree=int(agree), of=27)

    # runtime
    main_runs = runs[runs.study != "tune"]
    summ["runtime"] = dict(cpu_hours_all=float(runs.runtime.sum() / 3600),
                           cpu_hours_main=float(main_runs.runtime.sum() / 3600),
                           sec_per_learner_run=float(runs[runs.study == "factorial"].runtime.mean()),
                           n_runs=int(len(runs)),
                           by_study={k: int(v) for k, v in runs.study.value_counts().items()})

    (RES / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    figures(summ, W, WLP, G)
    print(json.dumps({k: summ[k] for k in ("lp_check", "runtime", "exp_vs_real_agreement")},
                     indent=1))


# ---------------------------------------------------------------- figures
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RHO_N, TAU_N = list(RHO), list(TAU)


def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 8.5,
        "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "svg.fonttype": "path", "figure.dpi": 150, "axes.titlesize": 9, "axes.titlecolor": INK,
        "axes.titleweight": "semibold"})


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "div", ["#b8322f", "#e34948", "#f3a6a2", "#f0efec", "#9ec5f4", "#3987e5", "#1c5cab"])


def _save(fig, name):
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.03, dpi=220)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.03)   # for the LaTeX report


def _heat(ax, vals, labels, title, vmax, cmap):
    from matplotlib.colors import TwoSlopeNorm
    im = ax.imshow(vals, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(3), TAU_N)
    ax.set_yticks(range(3), RHO_N)
    ax.set_xlabel("Budget tightness τ (Slack 0.5 · Moderate 1 · Tight 2)")
    ax.set_ylabel("Contention ρ")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    for i in range(3):
        for j in range(3):
            v = vals[i, j]
            light = abs(v) / vmax > 0.6
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8,
                    color="white" if light else INK)
    ax.set_title(title, loc="left")
    return im


def figures(summ, W, WLP, G):
    import matplotlib.pyplot as plt
    _style()
    cmap = _diverging()

    # Figure 4: Delta W heatmap (pp of W_LP) with CIs
    vals = np.array([[summ["dW"][f"{r}/{t}"]["pct"] * 100 for t in TAU_N] for r in RHO_N])
    labs = [[f"{summ['dW'][f'{r}/{t}']['pct'] * 100:+.1f} pp\n"
             f"[{summ['dW'][f'{r}/{t}']['pct_lo'] * 100:+.1f}, {summ['dW'][f'{r}/{t}']['pct_hi'] * 100:+.1f}]"
             for t in TAU_N] for r in RHO_N]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    vmax = np.abs(vals).max() * 1.05
    im = _heat(ax, vals, labs, "ΔW = W(PACE) − W(vanilla), % of W$^{LP}$", vmax, cmap)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK2)
    cb.set_label("percentage points of W$^{LP}$", color=INK2, fontsize=7.5)
    _save(fig, "fig4_dW_heatmap")
    plt.close(fig)

    # Figure 2: main effects forest plot
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9), sharey=True)
    for ax, (term, label), col in zip(axes, (("P", "θ$_P$ pacing (H3)"), ("C", "θ$_C$ context (H1)"),
                                             ("S", "θ$_S$ window (H2)")), SERIES):
        for y, c in enumerate(CELLS):
            d = summ["h13"]["exp"][c][term]
            x, lo, hi = (d["pct_lp"] * 100, d["lo"] / summ["W_LP_mean"][c] * 100,
                         d["hi"] / summ["W_LP_mean"][c] * 100)
            ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(x, y, "o", ms=5.5, color=col, mec="white", mew=1.2,
                    mfc=col if d["reject"] else "white")
            if not d["reject"]:
                ax.plot(x, y, "o", ms=5.5, mfc="white", mec=col, mew=1.4)
        ax.axvline(0, color=BASE, lw=1)
        ax.set_title(label, loc="left")
        ax.set_xlabel("effect, % of W$^{LP}$")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(9), CELLS)
    axes[0].invert_yaxis()
    fig.text(0.01, -0.04, "Filled marker: one-sided test significant after Bonferroni (p < 0.05/9); "
             "hollow: not significant. Bars: 95% CI.", color=INK2, fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig2_main_effects")
    plt.close(fig)

    # Figure 5: incentive map (Table 2 quadrants) + stability
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    ax = axes[0]
    # short codes: contention L/M/H, then tightness S/M/T; offsets keep the Slack cluster legible
    offs = {"Low/Slack": (5, 5, "left"), "Medium/Slack": (-2, 7, "center"),
            "High/Slack": (5, -11, "left"), "High/Moderate": (5, 2, "left")}
    for c in CELLS:
        dw = summ["dW"][c]
        g = summ["h5"]["exp"]["invasion"][c]
        ax.errorbar(dw["mean"] / 1000, g["mean"], xerr=[[dw["mean"] / 1000 - dw["lo"] / 1000],
                                                        [dw["hi"] / 1000 - dw["mean"] / 1000]],
                    yerr=[[g["mean"] - g["lo"]], [g["hi"] - g["mean"]]], fmt="o", ms=5,
                    color=SERIES[0], mec="white", mew=1, elinewidth=1.2, capsize=0)
        code = "/".join(p[0] for p in c.split("/"))
        dx, dy, ha = offs.get(c, (5, 4, "left"))
        ax.annotate(code, (dw["mean"] / 1000, g["mean"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.5, color=INK2, ha=ha)
    ax.axhline(0, color=BASE, lw=1)
    ax.axvline(0, color=BASE, lw=1)
    yl = ax.get_ylim()
    ax.set_ylim(yl[0] - 0.08 * (yl[1] - yl[0]), yl[1])
    xl, yl = ax.get_xlim(), ax.get_ylim()
    for (x, y, t, ha, va) in ((xl[1], yl[1], "Aligned", "right", "top"),
                              (xl[1], yl[0], "Social dilemma", "right", "bottom"),
                              (xl[0], yl[1], "Tragedy", "left", "top"),
                              (xl[0], yl[0], "Not worth adopting", "left", "bottom")):
        ax.text(x, y, t, ha=ha, va=va, fontsize=7, color=MUTED, style="italic")
    ax.set_xlabel("ΔW, self-play surplus gain ($ thousands)")
    ax.set_ylabel("Ḡ(Van): gain of a lone PACE adopter ($)")
    ax.set_title("Invasion (H5a) vs market gain", loc="left")
    ax = axes[1]
    inv = [summ["h5"]["exp"]["invasion"][c] for c in CELLS]
    stb = [summ["h5"]["exp"]["stability"][c] for c in CELLS]
    y = np.arange(9)
    for off, data, col, lab in ((-0.17, inv, SERIES[0], "Ḡ(Van): adopt PACE in a vanilla market"),
                                (0.17, stb, SERIES[1], "Ḡ(PACE): stay with PACE in a PACE market")):
        for yi, d in zip(y, data):
            ax.plot([d["lo"], d["hi"]], [yi + off] * 2, color=col, lw=2, solid_capstyle="round")
            ax.plot(d["mean"], yi + off, "o", ms=5, color=col, mec="white", mew=1,
                    mfc=col if d["sig_pos"] else "white")
            if not d["sig_pos"]:
                ax.plot(d["mean"], yi + off, "o", ms=5, mfc="white", mec=col, mew=1.3)
        ax.plot([], [], "o", color=col, label=lab)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(y, CELLS)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("unilateral gain from PACE ($ per buyer)")
    ax.set_title("Invasion and stability (H5)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), fontsize=7, ncol=1)
    fig.tight_layout()
    _save(fig, "fig5_incentives")
    plt.close(fig)

    # Figure 3: diagnostics small multiples
    show = [("000", "Vanilla"), ("100", "P only"), ("010", "C only"), ("111", "PACE")]
    metrics = [("hhi", "Herding (HHI)"), ("load_cv", "Load volatility (CV)"),
               ("vol_abstain_rate", "Voluntary abstention rate"),
               ("exhaust_frac", "Share broke before T")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
    markers = ["o", "s", "^", "D"]
    for ax, (mk, title) in zip(axes, metrics):
        for (v, lab), col, mkr in zip(show, SERIES, markers):
            xs = [summ["diagnostics"][v][c][mk] for c in CELLS]
            ax.plot(xs, range(9), mkr, color=col, ms=4.5, mec="white", mew=0.8, label=lab)
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(9), CELLS)
    axes[0].invert_yaxis()
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.3, -0.13), ncol=4, fontsize=7.5)
    fig.tight_layout()
    _save(fig, "fig3_diagnostics")
    plt.close(fig)

    # Figure 6: population-share payoff curves
    fig, axes = plt.subplots(3, 3, figsize=(7.0, 5.0), sharex=True)
    for ax, c in zip(axes.ravel(), CELLS):
        d = summ["shares"][c]
        for key, col, lab in (("pace", SERIES[0], "PACE adopters"),
                              ("van", SERIES[1], "Vanilla buyers")):
            fs = sorted(float(f) for f in d[key])
            ys = np.array([d[key][f"{f:.2f}"] / 1000 for f in fs])
            hw = np.array([summ["shares_hw"][c][key][f"{f:.2f}"] / 1000 for f in fs])
            ax.fill_between(fs, ys - hw, ys + hw, color=col, alpha=0.14, lw=0)
            ax.plot(fs, ys, "-o", color=col, lw=2, ms=4, mec="white", mew=0.8, label=lab)
        ax.axhline(0, color=BASE, lw=1, zorder=0)
        ax.set_title(c, loc="left", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("share of buyers running PACE")
    for ax in axes[:, 0]:
        ax.set_ylabel("gain vs all-vanilla ($K)")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    _save(fig, "fig6_shares")
    plt.close(fig)

    # Figure A1: tuning grid
    grid = summ["tuning"]["grid"]
    etas = sorted(grid, key=float)
    Ls = sorted(grid[etas[0]], key=int)
    vals = np.array([[grid[e][l] * 100 for l in Ls] for e in etas])
    fig, ax = plt.subplots(figsize=(4.0, 2.6))
    im = ax.imshow(vals, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(Ls)), Ls)
    ax.set_yticks(range(len(etas)), etas)
    ax.set_xlabel("window length L (rounds)")
    ax.set_ylabel("pacing step η")
    ax.grid(False)
    best = np.unravel_index(vals.argmax(), vals.shape)
    for i in range(len(etas)):
        for j in range(len(Ls)):
            ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=7,
                    color="white" if vals[i, j] > vals.min() + 0.6 * np.ptp(vals) else INK,
                    weight="bold" if (i, j) == best else "normal")
    ax.set_title("Validation seeds: mean % of W$^{LP}$ (PACE)", loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, "figA1_tuning")
    plt.close(fig)


if __name__ == "__main__":
    main()
