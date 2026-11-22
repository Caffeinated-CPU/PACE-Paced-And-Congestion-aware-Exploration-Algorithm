# Numbers quoted in the prose, derived from results/summary.json (exec'd by build_tex.py,
# which provides S, money, signed, pval, CELLS and ROOT). Keys use two-letter cell codes such
# as HT = High contention / Tight budgets.
import datetime as _dt
from pathlib import Path as _P


def extra(S):
    v = {}
    code = {c: c.split("/")[0][0] + c.split("/")[1][0] for c in CELLS}
    for c, cc in code.items():
        for var, d in S["pct_lp"].items():
            v[f"pct_{var}_{cc}"] = f"{100 * d[c]['mean']:.1f}"
        dw = S["dW"][c]
        v[f"dwpct_{cc}"] = signed(100 * dw["pct"], 1, "&thinsp;pp")
        v[f"dwci_{cc}"] = f"[{signed(100 * dw['pct_lo'], 1)}, {signed(100 * dw['pct_hi'], 1)}]"
        v[f"dwusd_{cc}"] = money(dw["mean"])
        v[f"dwrel_{cc}"] = signed(100 * dw["rel"], 1, "%")
        for t in ("P", "C", "S"):
            d = S["h13"]["exp"][c][t]
            v[f"th{t}_{cc}"] = signed(100 * d["pct_lp"], 2, "&thinsp;pp")
            v[f"th{t}usd_{cc}"] = money(d["theta"])
            v[f"th{t}p_{cc}"] = pval(d["p"])
        for pop, k in (("invasion", "ginv"), ("stability", "gstab")):
            d = S["h5"]["exp"][pop][c]
            v[f"{k}_{cc}"] = money(d["mean"])
            v[f"{k}ci_{cc}"] = f"[{money(d['lo'])}, {money(d['hi'])}]"
            v[f"{k}rel_{cc}"] = signed(100 * d["rel"], 0, "%")
            v[f"{k}abs_{cc}"] = money(abs(d["mean"]))
            for q, x in d["by_q"].items():
                v[f"{k}q{q}_{cc}"] = money(x)
        for var, d in S["diagnostics"].items():
            dd = d[c]
            v[f"exh_{var}_{cc}"] = f"{100 * dd['exhaust_frac']:.1f}%"
            v[f"exhmed_{var}_{cc}"] = f"{dd['exhaust_median']:,.0f}"
            v[f"vab_{var}_{cc}"] = f"{100 * dd['vol_abstain_rate']:.1f}%"
            v[f"ab_{var}_{cc}"] = f"{100 * dd['abstain_rate']:.1f}%"
            v[f"hhi_{var}_{cc}"] = f"{dd['hhi']:.3f}"
            v[f"cv_{var}_{cc}"] = f"{dd['load_cv']:.2f}"
        v[f"vr_{cc}"] = str(S["van_vs_random"][c])
        v[f"pr_{cc}"] = str(S["pace_vs_random"][c])
        for grp in ("pace", "van"):
            for f, x in S["shares"][c][grp].items():
                v[f"sh_{grp}_{f.replace('.', '')}_{cc}"] = money(x)
        for f, x in S["shares"][c]["W"].items():
            v[f"shW_{f.replace('.', '')}_{cc}"] = f"{100 * x:.1f}"
    alpha = 0.05 / 9
    for t in ("P", "C", "S"):
        neg = [c for c in CELLS if 1 - S["h13"]["exp"][c][t]["p"] < alpha]
        pos = [c for c in CELLS if S["h13"]["exp"][c][t]["reject"]]
        v[f"n{t}neg"], v[f"n{t}pos"] = str(len(neg)), str(len(pos))
    for c, cc in code.items():
        v[f"th2P_{cc}"] = signed(200 * S["h13"]["exp"][c]["P"]["pct_lp"], 1, "&thinsp;pp")
        v[f"bestfi_{cc}"] = f"{100 * max(S['pct_lp']['greedy'][c]['mean'], S['pct_lp']['lp_guided'][c]['mean']):.1f}"
    gaps = [100 - 100 * max(S["pct_lp"]["greedy"][c]["mean"], S["pct_lp"]["lp_guided"][c]["mean"]) for c in CELLS]
    v["max_fi_gap"] = f"{max(gaps):.1f}"
    agree = sum((S["h5"]["exp"][p][c]["sig_pos"], S["h5"]["exp"][p][c]["sig_neg"]) ==
                (S["h5"]["real"][p][c]["sig_pos"], S["h5"]["real"][p][c]["sig_neg"])
                for p in ("invasion", "stability") for c in CELLS)
    v["h5_agree"] = str(agree)
    ds = S["diag_max_se"]
    v.update(se_hhi=f"{ds['hhi']:.3f}", se_vab=f"{ds['vol_abstain_rate']:.3f}", se_exh=f"{ds['exhaust_frac']:.3f}", se_cv=f"{ds['load_cv']:.2f}")
    v["n_p_beats"] = str(sum(S["pct_lp"]["100"][c]["mean"] >= S["pct_lp"]["111"][c]["mean"] for c in CELLS))
    for key in ("abs", "rel", "pct_lp"):
        for j in range(4):
            d = S["h4"][key][f"lambda{j}"]
            fmt = (lambda x: money(x)) if key == "abs" else (
                (lambda x: signed(100 * x, 1, "%")) if key == "rel" else (lambda x: signed(100 * x, 2, "&thinsp;pp")))
            v[f"l{j}_{key}"] = fmt(d["est"])
            v[f"l{j}ci_{key}"] = f"[{fmt(d['lo'])}, {fmt(d['hi'])}]"
            v[f"l{j}p_{key}"] = pval(d["p"])
    _p = S["h4"]["abs"]["lambda3"]["p"]
    v["l3p_abs_txt"] = "&lt;&nbsp;10<sup>&minus;4</sup>" if _p < 1e-4 else f"= {_p:.4f}"
    v["l3_real"] = money(S["h4"]["real"]["lambda3"]["est"])
    for s, d in S["robust"].items():
        k = s.replace("=", "").replace("/", "half").replace(".", "")
        for c, cc in (("High/Tight", "HT"), ("Low/Slack", "LS")):
            v[f"rob_{k}_{cc}"] = money(d[c]["mean"])
            v[f"robrel_{k}_{cc}"] = signed(100 * d[c]["rel"], 1, "%")
    lc = S["lp_check"]
    v["max_overspend"] = ("no buyer ever spent more than its budget" if lc["max_overspend"] <= 1e-6
                          else "the largest overspend was " + money(lc["max_overspend"], 2))
    root = _P(ROOT) / "results" / "logs"
    try:
        t0 = _dt.datetime.fromisoformat((root / "main_start.txt").read_text(encoding="utf-8-sig").strip())
        t1 = _dt.datetime.fromisoformat((root / "main_end.txt").read_text(encoding="utf-8-sig").strip())
        v["wall_min"] = f"{(t1 - t0).total_seconds() / 60 + 9.2:.0f}"   # + two tuning passes (4.6 min each)
    except Exception:
        v["wall_min"] = "about 50"
    g = S["tuning"]["grid"]
    v["tune_best_L1600"] = f"{100 * g['1.0']['1600']:.1f}"
    v["tune_eta3"] = f"{100 * g['3.0']['3200']:.1f}"
    v["tune_eta10"] = f"{100 * g['10.0']['3200']:.1f}"
    v["tune_L100"] = f"{100 * g['1.0']['100']:.1f}"
    v["tune_se"] = f"{100 * S['tuning']['max_se']:.1f}"
    v["n_at_bound"] = str(lc["n_at_bound"])
    v["lp_excess_real"] = f"{100 * (lc['max_pct_real'] - 1):.1f}"
    v["n_pr40"] = str(sum(n == S["n_seeds"] for n in S["pace_vs_random"].values()))
    # effect of context with pacing off / on (mean over S), in points of W_LP
    pct = lambda var, c: 100 * S["pct_lp"][var][c]["mean"]
    for c, cc in code.items():
        v[f"cP0_{cc}"] = signed((pct("010", c) + pct("011", c) - pct("000", c) - pct("001", c)) / 2, 1, "&thinsp;pp")
        v[f"cP1_{cc}"] = signed((pct("110", c) + pct("111", c) - pct("100", c) - pct("101", c)) / 2, 1, "&thinsp;pp")
        for var in S["diagnostics"]:
            v[f"spa_{var}_{cc}"] = f"{S['diagnostics'][var][c]['spend_per_attempt']:.2f}"
    from scipy import stats as _st
    sig = [c for c in CELLS if S["h13"]["exp"][c]["PC"]["theta"] > 0 and
           2 * _st.t.sf(abs(S["h13"]["exp"][c]["PC"]["theta"] / S["h13"]["exp"][c]["PC"]["se"]),
                        S["h13"]["exp"][c]["PC"]["dof"]) < alpha]
    v["pc_sig_cells"] = ", ".join(sig[:-1]) + " and " + sig[-1] if len(sig) > 1 else (sig[0] if sig else "no cell")
    return v
