"""Fill main_template.tex with numbers from results/summary.json and compile it.

    python report/build_tex.py     -> report/main.tex and report/main.pdf
                                      (needs tectonic on PATH, or set TECTONIC to its path)

The prose numbers come from report_values.py, formatted here for LaTeX; the results tables
are generated below. Figures are copied from figures/ (written by src/analysis.py).
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S = json.loads((ROOT / "results" / "summary.json").read_text())
CELLS = [f"{r}/{t}" for r in ("Low", "Medium", "High") for t in ("Slack", "Moderate", "Tight")]
CODE = {c: c.split("/")[0][0] + "/" + c.split("/")[1][0] for c in CELLS}
FIGURES = ["fig2_main_effects", "fig3_diagnostics", "fig4_dW_heatmap", "fig5_incentives", "fig6_shares"]


def base_values():
    """Run-level numbers: tuning choice, LP check, runtime."""
    ch, rt, lc = S["tuning"]["choice"], S["runtime"], S["lp_check"]
    return dict(eta=f"{ch['eta']:g}", tune_score=f"{100 * ch['score']:.1f}",
                lp_ms=f"{S['lp_solve_ms']['median']:.0f}", n_seeds=str(S["n_seeds"]),
                cpu_h=f"{rt['cpu_hours_all']:.1f}", sec_run=f"{rt['sec_per_learner_run']:.1f}",
                n_runs=f"{rt['n_runs']:,}", lp_check_runs=f"{lc['n_runs']:,}",
                lp_max_pct=f"{100 * lc['max_pct_exp']:.1f}",
                lp_max_pct_real=f"{100 * lc['max_pct_real']:.1f}",
                agree=str(S["exp_vs_real_agreement"]["h13_decisions_agree"]))


def money(x, d=0):
    s = f"{abs(x):,.{d}f}"
    return ("$-$" if x < 0 and float(s.replace(",", "")) > 0 else "") + "\\$" + s


def signed(x, d=1, unit=""):
    s = f"{abs(x):.{d}f}"
    if float(s) == 0:
        return s + unit
    return ("+" if x > 0 else "$-$") + s + unit


def pval(p):
    return "<10^{-4}" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def tex(s):
    """Turn the HTML-flavoured strings of report_values.py into LaTeX."""
    s = str(s).replace("&thinsp;", "\\,").replace("&nbsp;", "~").replace("&minus;", "$-$")
    s = s.replace("&ndash;", "--").replace("&lt;", "<")
    return re.sub(r"(?<!\\)%", r"\\%", s)


def values():
    v = {k: tex(x) for k, x in base_values().items()}
    ns = {"S": S, "money": money, "signed": signed, "pval": pval, "CELLS": CELLS, "ROOT": ROOT}
    exec((HERE / "report_values.py").read_text(encoding="utf-8-sig"), ns)
    v.update({k: tex(x) for k, x in ns["extra"](S).items()})
    pct = lambda var, c: 100 * S["pct_lp"][var][c]["mean"]
    for c in CELLS:
        cc = c.split("/")[0][0] + c.split("/")[1][0]
        s1 = (pct("101", c) + pct("111", c) - pct("100", c) - pct("110", c)) / 2
        v[f"sP1_{cc}"] = signed(s1, 1, "\\,pp")
        v[f"dwabs_{cc}"] = money(abs(S["dW"][c]["mean"]))
    others = [c for c in CELLS if c not in ("Low/Tight", "Medium/Tight")]
    v["n_p_beats_other"] = str(sum(pct("100", c) >= pct("111", c) for c in others))
    v["L"] = f"{S['tuning']['choice']['L']:,}"
    v.update(bench_table=bench_table(), incentive_table=incentive_table(),
             robust_table=robust_table())
    return v


def bench_table():
    order = ["000", "100", "010", "001", "110", "101", "011", "111", "random", "greedy", "lp_guided"]
    name = {"000": "Vanilla (0,0,0)", "100": "P (1,0,0)", "010": "C (0,1,0)", "001": "S (0,0,1)",
            "110": "P+C (1,1,0)", "101": "P+S (1,0,1)", "011": "C+S (0,1,1)",
            "111": "\\textbf{PACE} (1,1,1)", "random": "Random", "greedy": "Greedy (full info)",
            "lp_guided": "LP-guided (full info)"}
    best = {c: max(round(100 * S["pct_lp"][u][c]["mean"], 1) for u in order[:8]) for c in CELLS}
    se = max(100 * S["pct_lp"][u][c]["se"] for u in order for c in CELLS)
    rows = []
    for v in order:
        vals = [100 * S["pct_lp"][v][c]["mean"] for c in CELLS]
        cells = [f"\\textbf{{{x:.1f}}}" if v in order[:8] and round(x, 1) == best[c] else f"{x:.1f}"
                 for x, c in zip(vals, CELLS)]
        rows.append(f"{name[v]} & " + " & ".join(cells) + f" & {sum(vals) / 9:.1f} \\\\")
        if v == "111":
            rows.append("\\midrule")
    head = " & ".join(CODE[c] for c in CELLS)
    return ("\\begin{table}[h]\n\\centering\\small\n"
            f"\\caption{{Surplus as \\% of $\\WLP$, mean over 40 holdout seeds (standard errors at most "
            f"{se:.1f}\\,pp). Columns: contention L/M/H, then tightness S/M/T. Bold: best "
            "decentralized learner in the cell. The last three rows are references.}\n"
            "\\label{tab:bench}\n\\begin{tabular}{@{}l*{9}{r}r@{}}\n\\toprule\n"
            f"\\textbf{{Policy}} & {head} & Mean \\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


def mark(d):
    return " \\checkmark" if d["sig_pos"] else (" $\\times$" if d["sig_neg"] else "")


def incentive_table():
    rows = []
    for c in CELLS:
        dw = S["dW"][c]
        gi, gs = S["h5"]["exp"]["invasion"][c], S["h5"]["exp"]["stability"][c]
        cl = S["h5_class"][c]
        cat = cl["category"] + ("; stable" if cl["category"] == "Aligned" and cl["stable"] else "")
        ci = lambda d: f"{{\\scriptsize[{money(d['lo'])}, {money(d['hi'])}]}}"
        rows.append(f"{c} & {money(dw['mean'])}{mark(dw)} & {money(gi['mean'])} {ci(gi)}{mark(gi)} & "
                    f"{money(gs['mean'])} {ci(gs)}{mark(gs)} & {cat} \\\\")
    return ("\\begin{table}[h]\n\\centering\\small\n"
            "\\caption{Incentive study: mean unilateral gain $\\bar G$ from PACE over 40 seeds $\\times$ 5 "
            "deviants (dollars per buyer, 95\\% bootstrap interval) and the self-play gain $\\Delta W$ (all 50 "
            "buyers). \\checkmark/$\\times$: significantly positive/negative, one-sided, Bonferroni over 9 cells.}\n"
            "\\label{tab:inc}\n\\begin{tabular}{@{}lrrrl@{}}\n\\toprule\n"
            "\\textbf{Cell} & $\\Delta W$ & $\\bar G(\\mathrm{Van})$ invasion & $\\bar G(\\mathrm{PACE})$ "
            "stability & \\textbf{Category}\\\\\n\\midrule\n" + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


def robust_table():
    order = ["primary", "delay=2", "delay=5", "L=50", "L=800", "eta/2", "2eta",
             "gamma=0.5", "gamma=2", "prior=1", "prior=20"]
    label = {"primary": "Primary configuration", "delay=2": "Information delay $d=2$",
             "delay=5": "Information delay $d=5$", "L=50": "Window $L=50$",
             "L=800": "Window $L=800$", "eta/2": "Pacing step $\\eta^\\star/2$",
             "2eta": "Pacing step $2\\eta^\\star$", "gamma=0.5": "Congestion $\\gamma=0.5$",
             "gamma=2": "Congestion $\\gamma=2$", "prior=1": "Prior strength 1",
             "prior=20": "Prior strength 20"}
    rows = []
    for s in order:
        cells = []
        for c in ("High/Tight", "Low/Slack"):
            d = S["robust"][s][c]
            cells += [f"{money(d['mean'])} {{\\scriptsize[{money(d['lo'])}, {money(d['hi'])}]}}",
                      signed(100 * d["rel"], 1, "\\%")]
        rows.append(f"{label[s]} & " + " & ".join(cells) + " \\\\")
    return ("\\begin{table}[h]\n\\centering\\small\n"
            "\\caption{One-at-a-time robustness checks: $\\Delta W=W(\\mathrm{PACE})-W(\\mathrm{Van})$ on the "
            "first 20 holdout seeds (95\\% bootstrap interval) and relative to vanilla.}\n"
            "\\label{tab:rob}\n\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n"
            "\\textbf{Setting} & High/Tight: $\\Delta W$ & rel. & Low/Slack: $\\Delta W$ & rel.\\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")


def build():
    v = values()
    body = (HERE / "main_template.tex").read_text(encoding="utf-8")
    for _ in range(2):
        body = re.sub(r"@@(\w+)@@", lambda m: v.get(m.group(1), m.group(0)), body)
    left = sorted(set(re.findall(r"@@(\w+)@@", body)))
    if left:
        sys.exit(f"unfilled placeholders: {left}")
    (HERE / "main.tex").write_text(body, encoding="utf-8")
    (HERE / "figures").mkdir(exist_ok=True)
    for name in FIGURES:
        shutil.copy(ROOT / "figures" / f"{name}.pdf", HERE / "figures" / f"{name}.pdf")
    if "--no-compile" in sys.argv:          # e.g. to upload main.tex + figures to Overleaf
        return
    exe = os.environ.get("TECTONIC", "tectonic")
    r = subprocess.run([exe, "--keep-logs", "main.tex"], cwd=HERE, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-3000:])
    if r.returncode:
        sys.exit("compile failed")


if __name__ == "__main__":
    build()
