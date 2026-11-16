"""
paper_numerics.py -- reproduces every table and figure of the paper
    "An optimal quadrature formula with derivative exact for exponential functions
     in the Hilbert space W_2^{(3,2)}(0,1)".

Run:  python paper_numerics.py      (about 1-2 minutes)
Output: tables/*.tex (table bodies) and figures/*.pdf
Requires: mpmath, sympy, matplotlib, and the modules kvadratura.py, sigma_tahlil.py.
"""
import os
import sys
import mpmath as mp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kvadratura as K                      # closed-form coefficients, classical rules, Funksiya
import sigma_tahlil as S                    # direct solution of the linear system, norm of the functional

mp.mp.dps = 40
os.makedirs(os.path.join(HERE, "..", "tables"), exist_ok=True)
os.makedirs(os.path.join(HERE, "..", "figures"), exist_ok=True)


def sci(v, n=3):
    v = mp.mpf(v)
    if v == 0:
        return "0"
    s = mp.nstr(v, n, min_fixed=1, max_fixed=0)
    if "e" not in s:
        return s
    m, k = s.split("e")
    return f"{m}\\cdot 10^{{{int(k)}}}"


def save(name, lines):
    open(os.path.join(HERE, "..", "tables", name), "w").write("\n".join(lines) + "\n")


# ---------------------------------------------------------------- Table: coefficients
s, N = mp.mpf(1), 10
Cy, Pr = S.koeff_yopiq(s, N)
Ce = S.koeff_etalon(s, N)
save("tab_coeff.tex", [f"{b} & ${sci(Cy[b], 16)}$ & ${sci(abs(Cy[b] - Ce[b]), 2)}$ \\\\" for b in range(N + 1)])
info = dict(lam=mp.nstr(Pr["lam"], 12), H1=sci(Pr["H1"], 10),
            sumC=sci(abs(sum(Cy)), 2),
            sumCe=sci(abs(sum(Cy[j] * mp.e**(-s * j / N) for j in range(N + 1)) - Pr["H1"]), 2),
            maxdiff=sci(max(abs(a - b) for a, b in zip(Cy, Ce)), 2))
print("coefficients:", info)

# ---------------------------------------------------------------- Table: exactness
rows = []
for s in [-2, 0.5, 1, 3, 10]:
    s = mp.mpf(s)
    for N in [1, 4, 16]:
        h = mp.mpf(1) / N
        C = K.C1(s, N)
        Q = lambda f, df: sum(K.C0(j, N) * f(h * j) for j in range(N + 1)) + sum(C[j] * df(h * j) for j in range(N + 1))
        Ie = (1 - mp.e**(-s)) / s
        errs = [1 - Q(lambda t: 1, lambda t: 0),
                mp.mpf(1) / 2 - Q(lambda t: t, lambda t: 1),
                Ie - Q(lambda t: mp.e**(-s * t), lambda t: -s * mp.e**(-s * t)),
                (2 + 5 * Ie) - Q(lambda t: 3 - 2 * t + 5 * mp.e**(-s * t), lambda t: -2 - 5 * s * mp.e**(-s * t))]
        F = dict(f=lambda t, s=s: mp.e**(-s * t), df=lambda t, s=s: -s * mp.e**(-s * t), I=Ie)
        simp = (Ie - K.simpson(F, N)) if N % 2 == 0 else None
        lab = f"${float(s):g}$" if N == 1 else ""
        rows.append(f"{lab} & {N} & ${sci(max(abs(e) for e in errs), 2)}$ & "
                    f"${sci(abs(Ie - K.trapetsiya(F, N)), 2)}$ & "
                    f"{'--' if simp is None else '$' + sci(abs(simp), 2) + '$'} & ${sci(abs(Ie - K.eyler_makloren(F, N)), 2)}$ \\\\")
    rows.append("\\midrule")
save("tab_exact.tex", rows[:-1])

# ---------------------------------------------------------------- Tables: exponential and trigonometric functions
EXP = ["exp(-0.5*x)", "exp(-2*x)", "exp(-20*x)", "exp(3*x)", "exp(-x) + exp(-3*x)", "cosh(2*x)", "x*exp(-2*x)",
       "(1 + x**2)*exp(-x)", "exp(-x**2)", "1/(1 + exp(-5*x))"]
TRIG = ["sin(x)", "cos(x)", "cos(2*x)", "sin(x)**2", "cos(5*x)", "exp(-x)*sin(5*x)", "exp(-2*x)*cos(3*x)",
        "exp(-3*x)*sin(10*x)", "cos(x) + exp(-2*x)", "sin(3*x)", "sin(20*x)", "sin(pi*x)"]


def latex_name(e):
    import sympy as sp
    return "$" + sp.latex(K.Funksiya(e)["ifoda"]) + "$"


def method_table(funcs, N, fname):
    """Columns: f | sigma* | optimal(sigma*) | Simpson | E-M | sigma_E^2 | fitted E-M(sigma_E^2)."""
    out = []
    for e in funcs:
        F = K.Funksiya(e)
        ss = F["sigma_nazariy"]
        sig = ss if ss is not None else mp.mpf(1)
        s2 = F["sigma2_E"]
        eo = abs(F["I"] - K.optimal(F, N, sig))
        es = abs(F["I"] - K.simpson(F, N))
        ee = abs(F["I"] - K.eyler_makloren(F, N))
        ef = abs(F["I"] - K.em_fitted(F, N, s2 if s2 is not None else 0))
        tiny = mp.mpf(10)**-35
        best = min(eo, es, ee, ef)
        def f(v):
            if v < tiny:
                return "\\mathbf{exact}" if best < tiny else "exact"
            return ("\\mathbf{" + sci(v) + "}") if v == best else sci(v)
        sig_s = "--" if ss is None else f"${float(ss):.4g}$"
        s2_s = "--" if s2 is None else f"${float(s2):.4g}$"
        out.append(f"{latex_name(e)} & {sig_s} & ${f(eo)}$ & ${f(es)}$ & ${f(ee)}$ & {s2_s} & ${f(ef)}$ \\\\")
    save(fname, out)


method_table(EXP, 16, "tab_exp.tex")
method_table(TRIG, 16, "tab_trig.tex")

# ---------------------------------------------------------------- Table: h^5 term after tuning (Theorem 7)
rows = []
r3 = mp.sqrt(3)
for e in ["exp(-x) + exp(-3*x)", "x*exp(-2*x)", "exp(-x**2)", "sin(x)", "exp(-2*x)*cos(3*x)", "cos(x) + exp(-2*x)"]:
    import sympy as sp
    F = K.Funksiya(e)
    ex = F["ifoda"]
    ss = F["sigma_nazariy"]
    d2 = sp.lambdify(K.x, sp.diff(ex, K.x, 2), "mpmath")
    d3 = sp.lambdify(K.x, sp.diff(ex, K.x, 3), "mpmath")
    L = lambda t: d3(t) + ss * d2(t)
    cells = []
    for N in (16, 64):
        h = mp.mpf(1) / N
        R = F["I"] - K.optimal(F, N, ss)
        pred = -ss * h**5 / (1440 * r3) * (L(0) + L(1))
        cells += [f"${sci(R)}$", f"{float(R / pred):.4f}"]
    rows.append(f"{latex_name(e)} & ${float(ss):.4g}$ & " + " & ".join(cells) + " \\\\")
save("tab_h5.tex", rows)

# ---------------------------------------------------------------- Table: norm of the error functional
rows = []
for s in (1, 3):
    s = mp.mpf(s)
    prev = None
    for N in (4, 8, 16, 32, 64, 128):
        h = mp.mpf(1) / N
        n_opt = mp.sqrt(S.norma2(s, N, K.C1(s, N)))
        c = K.c_sigma(s**2, N)
        Cem = [mp.mpf(0)] * (N + 1)
        Cem[0], Cem[N] = c, -c
        n_em = mp.sqrt(S.norma2(s, N, Cem))
        p = "--" if prev is None else f"{float(mp.log(prev / n_opt, 2)):.4f}"
        lab = f"${int(s)}$" if N == 4 else ""
        rows.append(f"{lab} & {N} & ${sci(n_opt, 5)}$ & {p} & {mp.nstr(mp.sqrt(30240) * N**3 * n_opt, 7)} & "
                    f"{mp.nstr(1 / mp.sqrt(1 + s**2 * h**2 / 40), 7)} & {mp.nstr(n_em / n_opt, 7)} \\\\")
        prev = n_opt
    rows.append("\\midrule")
save("tab_norm.tex", rows[:-1])

# ---------------------------------------------------------------- Table: comparison with classical rules, sigma = 1
rows = []
for e, nm in [("exp(x)", "$e^{x}$"), ("log(1+x)", "$\\ln(1+x)$"), ("exp(-x)*cos(10*x)", "$e^{-x}\\cos 10x$"), ("x**(5/2)", "$x^{5/2}$")]:
    F = K.Funksiya(e)
    errs = {k: [] for k in ("o", "t", "s", "e")}
    Ns = (8, 16, 32, 64)
    for j, N in enumerate(Ns):
        vals = (abs(F["I"] - K.optimal(F, N, 1)), abs(F["I"] - K.trapetsiya(F, N)),
                abs(F["I"] - K.simpson(F, N)), abs(F["I"] - K.eyler_makloren(F, N)))
        for k, v in zip(("o", "t", "s", "e"), vals):
            errs[k].append(v)
        lab = f"\\multirow{{4}}{{*}}{{{nm}}}" if j == 0 else ""
        rows.append(f"{lab} & {N} & " + " & ".join(f"${sci(v)}$" for v in vals) + " \\\\")
    rows.append(" & $p$ & " + " & ".join(f"{float(mp.log(errs[k][-2] / errs[k][-1], 2)):.2f}" for k in ("o", "t", "s", "e")) + " \\\\")
    rows.append("\\midrule")
save("tab_classic.tex", rows[:-1])

# ---------------------------------------------------------------- Table: leading term check
rows = []
for e in ["x**4", "1/(1+x)", "sqrt(1+x)", "log(1+x)", "sin(x)", "exp(-x)*cos(x)", "besselj(0, 5*x)"]:
    import sympy as sp
    F = K.Funksiya(e)
    ex = F["ifoda"]
    d2 = sp.lambdify(K.x, sp.diff(ex, K.x, 2), "mpmath")
    d3 = sp.lambdify(K.x, sp.diff(ex, K.x, 3), "mpmath")
    D2, D3 = d2(1) - d2(0), d3(1) - d3(0)
    r = []
    for s in (mp.mpf("0.5"), mp.mpf(1), mp.mpf(2)):
        pred = (D3 + s * D2) / 720 / mp.mpf(64)**4
        r.append(f"{float((F['I'] - K.optimal(F, 64, s)) / pred):.4f}")
    rows.append(f"{latex_name(e)} & ${float(D2):.4g}$ & ${float(D3):.4g}$ & {r[0]} & {r[1]} & {r[2]} & ${float(-D3 / D2):.4g}$ \\\\")
save("tab_leading.tex", rows)

# ---------------------------------------------------------------- Figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
K.uslub()
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "figure.dpi": 150})
COL, MK = K.RANG, K.MARKER

# Fig. 1: |C_{beta,1}|, N = 20
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for k, s in enumerate([1, 3, -2]):
    C = K.C1(s, 20)
    ax.semilogy(range(21), [float(abs(c)) for c in C], color=COL[k], marker=MK[k], ms=4, label=f"$\\sigma={s}$")
ax.set_xlabel("$\\beta$"); ax.set_ylabel("$|C_{\\beta,1}|$"); ax.legend()
ax.set_title("Optimal coefficients, $N=20$")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "figures", "fig_coeff.pdf"))

# Fig. 2: exactness only for the own sigma
N = 8
taus = [mp.mpf(k) / 10 for k in range(-60, 61)]
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for k, s in enumerate([-2, 1, 3]):
    y = []
    for t in taus:
        I = (1 - mp.e**(-t)) / t if t != 0 else mp.mpf(1)
        F = dict(f=lambda z, t=t: mp.e**(-t * z), df=lambda z, t=t: -t * mp.e**(-t * z), I=I)
        y.append(max(float(abs(I - K.optimal(F, N, s))), 1e-42))
    ax.semilogy([float(t) for t in taus], y, color=COL[k], label=f"formula with $\\sigma={s}$")
ax.set_xlabel("$\\tau$  (integrand $e^{-\\tau x}$)"); ax.set_ylabel("$|R_N(e^{-\\tau x})|$")
ax.set_title(f"Error for $f(x)=e^{{-\\tau x}}$, $N={N}$"); ax.legend(loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "figures", "fig_tau.pdf"))

# Fig. 3: convergence
sel = ["exp(-x) + exp(-3*x)", "x*exp(-2*x)", "exp(-x**2)", "sin(x)", "exp(-2*x)*cos(3*x)", "cos(x) + exp(-2*x)"]
Ns = [4, 8, 16, 32, 64]
COL5, MK5 = COL + ["#7b4fc8"], MK + ["v"]
fig, axs = plt.subplots(2, 3, figsize=(10, 5.8))
for ax, e in zip(axs.flat, sel):
    F = K.Funksiya(e)
    ss = F["sigma_nazariy"]
    s2 = F["sigma2_E"]
    ser = [(f"optimal, $\\sigma^*={float(ss):.3g}$", lambda n: K.optimal(F, n, ss)),
           ("trapezoidal", lambda n: K.trapetsiya(F, n)), ("Simpson", lambda n: K.simpson(F, n)),
           ("Euler--Maclaurin", lambda n: K.eyler_makloren(F, n)),
           (f"fitted E--M, $\\sigma_E^2={float(s2):.3g}$", lambda n: K.em_fitted(F, n, s2))]
    for k, (lab, fn) in enumerate(ser):
        ys = [float(abs(F["I"] - fn(n))) for n in Ns]
        if max(ys) < 1e-30:          # exact rule: do not squash the axis, mark it in the legend
            ax.plot([], [], color=COL5[k], marker=MK5[k], ms=3.5, label=lab + " (exact)")
            continue
        ax.loglog(Ns, ys, color=COL5[k], marker=MK5[k], ms=3.5, label=lab)
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_title(latex_name(e), fontsize=10); ax.set_xlabel("$N$"); ax.legend(fontsize=6.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "figures", "fig_conv.pdf"))

# Fig. 4: leading term law
N = 32
ss_ = [mp.mpf(k) / 2 for k in range(-12, 13) if k != 0]
fig, ax = plt.subplots(figsize=(6.4, 3.8))
import sympy as sp
for k, e in enumerate(["1/(1+x)", "x**4", "sin(x)", "exp(-x)*cos(x)"]):
    F = K.Funksiya(e)
    ex = F["ifoda"]
    d2 = sp.lambdify(K.x, sp.diff(ex, K.x, 2), "mpmath"); d3 = sp.lambdify(K.x, sp.diff(ex, K.x, 3), "mpmath")
    D2, D3 = d2(1) - d2(0), d3(1) - d3(0)
    ax.plot([float(v) for v in ss_], [float(D3 + v * D2) for v in ss_], color=COL[k], lw=1.1, ls="--")
    ax.scatter([float(v) for v in ss_], [float(720 * N**4 * (F["I"] - K.optimal(F, N, v))) for v in ss_],
               s=14, color=COL[k], marker=MK[k], label=latex_name(e), zorder=3)
ax.axhline(0, color="#52514e", lw=0.8)
ax.set_xlabel("$\\sigma$"); ax.set_ylabel("$720\\,N^4 R_N(f)$")
ax.set_title("Markers: computed; dashed: $[f'''+\\sigma f'']_0^1$ ($N=32$)", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "..", "figures", "fig_leading.pdf"))
print("done")
