"""
paper_numerics.py -- reproduces every table and figure of the paper
    "An optimal quadrature formula with derivative exact for exponential functions
     in the Hilbert space W_2^{(3,2)}(0,1)".

Run:  python paper_numerics.py      (a few minutes)
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

# ---------------------------------------------------------------- Table: norm of the error functional
rows = []
for s in [1, 3]:
    prev = None
    for N in [4, 8, 16, 32, 64, 128]:
        C = K.C1(s, N)
        n_opt = mp.sqrt(S.norma2(s, N, C))
        Cem = [mp.mpf(0)] * (N + 1)
        H1 = sum(C[j] * mp.e**(-mp.mpf(s) * j / N) for j in range(N + 1))          # = H_1 by (3.7)
        Cem[0] = H1 / (1 - mp.e**(-mp.mpf(s)))
        Cem[N] = -Cem[0]
        n_em = mp.sqrt(S.norma2(s, N, Cem))
        p_ = "--" if prev is None else f"{float(mp.log(prev / n_opt, 2)):.4f}"
        rows.append(f"{'$' + str(s) + '$' if N == 4 else ''} & {N} & ${sci(n_opt, 5)}$ & {p_} & "
                    f"{mp.nstr(mp.sqrt(30240) * N**3 * n_opt, 7)} & {mp.nstr(n_em / n_opt, 7)} \\\\")
        prev = n_opt
    rows.append("\\midrule")
save("tab_norm.tex", rows[:-1])


def latex_name(e):
    import sympy as sp
    return "$" + sp.latex(K.Funksiya(e)["ifoda"]) + "$"


# ---------------------------------------------------------------- Table: comparison with classical rules
# Integrands that are NOT of exponential type (polynomials, rational, algebraic, logarithmic functions).
# sigma_R is found from the values of f and f' at the nodes only (root of Q_N - Q_{N/2}).
CMP = ["x**4", "x**6", "1/(1+x)", "1/(1+x)**2", "sqrt(1+x)", "(1+x)**Rational(3,2)", "log(1+x)", "x*log(1+x)",
       "atan(x)", "tan(x)", "x**Rational(7,2)", "1/(1+25*x**2)"]
NAME = {"x**Rational(7,2)": "$x^{7/2}$", "log(1+x)": "$\\ln(1+x)$", "atan(x)": "$\\arctan x$", "tan(x)": "$\\tan x$",
        "1/(1+x)": "$1/(1+x)$", "sqrt(1+x)": "$\\sqrt{1+x}$", "1/(1+25*x**2)": "$1/(1+25x^2)$", "1/(1+x**2)": "$1/(1+x^2)$",
        "x**2*(1-x)**2": "$x^2(1-x)^2$", "x**4": "$x^4$", "x**6": "$x^6$",
        "1/(1+x)**2": "$1/(1+x)^2$", "(1+x)**Rational(3,2)": "$(1+x)^{3/2}$", "x*log(1+x)": "$x\\ln(1+x)$"}
RULES = [("Simpson", K.simpson), ("Gauss-2", K.gauss2), ("E--M", K.eyler_makloren), ("Boole", K.boole)]
NC = 32
rat = lambda v: f"{float(v):.3g}" if v < 1000 else str(int(float(mp.nstr(v, 3))))
rows = []
for e in CMP:
    F = K.Funksiya(e)
    ss = F["sigma_nazariy"]
    sr = K.sigma_runge(F, NC)
    errs = [abs(F["I"] - K.optimal(F, NC, sr))] + [abs(F["I"] - fn(F, NC)) for _, fn in RULES]
    best = min(errs)
    cell = lambda v: "exact" if v < mp.mpf(10)**-35 else ("$\\mathbf{" + sci(v) + "}$" if v == best else "$" + sci(v) + "$")
    if best < mp.mpf(10)**-35:
        cell = lambda v: "\\textbf{exact}" if v < mp.mpf(10)**-35 else "$" + sci(v) + "$"
    sstar = "--" if ss is None else f"${float(ss):.4g}$"
    r16, r64 = [], []
    for n_, store in ((16, r16), (64, r64)):
        s_ = K.sigma_runge(F, n_)
        store.append(abs(F["I"] - K.eyler_makloren(F, n_)) / abs(F["I"] - K.optimal(F, n_, s_)))
    rows.append(f"{NAME[e]} & ${float(sr):.4g}$ & " + " & ".join(cell(v) for v in errs)
                + f" & {rat(r16[0])} & {rat(errs[3] / errs[0])} & {rat(r64[0])} \\\\")
save("tab_classic.tex", rows)

# sigma_R(N) -> sigma*, and the observed order of the self-tuned formula
rows = []
for e in ["x**6", "1/(1+x)", "sqrt(1+x)", "log(1+x)", "x**Rational(7,2)"]:
    F = K.Funksiya(e)
    Ns = [8, 16, 32, 64, 128]
    sr = [K.sigma_runge(F, n) for n in Ns]
    er = [abs(F["I"] - K.optimal(F, n, s_)) for n, s_ in zip(Ns, sr)]
    em = [abs(F["I"] - K.eyler_makloren(F, n)) for n in Ns]
    rows.append(f"{NAME[e]} & ${float(F['sigma_nazariy']):.4g}$ & " + " & ".join(f"${float(v):.4f}$" for v in sr[1:])
                + f" & {float(mp.log(er[-2] / er[-1], 2)):.2f} & {float(mp.log(em[-2] / em[-1], 2)):.2f} \\\\")
save("tab_sigmaR.tex", rows)

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

# Fig. 3: convergence on non-exponential integrands, sigma = sigma_R(N)
sel = ["x**6", "1/(1+x)", "log(1+x)", "sqrt(1+x)", "x**Rational(7,2)", "1/(1+25*x**2)"]
Ns = [8, 16, 32, 64, 128]
COL5 = ["#2a78d6", "#eb6834", "#eda100", "#1baf7a", "#7b5bd6"]
MK5 = ["o", "s", "^", "D", "v"]
fig, axs = plt.subplots(2, 3, figsize=(10, 6.0))
for ax, e in zip(axs.flat, sel):
    F = K.Funksiya(e)
    ser = [("optimal, $\\sigma=\\sigma_R(N)$", lambda n: K.optimal(F, n, K.sigma_runge(F, n))),
           ("Simpson", lambda n: K.simpson(F, n)),
           ("Gauss-2", lambda n: K.gauss2(F, n)), ("Euler--Maclaurin", lambda n: K.eyler_makloren(F, n)),
           ("Boole", lambda n: K.boole(F, n))]
    for k, (lab, fn) in enumerate(ser):
        ax.loglog(Ns, [float(abs(F["I"] - fn(n))) for n in Ns], color=COL5[k], marker=MK5[k], ms=3.5,
                  lw=1.8 if k == 0 else 1.1, label=lab)
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_title(NAME[e], fontsize=10); ax.set_xlabel("$N$")
axs.flat[0].set_ylabel("$|R_N(\\varphi)|$"); axs.flat[3].set_ylabel("$|R_N(\\varphi)|$")
hd, lb = axs.flat[0].get_legend_handles_labels()
fig.legend(hd, lb, loc="lower center", ncol=5, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(os.path.join(HERE, "..", "figures", "fig_conv.pdf"))

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
