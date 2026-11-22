# Optimal quadrature formula with derivative in $W_2^{(3,2)}(0,1)$ — numerical experiments

Reproducible code for the paper

> A. R. Hayotov, S. Z. Polvonov, *An optimal quadrature formula with derivative exact for exponential functions in the Hilbert space $W_2^{(3,2)}$* (2026, submitted).

The paper studies the formula

$$\int_0^1 \varphi(x)\,dx \cong \sum_{\beta=0}^{N} C_{\beta,0}\,\varphi(h\beta) + \sum_{\beta=0}^{N} C_{\beta,1}\,\varphi'(h\beta), \qquad h = 1/N,$$

in the Hilbert space with the norm $\|\varphi\|^2=\int_0^1(\varphi'''+\sigma\varphi'')^2dx$. The weights $C_{\beta,0}$ are those of the trapezoidal rule; the derivative weights $C_{\beta,1}$ are optimal in the sense of Sard and are given in closed form.

## Online calculator

**https://sarvarbek-zafar.github.io/w32-optimal-quadrature/**

A web page for non-specialists (in Uzbek): type any integrand f(x), see it typeset as a formula,
and get the exact integral, the errors of the trapezoidal, Simpson, Euler–Maclaurin, fitted
Euler–Maclaurin and optimal rules for N = 4, 8, …, convergence orders, four plots and a plain-language
conclusion (which rule wins and why, and whether the optimal formula suits this integrand).

The page lives in `docs/` (`index.html`, `kvadratura.js` with the formulas of Theorem 4, `app.js`
with the interface) and runs entirely in the browser in double precision; errors below about
1e-14 are shown as ≈ 0. For 40-digit results use `code/taqqoslash.py`. The JavaScript coefficients
agree with the Python implementation to about 2e-15.

## Contents

| Path | Contents |
|---|---|
| `code/kvadratura.py` | Closed-form optimal coefficients (Theorems 3 and 4), the fitted Euler–Maclaurin rule, classical rules, test functions |
| `code/sigma_tahlil.py` | Direct solution of the linear system (5.1)–(5.3), norm of the error functional |
| `code/paper_numerics.py` | Regenerates every table (`tables/*.tex`, Tables 1–8) and figure (`figures/*.pdf`, Figures 1–4) of the paper, about 15 s |
| `code/verify_paper.py` | Independent high-precision check (40–60 digits) of every formula of the paper against the direct solution of the system, about 40 s |
| `code/taqqoslash.py` | Compares the optimal formula with the trapezoidal, Simpson, Euler–Maclaurin and fitted Euler–Maclaurin rules (optionally Gauss–Legendre) on **any integrand you type in**: errors, observed orders, check of Theorem 7, CSV table, plots and an automatic conclusion |
| `tables/`, `figures/` | Output of `paper_numerics.py`, as used in the paper |
| `docs/` | Online calculator (GitHub Pages): `index.html`, `kvadratura.js`, `app.js` |

## Usage

```bash
pip install -r requirements.txt
cd code
python paper_numerics.py     # writes ../tables/*.tex and ../figures/*.pdf
python verify_paper.py       # prints OK/FAIL for each formula; ends with "ALL OK"
```

Quick start in Python:

```python
import mpmath as mp
import kvadratura as K

sigma, N = 1, 16
C1 = K.C1_sodda(sigma, N)        # optimal derivative weights C_{beta,1}, Theorem 4
F  = K.Funksiya("log(1+x)")      # integrand with exact integral and sigma*
print(mp.nstr(F["I"] - K.optimal(F, N, F["sigma_nazariy"]), 5))   # error with sigma = sigma*
```

### Your own integrand

```bash
cd code
python taqqoslash.py "exp(-x)*cos(3*x)"                 # sigma = sigma* chosen automatically
python taqqoslash.py "log(1+x)" --sigma 1 --N 4 8 16 32 64 128
python taqqoslash.py "1/(1+25*x**2)" --oraliq -1 1 --gauss
python taqqoslash.py                                    # asks for f(x)
```

The script prints a table of errors |R_N| with observed orders for every method, the
parameters sigma* = -[f''']/[f''] and sigma_E^2 = [f''']/[f'], and a check of Theorem 7
(720 N^4 R_N -> [f''' + sigma f'']_0^1, or the h^5 term when sigma = sigma*). It ends with a **conclusion** (also saved as `xulosa.txt`): a ranking of the methods, which rule
integrates f exactly (Theorem 5, Proposition 2), whether the observed orders match the theory
(a drop signals insufficient smoothness), whether the optimal formula suits this integrand and,
if not, why (no sigma*, too large sigma*, singular f''/f''' at an end point, no gain over
Euler-Maclaurin), and why the winning method wins. It writes
`natijalar/<function>/xatoliklar.csv` and a four-panel figure `taqqoslash.png` / `.pdf`:
the integrand and its derivative, convergence of all methods, the error as a function of
sigma, and the optimal coefficients C_{beta,1}. Run `python taqqoslash.py -h` for all options.
The integrand is parsed with sympy (`exp, log, sin, cos, sqrt, sinh, cosh, atan, erf, pi, ...`,
powers with `**`); f and f' must be finite at the end points. On an interval [a, b] the
substitution x = a + (b - a) t is used, and sigma refers to the variable t.

All computations use `mpmath` with at least 40 significant digits; results do not depend on the platform.

## Numbering

Theorem, equation and table numbers in the code refer to the numbering of the manuscript (version 3, November 2026).

## License

MIT, see `LICENSE`.
