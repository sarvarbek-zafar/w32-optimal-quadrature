# Optimal quadrature formula with derivative in $W_2^{(3,2)}(0,1)$: code and data

This repository contains the code and data for the paper

> A. R. Hayotov, S. Z. Polvonov, *An exponentially fitted optimal quadrature formula with first derivative in the space $W_2^{(3,2)}(0,1)$*, submitted to *Calcolo* (2026).

The paper studies the quadrature formula

$$\int_0^1 \varphi(x)\,dx \cong \sum_{\beta=0}^{N} C_{\beta,0}\,\varphi(h\beta) + \sum_{\beta=0}^{N} C_{\beta,1}\,\varphi'(h\beta), \qquad h = 1/N,$$

in the Hilbert space with the norm $\|\varphi\|^2=\int_0^1(\varphi'''+\sigma\varphi'')^2\,dx$, $\sigma\neq0$.
The weights $C_{\beta,0}$ are those of the trapezoidal rule. The derivative weights $C_{\beta,1}$ are optimal in the sense of Sard and are given in closed form (Theorem 3 of the paper).

## Reproducing the results of the paper

```bash
pip install -r requirements.txt
cd code
python paper_numerics.py     # writes ../tables/*.tex and ../figures/*.pdf (about 20 s)
python verify_paper.py       # independent check of the formulas; ends with "ALL OK" (about 20 s)
```

Both scripts use `mpmath` with at least 40 significant digits, so the results do not depend on the platform.
The files in `tables/` and `figures/` are exactly the files produced by `paper_numerics.py` and used in the paper.

| Output | Used in the paper |
|---|---|
| `tables/tab_coeff.tex` | Table 1: optimal coefficients for $\sigma=1$, $N=10$ and their deviation from the direct solution |
| `tables/tab_norm.tex` | Table 2: norm of the error functional, observed order, comparison with the Euler–Maclaurin-type formula |
| `tables/tab_leading.tex` | Table 3: check of the leading error term (Conjecture 2) |
| `tables/tab_sigmaR.tex` | Table 4: the parameter $\sigma_R(N)$ and observed orders |
| `tables/tab_classic.tex` | Table 5: comparison with the Simpson, two-point Gauss, Euler–Maclaurin and Boole rules |
| `tables/tab_exact.tex` | Data for Section 11.2 (exactness for $1$, $x$, $e^{-\sigma x}$) |
| `figures/fig_coeff.pdf` | Figure 1: optimal coefficients |
| `figures/fig_leading.pdf` | Figure 2: leading error term as a function of $\sigma$ |
| `figures/fig_conv.pdf` | Figure 3: convergence for integrands of non-exponential type |
| `figures/fig_tau.pdf` | Additional figure: error for $e^{-\tau x}$ with fixed $\sigma$ (not used in the paper) |

## Code

| File | Contents |
|---|---|
| `code/kvadratura.py` | Optimal coefficients (Theorem 3), the classical rules (trapezoidal, Simpson, two-point Gauss, Euler–Maclaurin, Boole), the parameter $\sigma_R(N)$ of Section 11.4, test functions |
| `code/sigma_tahlil.py` | Direct solution of the linear system (5.1)–(5.3) and the norm of the error functional (Lemma 1) |
| `code/paper_numerics.py` | Generates all tables and figures of the paper |
| `code/verify_paper.py` | Implements the formulas of the paper independently and compares them with the direct solution of the system for several values of $\sigma$ and $N$ |
| `code/taqqoslash.py` | Companion tool: compares the optimal formula with classical rules on any integrand you type in |
| `docs/` | Companion tool: online calculator (GitHub Pages) |

Quick start in Python:

```python
import mpmath as mp
import kvadratura as K

sigma, N = 1, 16
C1 = K.C1(sigma, N)                     # optimal derivative weights C_{beta,1} (Theorem 3)
F  = K.Funksiya("log(1+x)")             # integrand, its derivative and the exact integral
s  = K.sigma_runge(F, N)                # sigma_R(N), Section 11.4
print(mp.nstr(F["I"] - K.optimal(F, N, s), 5))   # error of the optimal formula
```

## Companion tools

These tools are not needed to reproduce the paper. Their interface is in Uzbek.

* **Online calculator:** https://sarvarbek-zafar.github.io/w32-optimal-quadrature/ .
  Type an integrand and get the errors of several rules, convergence orders and plots. It runs in the browser in double precision.
* **`code/taqqoslash.py`:** the same comparison with 40-digit arithmetic, CSV output and plots, for example
  `python taqqoslash.py "log(1+x)" --N 8 16 32 64`.

Both tools also contain a fitted Euler–Maclaurin rule and a simplified, equivalent form of the coefficient formulas. These are not part of the submitted paper.
The state of the repository before it was aligned with the submitted paper is kept in the branch [`extended-version`](https://github.com/sarvarbek-zafar/w32-optimal-quadrature/tree/extended-version).

## Numbering

Theorem, equation, table and section numbers in the code refer to the manuscript submitted to *Calcolo* (November 2026).

## How to cite

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
