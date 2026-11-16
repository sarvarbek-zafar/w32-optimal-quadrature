# Optimal quadrature formula with derivative in $W_2^{(3,2)}(0,1)$ — numerical experiments

Reproducible code for the paper

> A. R. Hayotov, S. Z. Polvonov, *An optimal quadrature formula with derivative exact for exponential functions in the Hilbert space $W_2^{(3,2)}$* (2026, submitted).

The paper studies the formula

$$\int_0^1 \varphi(x)\,dx \cong \sum_{\beta=0}^{N} C_{\beta,0}\,\varphi(h\beta) + \sum_{\beta=0}^{N} C_{\beta,1}\,\varphi'(h\beta), \qquad h = 1/N,$$

in the Hilbert space with the norm $\|\varphi\|^2=\int_0^1(\varphi'''+\sigma\varphi'')^2dx$. The weights $C_{\beta,0}$ are those of the trapezoidal rule; the derivative weights $C_{\beta,1}$ are optimal in the sense of Sard and are given in closed form.

## Contents

| Path | Contents |
|---|---|
| `code/kvadratura.py` | Closed-form optimal coefficients (Theorems 3 and 4), the fitted Euler–Maclaurin rule, classical rules, test functions |
| `code/sigma_tahlil.py` | Direct solution of the linear system (5.1)–(5.3), norm of the error functional |
| `code/paper_numerics.py` | Regenerates every table (`tables/*.tex`, Tables 1–8) and figure (`figures/*.pdf`, Figures 1–4) of the paper, about 15 s |
| `code/verify_paper.py` | Independent high-precision check (40–60 digits) of every formula of the paper against the direct solution of the system, about 40 s |
| `tables/`, `figures/` | Output of `paper_numerics.py`, as used in the paper |

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

All computations use `mpmath` with at least 40 significant digits; results do not depend on the platform.

## Numbering

Theorem, equation and table numbers in the code refer to the numbering of the manuscript (version 3, November 2026).

## License

MIT, see `LICENSE`.
