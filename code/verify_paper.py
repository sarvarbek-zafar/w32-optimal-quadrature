"""
verify_paper.py -- independent numerical check of the formulas of paper.tex for several (sigma, N).
It complements, but does not replace, the proofs given in the paper.
The formulas are transcribed from the paper itself (not imported from earlier modules);
the reference is the direct LU solution of system (5.1)-(5.3) assembled from the definitions.
"""
import sys
import mpmath as mp

mp.mp.dps = 50


def sgn(x):
    return (x > 0) - (x < 0)


def G3(x, s):
    return sgn(x) / (2 * s**5) * (mp.sinh(s * x) - s * x - (s * x)**3 / 6)


def dG3(x, s):
    return sgn(x) / (2 * s**4) * (mp.cosh(s * x) - 1 - (s * x)**2 / 2)


def G2(x, s):
    return sgn(x) / (2 * s**3) * (mp.sinh(s * x) - s * x)


def Phi1(t, s):
    return (mp.cosh(s * t) - 1 - (s * t)**2 / 2 - (s * t)**4 / 24) / (2 * s**6)


def Phi2(t, s):
    return (mp.sinh(s * t) - s * t - (s * t)**3 / 6 - (s * t)**5 / 120) / (2 * s**7)


def C0(b, N):
    h = mp.mpf(1) / N
    return h / 2 if b in (0, N) else h


# ------------------------------------------------------------------ definitions
def F1_def(b, s, N):
    h = mp.mpf(1) / N
    X = h * b
    F11 = mp.quad(lambda x: dG3(x - X, s), [0, X, 1])          # integral by quadrature
    F12 = sum(C0(g, N) * dG3(h * b - h * g, s) for g in range(N + 1))
    return F11 + F12, F11, F12


def H1_def(s, N):
    h = mp.mpf(1) / N
    return sum(C0(b, N) * mp.e**(-s * h * b) for b in range(N + 1)) / s + (mp.e**(-s) - 1) / s**2


def direct(s, N):
    """System (5.1)-(5.3) exactly as written; unknowns C_0..C_N, A, B."""
    h = mp.mpf(1) / N
    n = N + 3
    A = mp.matrix(n, n)
    r = mp.matrix(n, 1)
    for i in range(N + 1):
        for j in range(N + 1):
            A[i, j] = G2(h * (i - j), s)
        A[i, N + 1] = 1
        A[i, N + 2] = mp.e**(-s * h * i)
        r[i] = F1_def(i, s, N)[0]
    for j in range(N + 1):
        A[N + 1, j] = 1
        A[N + 2, j] = mp.e**(-s * h * j)
    r[N + 2] = H1_def(s, N)
    X = mp.lu_solve(A, r)
    return [X[i] for i in range(N + 1)], X[N + 1], X[N + 2]


# ------------------------------------------------------------------ paper formulas
def par(s, N):
    s = mp.mpf(s)
    h = mp.mpf(1) / N
    q = mp.e**(s * h)
    p2 = 1 - mp.e**(2 * s * h) + 2 * s * h * mp.e**(s * h)
    p1 = 2 * (mp.e**(2 * s * h) - 1) - 2 * s * h * (1 + mp.e**(2 * s * h))
    lam = (-p1 - mp.sqrt(p1**2 - 4 * p2**2)) / (2 * p2)               # (6.x), sigma > 0
    K1 = 2 * mp.e**(s * h) + (1 + mp.e**(2 * s * h)) + mp.e**(s * h) * p1 / p2
    A1 = (lam - 1) * (lam * (1 + mp.e**(2 * s * h)) - mp.e**(s * h) * (lam**2 + 1)) / (lam + 1)
    H1 = (1 - mp.e**(-s)) / (2 * s**2) * (s * h * (mp.e**(s * h) + 1) / (mp.e**(s * h) - 1) - 2)
    M = (s * h * (mp.e**(s * h) + 1) - 2 * (mp.e**(s * h) - 1)) / (8 * s**5 * (mp.e**(s * h) - 1))
    return dict(s=s, N=N, h=h, q=q, p1=p1, p2=p2, lam=lam, K1=K1, A1=A1, c=2 * s**3 / p2, H1=H1,
                kap=H1 / (4 * s**3), M=M, P=M * (1 + mp.e**(-s)), Q=-M * (1 + mp.e**s),
                R=-h**2 / (12 * s**2), T=h**2 / (24 * s**2),
                Sp=lam * q / (1 - lam * q), S0=lam / (1 - lam), Sm=lam / (q - lam), S1=lam / (1 - lam)**2)


def D2(m, p):
    """Theorem 1 (6.2)."""
    m = abs(m)
    if m >= 2:
        return p["c"] * p["A1"] * p["lam"]**(m - 1)
    if m == 1:
        return p["c"] * (-mp.e**(p["s"] * p["h"]) + p["A1"])
    return p["c"] * (p["K1"] + p["A1"] / p["lam"])


def F1_closed(b, p):
    X = p["h"] * b
    return p["P"] * mp.e**(p["s"] * X) + p["Q"] * mp.e**(-p["s"] * X) + p["R"] * X + p["T"]


def F11_closed(X, s):
    return 1 / (2 * s**5) * (mp.e**(-s * X) * (mp.e**s + 1) / 2 - mp.e**(s * X) * (mp.e**(-s) + 1) / 2
                             + s**3 * X**3 / 3 - s**3 * X**2 / 2 + (s**3 / 2 + 2 * s) * X - s - s**3 / 6)


def F12_closed(X, s, h):
    q = mp.e**(s * h)
    return 1 / (2 * s**4) * (h * (q + 1) / (4 * (q - 1)) * (mp.e**(s * X) * (1 + mp.e**(-s)) - mp.e**(-s * X) * (1 + mp.e**s))
                             - s**2 * X**3 / 3 + s**2 * X**2 / 2 - (s**2 / 2 + 2 + s**2 * h**2 / 6) * X
                             + 1 + s**2 / 6 + s**2 * h**2 / 12)


def Bsys(p):
    s, N, h, q, lam, K1, A1 = p["s"], p["N"], p["h"], p["q"], p["lam"], p["K1"], p["A1"]
    kap, P, Q, R, T = p["kap"], p["P"], p["Q"], p["R"], p["T"]
    Sp, S0, Sm = p["Sp"], p["S0"], p["Sm"]
    es = mp.e**s
    F0, F1 = P + Q + T, P * es + Q / es + R + T                      # (7.4)
    GN = lambda z: (1 - z**(N + 1)) / (1 - z)
    LN = lam * (1 - (N + 1) * lam**N + N * lam**(N + 1)) / (1 - lam)**2
    Phim = P * GN(lam * q) + Q * GN(lam / q) + R * h * LN + T * GN(lam)
    Phip = P * es * GN(lam / q) + Q / es * GN(lam * q) + (R + T) * GN(lam) - R * h * LN
    a1 = K1 * (q - 1) - q * (q**2 - 1) + A1 / lam**2 * (Sp - S0)
    b1 = A1 * lam**N / es * (Sm - S0)
    T1 = (-K1 * (kap * (1 - 1 / q) + F0) + q * (kap * (1 - q**-2) + 2 * F0) - A1 / lam**2 * (kap * (S0 - Sm) + F0 * S0)
          - A1 * Phim - A1 * lam**N * (kap * es * (Sp - S0) + F1 * S0))
    a2 = A1 * lam**N * (Sp - S0)
    b2 = (K1 * (1 / q - 1) + q - 1 / q + A1 / lam**2 * (Sm - S0)) / es
    T2 = (-K1 * (kap * es * (q - 1) + F1) + q * (kap * es * (q**2 - 1) + 2 * F1) - A1 / lam**2 * (kap * es * (Sp - S0) + F1 * S0)
          - A1 * Phip - A1 * lam**N * (kap * (S0 - Sm) + F0 * S0))
    det = a1 * b2 - a2 * b1
    return (T1 * b2 - T2 * b1) / det, (a1 * T2 - a2 * T1) / det, dict(Phim=Phim, Phip=Phip)


def theorem(p):
    Bm, Bp, _ = Bsys(p)
    s, N, h, q, lam, A1, c = p["s"], p["N"], p["h"], p["q"], p["lam"], p["A1"], p["c"]
    kap, P, Q, R = p["kap"], p["P"], p["Q"], p["R"]
    Sp, S0, Sm, S1 = p["Sp"], p["S0"], p["Sm"], p["S1"]
    es = mp.e**s
    Wm = (kap + P) * (S0 - Sm) + (Bm - Q) * (Sp - S0) + R * h * S1
    Wp = (kap - P) * es * (Sp - S0) + (Bp - Q) / es * (Sm - S0) - R * h * S1
    wm = (kap + P) * (1 - 1 / q) + (Bm - Q) * (q - 1) + R * h
    wp = (kap - P) * es * (q - 1) + (Bp - Q) / es * (1 / q - 1) - R * h
    C = [c * A1 / lam * (lam**b * Wm + lam**(N - b) * Wp) for b in range(N + 1)]
    C[0] -= c * q * wm
    C[N] -= c * q * wp
    return C, Bm, Bp


def rel(a, b):
    return max(abs(x - y) for x, y in zip(a, b)) / max(abs(y) for y in b)


report = []


def chk(name, val, tol):
    ok = val < tol
    report.append(ok)
    print(f"{'OK ' if ok else 'FAIL'} {name:60s} {mp.nstr(val, 3)}")


for s, N in [(1, 10), (0.5, 7), (3, 12), (2, 3)]:
    s = mp.mpf(s)
    p = par(s, N)
    h = p["h"]
    print(f"\n=== sigma={s}, N={N}, lambda1={mp.nstr(p['lam'], 12)}, |lam|e^(sh)={mp.nstr(abs(p['lam'])*p['q'], 4)}")
    # lambda1 is a root, |lambda1|<1, asymptotics
    chk("P2(lambda1)=0", abs(p["p2"] * p["lam"]**2 + p["p1"] * p["lam"] + p["p2"]) / abs(p["p2"]), 1e-40)
    chk("lambda1 asymptotics (6.1), error / (sigma h)^4 bounded", abs(p["lam"] - (mp.sqrt(3) - 2 + (mp.sqrt(3) / 15 - mp.mpf(1) / 10) * (s * h)**2)) / (s * h)**4, 0.05)
    # (6.6): K1, A1 alternative forms
    lam, q = p["lam"], p["q"]
    chk("K1 form (6.6)", abs(p["K1"] - ((1 + q)**2 - q * (lam + 1 / lam))), 1e-35)
    chk("A1 form (6.6)", abs(p["A1"] - (1 - lam) * (q - lam) * (1 - q * lam) / (1 + lam)), 1e-40)
    # D2 properties (6.3)
    L = 600
    ms = range(-L, L + 1)
    chk("D2*1 = 0", abs(sum(D2(m, p) for m in ms)), 1e-25)
    chk("D2*(h beta) = 0 (at beta=3)", abs(sum(D2(3 - m, p) * h * m for m in ms)), 1e-25)
    chk("D2*exp(+s h beta) = 0 (at beta=2)", abs(sum(D2(2 - m, p) * mp.e**(s * h * m) for m in ms)), 1e-20)
    chk("D2*exp(-s h beta) = 0 (at beta=2)", abs(sum(D2(2 - m, p) * mp.e**(-s * h * m) for m in ms)), 1e-20)
    for b in (0, 1, 2):
        chk(f"D2*G2 = delta at beta={b}", abs(sum(D2(b - m, p) * G2(h * m, s) for m in ms) - (1 if b == 0 else 0)), 1e-15)
    # H1
    chk("H1 closed (7.1) vs definition (3.8)", abs(p["H1"] - H1_def(s, N)) / p["H1"], 1e-40)
    # F1, F11, F12
    e1 = e11 = e12 = 0
    for b in range(N + 1):
        F, F11, F12 = F1_def(b, s, N)
        X = h * b
        e1 = max(e1, abs(F - F1_closed(b, p)))
        e11 = max(e11, abs(F11 - F11_closed(X, s)))
        e12 = max(e12, abs(F12 - F12_closed(X, s, h)))
    chk("F11 closed (proof of Lemma 3)", e11, 1e-35)
    chk("F12 closed (proof of Lemma 3)", e12, 1e-35)
    chk("F1 closed (7.2)-(7.3) vs definition (4.6)", e1, 1e-35)
    # direct solution, theorem
    Cd, cA, cB = direct(s, N)
    Ct, Bm, Bp = theorem(p)
    chk("Theorem 2 coefficients vs direct solution (relative)", rel(Ct, Cd), 1e-35)
    Bt = sum(Cd[g] * mp.e**(s * h * g) for g in range(N + 1)) / (4 * s**3)
    chk("B^- = calB + tilde B", abs(Bm - (cB + Bt)) / abs(Bm), 1e-30)
    chk("B^+ = calB - tilde B", abs(Bp - (cB - Bt)) / abs(Bp), 1e-30)
    # u2 outside [0,N] by definition (8.1) vs (8.4)
    eu = 0
    for k in (1, 2, 5):
        u_def = sum(Cd[g] * G2(-h * k - h * g, s) for g in range(N + 1)) + cA + cB * mp.e**(s * h * k)
        u_f = p["kap"] * (1 - mp.e**(-s * h * k)) + Bm * (mp.e**(s * h * k) - 1) + F1_closed(0, p)
        eu = max(eu, abs(u_def - u_f))
        u_def = sum(Cd[g] * G2(h * (N + k) - h * g, s) for g in range(N + 1)) + cA + cB * mp.e**(-s * h * (N + k))
        u_f = p["kap"] * mp.e**s * (mp.e**(s * h * k) - 1) + Bp * mp.e**(-s) * (mp.e**(-s * h * k) - 1) + F1_closed(N, p)
        eu = max(eu, abs(u_def - u_f))
    chk("u2 outside [0,N]: definition vs (8.4)", eu, 1e-35)
    # norm: Lemma 1 vs -(ell, ell*G3) computed with quadrature
    C1 = Cd
    norm_lemma = (sum(C1[b] * C1[g] * G2(h * (b - g), s) for b in range(N + 1) for g in range(N + 1))
                  - 2 * sum(C1[b] * F1_def(b, s, N)[0] for b in range(N + 1))
                  - sum(C0(b, N) * C0(g, N) * G3(h * (b - g), s) for b in range(N + 1) for g in range(N + 1))
                  + 2 * sum(C0(b, N) * (Phi1(h * b, s) + Phi1(1 - h * b, s)) for b in range(N + 1)) - 2 * Phi2(1, s))

    def g(x):
        return (mp.quad(lambda y: G3(x - y, s), [0, x, 1]) - sum(C0(j, N) * G3(x - h * j, s) for j in range(N + 1))
                + sum(C1[j] * dG3(x - h * j, s) for j in range(N + 1)))

    def dg(x):
        return (mp.quad(lambda y: dG3(x - y, s), [0, x, 1]) - sum(C0(j, N) * dG3(x - h * j, s) for j in range(N + 1))
                + sum(C1[j] * G2(x - h * j, s) for j in range(N + 1)))

    if N <= 7:
        with mp.workdps(30):
            pts = [h * j for j in range(N + 1)]
            Ig = mp.quad(g, pts)
            ellg = Ig - sum(C0(j, N) * g(h * j) for j in range(N + 1)) - sum(C1[j] * dg(h * j) for j in range(N + 1))
        chk("norm^2: Lemma 1 vs -(ell, ell*G3) by quadrature (relative)", abs(norm_lemma + ellg) / norm_lemma, 1e-15)
    # Lemma (lambda_1) and Lemma (determinant of the reduced system)
    t = s * h
    chk("p2 = -2e^t(sinh t - t), p1 = 4e^t(sinh t - t cosh t)", max(abs(p["p2"] + 2 * mp.e**t * (mp.sinh(t) - t)),
                                                                   abs(p["p1"] - 4 * mp.e**t * (mp.sinh(t) - t * mp.cosh(t)))), 1e-40)
    chk("-1 < lambda1 < 0 and p1^2 > 4 p2^2", 0 if (-1 < p["lam"] < 0 and p["p1"]**2 > 4 * p["p2"]**2) else 1, 0.5)
    es_ = mp.e**s
    Sp_, S0_, Sm_ = p["Sp"], p["S0"], p["Sm"]
    a1_ = p["K1"] * (q - 1) - q * (q**2 - 1) + p["A1"] / lam**2 * (Sp_ - S0_)
    b1_ = p["A1"] * lam**N / es_ * (Sm_ - S0_)
    a2_ = p["A1"] * lam**N * (Sp_ - S0_)
    b2_ = (p["K1"] * (1 / q - 1) + q - 1 / q + p["A1"] / lam**2 * (Sm_ - S0_)) / es_
    Dl = -lam**2 * (q - lam) * (q - 1)**2 * (1 - lam * q) * (1 - lam**(2 * N)) / (es_ * (1 + lam)**2)
    chk("Delta closed form (Lemma on the reduced system)", abs(a1_ * b2_ - a2_ * b1_ - Dl) / abs(Dl), 1e-35)
    chk("a1, b2 closed forms", max(abs(a1_ - lam * (q - 1) * (1 - lam * q) / (1 + lam)),
                                    abs(b2_ + lam * (q - lam) * (q - 1) / (es_ * (1 + lam)))), 1e-40)
    # exactness and reflection
    Q = lambda f, df: sum(C0(j, N) * f(h * j) for j in range(N + 1)) + sum(Ct[j] * df(h * j) for j in range(N + 1))
    chk("exact for 1, x, e^{-sx}", max(abs(1 - Q(lambda t: 1, lambda t: 0)), abs(mp.mpf(1) / 2 - Q(lambda t: t, lambda t: 1)),
                                       abs((1 - mp.e**(-s)) / s - Q(lambda t: mp.e**(-s * t), lambda t: -s * mp.e**(-s * t)))), 1e-40)
    Cneg = direct(-s, N)[0]
    chk("reflection C(-s)_b = -C(s)_{N-b} (Prop. 1)", rel(Cneg, [-Cd[N - b] for b in range(N + 1)]), 1e-35)

# closed formulas outside the convergence range |lambda1| e^{sigma h} < 1 (analytic continuation, Theorem 3)
for s, N in [(2, 1), (10, 3), (50, 5)]:
    p = par(mp.mpf(s), N)
    Cd = direct(mp.mpf(s), N)[0]
    Ct = theorem(p)[0]
    chk(f"sigma={s}, N={N}: |lam|e^(sh)={mp.nstr(abs(p['lam']) * p['q'], 4)} > 1, Theorem 3 vs direct (relative)", rel(Ct, Cd), 1e-15)

print("\nALL OK" if all(report) else f"\n{report.count(False)} FAILED")
sys.exit(0 if all(report) else 1)
