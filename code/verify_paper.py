"""
verify_paper.py -- independent check of every formula stated in paper.tex.
The formulas are transcribed from the paper itself (not imported from earlier modules);
the reference is the direct LU solution of system (5.1)-(5.3) assembled from the definitions.
"""
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
    chk("lambda1 asymptotics (6.2), error / (sigma h)^4 bounded", abs(p["lam"] - (mp.sqrt(3) - 2 + (mp.sqrt(3) / 15 - mp.mpf(1) / 10) * (s * h)**2)) / (s * h)**4, 0.05)
    # (6.6): K1, A1 alternative forms
    lam, q = p["lam"], p["q"]
    chk("K1 form (6.6)", abs(p["K1"] - ((1 + q)**2 - q * (lam + 1 / lam))), 1e-35)
    chk("A1 form (6.6)", abs(p["A1"] - (1 - lam) * (q - lam) * (1 - q * lam) / (1 + lam)), 1e-40)
    # D2 properties (6.4)
    L = 600
    ms = range(-L, L + 1)
    chk("D2*1 = 0", abs(sum(D2(m, p) for m in ms)), 1e-25)
    chk("D2*(h beta) = 0 (at beta=3)", abs(sum(D2(3 - m, p) * h * m for m in ms)), 1e-25)
    chk("D2*exp(+s h beta) = 0 (at beta=2)", abs(sum(D2(2 - m, p) * mp.e**(s * h * m) for m in ms)), 1e-20)
    chk("D2*exp(-s h beta) = 0 (at beta=2)", abs(sum(D2(2 - m, p) * mp.e**(-s * h * m) for m in ms)), 1e-20)
    for b in (0, 1, 2):
        chk(f"D2*G2 = delta at beta={b}", abs(sum(D2(b - m, p) * G2(h * m, s) for m in ms) - (1 if b == 0 else 0)), 1e-15)
    # H1
    chk("H1 closed (7.1) vs definition (3.6)", abs(p["H1"] - H1_def(s, N)) / p["H1"], 1e-40)
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
    chk("F1 closed (7.2)-(7.3) vs definition (4.4)", e1, 1e-35)
    # direct solution, theorem
    Cd, cA, cB = direct(s, N)
    Ct, Bm, Bp = theorem(p)
    chk("Theorem 3 coefficients vs direct solution (relative)", rel(Ct, Cd), 1e-35)
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
    chk("u2 outside [0,N]: definition vs (8.3)", eu, 1e-35)
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
    # exactness and reflection
    Q = lambda f, df: sum(C0(j, N) * f(h * j) for j in range(N + 1)) + sum(Ct[j] * df(h * j) for j in range(N + 1))
    chk("exact for 1, x, e^{-sx}", max(abs(1 - Q(lambda t: 1, lambda t: 0)), abs(mp.mpf(1) / 2 - Q(lambda t: t, lambda t: 1)),
                                       abs((1 - mp.e**(-s)) / s - Q(lambda t: mp.e**(-s * t), lambda t: -s * mp.e**(-s * t)))), 1e-40)
    Cneg = direct(-s, N)[0]
    chk("reflection C(-s)_b = -C(s)_{N-b} (Prop. 1)", rel(Cneg, [-Cd[N - b] for b in range(N + 1)]), 1e-35)

# ================================================================== new results (Sections 10 and 12)
print("\n=== Theorem 4 (simple closed form), Lemma 4, Corollary 1, Theorems 6-7, Proposition 2")


def lam_root(s, h):
    q = mp.e**(s * h)
    p2 = 1 - q**2 + 2 * s * h * q
    p1 = 2 * (q**2 - 1) - 2 * s * h * (1 + q**2)
    r = [(-p1 - mp.sqrt(p1**2 - 4 * p2**2)) / (2 * p2), (-p1 + mp.sqrt(p1**2 - 4 * p2**2)) / (2 * p2)]
    return min(r, key=abs)


def c_sig(s, h):
    return h / (2 * s) * mp.coth(s * h / 2) - 1 / s**2


def simple(s, N):
    """Theorem 4: (10.4)-(10.6)."""
    h = mp.mpf(1) / N
    q, lam = mp.e**(s * h), lam_root(s, h)
    D = h**2 / 12 - c_sig(s, h)
    mm = D * (1 - lam) * (q - lam) / (lam * (q - 1) * (1 + lam**N))
    mp_ = D * (1 - lam) * (1 - lam * q) / (lam * (q - 1) * (1 + lam**N))
    C = [mm * lam**b + mp_ * lam**(N - b) for b in range(N + 1)]
    C[0] = h**2 / 12 - (lam * mm - lam**N * mp_) / (1 - lam)
    C[N] = -h**2 / 12 - (lam * mp_ - lam**N * mm) / (1 - lam)
    return C, mm, mp_, lam, q, D


with mp.workdps(60):
    worst, worst50 = mp.mpf(0), mp.mpf(0)
    for s in [mp.mpf("0.01"), mp.mpf("0.5"), mp.mpf("-0.5"), mp.mpf(1), mp.mpf(3), mp.mpf(-2), mp.mpf(10), mp.mpf(-10)]:
        for N in [1, 2, 3, 5, 8, 13, 20, 30]:
            worst = max(worst, rel(simple(s, N)[0], direct(s, N)[0]))
    for N in range(2, 31):
        worst50 = max(worst50, rel(simple(mp.mpf(50), N)[0], direct(mp.mpf(50), N)[0]))
chk("Theorem 4 vs direct solution, sigma in {0.01,+-0.5,1,3,-2,+-10}, N<=30", worst, 1e-40)
chk("Theorem 4 vs direct solution, sigma=50, N=2..30 (60 digits)", worst50, 1e-37)
print("    worst deviations:", mp.nstr(worst, 3), mp.nstr(worst50, 3))

for s, N in [(1, 10), (3, 7), (-2, 9), (0.5, 3)]:
    s = mp.mpf(s); h = mp.mpf(1) / N
    C, mm, mp_, lam, q, D = simple(s, N)
    f = lambda m: mp.sinh(s * h * m) - s * h * m
    U1 = (q + lam**(N + 1) * q**(-N)) / (2 * (q - lam))
    U2 = -(1 + (lam * q)**(N + 1)) / (2 * (1 - lam * q))
    U3 = -s * h * (1 + lam**(N + 1)) / (1 - lam)
    U4 = s * h * lam * (1 + lam**N * (N * (1 - lam) + 1)) / (1 - lam)**2
    eA = max(abs(2 * s**3 * sum(lam**g * G2(h * (b - g), s) for g in range(N + 1)) - (U1 * q**b + U2 * q**(-b) + U3 * b + U4))
             for b in range(N + 1))
    chk(f"Lemma 4, sigma={s}, N={N}", eA, 1e-40)
    cs = c_sig(s, h)
    e1 = max(abs(sum(C[:m + 1]) - (h**2 / 12 - (lam**(m + 1) * mm - lam**(N - m) * mp_) / (1 - lam))) for m in range(N))
    e2 = max(abs(sum(C[b] * q**(-b) for b in range(m + 1)) - (cs - q**(-m) * (lam**(m + 1) * mm / (q - lam) - lam**(N - m) * mp_ / (1 - lam * q))))
             for m in range(N))
    chk(f"Corollary 1, partial sums (10.6)-(10.7), sigma={s}, N={N}", max(e1, e2), 1e-45)
    chk(f"sum lam^k f(k) = 0 (proof of Theorem 2), sigma={s}", abs(mp.nsum(lambda k: lam**k * f(k), [1, mp.inf])), 1e-40)
    chk(f"(10.2): H1=(1-e^-s)c_s, sigma={s}", abs(H1_def(s, N) - (1 - mp.e**(-s)) * cs), 1e-40)

# Corollary 1, asymptotics of C_{0,1}: (C0 - h^2/12 + s(3-r3)h^3/720)/h^4 -> -s^2(r3-1)/1440
r3 = mp.sqrt(3)
for s in (mp.mpf(1), mp.mpf(2)):
    vals = []
    for N in (64, 128, 256):
        h = mp.mpf(1) / N
        C0v = simple(s, N)[0][0]
        vals.append((C0v - h**2 / 12 + s * (3 - r3) * h**3 / 720) / h**4)
    chk(f"(10.9) h^4 coefficient of C_01, sigma={s} (N=256 vs limit)", abs(vals[-1] + s**2 * (r3 - 1) / 1440) / (s**2 * (r3 - 1) / 1440), 0.01)

# Proposition 2: fitted E-M exact for e^{+-s x}
for s, N in [(1, 10), (2.5, 7)]:
    s = mp.mpf(s); h = mp.mpf(1) / N; cs = c_sig(s, h)
    Qe = lambda f, df: sum(C0(j, N) * f(h * j) for j in range(N + 1)) + cs * (df(0) - df(1))
    err = max(abs((mp.e**(t * s) - 1) / (t * s) - Qe(lambda x: mp.e**(t * s * x), lambda x: t * s * mp.e**(t * s * x))) for t in (1, -1))
    chk(f"Prop. 2(ii): fitted E-M exact for e^(+-sigma x), sigma={s}", err, 1e-40)

# Theorem 6(i): lower bound; test function phi_* gives (l, phi_*) = -h^6/30240 for any C
def Bt(n, t):
    return mp.bernpoly(n, t - mp.floor(t))
for s, N in [(1, 4), (3, 8), (-2, 6)]:
    s = mp.mpf(s); h = mp.mpf(1) / N
    C = direct(s, N)[0]
    n2 = (sum(C[b] * C[g] * G2(h * (b - g), s) for b in range(N + 1) for g in range(N + 1))
          - 2 * sum(C[b] * F1_def(b, s, N)[0] for b in range(N + 1))
          - sum(C0(b, N) * C0(g, N) * G3(h * (b - g), s) for b in range(N + 1) for g in range(N + 1))
          + 2 * sum(C0(b, N) * (Phi1(h * b, s) + Phi1(1 - h * b, s)) for b in range(N + 1)) - 2 * Phi2(1, s))
    lb = h**6 / 30240 / (1 + s**2 * h**2 / 40)
    chk(f"Theorem 6(i): ||l||^2 >= lower bound, sigma={s}, N={N}", 0 if n2 >= lb else 1, 0.5)
    phi = lambda x: h**6 / 720 * Bt(6, x / h)
    val = mp.quad(phi, [h * k for k in range(N + 1)]) - sum(C0(b, N) * phi(h * b) for b in range(N + 1))
    chk(f"Theorem 6(i): (l, phi_*) = -h^6/30240, sigma={s}, N={N}", abs(val + h**6 / 30240) / (h**6 / 30240), 1e-25)
    nphi = mp.quad(lambda x: (h**3 / 6 * Bt(3, x / h) + s * h**4 / 24 * Bt(4, x / h))**2, [h * k for k in range(N + 1)])
    chk(f"Theorem 6(i): ||phi_*||^2 formula, sigma={s}, N={N}", abs(nphi - h**6 / 30240 * (1 + s**2 * h**2 / 40)) / nphi, 1e-25)

# Theorem 7: (R - h^4 term - h^5 term)/h^6 stays bounded (ratio of consecutive values ~ 1)
f_, df_, d2_, d3_ = (lambda x: mp.log(1 + x)), (lambda x: 1 / (1 + x)), (lambda x: -1 / (1 + x)**2), (lambda x: 2 / (1 + x)**3)
I_ = 2 * mp.log(2) - 1
s = mp.mpf(1)
L_ = lambda x: d3_(x) + s * d2_(x)
res = []
for N in (16, 32, 64, 128):
    h = mp.mpf(1) / N
    C = simple(s, N)[0]
    R = I_ - sum(C0(j, N) * f_(h * j) for j in range(N + 1)) - sum(C[j] * df_(h * j) for j in range(N + 1))
    res.append((R - h**4 / 720 * (L_(1) - L_(0)) + s * h**5 / (1440 * r3) * (L_(0) + L_(1))) / h**6)
chk("Theorem 7: remainder/h^6 bounded for ln(1+x), sigma=1 (N=64 vs 128)", abs(res[-1] - res[-2]) / abs(res[-1]), 0.05)

print("\nALL OK" if all(report) else f"\n{report.count(False)} FAILED")
