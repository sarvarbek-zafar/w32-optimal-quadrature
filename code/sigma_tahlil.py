"""
sigma_tahlil.py
===============
Hosilali optimal kvadratura formulasini sigma ning turli qiymatlari bo'yicha baholash:
    sigma < 0,   sigma = 0,   0 < sigma < 1,   sigma > 1.

Kvadratura formulasi:
    int_0^1 f dx ~ sum C_{b,0} f(hb) + sum C_{b,1} f'(hb),   C_{b,0} -- trapetsiya.

Asosiy funksiyalar
------------------
koeff_etalon(s, N)    -- C_{b,1}: (28)-(30) sistemani bevosita yechish (har qanday sigma, sigma=0 ham)
koeff_yopiq(s, N)     -- C_{b,1}: yopiq formulalar (C_toliq_hisob.pdf, 9-bo'lim), sigma != 0
norma2(s, N, C1)      -- ||ell||^2 (xato funksionali normasining kvadrati), integralsiz formula
kvadratura(f, df, s, N, C1) -- formulaning qiymati

sigma = 0 holi: sigma -> 0 limitida yadrolar ko'phadga aylanadi:
    G_2 -> |x|^3/12,  G_3' -> sgn(x) x^4/48,  G_3 -> |x|^5/240,
    (ell, e^{-sigma x}) = 0 sharti esa (ell, x^2) = 0 shartiga aylanadi
    (fazo: int (f''')^2 dx normali W_2^{(3)}; formula 1, x, x^2 uchun aniq).
Hisoblar mpmath bilan (40 xona) bajariladi, shuning uchun kichik |sigma| da ham yo'qotish bo'lmaydi.
"""
import mpmath as mp

mp.mp.dps = 40


def sgn(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def C0(j, N):
    h = mp.mpf(1) / N
    return h / 2 if j in (0, N) else h


# --- yadrolar ---------------------------------------------------------------
def G2(x, s):
    if s == 0:
        return abs(x)**3 / 12
    return sgn(x) / (2 * s**3) * (mp.sinh(s * x) - s * x)


def G3(x, s):
    if s == 0:
        return abs(x)**5 / 240
    return sgn(x) / (2 * s**5) * (mp.sinh(s * x) - s * x - (s * x)**3 / 6)


def dG3(x, s):
    """G_3'(x)"""
    if s == 0:
        return sgn(x) * x**4 / 48
    return sgn(x) / (2 * s**4) * (mp.cosh(s * x) - 1 - (s * x)**2 / 2)


def Phi1(t, s):
    """int_0^t G_3"""
    if s == 0:
        return t**6 / 1440
    return (mp.cosh(s * t) - 1 - (s * t)**2 / 2 - (s * t)**4 / 24) / (2 * s**6)


def Phi2(t, s):
    """int_0^t Phi1"""
    if s == 0:
        return t**7 / 10080
    return (mp.sinh(s * t) - s * t - (s * t)**3 / 6 - (s * t)**5 / 120) / (2 * s**7)


# --- o'ng tomonlar (ta'rif bo'yicha, umumiy sigma uchun) --------------------
def F1_tarif(b, s, N):
    """F_1(hb) = int_0^1 G_3'(x-hb) dx + sum_g C_{g,0} G_3'(hb-hg) = G_3(1-hb) - G_3(hb) + ..."""
    h = mp.mpf(1) / N
    X = h * b
    return G3(1 - X, s) - G3(X, s) + sum(C0(g, N) * dG3(h * (b - g), s) for g in range(N + 1))


def kernel_fn(x, s):
    """Normaning yadrosidagi 3-funksiya: e^{-s x} (s != 0) yoki x^2 (s = 0)."""
    return x**2 if s == 0 else mp.e**(-s * x)


def kernel_dfn(x, s):
    return 2 * x if s == 0 else -s * mp.e**(-s * x)


# --- etalon: (28)-(30) sistemani bevosita yechish ---------------------------
def _dps(s):
    """Kichik |sigma| da 1/sigma^7 tartibli qisqarishlar uchun aniqlikni oshiramiz."""
    s = mp.mpf(s)
    if s == 0:
        return mp.mp.dps
    return mp.mp.dps + max(0, int(-8 * mp.log10(abs(s)))) + 8


def koeff_etalon(s, N):
    """Noma'lumlar C_0..C_N, A, B. Uchinchi shart: (ell, kernel_fn) = 0."""
    with mp.workdps(_dps(s)):
        return [+x for x in _koeff_etalon(s, N)]


def _koeff_etalon(s, N):
    s = mp.mpf(s)
    h = mp.mpf(1) / N
    n = N + 3
    A = mp.matrix(n, n)
    r = mp.matrix(n, 1)
    for i in range(N + 1):
        for j in range(N + 1):
            A[i, j] = G2(h * (i - j), s)
        A[i, N + 1] = 1
        A[i, N + 2] = kernel_dfn(h * i, s)
        r[i] = F1_tarif(i, s, N)
    for j in range(N + 1):
        A[N + 1, j] = 1                       # sum C_{j,1} = 0
        A[N + 2, j] = kernel_dfn(h * j, s)    # (ell, kernel_fn) = 0
    # (ell, k) = int_0^1 k - sum C0 k(hj) - sum C1 k'(hj) = 0
    if s == 0:
        intk = mp.mpf(1) / 3
    else:
        intk = (1 - mp.e**(-s)) / s
    r[N + 2] = intk - sum(C0(j, N) * kernel_fn(h * j, s) for j in range(N + 1))
    X = mp.lu_solve(A, r)
    return [X[i] for i in range(N + 1)]


# --- yopiq formulalar (sigma != 0) ------------------------------------------
def parametrlar(s, N):
    s = mp.mpf(s)
    h = mp.mpf(1) / N
    q = mp.e**(s * h)
    es = mp.e**s
    p2 = 1 - q**2 + 2 * s * h * q
    p1 = 2 * (q**2 - 1) - 2 * s * h * (1 + q**2)
    r1 = (-p1 - mp.sqrt(p1**2 - 4 * p2**2)) / (2 * p2)
    r2 = (-p1 + mp.sqrt(p1**2 - 4 * p2**2)) / (2 * p2)
    lam = r1 if abs(r1) < 1 else r2          # |lambda1| < 1 ildiz (sigma < 0 da r2)
    A1 = (lam - 1) * (lam * (1 + q**2) - q * (lam**2 + 1)) / (lam + 1)
    K1 = 2 * q + (1 + q**2) + q * p1 / p2
    H1 = (1 - 1 / es) / (2 * s**2) * (s * h * (q + 1) / (q - 1) - 2)
    M = (s * h * (q + 1) - 2 * (q - 1)) / (8 * s**5 * (q - 1))
    return dict(s=s, N=N, h=h, q=q, es=es, p1=p1, p2=p2, lam=lam, lam_maqola=r1, A1=A1, K1=K1,
                c=2 * s**3 / p2, Sp=lam * q / (1 - lam * q), S0=lam / (1 - lam), Sm=lam / (q - lam),
                S1=lam / (1 - lam)**2, H1=H1, kap=H1 / (4 * s**3), M=M,
                P=M * (1 + 1 / es), Q=-M * (1 + es), R=-h**2 / (12 * s**2), T=h**2 / (24 * s**2),
                yaqinlashish=abs(lam) * mp.e**(abs(s) * h))   # < 1 bo'lishi kerak


def koeff_yopiq(s, N, lam_tanlov="moduli<1"):
    """C_toliq_hisob.pdf, 9-bo'lim. lam_tanlov="maqola" -- maqoladagi ildiz formulasi."""
    with mp.workdps(_dps(s)):
        C, Pr = _koeff_yopiq(s, N, lam_tanlov)
        return [+x for x in C], Pr


def _koeff_yopiq(s, N, lam_tanlov):
    Pr = parametrlar(s, N)
    if lam_tanlov == "maqola":
        lam = Pr["lam_maqola"]
        q = Pr["q"]
        Pr.update(lam=lam, A1=(lam - 1) * (lam * (1 + q**2) - q * (lam**2 + 1)) / (lam + 1),
                  Sp=lam * q / (1 - lam * q), S0=lam / (1 - lam), Sm=lam / (q - lam), S1=lam / (1 - lam)**2)
    s, N, h, q, es = Pr["s"], Pr["N"], Pr["h"], Pr["q"], Pr["es"]
    lam, A1, K1, c = Pr["lam"], Pr["A1"], Pr["K1"], Pr["c"]
    Sp, S0, Sm, S1, k = Pr["Sp"], Pr["S0"], Pr["Sm"], Pr["S1"], Pr["kap"]
    P, Q, R, T = Pr["P"], Pr["Q"], Pr["R"], Pr["T"]
    F0 = P + Q + T
    F1v = P * es + Q / es + R + T
    GN = lambda x: (1 - x**(N + 1)) / (1 - x)
    LN = lam * (1 - (N + 1) * lam**N + N * lam**(N + 1)) / (1 - lam)**2
    Phm = P * GN(lam * q) + Q * GN(lam / q) + R * h * LN + T * GN(lam)
    Php = P * es * GN(lam / q) + Q / es * GN(lam * q) + (R + T) * GN(lam) - R * h * LN
    a1 = K1 * (q - 1) - q * (q**2 - 1) + A1 / lam**2 * (Sp - S0)
    b1 = A1 * lam**N / es * (Sm - S0)
    T1 = (-K1 * (k * (1 - 1 / q) + F0) + q * (k * (1 - q**-2) + 2 * F0)
          - A1 / lam**2 * (k * (S0 - Sm) + F0 * S0) - A1 * Phm
          - A1 * lam**N * (k * es * (Sp - S0) + F1v * S0))
    a2 = A1 * lam**N * (Sp - S0)
    b2 = (K1 * (1 / q - 1) + q - 1 / q + A1 / lam**2 * (Sm - S0)) / es
    T2 = (-K1 * (k * es * (q - 1) + F1v) + q * (k * es * (q**2 - 1) + 2 * F1v)
          - A1 / lam**2 * (k * es * (Sp - S0) + F1v * S0) - A1 * Php
          - A1 * lam**N * (k * (S0 - Sm) + F0 * S0))
    Dl = a1 * b2 - a2 * b1
    Bm = (T1 * b2 - T2 * b1) / Dl
    Bp = (a1 * T2 - a2 * T1) / Dl
    Wm = (k + P) * (S0 - Sm) + (Bm - Q) * (Sp - S0) + R * h * S1
    Wp = (k - P) * es * (Sp - S0) + (Bp - Q) / es * (Sm - S0) - R * h * S1
    wm1 = (k + P) * (1 - 1 / q) + (Bm - Q) * (q - 1) + R * h
    wp1 = (k - P) * es * (q - 1) + (Bp - Q) / es * (1 / q - 1) - R * h
    C = [c * A1 / lam * (lam**b * Wm + lam**(N - b) * Wp) for b in range(N + 1)]
    C[0] -= c * q * wm1
    C[N] -= c * q * wp1
    return C, Pr


# --- xato funksionali normasi -----------------------------------------------
def norma2(s, N, C1):
    """||ell||^2 = sum sum C1 C1 G2 - 2 sum C1 F1 + K0   (maqola, Lemma 3.2)."""
    with mp.workdps(_dps(s)):
        return +_norma2(s, N, C1)


def _norma2(s, N, C1):
    s = mp.mpf(s)
    h = mp.mpf(1) / N
    t1 = sum(C1[i] * C1[j] * G2(h * (i - j), s) for i in range(N + 1) for j in range(N + 1))
    t2 = sum(C1[i] * F1_tarif(i, s, N) for i in range(N + 1))
    K0 = (-sum(C0(i, N) * C0(j, N) * G3(h * (i - j), s) for i in range(N + 1) for j in range(N + 1))
          + 2 * sum(C0(i, N) * (Phi1(h * i, s) + Phi1(1 - h * i, s)) for i in range(N + 1))
          - 2 * Phi2(1, s))
    return t1 - 2 * t2 + K0


# --- kvadratura -------------------------------------------------------------
def kvadratura(f, df, N, C1):
    h = mp.mpf(1) / N
    return sum(C0(j, N) * f(h * j) for j in range(N + 1)) + sum(C1[j] * df(h * j) for j in range(N + 1))


def em_koeff(N):
    """Eyler--Makloren: C_{0,1} = h^2/12, C_{N,1} = -h^2/12."""
    h = mp.mpf(1) / N
    C = [mp.mpf(0)] * (N + 1)
    C[0], C[N] = h**2 / 12, -h**2 / 12
    return C
