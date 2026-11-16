"""
kvadratura.py -- W_2^{(3,2)} fazosidagi hosilali optimal kvadratura formulasi (umumiy modul)
==========================================================================================
    int_0^1 f dx ~ sum C_{b,0} f(hb) + sum C_{b,1}(sigma) f'(hb),   h = 1/N,  sigma != 0.

Bu modulni 01_, 02_, 03_ notebooklar ishlatadi. Asosiy funksiyalar:
    C1(sigma, N)            -- optimal koeffitsiyentlar C_{b,1} (yopiq formulalar)
    optimal(F, N, sigma)    -- formula qiymati;  F = Funksiya("exp(-2*x)")
    trapetsiya, simpson, eyler_makloren, gauss -- taqqoslash usullari
    Funksiya(ifoda)         -- f, f', aniq integral va sigma* = -(f3(1)-f3(0))/(f2(1)-f2(0))
    sigma_tori(a, b, qadam), Dastur(...)  -- sigma ni tanlash dasturi
    uslub(), jadval(...)    -- grafik uslubi va Markdown jadval (notebook uchun)
Kerakli kutubxonalar: mpmath, sympy, matplotlib.
"""

import math
import csv
import mpmath as mp
import sympy as sp

mp.mp.dps = 40
x = sp.Symbol("x", real=True)
_LOC = {"x": x, "besselj": sp.besselj, "erf": sp.erf, "gamma": sp.gamma, "e": sp.E, "pi": sp.pi}


# =============================================================================
# 1. Optimal koeffitsiyentlar (yopiq formulalar, C_toliq_hisob.pdf, 9-bo'lim)
# =============================================================================
def _dps(s):
    s = mp.mpf(s)
    return mp.mp.dps + max(0, int(-8 * mp.log10(abs(s)))) + 8


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



_kesh = {}


def C1(s, N):
    """Optimal koeffitsiyentlar C_{b,1}(sigma), b = 0..N (keshlangan)."""
    k = (mp.nstr(mp.mpf(s), 30), N)
    if k not in _kesh:
        if mp.mpf(s) == 0:
            raise ValueError("sigma = 0 bo'lishi mumkin emas")
        _kesh[k] = koeff_yopiq(s, N)[0]
    return _kesh[k]


def C0(j, N):
    h = mp.mpf(1) / N
    return h / 2 if j in (0, N) else h


# =============================================================================
# 2. Usullar
# =============================================================================
def optimal(F, N, s):
    h = mp.mpf(1) / N
    C = C1(s, N)
    return sum(C0(j, N) * F["f"](h * j) for j in range(N + 1)) + sum(C[j] * F["df"](h * j) for j in range(N + 1))


def trapetsiya(F, N):
    h = mp.mpf(1) / N
    return h * (sum(F["f"](h * j) for j in range(1, N)) + (F["f"](0) + F["f"](1)) / 2)


def simpson(F, N):
    if N % 2:
        return None
    h = mp.mpf(1) / N
    f = F["f"]
    return h / 3 * (f(0) + f(1) + 4 * sum(f(h * j) for j in range(1, N, 2)) + 2 * sum(f(h * j) for j in range(2, N, 2)))


def eyler_makloren(F, N):
    h = mp.mpf(1) / N
    return trapetsiya(F, N) + h**2 / 12 * (F["df"](0) - F["df"](1))


_gl = {}


def gauss(F, N):
    """Gauss-Lejandr, N+1 tugun (boshqa tugunlar -- faqat ma'lumot uchun)."""
    n = N + 1
    if n not in _gl:
        t, w = [], []
        for i in range(1, n + 1):
            z = mp.mpf(math.cos(math.pi * (i - 0.25) / (n + 0.5)))
            for _ in range(100):
                p0, p1 = mp.mpf(1), z
                for k in range(2, n + 1):
                    p0, p1 = p1, ((2 * k - 1) * z * p1 - (k - 1) * p0) / k
                dp = n * (z * p1 - p0) / (z**2 - 1)
                dz = p1 / dp
                z -= dz
                if abs(dz) < mp.mpf(10)**(-mp.mp.dps + 2):
                    break
            t.append((1 + z) / 2)
            w.append(1 / ((1 - z**2) * dp**2))
        _gl[n] = (t, w)
    t, w = _gl[n]
    return sum(wi * F["f"](ti) for ti, wi in zip(t, w))


def c_sigma(s2, N):
    """End weight of the exponentially fitted Euler--Maclaurin rule.
    s2 = sigma^2 > 0: c = h/(2 sigma) coth(sigma h/2) - 1/sigma^2 (exact for e^{+-sigma x});
    s2 = -omega^2 < 0: c = 1/omega^2 - h/(2 omega) cot(omega h/2)  (exact for cos, sin omega x);
    s2 = 0: c = h^2/12 (classical Euler--Maclaurin)."""
    h = mp.mpf(1) / N
    s2 = mp.mpf(s2)
    if s2 == 0:
        return h**2 / 12
    if s2 > 0:
        s = mp.sqrt(s2)
        return h / (2 * s) * mp.coth(s * h / 2) - 1 / s2
    w = mp.sqrt(-s2)
    return 1 / w**2 - h / (2 * w) * mp.cot(w * h / 2)


def em_fitted(F, N, s2):
    """Fitted Euler--Maclaurin rule: trapezoidal rule + c (f'(0) - f'(1))."""
    return trapetsiya(F, N) + c_sigma(s2, N) * (F["df"](0) - F["df"](1))


def C1_sodda(s, N):
    """Simple closed form of the optimal coefficients (Theorem 4 of the paper)."""
    with mp.workdps(_dps(s)):
        p = parametrlar(s, N)
        h, q, lam, s_ = p["h"], p["q"], p["lam"], p["s"]
        D = h**2 / 12 - c_sigma(s_**2, N)
        am = D * (1 - lam) * (q - lam) / (lam * (q - 1) * (1 + lam**N))
        ap = D * (1 - lam) * (1 - lam * q) / (lam * (q - 1) * (1 + lam**N))
        C = [am * lam**b + ap * lam**(N - b) for b in range(N + 1)]
        C[0] = h**2 / 12 - (lam * am - lam**N * ap) / (1 - lam)
        C[N] = -h**2 / 12 - (lam * ap - lam**N * am) / (1 - lam)
        return [+c for c in C]


USULLAR = {"Trapetsiya": trapetsiya, "Simpson": simpson, "Eyler-Makloren": eyler_makloren, "Gauss*": gauss}


# =============================================================================
# 3. Funksiya
# =============================================================================
def Funksiya(ifoda, uzilish=()):
    """Matndan f, f', aniq integral va sigma* ni tayyorlaydi. Masalan: "1/(1+x)", "sqrt(1+x)"."""
    e = sp.sympify(ifoda, locals=_LOC)
    f = sp.lambdify(x, e, "mpmath")
    df = sp.lambdify(x, sp.diff(e, x), "mpmath")
    for t in (0, 1):                                  # formula f'(0), f'(1) dan foydalanadi
        try:
            v = df(mp.mpf(t))
            ok = mp.isfinite(v) and mp.isfinite(f(mp.mpf(t)))
        except (ZeroDivisionError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"{ifoda}: f({t}) yoki f'({t}) cheksiz -- bu formula uchun yaroqsiz")
    I = mp.quad(f, [0] + [mp.mpf(u) for u in uzilish] + [1])
    try:
        Is = sp.integrate(e, (x, 0, 1))
        if Is.is_finite and not Is.has(sp.Integral):
            Is = mp.mpf(sp.N(Is, mp.mp.dps + 5))
            if abs(Is - I) < mp.mpf(10)**-20:
                I = Is
    except Exception:
        pass
    # sigma* = -(f'''(1)-f'''(0))/(f''(1)-f''(0))
    ss = None
    try:
        nd = lambda ex: ex.replace(sp.DiracDelta, lambda *a: sp.S.Zero)
        d2 = sp.lambdify(x, nd(sp.diff(e, x, 2)), "mpmath")
        d3 = sp.lambdify(x, nd(sp.diff(e, x, 3)), "mpmath")
        D2, D3 = d2(1) - d2(0), d3(1) - d3(0)
        if abs(D2) > mp.mpf(10)**-25 * max(1, abs(D3)):   # D2 = 0 bo'lsa sigma* aniqlanmaydi
            ss = mp.mpf(-D3 / D2)
    except Exception:
        pass
    # sigma_E^2 = [f''']/[f'] : tuning of the fitted Euler--Maclaurin rule
    s2E = None
    try:
        d1 = sp.lambdify(x, sp.diff(e, x, 1), "mpmath")
        d3 = sp.lambdify(x, nd(sp.diff(e, x, 3)), "mpmath")
        D1, D3 = d1(1) - d1(0), d3(1) - d3(0)
        if abs(D1) > mp.mpf(10)**-25 * max(1, abs(D3)):
            s2E = mp.mpf(D3 / D1)
    except Exception:
        pass
    return dict(nom=ifoda, ifoda=e, f=f, df=df, I=I, sigma_nazariy=ss, sigma2_E=s2E)


# =============================================================================
# 4. Sigma to'ri va tanlash
# =============================================================================
def sigma_tori(a, b, qadam, chegaralar=True):
    """[a, b] (yoki (a, b)) oraliqda qadam bilan to'r; sigma = 0 chiqariladi."""
    a, b, qadam = mp.mpf(str(a)), mp.mpf(str(b)), mp.mpf(str(qadam))
    n = int(mp.nint((b - a) / qadam))
    tor = [a + k * qadam for k in range(n + 1)]
    if not chegaralar:
        tor = tor[1:-1]
    return [s for s in tor if abs(s) > mp.mpf(10)**-12]


def xato_bahosi(F, N, s, usul):
    """usul="aniq": I - Q_N;  usul="runge": (Q_2N - Q_N) * 16/15 (aniq integralsiz baho)."""
    if usul == "aniq":
        return F["I"] - optimal(F, N, s)
    Q1, Q2 = optimal(F, N, s), optimal(F, 2 * N, s)
    return (Q2 - Q1) * 16 / 15


class Dastur:
    def __init__(self, a=1, b=2, qadam=0.2, N_tanlash=16, Nlar=(8, 16, 32, 64), usul="aniq", chegaralar=True):
        assert usul in ("aniq", "runge", "nazariy")
        self.a, self.b, self.qadam = a, b, qadam
        self.N_tanlash, self.Nlar, self.usul = N_tanlash, tuple(Nlar), usul
        self.tor = sigma_tori(a, b, qadam, chegaralar)
        if not self.tor:
            raise ValueError("sigma to'ri bo'sh")
        self.natija = []

    # --- bitta funksiya ---------------------------------------------------
    def bitta(self, ifoda, uzilish=()):
        F = Funksiya(ifoda, uzilish)
        N = self.N_tanlash
        egri = [(s, xato_bahosi(F, N, s, "aniq" if self.usul == "nazariy" else self.usul)) for s in self.tor]
        if self.usul == "nazariy" and F["sigma_nazariy"] is not None:
            s_opt = min(max(F["sigma_nazariy"], self.tor[0]), self.tor[-1])      # oraliqqa cheklash
            s_opt = min(self.tor, key=lambda s: abs(s - s_opt))                 # eng yaqin to'r nuqtasi
        else:
            s_opt = min(egri, key=lambda t: abs(t[1]))[0]
        if self.usul == "nazariy" and F["sigma_nazariy"] is not None:
            chegarada = not (self.tor[0] <= F["sigma_nazariy"] <= self.tor[-1])   # sigma* oraliqdan tashqarida
        else:
            chegarada = s_opt in (self.tor[0], self.tor[-1])
        jad = {}
        for N_ in self.Nlar:
            jad[(f"Optimal (σ={mp.nstr(s_opt, 4)})", N_)] = F["I"] - optimal(F, N_, s_opt)
            for nom, fn in USULLAR.items():
                Q = fn(F, N_)
                jad[(nom, N_)] = None if Q is None else F["I"] - Q
        r = dict(F=F, egri=egri, s_opt=s_opt, chegarada=chegarada, jadval=jad)
        self.natija.append(r)
        return r

    def ishga_tushir(self, funksiyalar):
        """funksiyalar: ["1/(1+x)", ...] yoki [("abs(x-1/3)**3", [1/3]), ...]"""
        for it in funksiyalar:
            ifoda, uz = (it, ()) if isinstance(it, str) else it
            try:
                self.bitta(ifoda, uz)
            except ValueError as e:
                print("  O'tkazib yuborildi:", e)
        return self.natija

    # --- natijalar ---------------------------------------------------------
    def usullar(self, r):
        return list(dict.fromkeys(k[0] for k in r["jadval"]))

    def xulosa_qatorlari(self, N=None):
        """Har bir funksiya: sigma_opt, sigma* (nazariy), har bir usulning |xato|si (N da), eng yaxshi usul."""
        N = N or self.N_tanlash
        if N not in self.Nlar:
            N = self.Nlar[len(self.Nlar) // 2]
        q = []
        for r in self.natija:
            us = self.usullar(r)
            xatolar = {u: r["jadval"][(u, N)] for u in us}
            teng = {u: abs(v) for u, v in xatolar.items() if v is not None and u != "Gauss*"}
            eng = min(teng, key=teng.get)
            q.append(dict(funksiya=r["F"]["nom"], s_opt=r["s_opt"], s_naz=r["F"]["sigma_nazariy"],
                          chegarada=r["chegarada"], xatolar=xatolar, eng_yaxshi=eng, N=N))
        return q

    def hisobot(self, N=None, chop=print):
        q = self.xulosa_qatorlari(N)
        if not q:
            return
        N = q[0]["N"]
        chop(f"\nσ to'ri: [{self.a}, {self.b}], qadam {self.qadam} ({len(self.tor)} nuqta);  tanlash usuli: {self.usul};"
             f"  σ tanlandi N = {self.N_tanlash} da;  jadval N = {N} da")
        us = ["Optimal", "Trapetsiya", "Simpson", "Eyler-Makloren", "Gauss*"]
        chop(f"{'funksiya':28s}{'σ_opt':>8s}{'σ* naz.':>9s}" + "".join(f"{u:>16s}" for u in us) + "   eng aniq (teng tugunlar)")
        for r in q:
            xs = list(r["xatolar"].values())
            yoz = "".join(f"{('-' if v is None else mp.nstr(abs(v), 3)):>16s}" for v in xs)
            snaz = "-" if r["s_naz"] is None else mp.nstr(r["s_naz"], 3)
            belgi = " (chegarada!)" if r["chegarada"] else ""
            chop(f"{r['funksiya'][:27]:28s}{mp.nstr(r['s_opt'], 3):>8s}{snaz:>9s}" + yoz + "   " +
                 r["eng_yaxshi"].split(" (")[0] + belgi)
        if any(r["chegarada"] for r in q):
            chop("\n(chegarada!) -- optimal σ oraliq chegarasiga tushdi: haqiqiy optimum oraliqdan tashqarida "
                 "bo'lishi mumkin, oraliqni kengaytirib ko'ring (σ* nazariy ustuniga qarang).")
        chop("Gauss* boshqa (teng uzoqlikda bo'lmagan) tugunlardan foydalanadi -- faqat ma'lumot uchun.")

    def csv(self, fayl="sigma_tanlash_natija.csv"):
        with open(fayl, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["funksiya", "sigma_opt", "sigma_nazariy", "usul", "N", "xato", "|xato|"])
            for r in self.natija:
                for (u, N_), v in r["jadval"].items():
                    w.writerow([r["F"]["nom"], mp.nstr(r["s_opt"], 6),
                                "" if r["F"]["sigma_nazariy"] is None else mp.nstr(r["F"]["sigma_nazariy"], 6),
                                u.split(" (")[0], N_, "" if v is None else mp.nstr(v, 10),
                                "" if v is None else mp.nstr(abs(v), 6)])
            w.writerow([])
            w.writerow(["funksiya", "sigma", f"xato(N={self.N_tanlash})" + (" [runge bahosi]" if self.usul == "runge" else "")])
            for r in self.natija:
                for s, e in r["egri"]:
                    w.writerow([r["F"]["nom"], mp.nstr(s, 6), mp.nstr(e, 10)])
        return fayl

    def grafik(self, fayl=None, ustun=3):
        import matplotlib.pyplot as plt
        RANG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
        MARK = ["o", "s", "D", "^"]
        n = len(self.natija)
        ustun = max(1, min(ustun, n))
        qator = (n + ustun - 1) // ustun
        figs = []
        # (1) xato -- sigma egri chiziqlari
        fig, axs = plt.subplots(qator, ustun, figsize=(3.8 * ustun, 3.0 * qator), squeeze=False)
        for ax, r in zip(axs.flat, self.natija):
            s = [float(t[0]) for t in r["egri"]]
            e = [max(float(abs(t[1])), 1e-30) for t in r["egri"]]
            ax.semilogy(s, e, color=RANG[0], marker="o", ms=3)
            em = abs(r["jadval"][("Eyler-Makloren", self.N_tanlash)]) if ("Eyler-Makloren", self.N_tanlash) in r["jadval"] else None
            if em:
                ax.axhline(float(em), color=RANG[1], lw=1.2, ls="--", label="Eyler–Makloren")
            ax.axvline(float(r["s_opt"]), color=RANG[2], lw=1.2, ls=":", label=f"σ_opt = {mp.nstr(r['s_opt'], 3)}")
            ax.set_title(r["F"]["nom"], fontsize=9.5)
            ax.set_xlabel("σ")
            ax.legend(fontsize=7)
        for ax in list(axs.flat)[n:]:
            ax.axis("off")
        fig.suptitle(f"|xato| σ ga qarab, N = {self.N_tanlash}" + (" (Runge bahosi)" if self.usul == "runge" else ""), fontsize=11)
        fig.tight_layout()
        figs.append(fig)
        # (2) yaqinlashish: optimal(σ_opt) va boshqa usullar
        fig, axs = plt.subplots(qator, ustun, figsize=(3.8 * ustun, 3.0 * qator), squeeze=False)
        for ax, r in zip(axs.flat, self.natija):
            us = [u for u in self.usullar(r) if u != "Gauss*"]
            us = [us[0], "Trapetsiya", "Simpson", "Eyler-Makloren"]
            for k, u in enumerate(us):
                pts = [(N_, abs(r["jadval"][(u, N_)])) for N_ in self.Nlar if r["jadval"][(u, N_)] not in (None, 0)]
                ax.loglog([p[0] for p in pts], [float(p[1]) for p in pts], color=RANG[k], marker=MARK[k], ms=3.5,
                          label=u.replace("Optimal ", "Opt. "))
            ax.set_title(r["F"]["nom"], fontsize=9.5)
            ax.set_xlabel("N")
            ax.set_xticks(list(self.Nlar))
            ax.set_xticklabels([str(n) for n in self.Nlar])
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.legend(fontsize=7, loc="lower left")
        for ax in list(axs.flat)[n:]:
            ax.axis("off")
        fig.suptitle("|xato|: tanlangan σ dagi optimal formula va boshqa usullar", fontsize=11)
        fig.tight_layout()
        figs.append(fig)
        if fayl:
            for i, f in enumerate(figs, 1):
                f.savefig(fayl.replace(".png", f"_{i}.png"), dpi=150)
        return figs


# =============================================================================
# 6. Notebook uchun yordamchilar
# =============================================================================
RANG = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # kategorik palitra, qat'iy tartibda
MARKER = ["o", "s", "D", "^"]


def uslub():
    """Grafiklar uchun yagona uslub."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "lines.linewidth": 2,
        "figure.dpi": 110,
    })


def son(v, n=3):
    if v is None:
        return "–"
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, (mp.mpf, float, int)) and not isinstance(v, bool):
        return mp.nstr(mp.mpf(v), n)
    return str(v)


def jadval(sarlavha, qatorlar, n=3):
    """Markdown jadvalni notebookda ko'rsatadi (IPython bo'lmasa -- matn)."""
    s = "| " + " | ".join(sarlavha) + " |\n|" + "---|" * len(sarlavha) + "\n"
    for r in qatorlar:
        s += "| " + " | ".join(son(v, n) for v in r) + " |\n"
    try:
        from IPython.display import display, Markdown
        display(Markdown(s))
    except ImportError:
        print(s)


def usullar_xatosi(F, N, s):
    """Bitta N da barcha usullarning xatosi (I - Q): optimal(s), trapetsiya, Simpson, Eyler-Makloren, Gauss."""
    I = F["I"]
    d = {"Optimal": I - optimal(F, N, s)}
    for nom, fn in USULLAR.items():
        Q = fn(F, N)
        d[nom] = None if Q is None else I - Q
    return d


def yaqinlashish_grafigi(ax, F, s, Nlar=(4, 8, 16, 32, 64), sarlavha=None):
    """|xato| -- N (log-log): optimal(s), trapetsiya, Simpson, Eyler-Makloren. Nol xato 1e-40 da chiziladi."""
    import matplotlib.pyplot as plt
    seriya = [(f"Optimal, σ={mp.nstr(mp.mpf(s), 3)}", lambda N: optimal(F, N, s)),
              ("Trapetsiya", lambda N: trapetsiya(F, N)),
              ("Simpson", lambda N: simpson(F, N)),
              ("Eyler-Makloren", lambda N: eyler_makloren(F, N))]
    for k, (nom, fn) in enumerate(seriya):
        pts = []
        for N in Nlar:
            Q = fn(N)
            if Q is not None:
                pts.append((N, max(float(abs(F["I"] - Q)), 1e-40)))
        ax.loglog([p[0] for p in pts], [p[1] for p in pts], color=RANG[k], marker=MARKER[k], ms=4, label=nom)
    ax.set_xticks(list(Nlar))
    ax.set_xticklabels([str(n) for n in Nlar])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("N")
    ax.set_title(sarlavha or F["nom"], fontsize=9.5)
    ax.legend(fontsize=7.5, loc="best")


# =============================================================================
# 5. Buyruq qatori
# =============================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Optimal sigma ni tanlash va usullarni taqqoslash")
    p.add_argument("--a", type=float, default=1.0, help="sigma oralig'ining boshi")
    p.add_argument("--b", type=float, default=2.0, help="sigma oralig'ining oxiri")
    p.add_argument("--qadam", type=float, default=0.2)
    p.add_argument("--ochiq", action="store_true", help="oraliq ochiq: a va b kirmaydi")
    p.add_argument("--N", type=int, default=16, help="sigma tanlanadigan N")
    p.add_argument("--Nlar", type=int, nargs="+", default=[8, 16, 32, 64])
    p.add_argument("--usul", choices=["aniq", "runge", "nazariy"], default="aniq")
    p.add_argument("--funksiyalar", nargs="+",
                   default=["1/(1+x)", "sqrt(1+x)", "log(1+x)", "x**4", "1/(1+25*x**2)", "exp(-1.5*x)"])
    p.add_argument("--csv", default="sigma_tanlash_natija.csv")
    p.add_argument("--grafik", default="sigma_tanlash.png")
    a = p.parse_args()
    d = Dastur(a.a, a.b, a.qadam, a.N, a.Nlar, a.usul, not a.ochiq)
    d.ishga_tushir(a.funksiyalar)
    d.hisobot()
    print("CSV:", d.csv(a.csv))
    try:
        import matplotlib
        matplotlib.use("Agg")
        d.grafik(a.grafik)
        print("Grafiklar:", a.grafik.replace(".png", "_1.png"), a.grafik.replace(".png", "_2.png"))
    except ImportError:
        pass
