#!/usr/bin/env python3
"""
Istalgan funksiya uchun optimal kvadratura formulasini klassik formulalar bilan taqqoslash.

Foydalanuvchi integral ostidagi funksiyani matn ko'rinishida kiritadi; skript

  * aniq integralni (sympy, bo'lmasa yuqori aniqlikdagi mpmath.quad) hisoblaydi;
  * sigma* = -[f''']/[f''] (Teorema 7 dagi h^4 hadini yo'qotadi) va
    sigma_E^2 = [f''']/[f'] (moslangan Euler--Maclaurin uchun) ni topadi;
  * trapetsiya, Simpson, Euler--Maclaurin, optimal formula, moslangan Euler--Maclaurin
    (va ixtiyoriy Gauss--Lejandr) xatoliklarini va yaqinlashish tartiblarini hisoblaydi;
  * Teorema 7 bosh hadini tekshiradi: 720 N^4 R_N  ->  [f''' + sigma f'']_0^1;
  * natijalarni jadval (ekranga va CSV) hamda grafiklar (PNG va PDF) ko'rinishida saqlaydi.

[a, b] oraliq x = a + (b-a) t almashtirish bilan [0, 1] ga keltiriladi; sigma parametri
t o'zgaruvchiga nisbatan olinadi (x bo'yicha sigma_x = sigma / (b - a)).

Misollar:
    python taqqoslash.py "exp(-x)*cos(3*x)"
    python taqqoslash.py "log(1+x)" --sigma 1 --N 4 8 16 32 64 128
    python taqqoslash.py "1/(1+25*x**2)" --oraliq -1 1 --gauss
    python taqqoslash.py                      # funksiya so'raladi

Funksiya sympy sintaksisida yoziladi: exp, log, sin, cos, tan, sinh, cosh, sqrt, atan,
erf, besselj, pi, e; daraja -- **. Matn sympy orqali Python ifodasi sifatida o'qiladi,
shuning uchun faqat o'zingiz yozgan funksiyalarni kiriting.
"""
import argparse
import csv
import os
import re
import sys

import mpmath as mp
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kvadratura as K  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RANG = {"Trapetsiya": "#7f7f7f", "Simpson": "#2ca02c", "Euler-Maclaurin": "#d0743c",
        "Optimal": "#1f5fa8", "Moslangan E-M": "#7b4fc8", "Gauss-Lejandr": "#b8860b"}
MARKER = {"Trapetsiya": "s", "Simpson": "^", "Euler-Maclaurin": "D", "Optimal": "o",
          "Moslangan E-M": "v", "Gauss-Lejandr": "x"}
NOL = mp.mpf(10)**-35          # bundan kichik xatolik "aniq" deb hisoblanadi


# ------------------------------------------------------------------------------ tayyorlash
def tayyorla(matn, a, b, aniq=None):
    """Matndan [0,1] dagi funksiya lug'atini (K.Funksiya) quradi."""
    e = sp.sympify(matn, locals=K._LOC)
    ortiqcha = e.free_symbols - {K.x}
    if ortiqcha:
        raise ValueError("Ifodada faqat x o'zgaruvchisi bo'lishi kerak; ortiqcha: "
                         + ", ".join(map(str, ortiqcha)))
    a, b = sp.nsimplify(a), sp.nsimplify(b)
    if b <= a:
        raise ValueError("Oraliqda a < b bo'lishi kerak")
    et = sp.simplify((b - a) * e.subs(K.x, a + (b - a) * K.x)) if (a, b) != (0, 1) else e
    F = K.Funksiya(et)
    F["matn"] = matn
    F["a"], F["b"] = mp.mpf(sp.N(a, 30)), mp.mpf(sp.N(b, 30))
    if aniq is not None:
        F["I"] = mp.mpf(aniq)
    # Teorema 7 uchun f'' va f''' (t o'zgaruvchida)
    F["d2"] = sp.lambdify(K.x, sp.diff(et, K.x, 2), "mpmath")
    F["d3"] = sp.lambdify(K.x, sp.diff(et, K.x, 3), "mpmath")
    return F


def usullar(F, sigma, gauss):
    """Usul nomi -> (Q_N funksiyasi, ishlatiladigan f va f' qiymatlari soni)."""
    d = {"Trapetsiya": (lambda N: K.trapetsiya(F, N), "(N+1) f"),
         "Simpson": (lambda N: K.simpson(F, N), "(N+1) f, N juft"),
         "Euler-Maclaurin": (lambda N: K.eyler_makloren(F, N), "(N+1) f + 2 f'"),
         "Optimal": (lambda N: K.optimal(F, N, sigma), "(N+1) f + (N+1) f'")}
    if F["sigma2_E"] is not None:
        s2 = F["sigma2_E"]
        d["Moslangan E-M"] = (lambda N: K.em_fitted(F, N, s2), "(N+1) f + 2 f'")
    if gauss:
        d["Gauss-Lejandr"] = (lambda N: K.gauss(F, N), "(N+1) f, boshqa tugunlarda")
    return d


# ------------------------------------------------------------------------------ hisoblash
def xatoliklar(F, U, Nlar):
    jad = {nom: [] for nom in U}
    for nom, (Q, _) in U.items():
        for N in Nlar:
            q = Q(N)
            jad[nom].append(None if q is None else F["I"] - q)
    return jad


def tartiblar(Nlar, R):
    p = [None]
    for i in range(1, len(Nlar)):
        r0, r1 = R[i - 1], R[i]
        if r0 is None or r1 is None or abs(r0) < NOL or abs(r1) < NOL:
            p.append(None)
        else:
            p.append(mp.log(abs(r0 / r1)) / mp.log(mp.mpf(Nlar[i]) / Nlar[i - 1]))
    return p


def son(v, n=3):
    if v is None:
        return "--"
    if abs(v) < NOL:
        return "aniq"
    return mp.nstr(v, n, min_fixed=0, max_fixed=0)


def chop_jadval(F, U, Nlar, jad, sigma):
    print()
    print(f"  Aniq integral: I = {mp.nstr(F['I'], 25)}")
    ss = F["sigma_nazariy"]
    ss_s = mp.nstr(ss, 8) if ss is not None else "aniqlanmaydi ([f'']=0)"
    print(f"  sigma* = -[f''']/[f''] = {ss_s};   ishlatilgan sigma = {mp.nstr(sigma, 8)}")
    if F["b"] - F["a"] != 1:
        print(f"  (x o'zgaruvchida: sigma_x = sigma/(b-a) = {mp.nstr(sigma / (F['b'] - F['a']), 8)})")
    s2 = F["sigma2_E"]
    print(f"  sigma_E^2 = [f''']/[f'] = {mp.nstr(s2, 8) if s2 is not None else 'aniqlanmaydi'}")
    print()
    sar = f"{'N':>5} | " + " | ".join(f"{nom:>16}" for nom in U)
    print("  |R_N| (qavs ichida -- kuzatilgan tartib p)")
    print("  " + sar)
    print("  " + "-" * len(sar))
    p = {nom: tartiblar(Nlar, jad[nom]) for nom in U}
    for i, N in enumerate(Nlar):
        kat = []
        for nom in U:
            r = jad[nom][i]
            s = son(None if r is None else abs(r))
            if p[nom][i] is not None:
                s += f" ({float(p[nom][i]):.2f})"
            kat.append(f"{s:>16}")
        print(f"  {N:>5} | " + " | ".join(kat))
    print()
    print("  Ishlatiladigan qiymatlar:")
    for nom, (_, soni) in U.items():
        print(f"     {nom:16s} {soni}")


def teorema7(F, Nlar, jad, sigma):
    L = lambda t: F["d3"](t) + sigma * F["d2"](t)
    try:
        bosh = L(1) - L(0)
    except Exception:
        return None
    qator = []
    for N, R in zip(Nlar, jad["Optimal"]):
        qator.append((N, 720 * mp.mpf(N)**4 * R))
    print()
    nol = abs(bosh) < mp.mpf(10)**-12 * max(1, abs(L(0)))
    print(f"  Teorema 7 tekshiruvi: 720 N^4 R_N -> [f'''+sigma f'']_0^1 = {'0' if nol else mp.nstr(bosh, 8)}")
    print("     " + "   ".join(f"N={N}: {mp.nstr(v, 7)}" for N, v in qator))
    if nol:
        # sigma = sigma*: h^4 hadi yo'q, bosh had -sigma h^5/(1440 sqrt3) [(Lf)(0)+(Lf)(1)]
        h5 = -sigma / (1440 * mp.sqrt(3)) * (L(0) + L(1))
        print("     h^4 hadi nolga teng (sigma = sigma*), xatolik O(h^5). Teorema 7 dagi h^5 hadi bilan nisbat"
              " (1 ga intilishi kerak):")
        print("     " + "   ".join(f"N={N}: {mp.nstr(R * mp.mpf(N)**5 / h5, 6)}"
                                   for N, R in zip(Nlar, jad["Optimal"])))
    return bosh


def sigma_tahlili(F, N, oraliq, nuqtalar=121):
    smin, smax = oraliq
    tor = [smin + (smax - smin) * k / (nuqtalar - 1) for k in range(nuqtalar)]
    tor = [mp.mpf(s) for s in tor if abs(s) > 1e-6]
    return tor, [abs(F["I"] - K.optimal(F, N, s)) for s in tor]


def sigma_oralig(F, sigma):
    m = max(3.0, 1.5 * abs(float(sigma)))
    ss = F["sigma_nazariy"]
    if ss is not None:
        m = max(m, 1.5 * abs(float(ss)))
    return (-m, m)


# ------------------------------------------------------------------------------ grafiklar
def _log(v):
    return max(float(abs(v)), 1e-40)


def grafiklar(F, U, Nlar, jad, sigma, Ns, sig_tor, sig_xato, fayl):
    try:
        K.uslub()
    except Exception:
        pass
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    a, b = float(F["a"]), float(F["b"])
    e0 = sp.sympify(F["matn"], locals=K._LOC)
    fx = sp.lambdify(K.x, e0, "mpmath")
    dfx = sp.lambdify(K.x, sp.diff(e0, K.x), "mpmath")
    xs = [a + (b - a) * k / 400 for k in range(401)]

    # (a) funksiya va hosilasi
    A = ax[0, 0]
    A.plot(xs, [float(mp.re(fx(v))) for v in xs], color="#1f5fa8", label="$f(x)$")
    A.plot(xs, [float(mp.re(dfx(v))) for v in xs], color="#d0743c", ls="--", label="$f'(x)$")
    A.set_title(f"(a) $f(x)={sp.latex(e0)}$", fontsize=10)
    A.set_xlabel("$x$"); A.legend(fontsize=8); A.grid(alpha=0.3)

    # (b) yaqinlashish
    B = ax[0, 1]
    for nom in U:
        ys = [(N, r) for N, r in zip(Nlar, jad[nom]) if r is not None]
        if not ys:
            continue
        if max(abs(r) for _, r in ys) < NOL:
            B.plot([], [], color=RANG[nom], marker=MARKER[nom], label=nom + " (aniq)")
            continue
        lab = nom + (f", $\\sigma={float(sigma):.3g}$" if nom == "Optimal" else "")
        if any(abs(r) < NOL for _, r in ys):      # ishchi aniqlikka yetgan nuqtalar chizilmaydi
            lab += " (keyin aniq)"
            ys = [(N, r) for N, r in ys if abs(r) >= NOL]
        B.loglog([N for N, _ in ys], [_log(r) for _, r in ys], color=RANG[nom], marker=MARKER[nom],
                 ms=4, label=lab)
    B.set_xticks(Nlar); B.set_xticklabels([str(N) for N in Nlar])
    B.xaxis.set_minor_formatter(plt.NullFormatter())
    B.set_xlabel("$N$"); B.set_ylabel("$|R_N(f)|$"); B.set_title("(b) Yaqinlashish", fontsize=10)
    B.legend(fontsize=7); B.grid(alpha=0.3, which="both")

    # (c) sigma ga sezgirlik
    C = ax[1, 0]
    C.semilogy([float(s) for s in sig_tor], [_log(v) for v in sig_xato], color="#1f5fa8", lw=1.3,
               label=f"Optimal, $N={Ns}$")
    em = abs(F["I"] - K.eyler_makloren(F, Ns))
    C.axhline(_log(em), color=RANG["Euler-Maclaurin"], ls="--", lw=1, label="Euler-Maclaurin")
    if "Moslangan E-M" in U:
        fe = abs(F["I"] - K.em_fitted(F, Ns, F["sigma2_E"]))
        if fe > NOL:
            C.axhline(_log(fe), color=RANG["Moslangan E-M"], ls=":", lw=1.2, label="Moslangan E-M")
    if F["sigma_nazariy"] is not None:
        C.axvline(float(F["sigma_nazariy"]), color="#555555", lw=0.8, ls="-.",
                  label=f"$\\sigma^*={float(F['sigma_nazariy']):.3g}$")
    C.set_xlabel("$\\sigma$"); C.set_ylabel(f"$|R_{{{Ns}}}(f)|$")
    C.set_title(f"(c) Xatolikning $\\sigma$ ga bog'liqligi, $N={Ns}$", fontsize=10)
    C.legend(fontsize=7); C.grid(alpha=0.3)

    # (d) optimal koeffitsiyentlar
    D = ax[1, 1]
    Cb = K.C1(sigma, Ns)
    h = 1 / Ns
    D.semilogy(range(Ns + 1), [_log(c) for c in Cb], "o-", color="#1f5fa8", ms=4, label="$|C_{\\beta,1}|$")
    D.axhline(h**2 / 12, color="#d0743c", ls="--", lw=1, label="$h^2/12$ (Euler-Maclaurin)")
    D.set_xlabel("$\\beta$"); D.set_title(f"(d) Optimal koeffitsiyentlar, $\\sigma={float(sigma):.3g}$, $N={Ns}$",
                                          fontsize=10)
    D.legend(fontsize=7); D.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(fayl + ".png", dpi=150)
    fig.savefig(fayl + ".pdf")
    plt.close(fig)


def csv_yoz(F, U, Nlar, jad, sigma, fayl):
    with open(fayl, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["funksiya", F["matn"], "a", mp.nstr(F["a"], 15), "b", mp.nstr(F["b"], 15),
                    "I", mp.nstr(F["I"], 30), "sigma", mp.nstr(sigma, 15)])
        w.writerow(["N"] + [f"R_N {nom}" for nom in U] + [f"p {nom}" for nom in U])
        p = {nom: tartiblar(Nlar, jad[nom]) for nom in U}
        for i, N in enumerate(Nlar):
            w.writerow([N] + [("" if jad[nom][i] is None else mp.nstr(jad[nom][i], 20)) for nom in U]
                       + [("" if p[nom][i] is None else f"{float(p[nom][i]):.4f}") for nom in U])


# ------------------------------------------------------------------------------ asosiy
def main(argv=None):
    ap = argparse.ArgumentParser(description="Optimal kvadratura formulasini istalgan funksiyada "
                                             "klassik formulalar bilan taqqoslash.")
    ap.add_argument("funksiya", nargs="?", help='masalan: "exp(-x)*cos(3*x)"')
    ap.add_argument("--oraliq", nargs=2, type=float, default=[0.0, 1.0], metavar=("A", "B"),
                    help="integrallash oralig'i (standart: 0 1)")
    ap.add_argument("--N", nargs="+", type=int, default=[4, 8, 16, 32, 64],
                    help="qadamlar soni (standart: 4 8 16 32 64)")
    ap.add_argument("--sigma", default="auto",
                    help='optimal formula parametri: son yoki "auto" (sigma*, bo\'lmasa 1)')
    ap.add_argument("--Ns", type=int, default=16, help="sigma tahlili va koeffitsiyentlar grafigi uchun N")
    ap.add_argument("--sigma-oraliq", nargs=2, type=float, default=None, metavar=("SMIN", "SMAX"),
                    help="(c) grafik uchun sigma oralig'i")
    ap.add_argument("--aniq", type=str, default=None, help="aniq integral qiymati (agar ma'lum bo'lsa)")
    ap.add_argument("--gauss", action="store_true", help="Gauss-Lejandr (N+1 tugun) ni ham qo'shish")
    ap.add_argument("--xonalar", type=int, default=40, help="hisoblash aniqligi, o'nli xonalar (standart 40)")
    ap.add_argument("--chiqish", default=None, help="natijalar papkasi (standart: natijalar/<funksiya>)")
    ap.add_argument("--grafiksiz", action="store_true", help="grafiklarni chizmaslik")
    arg = ap.parse_args(argv)

    mp.mp.dps = arg.xonalar
    matn = arg.funksiya or input("f(x) = ").strip()
    if not matn:
        ap.error("funksiya kiritilmadi")
    a, b = arg.oraliq
    Nlar = sorted(set(arg.N))
    if Nlar[0] < 1:
        ap.error("N >= 1 bo'lishi kerak")

    print(f"\n  f(x) = {matn},  oraliq [{a:g}, {b:g}],  {mp.mp.dps} xonali aniqlik")
    try:
        F = tayyorla(matn, a, b, arg.aniq)
    except Exception as ex:
        print(f"  Xato: {ex}")
        return 1

    if arg.sigma == "auto":
        sigma = F["sigma_nazariy"] if F["sigma_nazariy"] is not None else mp.mpf(1)
        if abs(sigma) < mp.mpf(10)**-8:
            sigma = mp.mpf(1)
    else:
        sigma = mp.mpf(arg.sigma)
    if abs(sigma) < mp.mpf(10)**-12:
        print("  Xato: sigma = 0 bo'lishi mumkin emas (bu holda formula Euler-Maclaurin ga aylanadi)")
        return 1
    if F["sigma_nazariy"] is not None and abs(F["sigma_nazariy"]) / Nlar[-1] > 1.4:
        print(f"  Ogohlantirish: sigma* = {mp.nstr(F['sigma_nazariy'], 5)} katta; asimptotik rejim "
              f"(sigma*h <~ 1) faqat N >~ {int(abs(F['sigma_nazariy']))} da boshlanadi.")

    U = usullar(F, sigma, arg.gauss)
    jad = xatoliklar(F, U, Nlar)
    chop_jadval(F, U, Nlar, jad, sigma)
    teorema7(F, Nlar, jad, sigma)

    slug = re.sub(r"[^A-Za-z0-9]+", "_", matn).strip("_")[:40] or "funksiya"
    papka = arg.chiqish or os.path.join(os.getcwd(), "natijalar", slug)
    os.makedirs(papka, exist_ok=True)
    csv_yoz(F, U, Nlar, jad, sigma, os.path.join(papka, "xatoliklar.csv"))
    if not arg.grafiksiz:
        oraliq = tuple(arg.sigma_oraliq) if arg.sigma_oraliq else sigma_oralig(F, sigma)
        tor, xato = sigma_tahlili(F, arg.Ns, oraliq)
        grafiklar(F, U, Nlar, jad, sigma, arg.Ns, tor, xato, os.path.join(papka, "taqqoslash"))
        print(f"\n  Grafiklar: {os.path.join(papka, 'taqqoslash.png')} (va .pdf)")
    print(f"  Jadval:    {os.path.join(papka, 'xatoliklar.csv')}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
