/* Optimal kvadratura formulasi (W_2^(3,2)) va taqqoslanadigan formulalar: double aniqlikdagi JavaScript.
   Python manbasi (40 xonali): ../code/kvadratura.py. Brauzerda ham, Node.js da ham ishlaydi. */
/* ---------- quadrature core (double precision), shared by the page and the tests ---------- */
const AK = [1/720, -1/30240, 1/1209600, -1/47900160, 691/1307674368000, -1/74724249600,
  3617/10670622842880000, -43867/5109094217170944000, 174611/802857662698291200000];
/* AK[k-2] = -B_{2k}/(2k)!,  k = 2..10 */
function dser(z) { let s = 0, p = z; for (const a of AK) { s += a * p; p *= z; } return s; }
/* end weight of the fitted Euler-Maclaurin rule divided by h^2; z = sigma^2 h^2 (negative: trigonometric) */
function cfun(z) {
  if (Math.abs(z) < 1) return 1 / 12 - dser(z);
  if (z > 0) { const x = Math.sqrt(z); return (x / 2 / Math.tanh(x / 2) - 1) / z; }
  const y = Math.sqrt(-z); return (1 - (y / 2) / Math.tan(y / 2)) / (-z);
}
/* D/h^2 = 1/12 - c_sigma/h^2 */
function dfun(z) { return Math.abs(z) < 1 ? dser(z) : 1 / 12 - cfun(z); }
const FACT = [1]; for (let k = 1; k < 40; k++) FACT[k] = FACT[k - 1] * k;
/* r = p1/p2 (even in sigma*h), lambda_1 = -2/(r + sqrt(r^2-4)) */
function lambda1(x) {
  x = Math.abs(x); let r;
  if (x < 2) {
    let P1 = 0, P2 = 0, p = 1;
    for (let k = 3; k < 36; k++) {
      P2 += (2 / FACT[k - 1] - 2 ** k / FACT[k]) * p;
      P1 += (2 ** (k + 1) / FACT[k] - 2 ** k / FACT[k - 1]) * p;
      p *= x;
    }
    r = P1 / P2;
  } else {
    const u = Math.exp(-x), u2 = u * u;
    r = (2 * (1 - u2) - 2 * x * (1 + u2)) / (u2 - 1 + 2 * x * u);
  }
  return -2 / (r + Math.sqrt(r * r - 4));
}
/* Optimal derivative weights (simplified closed form, equivalent to Theorem 3 of the paper): C_{beta,1}, beta = 0..N, on [0,1] */
function C1opt(s, N) {
  const h = 1 / N, x = s * h, lam = lambda1(x), D = h * h * dfun(x * x);
  let A, B;                       /* A = (q-lam)/(q-1), B = (1-lam q)/(q-1) */
  if (x > 0) { const iq = Math.exp(-x), den = -Math.expm1(-x); A = (1 - lam * iq) / den; B = (iq - lam) / den; }
  else { const q = Math.exp(x), den = Math.expm1(x); A = (q - lam) / den; B = (1 - lam * q) / den; }
  const lN = lam ** N, k = D * (1 - lam) / (lam * (1 + lN)), mm = k * A, mp = k * B;
  const C = new Array(N + 1);
  for (let b = 1; b < N; b++) C[b] = mm * lam ** b + mp * lam ** (N - b);
  C[0] = h * h / 12 - (lam * mm - lN * mp) / (1 - lam);
  C[N] = -h * h / 12 - (lam * mp - lN * mm) / (1 - lam);
  return C;
}
/* rules on [0,1] for g with derivative dg */
function trap(g, N) { const h = 1 / N; let s = (g(0) + g(1)) / 2; for (let b = 1; b < N; b++) s += g(b * h); return s * h; }
function simpson(g, N) { if (N % 2) return null; const h = 1 / N; let s = g(0) + g(1); for (let b = 1; b < N; b++) s += (b % 2 ? 4 : 2) * g(b * h); return s * h / 3; }
function em(g, dg, N) { const h = 1 / N; return trap(g, N) + h * h / 12 * (dg(0) - dg(1)); }
function emFit(g, dg, N, s2) { const h = 1 / N; return trap(g, N) + h * h * cfun(s2 * h * h) * (dg(0) - dg(1)); }
function optimal(g, dg, N, s) { const h = 1 / N, C = C1opt(s, N); let t = trap(g, N); for (let b = 0; b <= N; b++) t += C[b] * dg(b * h); return t; }
/* adaptive Gauss-Kronrod 7-15 */
const XK = [0.991455371120812639, 0.949107912342758525, 0.864864423359769073, 0.741531185599394440, 0.586087235467691130, 0.405845151377397167, 0.207784955007898468, 0];
const WK = [0.022935322010529225, 0.063092092629978553, 0.104790010322250184, 0.140653259715525919, 0.169004726639267903, 0.190350578064785410, 0.204432940075298892, 0.209482141084727828];
const WG = [0.129484966168869693, 0.279705391489276668, 0.381830050505118945, 0.417959183673469388];
function gk(f, a, b) {
  const c = (a + b) / 2, r = (b - a) / 2; let k = WK[7] * f(c), g = WG[3] * f(c);
  for (let i = 0; i < 7; i++) { const v = f(c - r * XK[i]) + f(c + r * XK[i]); k += WK[i] * v; if (i % 2) g += WG[(i - 1) / 2] * v; }
  return [k * r, Math.abs((k - g) * r)];
}
function integrate(f, a, b) {
  let segs = [[a, b, ...gk(f, a, b)]];
  for (let it = 0; it < 2000; it++) {
    let tot = 0, err = 0, w = 0;
    for (let i = 0; i < segs.length; i++) { tot += segs[i][2]; err += segs[i][3]; if (segs[i][3] > segs[w][3]) w = i; }
    if (err <= 1e-15 * Math.max(1e-300, Math.abs(tot)) || err < 1e-300) return tot;
    const [p, q] = segs[w], m = (p + q) / 2;
    if (m <= p || m >= q) return tot;
    segs.splice(w, 1, [p, m, ...gk(f, p, m)], [m, q, ...gk(f, m, q)]);
  }
  return segs.reduce((s, v) => s + v[2], 0);
}
if (typeof module !== "undefined") module.exports = { C1opt, lambda1, cfun, dfun, trap, simpson, em, emFit, optimal, integrate };
