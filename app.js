/* Kalkulyator sahifasining interfeysi: kiritish, hisoblash, xulosa, jadval va grafiklar. */
{
const $ = id => document.getElementById(id);
class UErr extends Error {}
const KEYS = [["sin(", "sin"], ["cos(", "cos"], ["tan(", "tan"], ["exp(", "eˣ"], ["ln(", "ln"], ["sqrt(", "√"],
  ["^", "xʸ"], ["pi", "π"], ["x", "x"], ["(", "("], [")", ")"], ["*", "×"], ["/", "÷"], ["+", "+"], ["-", "−"]];
const EX = ["exp(-x)*cos(3x)", "ln(1+x)", "sqrt(1+x)", "exp(-x^2)", "1/(1+x^2)", "cosh(2x)", "exp(-2x)", "sin(pi*x)", "x^(5/2)", "sin(20x)"];
const ALLOWED = new Set(["x", "pi", "e", "E", "PI"]);
const M = [
  { id: "tr", name: "Trapetsiya", c: "--c1", exp: 2, uses: "N+1 ta f qiymati" },
  { id: "si", name: "Simpson", c: "--c2", exp: 4, uses: "N+1 ta f qiymati (N juft)" },
  { id: "em", name: "Euler–Maclaurin", c: "--c3", exp: 4, uses: "N+1 ta f va 2 ta f′" },
  { id: "op", name: "Optimal", c: "--c4", exp: 4, uses: "N+1 ta f va N+1 ta f′" },
  { id: "fe", name: "Moslangan E–M", c: "--c5", exp: 6, uses: "N+1 ta f va 2 ta f′" }];
const SUP = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const pw = k => "10" + String(k).split("").map(c => SUP[c]).join("");
const sci = (v, html = true) => {
  const [m, e] = Math.abs(v).toExponential(2).split("e");
  return html ? `${m}·10<sup>${(+e).toString().replace("-", "−")}</sup>` : `${m}·${pw(+e)}`;
};
const num = v => v === 0 ? "0" : Math.abs(v) >= 1e-3 && Math.abs(v) < 1e5 ? (+v.toPrecision(5)).toString() : sci(v);
const msg = (t, err, src = "") => { const m = $("msg"); m.innerHTML = t; m.className = "msg" + (err ? " err" : ""); m.dataset.src = src; };
const hideRes = () => { $("res").style.display = "none"; };

/* ------------- input helpers ------------- */
$("keys").innerHTML = KEYS.map(([v, l], i) => `<button type="button" data-k="${i}" aria-label="${v}">${l}</button>`).join("");
$("chips").innerHTML = "Misollar: " + EX.map(e => `<button type="button" data-e="${e}">${e}</button>`).join("");
const fx = $("fx");
$("keys").addEventListener("click", ev => {
  const b = ev.target.closest("button"); if (!b) return;
  const v = KEYS[+b.dataset.k][0], s = fx.selectionStart ?? fx.value.length, e = fx.selectionEnd ?? s;
  fx.value = fx.value.slice(0, s) + v + fx.value.slice(e); fx.focus(); fx.setSelectionRange(s + v.length, s + v.length); preview();
});
$("chips").addEventListener("click", ev => { const b = ev.target.closest("button"); if (!b) return; fx.value = b.dataset.e; preview(); run(); });
$("clr").addEventListener("click", () => { fx.value = ""; fx.focus(); preview(); msg(""); });
$("smode").addEventListener("change", () => { $("sval").disabled = $("smode").value !== "hand"; });
let tmr; fx.addEventListener("input", () => { clearTimeout(tmr); tmr = setTimeout(preview, 250); });
$("frm").addEventListener("submit", ev => { ev.preventDefault(); run(); });

function prep(s) {
  return s.replace(/(\d),(\d)/g, "$1.$2").replace(/\bln\b/g, "log").replace(/\barctg\b|\barctan\b/g, "atan")
    .replace(/\barcsin\b/g, "asin").replace(/\barccos\b/g, "acos").replace(/\btg\b/g, "tan").replace(/\bctg\b/g, "cot")
    .replace(/\bsh\b/g, "sinh").replace(/\bch\b/g, "cosh").replace(/\bth\b/g, "tanh");
}
function parse(s) {
  const node = math.parse(prep(s));
  node.traverse((n, path, parent) => {
    if (n.isSymbolNode && !(parent && parent.isFunctionNode && path === "fn") && !ALLOWED.has(n.name))
      throw new UErr(`“${n.name}” noma'lum. Faqat x o'zgaruvchisi va pi, e o'zgarmaslaridan foydalaning.`);
  });
  return node;
}
const texOf = node => node.toTex({ parenthesis: "auto", implicit: "hide" }).replace(/\\mathrm\{x\}/g, "x");
function typeset(el, tex) {
  if (window.__mj && MathJax.tex2svgPromise) MathJax.tex2svgPromise(tex, { display: true }).then(n => el.replaceChildren(n)).catch(() => { el.textContent = tex; });
  else el.textContent = tex;
}
function preview() {
  const s = fx.value.trim(); if (!s) { $("tex").textContent = "—"; return; }
  try {
    const node = parse(s), a = $("a").value || "0", b = $("b").value || "1";
    typeset($("tex"), `\\int_{${a}}^{${b}} ${texOf(node)}\\,dx`); if ($("msg").dataset.src === "parse") msg("");
  } catch (e) { $("tex").textContent = "—"; msg(e instanceof UErr ? e.message : "Ifodani o'qib bo'lmadi: qavslar va belgilarni tekshiring.", true, "parse"); }
}
document.addEventListener("mjready", preview);
["a", "b"].forEach(i => $(i).addEventListener("input", () => { clearTimeout(tmr); tmr = setTimeout(preview, 250); }));

/* ------------- computation ------------- */
function realf(code) { return x => { let v = code.evaluate({ x }); if (typeof v !== "number") v = v && v.im === 0 ? v.re : NaN; return v; }; }
function compute(s, a, b, nmax, mode, sval) {
  const node = parse(s), L = b - a;
  let d1, d2, d3;
  try { d1 = math.derivative(node, "x"); d2 = math.derivative(d1, "x"); d3 = math.derivative(d2, "x"); }
  catch (e) { throw new UErr("Bu funksiyaning hosilasini simvolik hisoblab bo'lmadi (masalan, |x| yoki bo'lakli funksiya)."); }
  const f = realf(node.compile()), f1 = realf(d1.compile()), f2 = realf(d2.compile()), f3 = realf(d3.compile());
  const X = t => a + L * t;
  const g = t => L * f(X(t)), dg = t => L * L * f1(X(t)), d2g = t => L ** 3 * f2(X(t)), d3g = t => L ** 4 * f3(X(t));
  for (let k = 0; k <= 8; k++) { const t = k / 8; if (!isFinite(g(t))) throw new UErr(`f(x) x = ${num(X(t))} nuqtada aniqlanmagan yoki cheksiz.`); }
  if (!isFinite(dg(0)) || !isFinite(dg(1))) throw new UErr("f′(x) oraliq chetida cheksiz: hosilali formulalarni qo'llab bo'lmaydi.");
  const I = integrate(g, 0, 1);
  const fin = v => isFinite(v);
  const smooth = fin(d2g(0)) && fin(d2g(1)) && fin(d3g(0)) && fin(d3g(1));
  const J1 = dg(1) - dg(0), J2 = smooth ? d2g(1) - d2g(0) : NaN, J3 = smooth ? d3g(1) - d3g(0) : NaN;
  const sStar = smooth && Math.abs(J2) > 1e-12 * Math.max(1, Math.abs(J3)) ? -J3 / J2 : null;   /* t-variable */
  const s2E = smooth && Math.abs(J1) > 1e-13 * Math.max(1, Math.abs(J3)) ? J3 / J1 : null;
  let sig, note = "";
  if (mode === "hand") { sig = sval * L; if (!(Math.abs(sig) > 1e-10)) throw new UErr("σ nolga teng bo'lmasligi kerak."); }
  else if (sStar !== null && Math.abs(sStar) > 1e-8) sig = sStar;
  else { sig = 1; note = "σ* aniqlanmadi, σ = 1 olindi."; }
  const Ns = []; for (let N = 4; N <= nmax; N *= 2) Ns.push(N);
  const fe = N => { if (s2E === null) return null; if (s2E < 0 && Math.sqrt(-s2E) / N > 6.2) return null; return emFit(g, dg, N, s2E); };
  const Q = { tr: N => trap(g, N), si: N => simpson(g, N), em: N => em(g, dg, N), op: N => optimal(g, dg, N, sig), fe };
  const R = {}; for (const m of M) R[m.id] = Ns.map(N => { const q = Q[m.id](N); return q === null || !isFinite(q) ? null : I - q; });
  const tol = 2e-14 * Math.max(1, Math.abs(I));
  return { node, L, a, b, I, g, dg, d2g, d3g, f, f1, sStar, s2E, sig, note, Ns, R, tol, smooth, J2, J3, mode,
    isStar: sStar !== null && Math.abs(sig - sStar) <= 1e-9 * Math.max(1, Math.abs(sStar)) };
}
function orders(r, Ns, tol) {
  return r.map((v, i) => {
    if (!i || v === null || r[i - 1] === null) return null;
    const A = Math.abs(r[i - 1]), B = Math.abs(v);
    return A > 50 * tol && B > 50 * tol ? Math.log(A / B) / Math.log(Ns[i] / Ns[i - 1]) : null;
  });
}
const lastOrder = p => { for (let i = p.length - 1; i >= 0; i--) if (p[i] !== null) return p[i]; return null; };

/* ------------- verdict ------------- */
const PREF = { op: 0, fe: 1, em: 2, si: 3, tr: 4 };
function ranked(S, n) {
  const e = id => { const v = Math.abs(S.R[id][n]); return v < S.tol ? 0 : v; };
  return M.filter(m => S.R[m.id][n] !== null).sort((p, q) => (e(p.id) - e(q.id)) || (PREF[p.id] - PREF[q.id]));
}
function verdict(S) {
  const n = S.Ns.length - 1, N = S.Ns[n], h = 1 / N, tol = S.tol, out = [];
  const err = id => S.R[id][n], ok = id => err(id) !== null;
  const exact = id => S.R[id].every(v => v !== null && Math.abs(v) < tol);
  const avail = M.filter(m => ok(m.id));
  const rank = ranked(S, n);
  const ord = Object.fromEntries(M.map(m => [m.id, lastOrder(orders(S.R[m.id], S.Ns, tol))]));
  const sx = v => num(v / S.L);
  const win = rank[0];
  /* suitability of the optimal formula */
  const why = []; let suits = true;
  if (exact("op")) why.push(`<span class="good">To'liq mos:</span> optimal formula bu funksiyani mashina aniqligida integrallaydi — f(x) ning ko'rinishi 1, x va e<sup>−σx</sup> kombinatsiyasi (maqoladagi Teorema 5).`);
  else {
    if (!S.smooth) { suits = false; why.push(`<span class="bad">Mos emas:</span> f″ yoki f‴ oraliq chetida cheksiz. Formula silliq (C<sup>5</sup>) funksiyalar uchun mo'ljallangan, shuning uchun σ ni tanlash yordam bermaydi va xatolik klassik formulalar darajasida qoladi.`); }
    else if (S.sStar === null) { suits = false; why.push(`<span class="bad">Foyda yo'q:</span> f″(a) = f″(b), shuning uchun σ* mavjud emas. Xatolikning asosiy qismi (h⁴/720)[f‴ + σf″] bu holda σ ga bog'liq bo'lmay qoladi — σ ni tanlab yutib bo'lmaydi.`); }
    else {
      if (Math.abs(S.sStar) * h > 1.4) { suits = false; why.push(`<span class="bad">Hali erta:</span> σ* = ${sx(S.sStar)} juda katta, bunday N larda formula o'z kuchiga kirmaydi; N ≳ ${Math.ceil(Math.abs(S.sStar))} dan keyin foydasi ko'rinadi.`); }
      else if (Math.abs(S.sStar) / S.Ns[0] > 1.4) why.push(`σ* = ${sx(S.sStar)} nisbatan katta: kichik N larda ustunlik sezilmaydi, N ≳ ${Math.ceil(Math.abs(S.sStar))} dan boshlab namoyon bo'ladi.`);
      if (S.isStar) why.push(`σ = σ* = ${sx(S.sStar)} tanlandi: bu xatolikning asosiy (h⁴) hadini yo'qotadi, shuning uchun xatolik h⁵ kabi kamayadi (Teorema 7).`);
      else why.push(`Siz σ = ${sx(S.sig)} tanladingiz, σ* = ${sx(S.sStar)} esa boshqa: xatolikning h⁴ hadi qoladi. „Avtomatik“ rejimda natija yaxshiroq bo'ladi.`);
    }
    if (ok("em") && ok("op") && Math.abs(err("op")) > tol) {
      const r = Math.abs(err("em") / err("op"));
      if (r >= 1.5) why.push(`N = ${N} da klassik Euler–Maclaurin formulasidan <b>${r >= 10 ? Math.round(r) : r.toFixed(1)} marta</b> aniqroq.`);
      else { suits = false; why.push(`Klassik Euler–Maclaurin formulasidan ${r > 1 / 1.5 ? "deyarli farq qilmaydi" : (1 / r).toFixed(1) + " marta yomonroq"} — qo'shimcha hosilalar o'zini oqlamaydi.`); }
    }
    why.push(suits ? `<b>Xulosa: optimal formula bu funksiya uchun mos.</b>` : `<b>Xulosa: bu funksiya uchun optimal formula yaxshi tanlov emas.</b>`);
  }
  /* why the winner wins */
  let wtxt = "";
  if (exact(win.id)) wtxt = win.id === "fe" ? `U bu funksiyani mashina aniqligida integrallaydi: f(x) 1, x va e<sup>±σx</sup> (yoki cos, sin) kombinatsiyasi.` : win.id === "op" ? `U bu funksiyani mashina aniqligida integrallaydi (Teorema 5).` : `U bu funksiyani mashina aniqligida integrallaydi.`;
  else if (win.id === "fe") wtxt = `Bu formula simmetrik: xatolik faqat h⁴, h⁶, … juft darajalardan iborat. σ<sub>E</sub>² = [f‴]/[f′] tanlanganda h⁴ hadi yo'qoladi va xatolik h⁶ kabi kamayadi. Optimal formula nosimmetrik bo'lgani uchun unda h⁵ hadi qoladi. Eng yomon holatdagi xatolik (norma) ikkalasida deyarli bir xil (Teorema 6), lekin silliq funksiyalarda moslangan E–M odatda yutadi. U faqat 2 ta hosila qiymatidan foydalanadi.`;
  else if (win.id === "op") wtxt = suits ? `σ parametri xatolikning asosiy hadini yo'qotadi (Teorema 7), shuning uchun formula 4-tartibli Simpson va Euler–Maclaurin formulalaridan o'zib ketadi.` : `Birinchi o'rinda, lekin farq kichik va nazariy ustunlik yo'q (yuqoriga qarang).`;
  else if (win.id === "em") wtxt = `Klassik Euler–Maclaurin 4-tartibli va faqat 2 ta hosila ishlatadi; bu funksiyada σ ni sozlash foyda bermadi.`;
  else if (win.id === "si") wtxt = `Simpson hosilasiz 4-tartib beradi; hosila ishlatuvchi formulalar bu funksiyada foyda bermadi (odatda funksiya chetlarda silliq bo'lmaganda).`;
  else wtxt = `Odatda bu funksiya davriy (chetlardagi hosilalar bir xil) yoki tugunlar orasida chiziqli ekanini bildiradi: bunday funksiyalarda trapetsiya formulasining o'zi juda aniq, tuzatmalar esa faqat xalaqit beradi.`;
  out.push(`<h3>Eng aniq: ${win.name}</h3><p>${wtxt}</p>`);
  out.push(`<p>O'rinlar (N = ${N}): ${rank.map((m, i) => `${i + 1}. ${m.name}${exact(m.id) ? " (≈ 0)" : ""}`).join(" · ")}</p>`);
  out.push(`<h3>Optimal formula bu funksiyaga mos keladimi?</h3>` + why.map(t => `<p>${t}</p>`).join(""));
  /* orders */
  const rows = M.filter(m => ok(m.id) && !exact(m.id) && ord[m.id] !== null).map(m => {
    const e = m.id === "op" && S.isStar ? 5 : m.exp, p = ord[m.id];
    return { m, e, p, low: p < e - 0.35 };
  });
  if (rows.length) {
    const low = rows.some(r => r.low);
    out.push(`<h3>Yaqinlashish tezligi</h3><p>${rows.map(r => `${r.m.name}: kutilgan ${r.e}, kuzatilgan <b class="${r.low ? "bad" : ""}">${r.p.toFixed(2)}</b>`).join(" · ")}</p>` +
      `<p>${low ? `<span class="bad">Ba'zi formulalar kutilgan tezlikka yetmayapti.</span> Odatda bu funksiya (yoki uning hosilalari) oraliqda yetarlicha silliq emasligini bildiradi, yoki N hali kichik. Oraliqni bo'lish yoki o'zgaruvchini almashtirish yordam beradi.` : `Barcha formulalar nazariy tezlikda yaqinlashyapti — funksiya yetarlicha silliq.`}</p>`);
  }
  if (S.note) out.push(`<p>${S.note}</p>`);
  return out.join("");
}

/* ------------- output ------------- */
const nm = id => M.find(m => m.id === id).name;
function figures(S) {
  const n = S.Ns.length - 1, N = S.Ns[n], tol = S.tol;
  const best = ranked(S, n)[0];
  const eo = S.R.op[n], ee = S.R.em[n];
  const gain = Math.abs(eo) < tol ? "aniq" : (Math.abs(ee / eo) >= 10 ? Math.round(Math.abs(ee / eo)) : Math.abs(ee / eo).toFixed(1)) + " ×";
  const F = [
    ["Aniq integral", S.I.toPrecision(12), `[${num(S.a)}, ${num(S.b)}] oraliqda`],
    ["σ* (optimal parametr)", S.sStar === null ? "yo'q" : num(S.sStar / S.L), S.mode === "hand" ? `ishlatildi: σ = ${num(S.sig / S.L)}` : (S.note ? "σ = 1 ishlatildi" : "avtomatik tanlandi")],
    ["Eng aniq formula", best.name, `N = ${N}: ${Math.abs(S.R[best.id][n]) < tol ? "≈ 0 (mashina aniqligi)" : sci(Math.abs(S.R[best.id][n]))}`],
    ["Optimal vs Euler–Maclaurin", gain, `N = ${N} dagi xatoliklar nisbati`]];
  $("figs").innerHTML = F.map(([k, v, s]) => `<div class="fig1"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div></div>`).join("");
}
function table(S) {
  const P = Object.fromEntries(M.map(m => [m.id, orders(S.R[m.id], S.Ns, S.tol)]));
  const head = `<thead><tr><th>N</th>${M.map(m => `<th>${m.name}</th>`).join("")}</tr></thead>`;
  const body = S.Ns.map((N, i) => {
    const vals = M.map(m => S.R[m.id][i]), abs = vals.map(v => v === null ? Infinity : Math.abs(v)), mn = Math.min(...abs);
    return `<tr><td>${N}</td>${M.map((m, j) => {
      const v = vals[j]; if (v === null) return `<td>—</td>`;
      const cell = Math.abs(v) < S.tol ? "≈ 0" : sci(Math.abs(v));
      const p = P[m.id][i] === null ? "" : ` <small>(${P[m.id][i].toFixed(2)})</small>`;
      return `<td class="${abs[j] <= mn * 1.0000001 ? "best" : ""}">${cell}${p}</td>`;
    }).join("")}</tr>`;
  }).join("");
  $("tab").innerHTML = head + `<tbody>${body}</tbody>`;
}

/* ------------- charts ------------- */
const CH = {};
function ticksLin(lo, hi, k = 5) {
  const st = Math.pow(10, Math.floor(Math.log10((hi - lo) / k))), m = [1, 2, 5, 10].find(m => (hi - lo) / (st * m) <= k) * st;
  const t = []; for (let v = Math.ceil(lo / m) * m; v <= hi + 1e-12 * m; v += m) t.push(+v.toPrecision(12)); return t;
}
function draw(id) {
  const c = CH[id]; if (!c) return; const s = $(id), W = s.clientWidth, H = s.clientHeight; if (!W || !H) return;
  const small = W < 420, pl = small ? 46 : 58, pr = 12, pt = 10, pb = 36;
  const tx = v => c.logx ? Math.log2(v) : v, ty = v => c.logy ? Math.log10(v) : v;
  const pts = c.series.flatMap(se => se.pts).filter(p => isFinite(tx(p[0])) && isFinite(ty(p[1])));
  const extra = (c.hl || []).map(l => ty(l.y)).filter(isFinite);
  if (!pts.length) { s.innerHTML = `<text x="${W / 2}" y="${H / 2}" text-anchor="middle" style="fill:var(--mut);font-size:14px">Ko'rsatadigan ma'lumot yo'q</text>`; return; }
  let x0 = Math.min(...pts.map(p => tx(p[0]))), x1 = Math.max(...pts.map(p => tx(p[0])));
  let y0 = Math.min(...pts.map(p => ty(p[1])), ...extra), y1 = Math.max(...pts.map(p => ty(p[1])), ...extra);
  if (c.logy) { y0 = Math.floor(y0); y1 = Math.ceil(y1); if (y1 === y0) y1++; } else { const d = (y1 - y0) || 1; y0 -= 0.06 * d; y1 += 0.06 * d; }
  if (x1 === x0) { x0 -= 1; x1 += 1; }
  const X = v => pl + (tx(v) - x0) / (x1 - x0) * (W - pl - pr), Y = v => pt + (y1 - ty(v)) / (y1 - y0) * (H - pt - pb);
  let o = "";
  const yt = c.logy ? (() => { const st = Math.max(1, Math.ceil((y1 - y0) / (small ? 4 : 6))); const t = []; for (let k = y0; k <= y1; k += st) t.push(k); return t; })() : ticksLin(y0, y1, small ? 4 : 5);
  for (const k of yt) { const yy = c.logy ? pt + (y1 - k) / (y1 - y0) * (H - pt - pb) : Y(k);
    o += `<line x1="${pl}" x2="${W - pr}" y1="${yy}" y2="${yy}" style="stroke:var(--line)"/><text x="${pl - 6}" y="${yy + 4}" text-anchor="end" style="fill:var(--mut);font-size:12px">${c.logy ? pw(k) : +k.toPrecision(4)}</text>`; }
  const xt = c.xt || ticksLin(x0, x1, small ? 4 : 6);
  for (const v of xt) { const xx = c.logx ? X(v) : pl + (v - x0) / (x1 - x0) * (W - pl - pr);
    o += `<line x1="${xx}" x2="${xx}" y1="${pt}" y2="${H - pb}" style="stroke:var(--line)"/><text x="${xx}" y="${H - pb + 16}" text-anchor="middle" style="fill:var(--mut);font-size:12px">${+(+v).toPrecision(4)}</text>`; }
  o += `<text x="${(pl + W - pr) / 2}" y="${H - 4}" text-anchor="middle" style="fill:var(--mut);font-size:12px">${c.xl}</text>`;
  for (const l of c.hl || []) { const yy = Y(l.y); o += `<line x1="${pl}" x2="${W - pr}" y1="${yy}" y2="${yy}" style="stroke:var(${l.c});stroke-width:1.5;stroke-dasharray:6 4"/>`; }
  if (c.vl !== undefined && c.vl !== null && c.vl >= Math.min(x0, x1) && c.vl <= Math.max(x0, x1)) { const xx = pl + (c.vl - x0) / (x1 - x0) * (W - pl - pr); o += `<line x1="${xx}" x2="${xx}" y1="${pt}" y2="${H - pb}" style="stroke:var(--ink);stroke-width:1;stroke-dasharray:2 3"/>`; }
  for (const se of c.series) {
    const P = se.pts.filter(p => isFinite(tx(p[0])) && isFinite(ty(p[1]))); if (!P.length) continue;
    o += `<polyline fill="none" style="stroke:var(${se.c});stroke-width:2" points="${P.map(p => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(" ")}"/>`;
    if (se.dots) o += P.map(p => `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="3.2" style="fill:var(${se.c})"/>`).join("");
  }
  s.innerHTML = o;
}
const RO = new ResizeObserver(es => es.forEach(e => draw(e.target.id)));
["ch1", "ch2", "ch3", "ch4"].forEach(i => RO.observe($(i)));
const legend = (el, items) => { $(el).innerHTML = items.map(([n, c]) => `<span style="--c:var(${c})">${n}</span>`).join(""); };
function charts(S) {
  const tol = S.tol;
  const ser = M.map(m => ({ c: m.c, dots: true, name: m.name, pts: S.Ns.map((N, i) => [N, S.R[m.id][i] === null ? NaN : Math.abs(S.R[m.id][i])]).filter(p => p[1] > tol) }));
  CH.ch1 = { logx: true, logy: true, xl: "N", xt: S.Ns, series: ser };
  legend("lg1", M.map(m => [m.name + (S.R[m.id].every(v => v !== null && Math.abs(v) < tol) ? " (≈ 0, ko'rinmaydi)" : ""), m.c]));
  const Nq = 16, sxs = S.sStar === null ? null : S.sStar / S.L, used = S.sig / S.L;
  const lim = Math.max(3, 1.5 * Math.abs(used), sxs === null ? 0 : 1.5 * Math.abs(sxs));
  const pts = []; for (let k = 0; k <= 200; k++) { const sv = -lim + 2 * lim * k / 200; if (Math.abs(sv) < 1e-6 * lim) continue;
    const e = Math.abs(S.I - optimal(S.g, S.dg, Nq, sv * S.L)); pts.push([sv, Math.max(e, tol)]); }
  const hl = [{ y: Math.max(Math.abs(S.I - em(S.g, S.dg, Nq)), tol), c: "--c3" }];
  if (S.s2E !== null && !(S.s2E < 0 && Math.sqrt(-S.s2E) / Nq > 6.2)) hl.push({ y: Math.max(Math.abs(S.I - emFit(S.g, S.dg, Nq, S.s2E)), tol), c: "--c5" });
  CH.ch2 = { logy: true, xl: "σ", series: [{ c: "--c4", pts }], hl, vl: sxs };
  legend("lg2", [["Optimal formula", "--c4"], ["Euler–Maclaurin (σ siz)", "--c3"], ...(hl.length > 1 ? [["Moslangan E–M", "--c5"]] : [])]);
  $("ch2cap").textContent = `Optimal formula, N = ${Nq}. ${sxs === null ? "σ* mavjud emas." : `Vertikal chiziq — σ* = ${num(sxs)}.`} Gorizontal chiziqlar — boshqa formulalar.`;
  const C = C1opt(S.sig, Nq);
  CH.ch3 = { logy: true, xl: "β", series: [{ c: "--c4", dots: true, pts: C.map((v, b) => [b, Math.max(Math.abs(v), 1e-30)]) }], hl: [{ y: 1 / Nq / Nq / 12, c: "--c3" }] };
  $("ch3cap").innerHTML = `|C<sub>β,1</sub>| (N = ${Nq}, [0,1] da): chetlarda ≈ h²/12 (shtrix chiziq), ichkariga qarab har qadamda ≈ 3.7 marta kamayadi.`;
  const fp = [], dp = []; for (let k = 0; k <= 300; k++) { const x = S.a + (S.b - S.a) * k / 300; fp.push([x, S.f(x)]); dp.push([x, S.f1(x)]); }
  CH.ch4 = { xl: "x", series: [{ c: "--c4", pts: fp }, { c: "--c3", pts: dp }] };
  legend("lg4", [["f(x)", "--c4"], ["f′(x)", "--c3"]]);
  ["ch1", "ch2", "ch3", "ch4"].forEach(draw);
}

function run() {
  const s = fx.value.trim(); if (!s) { msg("Avval funksiyani kiriting.", true); hideRes(); fx.focus(); return; }
  const a = parseFloat($("a").value), b = parseFloat($("b").value);
  if (!isFinite(a) || !isFinite(b) || !(b > a)) { msg("Oraliqni tekshiring: a < b bo'lishi kerak.", true); hideRes(); return; }
  let S;
  try { S = compute(s, a, b, +$("nmax").value, $("smode").value, parseFloat($("sval").value)); }
  catch (e) { msg(e instanceof UErr ? e.message : "Ifodani o'qib bo'lmadi: qavslar va belgilarni tekshiring.", true); hideRes(); return; }
  try { localStorage.setItem("okf_fx", s); } catch (e) { }
  msg(`Hisoblandi: N = ${S.Ns.join(", ")}.`);
  figures(S); $("verdict").innerHTML = verdict(S); table(S);
  const r = $("res"); r.style.display = "block"; r.classList.remove("show"); void r.offsetWidth; r.classList.add("show");
  charts(S);
}
let start = "exp(-x)*cos(3x)"; try { start = localStorage.getItem("okf_fx") || start; } catch (e) { }
fx.value = start; preview(); run();
}
