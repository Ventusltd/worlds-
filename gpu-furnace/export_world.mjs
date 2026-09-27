// export_world.mjs: runs the world's own JS (web/world, read-only import) and prints JSON for the furnace.
// node export_world.mjs <request.json> <out.json>
// request.mode: "layout" {boundary, targetMW, layout, options, grid, ground:{file,e0,n0,w,h}} | "cables" {kv}
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
// Paths come from the environment (see gpu-furnace/README.md); nothing local is written here.
const need = k => { const v = process.env[k]; if (!v) throw Error(`set ${k} (see gpu-furnace/README.md)`); return v.replace(/\\/g, '/').replace(/\/$/, ''); };
const worldDir = () => need('GRID_WORLD_JS') + '/';
const W = worldDir();
const imp = m => import(pathToFileURL(W + m).href);
const req = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = { mode: req.mode, source: 'web/world (v08)' };
const t0 = performance.now();
if (req.mode === 'layout') {
  const { layoutPlant } = await imp('plant-layout.mjs');
  const G = req.ground, buf = readFileSync(G.file);
  const a = new Float32Array(buf.buffer, buf.byteOffset, G.w * G.h);
  const groundAt = (e, n) => {            // bilinear, rows south to north, 1 m spacing
    const x = e - G.e0, y = n - G.n0, i = Math.floor(x), j = Math.floor(y);
    if (i < 0 || j < 0 || i >= G.w - 1 || j >= G.h - 1) return NaN;
    const fx = x - i, fy = y - j, r0 = j * G.w, r1 = r0 + G.w;
    const h0 = a[r0 + i] + (a[r0 + i + 1] - a[r0 + i]) * fx, h1 = a[r1 + i] + (a[r1 + i + 1] - a[r1 + i]) * fx;
    return h0 + (h1 - h0) * fy;
  };
  const r = layoutPlant({ boundary: req.boundary, targetMW: req.targetMW, layout: req.layout, options: req.options || {},
    grid: req.grid || null, water: [], groundAt });
  const T = r.table, F = r.frame, nT = r.tables.length / 6, tables = [];
  for (let q = 0; q < nT; q++) {
    const [ua, va, ...h] = r.tables.slice(q * 6, q * 6 + 6), ub = ua + T.lenU, vb = va + T.depth;
    const c = [[ua, va], [ub, va], [ub, vb], [ua, vb]].map(([u, v]) => F.en(u, v));
    tables.push([...c.flat(), ...h]);
  }
  const lines = [];
  const seg = (u0, v0, u1, v1, kind) => lines.push([...F.en(u0, v0), ...F.en(u1, v1), kind]);
  if (r.spine) seg(r.spine.u ?? 0, r.spine.v0 ?? 0, r.spine.u ?? 0, r.spine.v1 ?? 0, 'spine');
  out.layout = { label: r.label, layout: r.layout, fits: r.fits, asked: r.asked, built: r.built, notes: r.notes,
    table: { lenU: T.lenU, depth: T.depth, pitch: T.pitch, kWp: T.kWp, modules: T.modules, tiltDeg: T.tilt * 180 / Math.PI, rise: T.rise },
    stations: (r.stations || []).map(s => ({ id: s.id, en: F.en(s.u, s.v) })),
    compound: { en: F.en(r.compound.u, r.compound.v), w: r.compound.w, d: r.compound.d },
    feeders: r.electrical ? r.electrical.feeders.map(f => ({ id: f.id, dropPct: f.dropPct,
      hops: (f.hops || []).map(h => ({ from: h.from, to: h.to, areaMm2: h.areaMm2, loadMVA: h.loadMVA, lengthM: h.lengthM })) })) : [],
    worstDropPct: r.electrical ? r.electrical.worstDropPct : null, cableM: r.electrical ? r.electrical.cableM : null,
    template: { counts: r.template.counts, ratings: r.template.ratings, nodes: r.template.nodes.length, links: r.template.links.length },
    tables, lines };
} else if (req.mode === 'cables') {
  const C = await imp('cable-rating.mjs'), kv = req.kv || 33;
  out.cables = C.SIZES.map(a => { const z = C.impedance(a, { kv }), r = C.rating(a, { kv });
    return { area: a, r_ohm_m: z.r, x_ohm_m: z.x, amps: r.amps, mva: r.mva }; });
  if (req.mva) { out.sizeFor = C.sizeFor(req.mva, { kv }); if (out.sizeFor && req.lengthM)
    out.dropPct = C.dropPercent(out.sizeFor.area, req.lengthM, req.mva, { kv, pf: req.pf || 0.95 }); }
}
out.ms = performance.now() - t0;
writeFileSync(process.argv[3], JSON.stringify(out));
