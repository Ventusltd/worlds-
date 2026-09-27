// p7_world.mjs: the world's own OHL layer (web/world (v08), read-only import) on open-land-01 with the resident
// 20 MW layout as design items. Prints the wires, spans, items, sample points and the world's GS6 flags as JSON.
// node p7_world.mjs <out.json> [height_m]
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
// Paths come from the environment (see gpu-furnace/README.md); nothing local is written here.
const need = k => { const v = process.env[k]; if (!v) throw Error(`set ${k} (see gpu-furnace/README.md)`); return v.replace(/\\/g, '/').replace(/\/$/, ''); };
const worldDir = () => need('GRID_WORLD_JS') + '/';
const FURNACE = need('FURNACE_ROOT');
const LIDAR = need('GRID_LIDAR');
const W = worldDir();
const imp = m => import(pathToFileURL(W + m).href);
const { buildOverhead } = await imp('overhead.mjs');
const { parseGridFile, buildGrid } = await imp('layers/grid.mjs');
const Z = await imp('ohl-zones.mjs');
const G6 = await imp('ohl-gs6.mjs');
const { layoutPlant } = await imp('plant-layout.mjs');

const OUT = process.argv[2], HEIGHT = Number(process.argv[3] || G6.WORKING_HEIGHT.defaultM);
const SRCJ = JSON.parse(readFileSync(LIDAR + '/open-land-01/source.json', 'utf8'));
const O = { e: SRCJ.origin_e_m, n: SRCJ.origin_n_m }, N = 2049;
const buf = readFileSync(FURNACE + '/work/p7-ground-open-land-01.f32');
const A = new Float32Array(buf.buffer, buf.byteOffset, N * N);
const heightAt = (x, y) => {  // local metres, bilinear, rows south to north
  const i = Math.floor(x), j = Math.floor(y);
  if (i < 0 || j < 0 || i >= N - 1 || j >= N - 1) return NaN;
  const fx = x - i, fy = y - j, r0 = j * N, r1 = r0 + N;
  const h0 = A[r0 + i] + (A[r0 + i + 1] - A[r0 + i]) * fx, h1 = A[r1 + i] + (A[r1 + i + 1] - A[r1 + i]) * fx;
  return h0 + (h1 - h0) * fy;
};
const t0 = performance.now();
const model = parseGridFile(JSON.parse(readFileSync(LIDAR + '/open-land-01/grid.odbl.json', 'utf8')));
const g = buildGrid(model, O, heightAt, buildOverhead, heightAt);
const spans = Z.buildSpans(g.wires);

// The resident layout, rebuilt exactly as the furnace asked for it (south, 20 MW, half 350 m about the box centre).
const ce = O.e + 1024, cn = O.n + 1024, h = 350;
const bnd = [[ce - h, cn - h], [ce + h, cn - h], [ce + h, cn + h], [ce - h, cn + h]];
const lp = layoutPlant({ boundary: bnd, targetMW: 20, layout: 'south', options: {}, grid: null, water: [],
  groundAt: (e, n) => heightAt(e - O.e, n - O.n) });
const T = lp.table, F = lp.frame, nT = lp.tables.length / 6, items = [];
const L = ([e, n]) => [e - O.e, n - O.n];
for (let q = 0; q < nT; q++) {        // each table's pile row: along its long axis through the middle of its depth
  const ua = lp.tables[q * 6], va = lp.tables[q * 6 + 1], vm = va + T.depth / 2;
  items.push({ id: `table-${q + 1}`, kind: 'piles', pts: [L(F.en(ua, vm)), L(F.en(ua + T.lenU, vm))] });
}
const st = new Map((lp.stations || []).map(s => [s.id, F.en(s.u, s.v)]));
const comp = F.en(lp.compound.u, lp.compound.v);
(lp.stations || []).forEach((s, i) => items.push({ id: `station-${i + 1}`, kind: 'pad', pts: [L(st.get(s.id))] }));
{ const [x, y] = L(comp), w = lp.compound.w / 2, d = lp.compound.d / 2;
  items.push({ id: 'compound-1', kind: 'pad', area: true, pts: [[x - w, y - d], [x + w, y - d], [x + w, y + d], [x - w, y + d]] }); }
(lp.electrical ? lp.electrical.feeders : []).forEach((f, i) => {
  const pts = [comp, ...(f.hops || []).map(h => st.get(h.to)).filter(Boolean)].map(L);
  items.push({ id: `feeder-${i + 1}`, kind: 'cable', pts });
});

const t1 = performance.now();
const res = Z.checkDesign(items, spans, heightAt, { height: HEIGHT });
const t2 = performance.now();
const samples = items.map(it => ({ id: it.id, half: Z.itemShape(it).half, pts: Z.samplePoints(it) }));
writeFileSync(OUT, JSON.stringify({
  source: 'web/world (v08): overhead.mjs, layers/grid.mjs, ohl-zones.mjs, ohl-gs6.mjs, plant-layout.mjs',
  origin: O, height_m: HEIGHT, disclaimer: G6.DISCLAIMER,
  constants: { zones: G6.ZONES, exclusion: G6.EXCLUSION.byClass, sway: G6.SWAY, position_m: G6.POSITION_ALLOWANCE.m,
    item_half: G6.ITEM_HALF, pad_radius_m: G6.PAD_RADIUS_M, sample_m: Z.SAMPLE_M, wire_step_m: Z.WIRE_STEP_M },
  layout: { built: lp.built, tables: nT, stations: (lp.stations || []).length },
  wires: g.wires.map(w => ({ line: w.line, span: w.span, conductor: w.conductor, cls: w.cls, a: w.a, b: w.b, param: w.param })),
  clearances: g.clearances.map(c => ({ span: c.span, from: c.from, to: c.to, min_m: c.minClearance, at: c.at, required_m: c.required, basis: c.basis })),
  spans: spans.map(s => ({ id: s.id, line: s.line, span: s.span, cls: s.cls, support: s.support, quad: s.quad, sway: s.sway,
    barrierReach: s.barrierReach, zoneReach: s.zoneReach, exclusion: s.exclusion })),
  items, samples, flags: res.flags, goalposts: res.goalposts, unmeasured: res.unmeasured,
  ms: { grid_layout: t1 - t0, check: t2 - t1 }
}));
console.log(JSON.stringify({ wires: g.wires.length, spans: spans.length, items: items.length, flags: res.flags.length,
  height: res.flags.filter(f => f.flag === 'height').length, ms: { grid_layout: t1 - t0, check: t2 - t1 } }));
