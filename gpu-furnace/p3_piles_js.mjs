// p3_piles_js.mjs: the world's own automatic piling (web/world v09 plant-layout.mjs + plant-piles.mjs, read-only import),
// run on a ground window, as the reference the VRAM piling is read against.
// node p3_piles_js.mjs <request.json> <out.json>
// request: { boundary, targetMW, layout, ground: { file, e0, n0, w, h } }
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
// Paths come from the environment (see gpu-furnace/README.md); nothing local is written here.
const need = k => { const v = process.env[k]; if (!v) throw Error(`set ${k} (see gpu-furnace/README.md)`); return v.replace(/\\/g, '/').replace(/\/$/, ''); };
const worldDir = () => need('GRID_WORLD_JS') + '/';
const W = worldDir();
const imp = m => import(pathToFileURL(W + m).href);
const req = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const { layoutPlant } = await imp('plant-layout.mjs');
const { pilesForLayout, STRIDE } = await imp('plant-piles.mjs');
const { PILE_RULES } = await imp('pile-rules.mjs');
const G = req.ground, buf = readFileSync(G.file);
const a = new Float32Array(buf.buffer, buf.byteOffset, G.w * G.h);
const groundAt = (e, n) => {
  const x = e - G.e0, y = n - G.n0, i = Math.floor(x), j = Math.floor(y);
  if (i < 0 || j < 0 || i >= G.w - 1 || j >= G.h - 1) return NaN;
  const fx = x - i, fy = y - j, r0 = j * G.w, r1 = r0 + G.w;
  const h0 = a[r0 + i] + (a[r0 + i + 1] - a[r0 + i]) * fx, h1 = a[r1 + i] + (a[r1 + i + 1] - a[r1 + i]) * fx;
  return h0 + (h1 - h0) * fy;
};
const t0 = performance.now();
const r = layoutPlant({ boundary: req.boundary, targetMW: req.targetMW, layout: req.layout, options: {}, grid: null, water: [], groundAt });
const t1 = performance.now();
const gen = pilesForLayout(r, groundAt);
let s; do { s = gen.next(); } while (!s.done);
const res = s.value, t2 = performance.now();
const T = r.table, out = {
  source: 'web/world (v09) plant-layout.mjs + plant-piles.mjs', built: r.built,
  frame: { swap: r.frame.swap }, table: { lenU: T.lenU, depth: T.depth, rise: T.rise, south: T.south, lowEdgeM: r.params.lowEdgeM },
  rules: PILE_RULES, stride: STRIDE, posts: res.posts, summary: res.summary, unmeasured: res.unmeasured,
  tables: res.tables.map(x => ({ ua: null, key: x.T.key, twistDeg: x.twistDeg, twisted: x.twisted, grading: x.grading, flags: x.flags,
    n: x.n, cutM3: x.cutM3, fillM3: x.fillM3, maxDepth: x.maxDepth, pa: Array.from(x.pa), pf: Array.from(x.pf) })),
  uv: [], ms: { layout: t1 - t0, piles: t2 - t1 } };
for (let q = 0; q < r.tables.length; q += 6) out.uv.push([r.tables[q], r.tables[q + 1]]);
writeFileSync(process.argv[3], JSON.stringify(out));
