// p5_export.mjs: runs the world's own layout (web/world (v08), read-only import) and dumps the earthworks geometry:
// tables, stations, compound, spine, access, track roads, feeder trench pieces. node p5_export.mjs <req.json> <out.json>
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
// Paths come from the environment (see gpu-furnace/README.md); nothing local is written here.
const need = k => { const v = process.env[k]; if (!v) throw Error(`set ${k} (see gpu-furnace/README.md)`); return v.replace(/\\/g, '/').replace(/\/$/, ''); };
const worldDir = () => need('GRID_WORLD_JS') + '/';
const W = worldDir();
const { layoutPlant } = await import(pathToFileURL(W + 'plant-layout.mjs').href);
const req = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const G = req.ground, buf = readFileSync(G.file);
const a = new Float32Array(buf.buffer, buf.byteOffset, G.w * G.h);
const groundAt = (e, n) => {
  const x = e - G.e0, y = n - G.n0, i = Math.floor(x), j = Math.floor(y);
  if (i < 0 || j < 0 || i >= G.w - 1 || j >= G.h - 1) return NaN;
  const fx = x - i, fy = y - j, r0 = j * G.w, r1 = r0 + G.w;
  const h0 = a[r0 + i] + (a[r0 + i + 1] - a[r0 + i]) * fx, h1 = a[r1 + i] + (a[r1 + i + 1] - a[r1 + i]) * fx;
  return h0 + (h1 - h0) * fy;
};
const r = layoutPlant({ boundary: req.boundary, targetMW: req.targetMW, layout: req.layout, options: {}, grid: null, water: [], groundAt });
const E = r.electrical;
const out = { swap: r.frame.swap, params: r.params, table: { lenU: r.table.lenU, depth: r.table.depth }, built: r.built,
  us: (r.spine.u), tracks: r.tracks, compound: r.compound, spine: r.spine, access: r.access, trackRoads: r.trackRoads,
  stations: r.stations.map(s => ({ id: s.id, u: s.u, v: s.v, k: s.k, track: s.track })),
  pieces: E ? E.pieces : [], crossings: E ? E.crossings : 0, trenchByType: E ? E.trenchByType : {}, trenchM: E ? E.trenchM : 0,
  cv: r.compound.v };
writeFileSync(process.argv[3], JSON.stringify(out));
