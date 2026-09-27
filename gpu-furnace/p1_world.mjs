// p1_world.mjs: the world's own JS (web/world (v08), read-only import) plays a game session and says what the
// world measures: measuredAt (terrain.mjs sampleTile on the site's .ght tiles), the Walk eye (views.mjs setView/step,
// eye 1.7 m), the drone floor (AERIAL_MIN_CLEARANCE) and horizon angles (shade.mjs horizonAt on the .ghz tiles).
// node p1_world.mjs <request.json> <out.json>
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
// Paths come from the environment (see gpu-furnace/README.md); nothing local is written here.
const need = k => { const v = process.env[k]; if (!v) throw Error(`set ${k} (see gpu-furnace/README.md)`); return v.replace(/\\/g, '/').replace(/\/$/, ''); };
const worldDir = () => need('GRID_WORLD_JS') + '/';
const W = worldDir();
const imp = m => import(pathToFileURL(W + m).href);
const { decodeTile, sampleTile } = await imp('layers/terrain.mjs');
const { decodeHorizonTile, horizonAt } = await imp('layers/shade.mjs');
const { VIEWS, step, setView, AERIAL_MIN_CLEARANCE } = await imp('views.mjs');
const { layoutPlant } = await imp('plant-layout.mjs');
const req = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const S = W + 'sites/' + req.site + '/';

const tj = JSON.parse(readFileSync(S + 'tiles.json', 'utf8'));
const tiles = tj.tiles.filter(t => !t.ring).map(t => ({ ...t, tile: decodeTile(readFileSync(S + t.file)), size: tj.tile_m }));
// measuredAt as the terrain layer's heightAt: the finest tile holding the point, bilinear, NaN off the tiles.
const measuredAt = (e, n) => {
  for (const t of tiles) {
    if (e < t.e0 || n < t.n0 || e > t.e0 + t.size || n > t.n0 + t.size) continue;
    return sampleTile(t.tile, e - t.e0, n - t.n0);
  }
  return NaN;
};
const hj = JSON.parse(readFileSync(S + 'horizon/horizon-tiles.json', 'utf8'));
const htiles = hj.tiles.map(t => ({ ...t, tile: decodeHorizonTile(readFileSync(S + 'horizon/' + t.file.replace(/^horizon\//, ''))) }));
// as shade.mjs cellAt: the tile holding the point, nearest 4 m cell
const cellAt = (e, n) => {
  for (const h of htiles) {
    const T = h.tile, size = (T.samples - 1) * T.spacing;
    if (e < T.e0 || n < T.n0 || e > T.e0 + size || n > T.n0 + size) continue;
    const s = T.samples - 1, i = Math.min(s, Math.round((e - T.e0) / T.spacing)), j = Math.min(s, Math.round((n - T.n0) / T.spacing));
    return { T, i, j, e: T.e0 + i * T.spacing, n: T.n0 + j * T.spacing };
  }
  return null;
};

const out = { source: 'web/world (v08) (terrain.mjs, views.mjs, shade.mjs, plant-layout.mjs)', site: req.site,
  walk_eye_m: VIEWS.standing.eye, walk_speed_m_s: VIEWS.standing.speed, aerial_min_clearance_m: AERIAL_MIN_CLEARANCE };

// 1. the plant (same request as the furnace's t-layout), ground from the world's own tiles
const r = layoutPlant({ boundary: req.boundary, targetMW: req.mw, layout: req.layout, options: {}, grid: null, water: [], groundAt: measuredAt });
const T = r.table, F = r.frame, nT = r.tables.length / 6, tabs = [];
for (let q = 0; q < nT; q++) {
  const [ua, va, ...h] = r.tables.slice(q * 6, q * 6 + 6);
  tabs.push({ ua, va, h, c: F.en(ua + T.lenU / 2, va + T.depth / 2) });
}
out.layout = { tables: nT, mwp: r.built.mwp, lenU: T.lenU, depth: T.depth, pitch: T.pitch, rise: T.rise, swap: F.swap };
const rows = [...new Set(tabs.map(t => t.va))].sort((a, b) => a - b);
out.layout.rows = rows.length;

// 2. WALK a row aisle with the world's own step(): 8 m/s, 30 fps
const k = Math.floor(rows.length / 2), row = tabs.filter(t => t.va === rows[k]);
const vA = rows[k] + T.depth + (rows[k + 1] - rows[k] - T.depth) / 2;                // the aisle behind row k
const u0 = Math.min(...row.map(t => t.ua)), u1 = Math.max(...row.map(t => t.ua + T.lenU));
const [e0, n0] = F.en(u0, vA), [e1, n1] = F.en(u1, vA);
const yaw = Math.atan2(e1 - e0, n1 - n0);
let st = setView({ view: 'standing', pos: [e0, n0, 0], yaw, pitch: 0 }, 'standing', measuredAt);
const walk = [];
const frames = Math.floor(Math.hypot(e1 - e0, n1 - n0) / (VIEWS.standing.speed / 30));
for (let f = 0; f <= frames; f++) {
  if (f % 15 === 0) walk.push({ f, e: st.pos[0], n: st.pos[1], ground_m: measuredAt(st.pos[0], st.pos[1]), eye_z_m: st.pos[2] });
  st = step(st, { forward: 1 }, 1 / 30, measuredAt, []);
}
out.walk = { row_index: k, aisle_v: vA, yaw_deg: yaw * 180 / Math.PI, frames, poses: walk };

// 3. FLY a lawnmower over a block of rows k-3..k+3 at 60 m above the take-off ground (drone keeps its z)
const blk = tabs.filter(t => t.va >= rows[Math.max(0, k - 3)] && t.va <= rows[Math.min(rows.length - 1, k + 3)]);
const bE0 = Math.min(...blk.map(t => t.c[0])), bE1 = Math.max(...blk.map(t => t.c[0]));
const bN0 = Math.min(...blk.map(t => t.c[1])), bN1 = Math.max(...blk.map(t => t.c[1]));
const z0 = measuredAt(bE0, bN0) + 60, fly = [];
for (let j = 0; j < 4; j++) for (let i = 0; i < 5; i++) {
  const e = bE0 + (bE1 - bE0) * (j % 2 ? 4 - i : i) / 4, n = bN0 + (bN1 - bN0) * j / 3;
  const s = setView({ view: 'aerial', pos: [e, n, z0], yaw: 0, pitch: -0.5 }, 'aerial', measuredAt);
  fly.push({ e, n, ground_m: measuredAt(e, n), eye_z_m: s.pos[2], agl_m: s.pos[2] - measuredAt(e, n) });
}
out.fly = { block_tables: blk.length, box: [bE0, bN0, bE1, bN1], z_m: z0, poses: fly };

// 4. DIVE to the table nearest the block centre: drone straight down to AERIAL_MIN_CLEARANCE
const cE = (bE0 + bE1) / 2, cN = (bN0 + bN1) / 2;
const tgt = blk.reduce((a, t) => (Math.hypot(t.c[0] - cE, t.c[1] - cN) < Math.hypot(a.c[0] - cE, a.c[1] - cN) ? t : a));
const g = measuredAt(tgt.c[0], tgt.c[1]), dive = [];
for (const agl of [120, 60, 30, 10, 3, 0.1]) {
  const s = setView({ view: 'aerial', pos: [tgt.c[0], tgt.c[1], g + agl], yaw: 0, pitch: -Math.PI / 2 + 0.02 }, 'aerial', measuredAt);
  dive.push({ asked_agl_m: agl, eye_z_m: s.pos[2], agl_m: s.pos[2] - g });
}
out.dive = { table: { ua: tgt.ua, va: tgt.va, centre: tgt.c, corner_ground_m: tgt.h }, ground_m: g, poses: dive };

// 5. HORIZON angles (ground-level observer, 32 azimuths) at the 4 m cells nearest the chosen points
const hpts = [...(req.horizon_points || []), [walk[0].e, walk[0].n], [walk[walk.length - 1].e, walk[walk.length - 1].n], tgt.c];
out.horizon = hpts.map(([e, n]) => {
  const c = cellAt(e, n);
  if (!c) return { e, n, cell: null };
  return { e, n, cell_e: c.e, cell_n: c.n, cell_ground_m: measuredAt(c.e, c.n),
    deg: Array.from({ length: 32 }, (_, a) => horizonAt(c.T, c.i, c.j, a * 11.25)) };
});
// 6. ground at arbitrary probe points
out.probes = (req.probes || []).map(([e, n]) => ({ e, n, ground_m: measuredAt(e, n) }));
writeFileSync(process.argv[3], JSON.stringify(out));
