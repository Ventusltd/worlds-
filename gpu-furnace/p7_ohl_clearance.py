"""p7_ohl_clearance.py: overhead line zones (GS6) and working-height reach for every design item near the lines on
open-land-01, in VRAM with CuPy on the Blackwell card. Pair method: electron = f64 catenary, 0.25 m along each wire,
f64 bilinear ground; positron = f32 parabola, 0.5 m along each wire, f32 ground. Photons = sample points whose
reach differs by more than the tolerance. CPU witness (numpy f64) on a random sample. Compared with the world's own
OHL layer (p7_world.mjs -> ohl-zones.mjs checkDesign). ILLUSTRATIVE ONLY: tower heights and sag are assumed."""
import furnace_paths as FP
import json, hashlib, time, os, sys
import numpy as np, cupy as cp

W = json.load(open((FP.FURNACE + "/work/p7-world.json"), encoding="utf-8"))
OUT = (FP.FURNACE + "/out/p7-ohl-clearance")
os.makedirs(OUT, exist_ok=True)
N = 2049
H = float(W["height_m"]); POS = float(W["constants"]["position_m"])
TOL = 0.10
t0 = time.perf_counter()
ground_np = np.fromfile((FP.FURNACE + "/work/p7-ground-open-land-01.f32"), "<f4").reshape(N, N)
G64 = cp.asarray(ground_np.astype(np.float64)); G32 = cp.asarray(ground_np)

# ---- wires and spans ---------------------------------------------------------------------------------------------
spans = W["spans"]; sid = {(s["line"], s["span"]): k for k, s in enumerate(spans)}
wires = W["wires"]; nw = len(wires)
wa = np.array([w["a"] for w in wires], np.float64); wb = np.array([w["b"] for w in wires], np.float64)
wpar = np.array([w["param"] for w in wires], np.float64)
wspan = np.array([sid[(w["line"], w["span"])] for w in wires], np.int64)
slack = np.array([spans[k]["sway"] + POS for k in wspan])
excl = np.array([spans[k]["exclusion"] for k in wspan], np.float64)
wL = np.hypot(wb[:, 0] - wa[:, 0], wb[:, 1] - wa[:, 1])

def coshm1(u, xp): s = xp.sinh(u / 2); return 2 * s * s

def wire_pts(step, dtype, form):
    """Points along every wire: (x, y, z) arrays padded to the longest wire; z by catenary or parabola."""
    m = int(np.ceil(wL.max() / step)) + 1
    f = cp.linspace(0, 1, m, dtype=cp.float64)[None, :]
    L = cp.asarray(wL)[:, None]; a = cp.asarray(wa); b = cp.asarray(wb); p = cp.asarray(wpar)[:, None]
    s = f * L
    x = a[:, 0:1] + (b[:, 0:1] - a[:, 0:1]) * f; y = a[:, 1:2] + (b[:, 1:2] - a[:, 1:2]) * f
    h1 = a[:, 2:3]; h2 = b[:, 2:3]
    if form == "catenary":
        s0 = L / 2 - p * cp.arcsinh((h2 - h1) / (2 * p * cp.sinh(L / (2 * p))))
        zlow = h1 - p * coshm1(s0 / p, cp)
        z = zlow + p * coshm1((s - s0) / p, cp)
    else:  # parabola with the same level-span sag: chord minus 4 sag s (L - s) / L^2
        sag = L * L / (8 * p)
        z = h1 + (h2 - h1) * f - 4 * sag * f * (1 - f)
    return x.astype(dtype), y.astype(dtype), z.astype(dtype)

def bilinear(G, x, y):
    i = cp.clip(cp.floor(x).astype(cp.int64), 0, N - 2); j = cp.clip(cp.floor(y).astype(cp.int64), 0, N - 2)
    fx = x - i; fy = y - j
    h0 = G[j, i] + (G[j, i + 1] - G[j, i]) * fx; h1 = G[j + 1, i] + (G[j + 1, i + 1] - G[j + 1, i]) * fx
    return h0 + (h1 - h0) * fy

# ---- per span ground clearance (electron catenary 0.25 m / positron parabola 0.5 m) vs the world's 1 m ---------------
def span_clear(step, dtype, form, G):
    x, y, z = wire_pts(step, dtype, form)
    gap = z - bilinear(G, x, y)
    valid = cp.ones(x.shape, bool)  # every wire is sampled uniformly end to end at <= step
    gap = cp.where(valid, gap, cp.inf)
    per_wire = cp.asnumpy(gap.min(axis=1)).astype(np.float64)
    out = np.full(len(spans), np.inf)
    for k in range(nw): out[wspan[k]] = min(out[wspan[k]], per_wire[k])
    return out
ce = span_clear(0.25, cp.float64, "catenary", G64); cpn = span_clear(0.5, cp.float32, "parabola", G32)
world_clear = {}
for c in W["clearances"]: world_clear[c["span"]] = c["min_m"]

# ---- sample points of every item ---------------------------------------------------------------------------------
P, owner = [], []
for k, s in enumerate(W["samples"]):
    for p in s["pts"]: P.append(p); owner.append(k)
P = np.array(P, np.float64); owner = np.array(owner); nP = len(P)
items = W["samples"]; half = np.array([s["half"] for s in items])

# plan distance, point to each span's quad (0 inside): electron f64, positron f32
def quad_dist(Px, Py, dtype):
    out = []
    for s in spans:
        q = np.array(s["quad"]["pts"], np.float64); closed = s["quad"]["closed"]
        qx = cp.asarray(q[:, 0], dtype); qy = cp.asarray(q[:, 1], dtype)
        n = len(q) if closed else len(q) - 1
        d = cp.full(Px.shape, cp.inf, dtype); inside = cp.zeros(Px.shape, bool)
        for e in range(n):
            ax, ay, bx, by = qx[e], qy[e], qx[(e + 1) % len(q)], qy[(e + 1) % len(q)]
            dx, dy = bx - ax, by - ay
            t = cp.clip(((Px - ax) * dx + (Py - ay) * dy) / (dx * dx + dy * dy), 0, 1)
            d = cp.minimum(d, cp.hypot(Px - ax - t * dx, Py - ay - t * dy))
            if closed:
                c = ((ay > Py) != (by > Py)) & (Px < (bx - ax) * (Py - ay) / (by - ay + 1e-30) + ax)
                inside ^= c
        out.append(cp.where(inside, 0, d))
    return cp.stack(out, 1)
Pe = cp.asarray(P); Pp = cp.asarray(P, cp.float32)
qd_e = cp.asnumpy(quad_dist(Pe[:, 0], Pe[:, 1], cp.float64)); qd_p = cp.asnumpy(quad_dist(Pp[:, 0], Pp[:, 1], cp.float32)).astype(np.float64)

# ---- working-height reach: plant of height H on the ground at each sample point to every conductor ------------------
def reach(step, dtype, form, G, Px, Py):
    x, y, z = wire_pts(step, dtype, form)
    valid = cp.ones(x.shape, bool)  # every wire is sampled uniformly end to end at <= step
    sl = cp.asarray(slack, dtype)[:, None]
    g = bilinear(G, Px, Py)
    best = cp.full((len(Px), len(spans)), cp.inf, dtype)
    B = 256
    for i in range(0, len(Px), B):
        px = Px[i:i + B, None, None]; py = Py[i:i + B, None, None]; gz = g[i:i + B, None, None]; top = gz + H
        dh = cp.maximum(0, cp.hypot(px - x[None], py - y[None]) - sl[None])
        dz = cp.where(z[None] > top, z[None] - top, cp.where(z[None] < gz, gz - z[None], 0))
        d = cp.where(valid[None], cp.hypot(dh, dz), cp.inf).min(axis=2)       # (B, wires)
        for k in range(len(spans)):
            m = cp.asarray(wspan == k)
            best[i:i + B, k] = d[:, m].min(axis=1)
    return cp.asnumpy(best).astype(np.float64), cp.asnumpy(g).astype(np.float64)
cp.cuda.Device().synchronize(); tg = time.perf_counter()
re, ge = reach(0.25, cp.float64, "catenary", G64, Pe[:, 0], Pe[:, 1])
rp, gp = reach(0.5, cp.float32, "parabola", G32, Pp[:, 0], Pp[:, 1])
cp.cuda.Device().synchronize(); gpu_ms = (time.perf_counter() - tg) * 1e3
vram_mib = cp.get_default_memory_pool().total_bytes() / 2**20

# photons: sample points near a line (reach < 30 m either way) whose reach differs beyond TOL
near = (re < 30) | (rp < 30)
diff = np.abs(re - rp)[near]
photons = int((diff > TOL).sum()); maxdiff = float(diff.max()) if diff.size else 0.0
zdiff = np.abs(qd_e - qd_p); zphot = int((zdiff > 0.01).sum())

# ---- CPU witness: numpy f64 catenary at 0.25 m on 48 random near-line points -------------------------------------
rng = np.random.default_rng(7); cand = np.where((re < 30).any(1))[0]
wit_idx = rng.choice(cand, min(48, len(cand)), replace=False) if len(cand) else np.array([], int)
wit_err = 0.0
def cpu_ground(x, y):
    i, j = int(np.floor(x)), int(np.floor(y)); fx, fy = x - i, y - j; a = ground_np.astype(np.float64)
    h0 = a[j, i] + (a[j, i + 1] - a[j, i]) * fx; h1 = a[j + 1, i] + (a[j + 1, i + 1] - a[j + 1, i]) * fx
    return h0 + (h1 - h0) * fy
for idx in wit_idx:
    x0, y0 = P[idx]; g = cpu_ground(x0, y0); top = g + H; best = np.full(len(spans), np.inf)
    for k in range(nw):
        L = wL[k]; p = wpar[k]; h1, h2 = wa[k, 2], wb[k, 2]
        s = np.linspace(0, L, int(np.ceil(L / 0.25)) + 1)
        s0 = L / 2 - p * np.arcsinh((h2 - h1) / (2 * p * np.sinh(L / (2 * p)))); zl = h1 - p * coshm1(s0 / p, np)
        z = zl + p * coshm1((s - s0) / p, np); x = wa[k, 0] + (wb[k, 0] - wa[k, 0]) * s / L; y = wa[k, 1] + (wb[k, 1] - wa[k, 1]) * s / L
        dh = np.maximum(0, np.hypot(x0 - x, y0 - y) - slack[k]); dz = np.where(z > top, z - top, np.where(z < g, g - z, 0))
        best[wspan[k]] = min(best[wspan[k]], np.hypot(dh, dz).min())
    wit_err = max(wit_err, float(np.nanmax(np.abs(np.where(np.isfinite(best), best - re[idx], 0)))))

# ---- per item results and the comparison with the world's flags ----------------------------------------------------
zoneR = np.array([s["zoneReach"] for s in spans]); barR = np.array([s["barrierReach"] for s in spans])
exS = np.array([s["exclusion"] for s in spans], np.float64)
ni = len(items)
def per_item(qd, r):
    pl = np.full((ni, len(spans)), np.inf); rc = np.full((ni, len(spans)), np.inf)
    np.minimum.at(pl, owner, qd); np.minimum.at(rc, owner, r)
    return pl - half[:, None], rc
pl_e, rc_e = per_item(qd_e, re); pl_p, rc_p = per_item(qd_p, rp)
gz_e = set(zip(*np.where(pl_e < zoneR))); gz_p = set(zip(*np.where(pl_p < zoneR)))
gb_e = set(zip(*np.where(pl_e < barR)))
gh_e = set(np.where((rc_e < exS).any(1))[0]); gh_p = set(np.where((rc_p < exS).any(1))[0])
iid = {s["id"]: k for k, s in enumerate(items)}; spid = {s["id"]: k for k, s in enumerate(spans)}
wz = {(iid[f["item"]], spid[f["span"]]) for f in W["flags"] if f["flag"] in ("inside", "crosses")}
wh = {iid[f["item"]] for f in W["flags"] if f["flag"] == "height"}
wh_reach = {iid[f["item"]]: f["reach_m"] for f in W["flags"] if f["flag"] == "height"}
kinds = {}
for k, s in enumerate(items):
    kd = s["id"].split("-")[0]; kinds.setdefault(kd, [0, 0, 0, 0])
    kinds[kd][0] += 1
    kinds[kd][1] += any((k, j) in gz_e for j in range(len(spans)))
    kinds[kd][2] += any((k, j) in gb_e for j in range(len(spans)))
    kinds[kd][3] += k in gh_e
worst = np.argsort(rc_e.min(1))[:10]
wr_diff = [abs(wh_reach[k] - round(rc_e[k].min(), 1)) for k in wh_reach]

res = {
  "role": "p7-ohl-clearance", "site": "open-land-01", "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
  "disclaimer": W["disclaimer"], "illustrative": True,
  "card": "RTX 5070 Ti (Blackwell) via CuPy " + cp.__version__, "vram_pool_mib": round(vram_mib, 1),
  "inputs": {"lines": len({w["line"] for w in wires}), "spans": len(spans), "conductors": nw, "items": ni,
             "sample_points": nP, "working_height_m": H, "layout_tables": W["layout"]["tables"],
             "zone_reach_m": sorted({round(float(v), 2) for v in zoneR}), "barrier_reach_m": sorted({round(float(v), 2) for v in barR}),
             "exclusion_m": sorted({float(v) for v in exS}), "sway_plus_position_m": sorted({round(float(v), 2) for v in slack}),
             "assumed": "tower heights, arm geometry, 3.5 % sag, 45 deg swing, 2 m position allowance, 12 m plant (user-set default)"},
  "span_ground_clearance_m": [{"span": s["id"], "electron_catenary_0.25m": round(float(ce[k]), 3),
      "positron_parabola_0.5m": round(float(cpn[k]), 3), "world_js_1m": (round(world_clear[s["span"]], 3) if world_clear.get(s["span"]) == world_clear.get(s["span"]) and world_clear.get(s["span"]) is not None else None),
      "esqcr_figure_m": 7.3} for k, s in enumerate(spans)],
  "zones": {"items_in_planning_zone_gpu": len({k for k, _ in gz_e}), "item_span_pairs_zone": {"electron": len(gz_e), "positron": len(gz_p), "world": len(wz),
             "gpu_only": len(gz_e - wz), "world_only": len(wz - gz_e)},
            "items_in_barrier_zone_gpu": len({k for k, _ in gb_e})},
  "working_height": {"items_within_exclusion": {"electron": len(gh_e), "positron": len(gh_p), "world": len(wh),
                     "gpu_only": len(gh_e - wh), "world_only": len(wh - gh_e)},
                     "world_vs_electron_reach_max_abs_m": round(max(wr_diff), 3) if wr_diff else None,
                     "closest_10": [{"item": items[k]["id"], "span": spans[int(np.argmin(rc_e[k]))]["id"],
                        "reach_m_electron": round(float(rc_e[k].min()), 3), "reach_m_positron": round(float(rc_p[k].min()), 3),
                        "exclusion_m": float(exS[int(np.argmin(rc_e[k]))]), "world_reach_m": wh_reach.get(k)} for k in worst]},
  "by_kind": {k: {"items": v[0], "in_planning_zone": v[1], "in_barrier_zone": v[2], "height_within_exclusion": v[3]} for k, v in kinds.items()},
  "pair": {"what": "plant reach to conductor per sample point: f64 catenary 0.25 m vs f32 parabola 0.5 m", "unit": "m",
           "tolerance": TOL, "compared_points": int(near.sum()), "photons": photons, "max_abs_diff": round(maxdiff, 4),
           "zone_plan_distance": {"what": "f64 vs f32 point-to-quad", "tolerance": 0.01, "photons": zphot, "max_abs_diff": float(zdiff.max())}},
  "witness": {"what": "numpy f64 catenary 0.25 m on random near-line points", "points": int(len(wit_idx)), "max_abs_err_m": wit_err},
  "timing_ms": {"gpu_reach_both": round(gpu_ms, 1), "total": round((time.perf_counter() - t0) * 1e3, 1),
                "world_js": W["ms"]},
}
body = json.dumps(res, indent=1)
res["sha256"] = hashlib.sha256(body.encode()).hexdigest()
with open(f"{OUT}/result.json", "w", encoding="utf-8", newline="\n") as f: f.write(json.dumps(res, indent=1) + "\n")
print(json.dumps({k: res[k] for k in ("zones", "working_height", "pair", "witness", "timing_ms")}, indent=1))
print(json.dumps(res["span_ground_clearance_m"]))
print(json.dumps(res["by_kind"]))
