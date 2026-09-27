"""p5_earthworks.py: earthworks for the exported plant design on real 1 m ground, in VRAM via CuPy.

Design from the world's own JS (web/world v08 plant-layout.mjs + plant-feeders.mjs via p5_export.mjs): feeder trench pieces,
spine / access / track roads, compound and station pads. Three quantities, each two ways (electron / positron),
photons = items whose two answers differ beyond tolerance, and a CPU f64 witness on a sample.

  trench spoil   w x d x ground-following length. e: f32 chord sum at 0.05 m on the GPU.  p: f64 exact piecewise-linear
                 length (an axis-aligned line through a bilinear grid is linear between node crossings).
  road cut/fill  formation = grade line through centreline ground at 25 m knots, minus 0.30 m stone; level across.
                 e: 0.25 m cell-centre raster, f32 ground.  p: 3x3 Gauss-Legendre per 1 m x (w/4) panel, f64 ground.
  pad cut/fill   level at the balance (cut = fill). e: 0.25 m raster f32 + 60-step bisection.  p: 4x4 Gauss per 1 m
                 panel f64, balance = closed-form area mean.
Illustrative engineering only: no bulking, no side slopes, no topsoil strip outside roads.

Run: python p5_earthworks.py <site> [mw] [half_m]
"""
import furnace_paths as FP
import hashlib, json, math, os, subprocess, sys, time
import numpy as np
import cupy as cp

LIDAR = FP.LIDAR
OUT = (FP.FURNACE + "/out/p5-earthworks")
SRC = FP.SRC
LOCK = FP.LOCK
SECT = {"one": (0.45, 0.91, "33kv-1-dno"), "two": (0.80, 0.91, "33kv-2"), "duct": (0.35, 1.02, "33kv-1-ducted")}
STONE_M, KNOT_M = 0.30, 25.0
GL3 = (np.array([-math.sqrt(0.6), 0.0, math.sqrt(0.6)]), np.array([5 / 9, 8 / 9, 5 / 9]))
GL4 = np.polynomial.legendre.leggauss(4)

site = sys.argv[1] if len(sys.argv) > 1 else "open-land-01"
MW = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
HALF = float(sys.argv[3]) if len(sys.argv) > 3 else 350.0
os.makedirs(f"{OUT}/work", exist_ok=True)

src = json.load(open(f"{LIDAR}/{site}/source.json", encoding="utf-8"))
A = np.load(f"{LIDAR}/{site}/source.npy", mmap_mode="r")
E0, N0, n = float(src["origin_e_m"]), float(src["origin_n_m"]), A.shape[0]

# ---------------------------------------------------------------- design from the world's JS
ce, cn = E0 + n / 2, N0 + n / 2
bnd = [[ce - HALF, cn - HALF], [ce + HALF, cn - HALF], [ce + HALF, cn + HALF], [ce - HALF, cn + HALF]]
i0, i1 = max(0, int(ce - HALF - E0) - 40), min(n, int(ce + HALF - E0) + 41)
j0, j1 = max(0, int(cn - HALF - N0) - 40), min(n, int(cn + HALF - N0) + 41)
gf = f"{OUT}/work/ground-{site}.f32"
np.ascontiguousarray(A[j0:j1, i0:i1], dtype="<f4").tofile(gf)
rq, rs = f"{OUT}/work/req-{site}.json", f"{OUT}/work/design-{site}.json"
json.dump({"boundary": bnd, "targetMW": MW, "layout": "south",
           "ground": {"file": gf, "e0": E0 + i0, "n0": N0 + j0, "w": i1 - i0, "h": j1 - j0}}, open(rq, "w"))
t = time.perf_counter()
subprocess.run(["node", f"{SRC}/p5_export.mjs", rq, rs], check=True)
node_ms = (time.perf_counter() - t) * 1e3
D = json.load(open(rs))
assert not D["swap"], "south layout expected (u = e, v = n)"
P = D["params"]

# ---------------------------------------------------------------- ground in VRAM + two bilinears
G32 = cp.asarray(np.ascontiguousarray(A, dtype=np.float32))
G64 = G32.astype(cp.float64)


def bil32(x, y):   # x, y local metres (f32), lerp form
    x = x.astype(cp.float32); y = y.astype(cp.float32)
    i = cp.floor(x).astype(cp.int32); j = cp.floor(y).astype(cp.int32); fx = x - i; fy = y - j
    a, b, c, d = G32[j, i], G32[j, i + 1], G32[j + 1, i], G32[j + 1, i + 1]
    h0 = a + (b - a) * fx; h1 = c + (d - c) * fx
    return h0 + (h1 - h0) * fy


def bil64(x, y):   # f64 polynomial form
    i = cp.floor(x).astype(cp.int32); j = cp.floor(y).astype(cp.int32); fx = x - i; fy = y - j
    a, b, c, d = G64[j, i], G64[j, i + 1], G64[j + 1, i], G64[j + 1, i + 1]
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy


def bil_cpu(x, y):
    i = np.floor(x).astype(int); j = np.floor(y).astype(int); fx = x - i; fy = y - j
    a = A[j, i].astype(np.float64); b = A[j, i + 1].astype(np.float64); c = A[j + 1, i].astype(np.float64); d = A[j + 1, i + 1].astype(np.float64)
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy


def lx(e): return e - E0
def ly(nn): return nn - N0


# ---------------------------------------------------------------- 1. trench spoil
def line_of(pc):
    k = pc["key"]
    if k.startswith("t"):
        v = D["tracks"][int(k[1:])]; return (pc["a"], v, pc["b"], v)
    if k == "a":
        return (pc["a"], D["cv"], pc["b"], D["cv"])
    if k == "s":
        return (D["us"], pc["a"], D["us"], pc["b"])
    return (pc["u"], pc["a"], pc["u"], pc["b"])       # stub:<station>


def sections(c):
    s = []
    s += [SECT["two"]] * (c // 2)
    if c % 2: s.append(SECT["one"])
    return s


def len_electron(L):   # f32 chord sum, 0.05 m
    e0, n0, e1, n1 = L; pl = math.hypot(e1 - e0, n1 - n0); m = max(2, int(math.ceil(pl / 0.05)) + 1)
    s = cp.linspace(0, 1, m, dtype=cp.float32)
    x = cp.float32(lx(e0)) + (cp.float32(lx(e1)) - cp.float32(lx(e0))) * s
    y = cp.float32(ly(n0)) + (cp.float32(ly(n1)) - cp.float32(ly(n0))) * s
    z = bil32(x, y)
    return float(cp.sqrt(cp.diff(x) ** 2 + cp.diff(y) ** 2 + cp.diff(z) ** 2).sum(dtype=cp.float64))


def len_exact(L, bil, xp):   # piecewise-linear exact: breaks at integer node crossings
    e0, n0, e1, n1 = L; x0, y0, x1, y1 = lx(e0), ly(n0), lx(e1), ly(n1)
    ts = [0.0, 1.0]
    for a0, a1 in ((x0, x1), (y0, y1)):
        if abs(a1 - a0) > 1e-12:
            lo, hi = sorted((a0, a1))
            ts += [(k - a0) / (a1 - a0) for k in range(int(math.floor(lo)) + 1, int(math.ceil(hi)))]
    ts = xp.asarray(np.unique(np.clip(ts, 0, 1)))
    x = x0 + (x1 - x0) * ts; y = y0 + (y1 - y0) * ts; z = bil(x, y)
    return float(xp.sqrt(xp.diff(x) ** 2 + xp.diff(y) ** 2 + xp.diff(z) ** 2).sum())


t = time.perf_counter()
trench = []
for pc in D["pieces"]:
    L = line_of(pc); le = len_electron(L); lp = len_exact(L, bil64, cp); plan = math.hypot(L[2] - L[0], L[3] - L[1])
    se = sum(w * d * le for w, d, _ in sections(pc["c"])); sp = sum(w * d * lp for w, d, _ in sections(pc["c"]))
    trench.append({"key": pc["key"], "circuits": pc["c"], "sections": [s[2] for s in sections(pc["c"])],
                   "e0": round(L[0], 3), "n0": round(L[1], 3), "e1": round(L[2], 3), "n1": round(L[3], 3),
                   "plan_m": plan, "ground_m": {"electron": le, "positron": lp}, "spoil_m3": {"electron": se, "positron": sp}})
cross_m = D["crossings"] * (P["spineWidthM"] + 2)
duct_m3 = SECT["duct"][0] * SECT["duct"][1] * cross_m      # crossings have no geometry in the JS: plan length only
trench_ms = (time.perf_counter() - t) * 1e3

# ---------------------------------------------------------------- 2. roads
roads = []
sp = D["spine"]; roads.append(("spine", (D["us"], sp["v0"], D["us"], sp["v1"]), P["spineWidthM"]))
if D["access"]: ac = D["access"]; roads.append(("access", (ac["u0"], ac["v"], ac["u1"], ac["v"]), P["trackWidthM"]))
for tr in D["trackRoads"]: roads.append((f"track{tr['k']}{'E' if tr['s'] > 0 else 'W'}", (tr["u0"], tr["v"], tr["u1"], tr["v"]), P["trackWidthM"]))


def road_frame(L):
    e0, n0, e1, n1 = L; pl = math.hypot(e1 - e0, n1 - n0); ux, uy = (e1 - e0) / pl, (n1 - n0) / pl
    m = max(1, int(math.ceil(pl / KNOT_M))); ks = np.linspace(0, pl, m + 1)
    kz = bil_cpu(lx(e0) + ux * ks, ly(n0) + uy * ks) - STONE_M      # formation knots (shared design input, f64 CPU)
    return pl, ux, uy, ks, kz


def road_electron(L, w):
    pl, ux, uy, ks, kz = road_frame(L); h = 0.25
    s = (cp.arange(int(round(pl / h))) + 0.5) * (pl / int(round(pl / h)))
    q = (cp.arange(int(round(w / h))) + 0.5) * (w / int(round(w / h))) - w / 2
    S, Q = cp.meshgrid(s, q, indexing="ij")
    x = lx(L[0]) + ux * S - uy * Q; y = ly(L[1]) + uy * S + ux * Q
    zf = cp.interp(S.ravel(), cp.asarray(ks), cp.asarray(kz)).reshape(S.shape)
    d = bil32(x, y).astype(cp.float64) - zf; da = (pl / s.size) * (w / q.size)
    return float(cp.clip(d, 0, None).sum() * da), float(cp.clip(-d, 0, None).sum() * da)


def gauss_grid(pl, w, na, nw, rule, xp):
    xg, wg = rule; ea = pl / na; ew = w / nw
    s = ((xp.arange(na)[:, None] + 0.5) * ea + xp.asarray(xg)[None, :] * ea / 2).ravel()
    q = ((xp.arange(nw)[:, None] + 0.5) * ew + xp.asarray(xg)[None, :] * ew / 2).ravel() - w / 2
    ws = xp.tile(xp.asarray(wg), na) * ea / 2; wq = xp.tile(xp.asarray(wg), nw) * ew / 2
    return s, q, ws, wq


def road_positron(L, w):
    pl, ux, uy, ks, kz = road_frame(L)
    s, q, ws, wq = gauss_grid(pl, w, int(math.ceil(pl)), 4, GL3, cp)
    S, Q = cp.meshgrid(s, q, indexing="ij"); Wt = ws[:, None] * wq[None, :]
    x = lx(L[0]) + ux * S - uy * Q; y = ly(L[1]) + uy * S + ux * Q
    zf = cp.interp(S.ravel(), cp.asarray(ks), cp.asarray(kz)).reshape(S.shape)
    d = bil64(x, y) - zf
    return float((cp.clip(d, 0, None) * Wt).sum()), float((cp.clip(-d, 0, None) * Wt).sum())


def road_witness(L, w):   # CPU f64, 0.1 m midpoint raster
    pl, ux, uy, ks, kz = road_frame(L); h = 0.1
    s = (np.arange(int(round(pl / h))) + 0.5) * (pl / int(round(pl / h))); q = (np.arange(int(round(w / h))) + 0.5) * (w / int(round(w / h))) - w / 2
    S, Q = np.meshgrid(s, q, indexing="ij")
    d = bil_cpu(lx(L[0]) + ux * S - uy * Q, ly(L[1]) + uy * S + ux * Q) - np.interp(S, ks, kz); da = (pl / s.size) * (w / q.size)
    return float(np.clip(d, 0, None).sum() * da), float(np.clip(-d, 0, None).sum() * da)


t = time.perf_counter()
road = []
for name, L, w in roads:
    pl = math.hypot(L[2] - L[0], L[3] - L[1])
    ce_, fe = road_electron(L, w); cpz, fp = road_positron(L, w)
    road.append({"road": name, "width_m": w, "plan_m": pl, "e0": round(L[0], 3), "n0": round(L[1], 3), "e1": round(L[2], 3), "n1": round(L[3], 3),
                 "stone_m3": STONE_M * w * pl, "cut_m3": {"electron": ce_, "positron": cpz}, "fill_m3": {"electron": fe, "positron": fp}})
road_ms = (time.perf_counter() - t) * 1e3

# ---------------------------------------------------------------- 3. pads
pads = [("compound", D["compound"]["u"], D["compound"]["v"], D["compound"]["w"], D["compound"]["d"])]
for s in D["stations"]:
    pads.append((f"station-{s['id']}", s["u"], s["v"], D["table"]["lenU"], D["table"]["depth"]))


def pad_electron(ce_, cn_, w, d):
    h = 0.25; nx, ny = int(round(w / h)), int(round(d / h))
    x = lx(ce_ - w / 2) + (cp.arange(nx) + 0.5) * (w / nx); y = ly(cn_ - d / 2) + (cp.arange(ny) + 0.5) * (d / ny)
    X, Y = cp.meshgrid(x, y); g = bil32(X, Y).astype(cp.float64); da = (w / nx) * (d / ny)
    lo, hi = float(g.min()), float(g.max())
    for _ in range(60):
        m = 0.5 * (lo + hi); lo, hi = (m, hi) if float((g - m).sum()) > 0 else (lo, m)
    lev = 0.5 * (lo + hi)
    return lev, float(cp.clip(g - lev, 0, None).sum() * da), float(cp.clip(lev - g, 0, None).sum() * da)


def pad_positron(ce_, cn_, w, d):
    sx, sy, wx, wy = gauss_grid(w, d, int(math.ceil(w)), int(math.ceil(d)), GL4, cp)
    X, Y = cp.meshgrid(lx(ce_ - w / 2) + sx, ly(cn_ - d / 2) + (sy + d / 2)); Wt = wy[:, None] * wx[None, :]
    g = bil64(X, Y); lev = float((g * Wt).sum() / Wt.sum())
    return lev, float((cp.clip(g - lev, 0, None) * Wt).sum()), float((cp.clip(lev - g, 0, None) * Wt).sum())


def pad_witness(ce_, cn_, w, d):
    h = 0.1; nx, ny = int(round(w / h)), int(round(d / h))
    X, Y = np.meshgrid(lx(ce_ - w / 2) + (np.arange(nx) + 0.5) * (w / nx), ly(cn_ - d / 2) + (np.arange(ny) + 0.5) * (d / ny))
    g = bil_cpu(X, Y); lev = float(g.mean()); da = (w / nx) * (d / ny)
    return lev, float(np.clip(g - lev, 0, None).sum() * da), float(np.clip(lev - g, 0, None).sum() * da)


t = time.perf_counter()
pad = []
for name, u, v, w, d in pads:
    le, ce_, fe = pad_electron(u, v, w, d); lp, cpz, fp = pad_positron(u, v, w, d)
    pad.append({"pad": name, "e": round(u, 3), "n": round(v, 3), "w_m": w, "d_m": d, "area_m2": w * d,
                "level_m": {"electron": le, "positron": lp}, "cut_m3": {"electron": ce_, "positron": cpz}, "fill_m3": {"electron": fe, "positron": fp}})
pad_ms = (time.perf_counter() - t) * 1e3
cp.cuda.Stream.null.synchronize()
vram_mib = cp.get_default_memory_pool().total_bytes() / 2 ** 20

# ---------------------------------------------------------------- pairs (tolerances fixed before the run)
def pair(items, key, tol_fn, what):
    ph, worst = [], 0.0
    for it in items:
        e, p = it[key]["electron"], it[key]["positron"]; tol = tol_fn(it); d = abs(e - p); worst = max(worst, d / max(tol, 1e-12))
        if not d <= tol: ph.append({"item": it.get("key") or it.get("road") or it.get("pad"), "electron": e, "positron": p, "tol": tol})
    return {"what": what, "photons": len(ph), "worst_diff_over_tol": worst, "photon_items": ph}


pairs = [
    pair(trench, "spoil_m3", lambda it: max(1e-3 * it["spoil_m3"]["positron"], 0.01), "trench spoil m3: f32 chord 0.05 m vs f64 exact piecewise (tol 0.1 %)"),
    pair(road, "cut_m3", lambda it: max(0.01 * (it["cut_m3"]["positron"] + it["fill_m3"]["positron"]), 0.5), "road cut m3: 0.25 m raster f32 vs Gauss 3x3 f64 (tol 1 % of cut+fill, min 0.5)"),
    pair(road, "fill_m3", lambda it: max(0.01 * (it["cut_m3"]["positron"] + it["fill_m3"]["positron"]), 0.5), "road fill m3: same"),
    pair(pad, "level_m", lambda it: 0.005, "pad balance level m: bisection f32 raster vs closed-form Gauss mean f64 (tol 5 mm)"),
    pair(pad, "cut_m3", lambda it: max(0.01 * it["cut_m3"]["positron"], 0.5), "pad cut m3 at balance (tol 1 %, min 0.5)"),
]

# ---------------------------------------------------------------- CPU witness on a sample
wit = {"trench": [], "road": [], "pad": []}
for it in trench[:: max(1, len(trench) // 5)][:5]:
    L = (it["e0"], it["n0"], it["e1"], it["n1"]); lc = len_exact(L, bil_cpu, np)
    wit["trench"].append({"key": it["key"], "cpu_ground_m": lc, "positron_m": it["ground_m"]["positron"], "diff_m": abs(lc - it["ground_m"]["positron"])})
for (name, L, w), it in list(zip(roads, road))[:3]:
    c_, f_ = road_witness(L, w)
    wit["road"].append({"road": name, "cpu_cut_m3": c_, "cpu_fill_m3": f_, "positron_cut_m3": it["cut_m3"]["positron"], "positron_fill_m3": it["fill_m3"]["positron"]})
for (name, u, v, w, d), it in list(zip(pads, pad))[:3]:
    l_, c_, f_ = pad_witness(u, v, w, d)
    wit["pad"].append({"pad": name, "cpu_level_m": l_, "cpu_cut_m3": c_, "positron_level_m": it["level_m"]["positron"], "positron_cut_m3": it["cut_m3"]["positron"]})

tot = lambda items, k, side: sum(it[k][side] for it in items)
summary = {
    "trench_ground_m": tot(trench, "ground_m", "positron"), "trench_plan_m": sum(it["plan_m"] for it in trench), "js_trench_m": D["trenchM"],
    "trench_spoil_m3": {"electron": tot(trench, "spoil_m3", "electron"), "positron": tot(trench, "spoil_m3", "positron")},
    "duct_crossings": D["crossings"], "duct_crossing_m": cross_m, "duct_spoil_m3_plan": duct_m3,
    "road_plan_m": sum(r["plan_m"] for r in road), "js_road_m": D["built"]["roadM"], "road_stone_m3": sum(r["stone_m3"] for r in road),
    "road_cut_m3": {"electron": tot(road, "cut_m3", "electron"), "positron": tot(road, "cut_m3", "positron")},
    "road_fill_m3": {"electron": tot(road, "fill_m3", "electron"), "positron": tot(road, "fill_m3", "positron")},
    "pad_cut_m3": {"electron": tot(pad, "cut_m3", "electron"), "positron": tot(pad, "cut_m3", "positron")},
    "pad_fill_m3": {"electron": tot(pad, "fill_m3", "electron"), "positron": tot(pad, "fill_m3", "positron")},
    "photons": sum(p["photons"] for p in pairs),
}
body = {"site": site, "design": {"mw": MW, "layout": "south", "half_m": HALF, "source": "web/world (v08) plant-layout.mjs + plant-feeders.mjs",
                                   "built": D["built"], "tables_s": None},
        "assumptions": {"trench_sections": {k: {"w_m": v[0], "d_m": v[1], "id": v[2]} for k, v in SECT.items()},
                        "road_stone_m": STONE_M, "road_knot_m": KNOT_M, "road_crossfall": "level", "pads": "levelled flat at balance, no batters",
                        "not_included": "bulking, side slopes, topsoil strip under pads, junction overlaps counted twice"},
        "summary": summary, "pairs": pairs, "witness": wit, "trench": trench, "road": road, "pad": pad,
        "timing_ms": {"node": node_ms, "trench": trench_ms, "road": road_ms, "pad": pad_ms}, "vram_pool_mib": vram_mib,
        "card": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}
body["design"].pop("tables_s")
raw = json.dumps(body, indent=1, default=float)
sha = hashlib.sha256(raw.encode()).hexdigest()
with open(f"{OUT}/{site}.json", "w", newline="\n") as f: f.write(raw[:-1] + f',\n "sha256": "{sha}"\n}}\n')
print(json.dumps({"site": site, "summary": summary, "photons_by_pair": [p["photons"] for p in pairs], "sha256": sha, "vram_mib": vram_mib}, default=float, indent=1))
