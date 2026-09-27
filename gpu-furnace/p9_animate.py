"""p9_animate.py: offline animation with the furnace's own CuPy kernels (render + lines), in VRAM.

1. Solar block, nine-stage Sequential build (block-sequence.mjs STAGES, 37 s at normal speed) fed by the staged
   prototype sequence on open-land-01 (gpu-prototype/prototype-sequence.json).
2. Follow the power: a panel on T01 -> string -> INV01 DC home run -> INV01 AC -> TX1 -> 33 kV (ASSUMED straight)
   -> hypothetical compound H0 -> tap to the nearest inferred 400 kV tower.
Writes PNG frames and text results to $FURNACE_ROOT/out/p9-animate/. Takes the GPU lock $GRID_GPU_LOCK.
ILLUSTRATIVE: geometry from the world's staged files; the MV route, the tap and all drawing heights are assumed.
"""
import furnace_paths as FP
import hashlib, json, math, os, subprocess, sys, time
import numpy as np
import cupy as cp
sys.path.insert(0, FP.SRC)
import furnace as F   # kernels only; the resident loop is not started

SITE = "open-land-01"
WT = (FP.WORLD + "/sites/open-land-01")
OUTD = (FP.FURNACE + "/out/p9-animate")
LOCK = FP.LOCK
W, H = F.W, F.H
STAGES = [("trenches", "Trenches dug (DC and AC)", 3), ("tables", "Piles driven, tables built", 6),
          ("strings", "Modules strung", 5), ("inverters", "Inverters mounted", 3), ("dc", "DC home cables pulled", 6),
          ("ac", "AC cables pulled to the transformers", 4), ("station", "Station transformers set", 2),
          ("mv", "33 kV cable pulled (route ASSUMED)", 5), ("grid", "Connection compound and 400 kV tap energised", 3)]
STARTS = np.cumsum([0] + [s[2] for s in STAGES])[:-1].astype(float)
TOTAL = float(sum(s[2] for s in STAGES))
NB, NP = 216, 144       # build frames, power frames
os.makedirs(f"{OUTD}/build", exist_ok=True); os.makedirs(f"{OUTD}/power", exist_ok=True)


def smi():
    r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10)
    t, m = [float(v) for v in r.stdout.strip().split(",")]
    return t, m


def main():
    seq = json.load(open(f"{WT}/gpu-prototype/prototype-sequence.json", encoding="utf-8"))
    grid = json.load(open(f"{WT}/grid.odbl.json", encoding="utf-8"))
    comp = json.load(open(f"{WT}/gpu-compounds.json", encoding="utf-8"))["compounds"][0]
    src = json.load(open((FP.LIDAR + f"/{SITE}/source.json")))
    e0, n0 = float(src["origin_e_m"]), float(src["origin_n_m"])
    a = np.load((FP.LIDAR + f"/{SITE}/source.npy"), mmap_mode="r")
    g = cp.asarray(np.ascontiguousarray(a, dtype=np.float32)); n = g.shape[0]
    MB = F.MB; mn = (n - 1) // MB
    P = cp.pad(g, ((0, (mn + 1) * MB - n), (0, (mn + 1) * MB - n)), mode="edge")
    A = P.reshape(mn + 1, MB, mn + 1, MB).max(axis=(1, 3))
    mip = cp.ascontiguousarray(cp.maximum(cp.maximum(A[:mn, :mn], A[1:, :mn]), cp.maximum(A[:mn, 1:], A[1:, 1:])), dtype=cp.float32)
    S = dict(g=g, n=n, e0=e0, n0=n0)
    L = lambda p: (p[0] - e0, p[1] - n0)

    # ---- items per stage: list of polylines (local x, y, lift) with colour; draped later
    items = {s[0]: [] for s in STAGES}
    col = lambda k: tuple(int(255 * c) for c in {"trench": (0.7, 0.6, 0.5), "pile": (0.75, 0.75, 0.78), "table": (0.62, 0.72, 0.86),
          "flag": (1, 0.55, 0.25), "string": (0.25, 0.35, 0.95), "inv": (0.95, 0.97, 1), "dc": (0.35, 0.8, 1), "ac": (0.55, 1, 0.62),
          "stn": (1, 0.72, 0.35), "mv": (1, 0.9, 0.3), "grid": (0.55, 0.66, 1.0)}[k])
    steps = sorted(seq["steps"], key=lambda s: (s["t_start_h"], s["step"]))
    tol = seq.get("twist_tol_deg", 1)
    for s in steps:
        k, gm = s["kind"], s["geometry"]
        if k == "trench-dug":
            items["trenches"].append(([[*L(p), 0.15] for p in gm["coordinates"]], col("trench")))
        elif k == "piles":
            posts = [[[*L(p), 0.0], [*L(p), 1.2]] for p in gm["coordinates"]]
            items["tables"].append((posts, col("pile"), "multi"))
        elif k == "table":
            ring = [[*L(p), 1.0] for p in gm["coordinates"][0]]
            c = col("flag") if abs(s.get("twist_deg") or 0) > tol else col("table")
            items["tables"].append((ring, c))
            r = np.array(ring[:4])   # rectangle: long axis between edge 0-1 midpoints... use the two long edges
            e01, e12 = np.linalg.norm(r[1] - r[0]), np.linalg.norm(r[2] - r[1])
            A0, A1, B0, B1 = (r[0], r[1], r[3], r[2]) if e01 < e12 else (r[1], r[2], r[0], r[3])
            # A0-A1 is a short edge; strings run from short edge A to the opposite short edge B
            strs = []
            for u in np.linspace(0.12, 0.88, 4):
                p0 = A0 + (A1 - A0) * u; p1 = B0 + (B1 - B0) * u
                strs.append([[p0[0], p0[1], 1.3], [p1[0], p1[1], 1.3]])
            items["strings"].append((strs, col("string"), "multi"))
        elif k == "inverter-set":
            x, y = L(gm["coordinates"])
            items["inverters"].append(([[[x, y, 0], [x, y, 2.5]], [[x - .6, y, 2.5], [x + .6, y, 2.5]], [[x, y - .6, 2.5], [x, y + .6, 2.5]]], col("inv"), "multi"))
        elif k == "dc-cables-pulled":
            items["dc"].append(([[[*L(p), 0.3] for p in ln] for ln in gm["coordinates"]], col("dc"), "multi"))
        elif k == "ac-cables-pulled":
            items["ac"].append(([[*L(p), 0.35] for p in gm["coordinates"]], col("ac")))
        elif k == "station-pad-earthworks":
            items["station"].append(([[*L(p), 0.2] for p in gm["coordinates"][0]], col("stn")))
        elif k == "station-set":
            x, y = L(gm["coordinates"])
            items["station"].append(([[[x, y, 0], [x, y, 3]], [[x - 1.5, y - 1.5, 3], [x + 1.5, y + 1.5, 3]], [[x - 1.5, y + 1.5, 3], [x + 1.5, y - 1.5, 3]]], col("stn"), "multi"))
    tx1 = [L(s["geometry"]["coordinates"]) for s in steps if s["kind"] == "station-set"][0]
    h0 = L(comp["centre"])
    mvline = [[tx1[0], tx1[1], 0.3], [h0[0], h0[1], 0.3]]
    items["mv"].append((mvline, col("mv")))
    ln400 = grid["lines"][0]; pts400 = np.array([L(p) for p in ln400["points"]])
    ti = int(np.argmin(np.hypot(pts400[:, 0] - h0[0], pts400[:, 1] - h0[1])))
    tower = pts400[ti]
    cpoly = [[*L(p), 0.3] for p in comp["poly"]] + [[*L(comp["poly"][0]), 0.3]]
    items["grid"].append((cpoly, col("grid")))
    items["grid"].append(([[[h0[0], h0[1], 0], [h0[0], h0[1], 12]], [[h0[0], h0[1], 12], [tower[0], tower[1], 44]]], (255, 255, 255), "multi"))

    # ---- drape: every polyline cut to <= 1 m pieces, ground sampled in VRAM by both formulations
    def pieces(poly):
        out = []
        for p, q in zip(poly[:-1], poly[1:]):
            p, q = np.array(p, float), np.array(q, float)
            k = max(1, int(math.ceil(math.hypot(*(q[:2] - p[:2])) / 1.0)))
            u = np.linspace(0, 1, k + 1)[:, None]; pts = p + (q - p) * u
            out.append(np.hstack([pts[:-1], pts[1:]]))
        return np.vstack(out)
    segs, cols, keys = [], [], []
    for si, (sid, _, sec) in enumerate(STAGES):
        its = items[sid]; ni = len(its)
        for ii, it in enumerate(its):
            polys = it[0] if len(it) > 2 else [it[0]]
            sg = np.vstack([pieces(p) for p in polys])
            m = len(sg)
            kk = STARTS[si] + (ii + (np.arange(m) + 1) / m) / ni * sec
            segs.append(sg); cols.append(np.tile(it[1], (m, 1))); keys.append(kk)
    # context: the 400 kV line (towers 44 m arm, 50 m posts) is there from frame 0
    ctx = []
    for p, q in zip(pts400[:-1], pts400[1:]):
        ctx.append(pieces([[p[0], p[1], 44], [q[0], q[1], 44]]))
    for p in pts400[1:-1]:
        ctx.append(pieces([[p[0], p[1], 0], [p[0], p[1], 50]]))
    ctx = np.vstack(ctx)
    inb = lambda X, Y: (X >= 1) & (Y >= 1) & (X <= n - 3) & (Y <= n - 3)
    ctx = ctx[inb(ctx[:, 0], ctx[:, 1]) & inb(ctx[:, 3], ctx[:, 4])]   # the line runs off the 2 km tile: clip
    segs = [ctx] + segs; cols = [np.tile(col("grid"), (len(ctx), 1))] + cols; keys = [np.full(len(ctx), -1.0)] + keys
    SG = np.vstack(segs); CO = np.vstack(cols).astype(np.uint8); KE = np.concatenate(keys)
    order = np.argsort(KE, kind="stable"); SG, CO, KE = SG[order], CO[order], KE[order]
    # drape pair
    ends = np.vstack([SG[:, 0:2], SG[:, 3:5]]) + [e0, n0]
    ge, gp = F.sample_pair(S, ends)
    drape_pair = F.pair_block(ge, gp, 1e-3, "m", "ground under every segment end: f32 lerp vs f64 polynomial")
    ns = len(SG)
    SG[:, 2] += ge[:ns]; SG[:, 5] += ge[ns:]
    rng = np.random.default_rng(9); pick = rng.choice(len(ends), 200, replace=False)
    wit = [abs(F.bilin_cpu(a, ends[i, 0] - e0, ends[i, 1] - n0) - gp[i]) for i in pick]
    drape_wit = {"sample": 200, "max_abs_m": float(max(wit)), "photons": int(sum(w > 1e-6 for w in wit))}
    seg_d = cp.asarray(SG.astype(np.float32).ravel()); col_d = cp.asarray(CO.ravel()); key_d = cp.asarray(KE.astype(np.float32))
    depth = cp.empty((H, W), cp.float32); rgba = cp.zeros((H, W, 4), cp.uint8)
    sun = F.Furnace.__init__.__code__ and (math.cos(math.radians(30)) * math.sin(math.radians(200)), math.cos(math.radians(30)) * math.cos(math.radians(200)), math.sin(math.radians(30)))
    ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()

    def ground(x, y):
        e, _ = F.sample_pair(S, np.array([[x + e0, y + n0]])); return float(e[0])

    def draw(cam, nseg, extra=None):
        ex, ey, ez, yaw, pitch, fov = cam
        ev0.record()
        F.K_RENDER(((W + 15) // 16, (H + 15) // 16), (16, 16), (g, mip, np.int32(n), np.int32(mn), np.int32(MB),
                   np.float32(ex), np.float32(ey), np.float32(ez), np.float32(yaw), np.float32(pitch), np.float32(fov),
                   np.int32(W), np.int32(H), depth, rgba, np.float32(sun[0]), np.float32(sun[1]), np.float32(sun[2]), np.float32(0)))
        for sd, cd, k in ([(seg_d, col_d, nseg)] + ([extra] if extra else [])):
            if k > 0:
                per = 4
                F.K_LINES(((k * per + 255) // 256,), (256,), (sd, cd, np.int32(k), np.int32(per), np.float32(ex), np.float32(ey),
                          np.float32(ez), np.float32(yaw), np.float32(pitch), np.float32(fov), np.int32(W), np.int32(H), depth, rgba))
        ev1.record(); ev1.synchronize()
        return cp.cuda.get_elapsed_time(ev0, ev1)

    def save(path):
        t0 = time.perf_counter(); img = cp.asnumpy(rgba[:, :, :3]); t1 = time.perf_counter()
        data = F.png_bytes(np.ascontiguousarray(img)); t2 = time.perf_counter()
        open(path, "wb").write(data); t3 = time.perf_counter()
        return (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, len(data)

    def look(eye, tgt, fov=math.radians(60)):
        dx, dy, dz = tgt[0] - eye[0], tgt[1] - eye[1], tgt[2] - eye[2]
        return (eye[0], eye[1], eye[2], math.atan2(dx, dy), math.atan2(dz, math.hypot(dx, dy)), fov)

    times = {"build": [], "power": []}; temps = []
    # ---- 1. nine-stage build
    ctr = np.array([(SG[:, 0].min() + SG[:, 0].max()) / 2, 0]); ctr[1] = (SG[KE >= 0, 1].min() + SG[KE >= 0, 1].max()) / 2
    ctr[0] = (SG[KE >= 0, 0].min() + SG[KE >= 0, 0].max()) / 2
    gc = ground(*ctr); counts_e, counts_p = [], []
    KE64 = KE.astype(np.float64)
    for i in range(NB):
        if i % 30 == 0:
            t_c, _ = smi(); temps.append(t_c)
            while t_c >= 80: time.sleep(5); t_c, _ = smi(); temps.append(t_c)
        t = TOTAL * (i + 1) / NB
        ne = int(cp.searchsorted(key_d, cp.asarray([t], cp.float32), side="right")[0])
        npz = int(np.count_nonzero(KE64 <= t))
        counts_e.append(ne); counts_p.append(npz)
        ang = math.radians(-60 + 100 * i / NB); r = 520
        eye = (ctr[0] - math.sin(ang) * r, ctr[1] - math.cos(ang) * r, gc + 260)
        ms = draw(look(eye, (ctr[0], ctr[1], gc)), ne)
        rb, enc, wr, sz = save(f"{OUTD}/build/b{i:04d}.png")
        times["build"].append((ms, rb, enc, wr, sz))
    # ---- 2. follow the power
    t01 = [s for s in steps if s["kind"] == "table"][0]; tr = np.array([L(p) for p in t01["geometry"]["coordinates"][0][:4]])
    dc1 = [s for s in steps if s["id"] == "INV01-DC"][0]["geometry"]["coordinates"][0]
    ac1 = [s for s in steps if s["id"] == "INV01-AC"][0]["geometry"]["coordinates"]
    pan = tr.mean(axis=0); st0 = np.array(L(dc1[0]))
    path = [[pan[0], pan[1], 1.3], [st0[0], st0[1], 1.3], [st0[0], st0[1], 0.3]] + [[*L(p), 0.3] for p in dc1[1:]] + \
           [[*L(p), 0.35] for p in ac1] + [[tx1[0], tx1[1], 0.3], [h0[0], h0[1], 0.3], [h0[0], h0[1], 12], [tower[0], tower[1], 44]]
    PP = pieces(path); pe, pp_ = F.sample_pair(S, np.vstack([PP[:, 0:2], PP[:, 3:5]]) + [e0, n0])
    m = len(PP); PP[:, 2] += pe[:m]; PP[:, 5] += pe[m:]
    PP_d = cp.asarray(PP.astype(np.float32))
    seglen_e = cp.sqrt(((PP_d[:, 3:] - PP_d[:, :3]) ** 2).sum(axis=1))
    len_e = float(seglen_e.sum()); len_p = math.fsum(math.dist(r[:3], r[3:]) for r in PP.astype(np.float32).astype(np.float64))
    cum = np.concatenate([[0], np.cumsum(np.hypot(np.hypot(PP[:, 3] - PP[:, 0], PP[:, 4] - PP[:, 1]), PP[:, 5] - PP[:, 2]))])
    def at(sv):
        j = min(int(np.searchsorted(cum, sv, side="right")) - 1, m - 1); u = (sv - cum[j]) / max(cum[j + 1] - cum[j], 1e-9)
        return PP[j, :3] + (PP[j, 3:] - PP[j, :3]) * u, j
    wcol = cp.asarray(np.tile([255, 255, 255], (m, 1)).astype(np.uint8).ravel())
    for i in range(NP):
        if i % 30 == 0:
            t_c, _ = smi(); temps.append(t_c)
            while t_c >= 80: time.sleep(5); t_c, _ = smi(); temps.append(t_c)
        u = i / (NP - 1); u = u * u * (3 - 2 * u); sv = cum[-1] * u
        p, j = at(sv); ahead, _ = at(min(cum[-1], sv + 60)); behind, _ = at(max(0, sv - 60))
        d = ahead[:2] - behind[:2]; d = d / max(np.linalg.norm(d), 1e-6)
        back = 40 + 260 * u
        eye = (p[0] - d[0] * back, p[1] - d[1] * back, ground(p[0] - d[0] * back, p[1] - d[1] * back) + 25 + 140 * u)
        ms = draw(look(eye, p), len(SG), (PP_d.ravel(), wcol, j + 1))
        rb, enc, wr, sz = save(f"{OUTD}/power/p{i:04d}.png")
        times["power"].append((ms, rb, enc, wr, sz))
    # ---- projection pair on the last power frame's camera: f32 kernel maths (CuPy) vs f64 numpy
    ex, ey, ez, yaw, pitch, fov = look(eye, p)
    def proj(X, xp, dt):
        X = X.astype(dt); x, y, z = X[:, 0] - dt(ex), X[:, 1] - dt(ey), X[:, 2] - dt(ez)
        cy, sy, cpp, sp = xp.cos(dt(yaw)), xp.sin(dt(yaw)), xp.cos(dt(pitch)), xp.sin(dt(pitch))
        fx, fy, fz, rx, ry = sy * cpp, cy * cpp, sp, cy, -sy
        ux, uy, uz = ry * fz, -rx * fz, rx * fy - ry * fx
        zf = x * fx + y * fy + z * fz; xr = x * rx + y * ry; yu = x * ux + y * uy + z * uz
        th = xp.tan(dt(fov) * dt(0.5)); asp = dt(W / H)
        return xp.stack([(xr / zf / (th * asp) + 1) * 0.5 * W, (1 - yu / zf / th) * 0.5 * H, zf], 1)
    smp = SG[rng.choice(len(SG), 4000, replace=False), :3]
    pe32 = cp.asnumpy(proj(cp.asarray(smp), cp, cp.float32)); pe64 = proj(smp, np, np.float64)
    vis = (pe64[:, 2] > 0.5) & (pe64[:, 0] >= 0) & (pe64[:, 0] < W) & (pe64[:, 1] >= 0) & (pe64[:, 1] < H)
    dpx = np.max(np.abs(pe32[vis, :2] - pe64[vis, :2]), axis=1)
    proj_pair = {"what": "pixel position of sampled segment points, f32 CuPy vs f64 numpy", "unit": "px", "tolerance": 0.5,
                 "points_on_screen": int(vis.sum()), "photons": int((dpx > 0.5).sum()), "max_abs_diff": float(dpx.max()) if vis.any() else 0.0}
    cnt_d = np.abs(np.array(counts_e) - np.array(counts_p))
    count_pair = {"what": "segments revealed per build frame: GPU searchsorted on f32 keys vs CPU count on f64 keys", "unit": "segments",
                  "tolerance": 0, "frames": NB, "photons": int((cnt_d > 0).sum()), "max_abs_diff": int(cnt_d.max())}
    len_pair = {"what": "follow-the-power path length: GPU f32 segment sum vs CPU f64 fsum", "unit": "m", "tolerance": 0.05,
                "electron": round(len_e, 3), "positron": round(len_p, 3), "photons": int(abs(len_e - len_p) > 0.05), "max_abs_diff": abs(len_e - len_p)}
    def st(v):
        v = np.array(v); return {"median": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3), "max": round(float(v.max()), 3)}
    tim = {}
    for k, v in times.items():
        v = np.array(v); tot = v[:, :4].sum(axis=1)
        tim[k] = {"frames": len(v), "gpu_draw_ms": st(v[:, 0]), "readback_ms": st(v[:, 1]), "png_encode_ms": st(v[:, 2]),
                  "write_ms": st(v[:, 3]), "total_ms_per_frame": st(tot), "offline_fps": round(1000 / float(np.median(tot)), 1),
                  "gpu_only_fps": round(1000 / float(np.median(v[:, 0])), 1), "png_mb_each": round(float(v[:, 4].mean()) / 2**20, 2)}
    stage_counts = {s[0]: int(np.count_nonzero((KE >= STARTS[i]) & (KE < STARTS[i] + s[2] + 1e-9))) for i, s in enumerate(STAGES)}
    return {"schema": "furnace.p9-animate/1", "site": SITE, "resolution": [W, H],
            "stages": [{"id": s[0], "label": s[1], "seconds": s[2], "segments": stage_counts[s[0]], "items": len(items[s[0]])} for s in STAGES],
            "segments_total": int(len(SG)), "context_400kv_segments": int(len(ctx)), "build_seconds_normal_speed": TOTAL,
            "power_path": {"legs": "panel T01 > string > INV01 DC > INV01 AC > TX1 > 33 kV ASSUMED straight > H0 hypothetical compound > tap to inferred 400 kV tower",
                           "length_m": round(len_p, 1), "pieces": m, "tower_local_m": [round(float(tower[0]), 1), round(float(tower[1]), 1)]},
            "pairs": {"drape": drape_pair, "reveal_count": count_pair, "path_length": len_pair, "projection": proj_pair},
            "witness": {"drape_cpu_bilinear": drape_wit}, "timing": tim, "gpu_temp_c_seen": [min(temps), max(temps)],
            "vram_pool_mib": round(cp.get_default_memory_pool().total_bytes() / 2**20, 1),
            "illustrative": "Drawing heights, the 33 kV route and the tap are ASSUMED; build rates from the staged sequence are ASSUMED there. Not a design."}


if __name__ == "__main__":
    t_all = time.perf_counter()
    for _ in range(120):
        try: os.mkdir(LOCK); break
        except FileExistsError: time.sleep(5)
    else:
        sys.exit("GPU lock busy for 10 min")
    try:
        cp.get_default_memory_pool().set_limit(size=2 * 2**30)
        res = main()
    finally:
        os.rmdir(LOCK)
    res["wall_s"] = round(time.perf_counter() - t_all, 2)
    res["script_sha256"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    body = json.dumps(res, sort_keys=True, default=float)
    res["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    with open(f"{OUTD}/RESULT.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, default=float); f.write("\n")
    print(json.dumps(res, indent=1, default=float))
