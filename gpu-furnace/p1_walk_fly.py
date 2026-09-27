"""p1_walk_fly.py: play the world as a game through the furnace, check it against the world's own JS.

Session (all on the resident 20 MW south plant's site, open-land-01):
  walk   the world's step() walks a row aisle at 8 m/s, 30 fps (p1_world.mjs); every 15th frame the same pose goes
         to the furnace as a camera at agl 1.7 m. Furnace eye = VRAM ground + 1.7; world eye = measuredAt + 1.7.
  fly    a 5 x 4 lawnmower over a block of rows at one drone height; then one animated fly (seconds) watched in STATUS.
  dive   straight down onto a table centre, 120 m to the drone floor.
  horizon at 4 poses the furnace draws a frame (sky fraction); this script's own CuPy (under the GPU lock) predicts
         the sky pixel by pixel two ways (electron: per-pixel fixed march; positron: 8192-azimuth horizon profile);
         and at ground level against the world's horizon tiles (shade.mjs horizonAt, 32 azimuths).
Text results: $FURNACE_ROOT/out/p1-walk-fly/.
"""
import furnace_paths as FP
import glob, hashlib, json, math, os, subprocess, sys, time
import numpy as np

ROOT = FP.FURNACE; IN = f"{ROOT}/in"; ANS = f"{ROOT}/out/answers"; OUTD = f"{ROOT}/out/p1-walk-fly"; WORK = f"{ROOT}/work/p1"
LOCK = FP.LOCK; SITE = "open-land-01"
os.makedirs(OUTD, exist_ok=True); os.makedirs(WORK, exist_ok=True)
src = json.load(open((FP.LIDAR + f"/{SITE}/source.json")))
E0, N0 = float(src["origin_e_m"]), float(src["origin_n_m"])
A = np.load((FP.LIDAR + f"/{SITE}/source.npy"), mmap_mode="r"); NN = A.shape[0]
W, H, FOV = 1280, 720, math.radians(70)


def status():
    for _ in range(20):
        try: return json.load(open(f"{ROOT}/out/STATUS.json", encoding="utf-8"))
        except Exception: time.sleep(0.05)


def send(tag, cmds, timeout=60):
    for c in cmds: c.setdefault("site", SITE)
    for c in cmds:
        p = f"{ANS}/{c['id']}.json"
        if os.path.exists(p): os.remove(p)
    with open(f"{IN}/{tag}.cmd.tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(json.dumps(c) for c in cmds) + "\n")
    os.replace(f"{IN}/{tag}.cmd.tmp", f"{IN}/{tag}.cmd")
    out, t0 = {}, time.time()
    while len(out) < len(cmds) and time.time() - t0 < timeout:
        for c in cmds:
            p = f"{ANS}/{c['id']}.json"
            if c["id"] not in out and os.path.exists(p):
                try: out[c["id"]] = json.load(open(p, encoding="utf-8"))
                except Exception: pass
        time.sleep(0.05)
    return [out.get(c["id"]) for c in cmds]


def pair(e, p, tol, unit, what):
    e, p = np.asarray(e, float), np.asarray(p, float); d = np.abs(e - p)
    return {"what": what, "unit": unit, "tolerance": tol, "n": int(d.size), "photons": int(np.count_nonzero(~(d <= tol))),
            "max_abs_diff": float(np.nanmax(d)) if d.size else 0.0, "mean_abs_diff": float(np.nanmean(d)) if d.size else 0.0}


def bil(x, y):  # CPU f64 witness on source.npy, lerp form
    i, j = int(math.floor(x)), int(math.floor(y)); fx, fy = x - i, y - j
    a, b, c, d = (float(np.float32(A[j, i])), float(np.float32(A[j, i + 1])), float(np.float32(A[j + 1, i])), float(np.float32(A[j + 1, i + 1])))
    return (a + (b - a) * fx) * (1 - fy) + (c + (d - c) * fx) * fy


R = {"site": SITE, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
st0 = status(); R["furnace_at_start"] = {k: st0.get(k) for k in ("pid", "alive", "uptime_s", "frame", "fps", "render_ms", "temperature_c", "card_vram_used_mib")}

# ---------------------------------------------------------------- 1. the world's own JS
half = 350.0; ce, cn = E0 + NN / 2, N0 + NN / 2
req = {"site": SITE, "mw": 20, "layout": "south", "boundary": [[ce - half, cn - half], [ce + half, cn - half], [ce + half, cn + half], [ce - half, cn + half]],
       "horizon_points": [[E0 + 896, N0 + 804]], "probes": []}
json.dump(req, open(f"{WORK}/req.json", "w"))
t0 = time.perf_counter()
r = subprocess.run(["node", f"{ROOT}/src/p1_world.mjs", f"{WORK}/req.json", f"{WORK}/world.json"], capture_output=True, text=True)
if r.returncode: sys.exit("world JS failed: " + r.stderr[-800:])
WJ = json.load(open(f"{WORK}/world.json")); R["world_js_ms"] = (time.perf_counter() - t0) * 1000
R["world"] = {k: WJ[k] for k in ("source", "walk_eye_m", "walk_speed_m_s", "aerial_min_clearance_m", "layout")}

# ---------------------------------------------------------------- 2. walk the aisle through the furnace
walk = WJ["walk"]; yaw = walk["yaw_deg"]
cm = [{"id": f"p1-w{q:03d}", "op": "camera", "e": p["e"], "n": p["n"], "agl": WJ["walk_eye_m"], "yaw": yaw, "pitch": 0} for q, p in enumerate(walk["poses"])]
ans = send("p1-walk", cm)
ok = [a for a in ans if a and a.get("ok")]
ge = np.array([a["ground_under_eye_m"]["electron"] for a in ok]); gp = np.array([a["ground_under_eye_m"]["positron"] for a in ok])
gw = np.array([p["ground_m"] for p, a in zip(walk["poses"], ans) if a and a.get("ok")])
ew = np.array([p["eye_z_m"] for p, a in zip(walk["poses"], ans) if a and a.get("ok")])
ef = ge + np.array([a["camera"]["agl_m"] for a in ok])
wit = np.array([bil(p["e"] - E0, p["n"] - N0) for p, a in zip(walk["poses"], ans) if a and a.get("ok")])
fph = sum(a["pair"]["photons"] for a in ok)
R["walk"] = {"poses": len(cm), "answered": len(ok), "aisle_n_m": walk["aisle_v"], "yaw_deg": yaw, "world_frames": walk["frames"],
             "length_m": walk["poses"][-1]["e"] - walk["poses"][0]["e"],
             "ground_m": {"world_js_first_last": [gw[0], gw[-1]], "furnace_first_last": [ge[0], ge[-1]], "min": float(gw.min()), "max": float(gw.max())},
             "furnace_internal_pair_photons": fph,
             "pair_ground": pair(gw, ge, 0.01, "m", "ground under the eye: world JS measuredAt (cm tiles, f64) vs furnace VRAM f32 lerp"),
             "pair_eye": pair(ew, ef, 0.01, "m", "eye height: world step() z (ground + 1.7) vs furnace ground + agl"),
             "pair_eye_above_ground": pair(ew - gw, ef - ge, 1e-6, "m", "eye above ground: world vs furnace"),
             "witness": pair(wit, gp, 1e-6, "m", "CPU f64 bilinear on source.npy vs furnace positron f64"),
             "timing_ms_mean": float(np.mean([a["timing_ms"] for a in ok]))}

# ---------------------------------------------------------------- 3. fly the block, then dive to a table
fly = WJ["fly"]
cm = [{"id": f"p1-f{q:02d}", "op": "camera", "e": p["e"], "n": p["n"], "agl": p["agl_m"], "yaw": 90 if (q // 5) % 2 == 0 else 270, "pitch": -30}
      for q, p in enumerate(fly["poses"])]
ans = send("p1-fly", cm); ok = [a for a in ans if a and a.get("ok")]
fg = np.array([a["ground_under_eye_m"]["electron"] for a in ok]); fz = fg + np.array([a["camera"]["agl_m"] for a in ok])
R["fly"] = {"poses": len(cm), "answered": len(ok), "block_tables": fly["block_tables"], "box_en_m": fly["box"], "drone_z_m": fly["z_m"],
            "agl_range_m": [float(min(p["agl_m"] for p in fly["poses"])), float(max(p["agl_m"] for p in fly["poses"]))],
            "pair_ground": pair([p["ground_m"] for p in fly["poses"]], fg, 0.01, "m", "ground: world measuredAt vs furnace VRAM"),
            "pair_eye": pair([p["eye_z_m"] for p in fly["poses"]], fz, 0.01, "m", "drone eye z: world (fixed z) vs furnace ground + agl"),
            "furnace_internal_pair_photons": sum(a["pair"]["photons"] for a in ok)}
dv = WJ["dive"]; tc = dv["table"]["centre"]
cm = [{"id": f"p1-d{q}", "op": "camera", "e": tc[0], "n": tc[1], "agl": p["asked_agl_m"], "yaw": 0, "pitch": -88.85} for q, p in enumerate(dv["poses"])]
ans = send("p1-dive", cm); ok = [a for a in ans if a and a.get("ok")]
dg = np.array([a["ground_under_eye_m"]["electron"] for a in ok])
R["dive"] = {"table_centre_en_m": tc, "table_corner_ground_world_m": dv["table"]["corner_ground_m"], "ground_world_m": dv["ground_m"],
             "ground_furnace_m": float(dg[0]),
             "asked_agl_m": [p["asked_agl_m"] for p in dv["poses"]],
             "world_eye_agl_m": [p["agl_m"] for p in dv["poses"]],
             "furnace_eye_agl_m": [a["camera"]["agl_m"] for a in ok],
             "pair_eye_agl": pair([p["agl_m"] for p in dv["poses"]], [a["camera"]["agl_m"] for a in ok], 0.01, "m",
                                  "eye above ground: world drone (floor 0.3 m) vs furnace agl"),
             "table_top_m_world": {"low_edge": min(dv["table"]["corner_ground_m"]) , "note": "tables stand on these corner grounds; the furnace draws the top outline at corner ground + 0.8 (+ rise at the back)"}}

# animated fly (seconds) watched in STATUS: smoothstep path, furnace follows the ground at constant agl
a0 = walk["poses"][0]; secs = 4.0
send("p1-anim0", [{"id": "p1-a0", "op": "camera", "e": a0["e"], "n": a0["n"], "agl": 1.7, "yaw": yaw, "pitch": 0}])
time.sleep(0.3)
send("p1-anim1", [{"id": "p1-a1", "op": "fly", "e": tc[0], "n": tc[1], "agl": 60, "yaw": 0, "pitch": -45, "seconds": secs}])
trace, t0 = [], time.time()
while time.time() - t0 < secs + 1.0:
    s = status(); c = s["camera"]; trace.append((time.time() - t0, s["frame"], c["x"], c["y"], c["agl"], s["fps"])); time.sleep(0.25)
tr = np.array([t[:5] for t in trace]); xs0, xs1 = a0["e"] - E0, tc[0] - E0
R["animated_fly"] = {"seconds": secs, "status_samples": len(trace), "frames_advanced": int(tr[-1, 1] - tr[0, 1]),
                     "fps_seen": [float(min(t[5] for t in trace)), float(max(t[5] for t in trace))],
                     "x_monotone": bool(np.all(np.diff(tr[:, 2]) * np.sign(xs1 - xs0) >= -1e-6)),
                     "end_xy_agl": [float(tr[-1, 2]), float(tr[-1, 3]), float(tr[-1, 4])], "asked_end_xy_agl": [xs1, tc[1] - N0, 60.0],
                     "trace_t_frame_x_y_agl": [[round(v, 3) for v in t[:5]] for t in trace]}

# ---------------------------------------------------------------- 4. horizon: furnace frames + own CuPy pair under the lock
poses = [("walk-start-east", a0["e"], a0["n"], 1.7, yaw, 0.0),
         ("walk-end-west", walk["poses"][-1]["e"], walk["poses"][-1]["n"], 1.7, (yaw + 180) % 360, 0.0),
         ("walk-mid-north", walk["poses"][len(walk["poses"]) // 2]["e"], walk["poses"][len(walk["poses"]) // 2]["n"], 1.7, 0.0, 2.0),
         ("dive-top-oblique", tc[0], tc[1], 120.0, 200.0, -20.0)]
frames = []
for q, (nm, e, n, agl, yw, pt) in enumerate(poses):
    for attempt in range(3):
        a = send(f"p1-h{q}", [{"id": f"p1-h{q}", "op": "camera", "e": e, "n": n, "agl": agl, "yaw": yw, "pitch": pt}])[0]
        f0 = status()["frame"]
        while status()["frame"] < f0 + 4: time.sleep(0.03)
        fr = send(f"p1-hf{q}", [{"id": f"p1-hf{q}", "op": "frame"}])[0]
        s = status(); c = s["camera"]
        same = abs(c["x"] - (e - E0)) < 1e-2 and abs(c["y"] - (n - N0)) < 1e-2 and abs(c["agl"] - agl) < 1e-3 and s["camera"]["site"] == SITE
        if same: break
    frames.append({"pose": nm, "e": e, "n": n, "agl": agl, "yaw_deg": yw, "pitch_deg": pt, "ground_furnace_m": a["ground_under_eye_m"]["electron"],
                   "sky_fraction_furnace": fr["sky_fraction"], "median_ground_distance_m": fr["median_ground_distance_m"],
                   "frame": fr["frame"], "camera_unchanged_at_frame": same, "attempts": attempt + 1})

import cupy as cp
KS = r"""
extern "C" {
__device__ float hat(const float* g, int n, float x, float y) {
  if (x < 0.f || y < 0.f || x > n - 1.001f || y > n - 1.001f) return -1e30f;
  int i = (int)x, j = (int)y; float fx = x - i, fy = y - j;
  const float* r0 = g + (size_t)j * n; const float* r1 = r0 + n;
  return (r0[i] * (1.f - fx) + r0[i+1] * fx) * (1.f - fy) + (r1[i] * (1.f - fx) + r1[i+1] * fx) * fy;
}
// electron: every pixel marched at a fixed 0.25 m, no skipping; 1 = sky (no ground hit inside the site)
__global__ void sky_march(const float* g, int n, double ex, double ey, double ez, double yaw, double pitch, double fov,
                          int w, int h, unsigned char* sky) {
  int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= w || py >= h) return;
  double th = tan(fov * 0.5), asp = (double)w / h;
  double sx = (2.0 * (px + 0.5) / w - 1.0) * th * asp, sy = (1.0 - 2.0 * (py + 0.5) / h) * th;
  double fx = sin(yaw) * cos(pitch), fy = cos(yaw) * cos(pitch), fz = sin(pitch), rx = cos(yaw), ry = -sin(yaw);
  double ux = ry * fz, uy = -rx * fz, uz = rx * fy - ry * fx;
  double dx = fx + sx * rx + sy * ux, dy = fy + sx * ry + sy * uy, dz = fz + sy * uz;
  double L = sqrt(dx*dx + dy*dy + dz*dz); float kx = dx / L, ky = dy / L, kz = dz / L;
  unsigned char s = 1;
  for (float t = 0.25f; t < 6000.f; t += 0.25f) {
    float x = (float)ex + kx * t, y = (float)ey + ky * t, z = (float)ez + kz * t;
    float gh = hat(g, n, x, y);
    if (gh < -1e29f) { if (x < 0.f || y < 0.f || x > n || y > n) break; else continue; }
    if (z <= gh) { s = 0; break; }
  }
  sky[py * w + px] = s;
}
}
"""
KM = cp.RawModule(code=KS).get_function("sky_march")


def ray_dirs(yaw, pitch):
    th, asp = math.tan(FOV / 2), W / H
    px, py = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    sx, sy = (2 * px / W - 1) * th * asp, (1 - 2 * py / H) * th
    fx, fy, fz = math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)
    rx, ry = math.cos(yaw), -math.sin(yaw); ux, uy, uz = ry * fz, -rx * fz, rx * fy - ry * fx
    dx, dy, dz = fx + sx * rx + sy * ux, fy + sx * ry + sy * uy, fz + sy * uz
    L = np.sqrt(dx * dx + dy * dy + dz * dz)
    return np.degrees(np.arctan2(dx, dy)) % 360, np.degrees(np.arcsin(dz / L))


def horizon_profile(G, x, y, z, az_deg, near=32.0, dn=0.05, df=0.5, xp=cp):
    """positron: max elevation angle (deg) of the bilinear ground seen from (x, y, z), per azimuth, to the site edge (f64)."""
    d = xp.concatenate([xp.arange(dn, near, dn), xp.arange(near, 3000.0, df)])
    a = xp.radians(xp.asarray(az_deg, xp.float64))
    out = xp.full(a.shape, -90.0)
    for s in range(0, a.size, 256):
        aa = a[s:s + 256, None]
        X, Y = x + xp.sin(aa) * d, y + xp.cos(aa) * d
        inside = (X >= 0) & (Y >= 0) & (X < NN - 1) & (Y < NN - 1)
        Xc, Yc = xp.clip(X, 0, NN - 1.0001), xp.clip(Y, 0, NN - 1.0001)
        i, j = xp.floor(Xc).astype(xp.int64), xp.floor(Yc).astype(xp.int64); fx, fy = Xc - i, Yc - j
        g00, g10, g01, g11 = G[j, i], G[j, i + 1], G[j + 1, i], G[j + 1, i + 1]
        h = g00 + (g10 - g00) * fx + (g01 - g00) * fy + (g00 - g10 - g01 + g11) * fx * fy
        ang = xp.where(inside, xp.degrees(xp.arctan2(h - z, d)), -90.0)
        out[s:s + 256] = ang.max(axis=1)
    return out


# the lock: the furnace is the one resident process; this script uses the card directly for seconds
t0 = time.time()
while True:
    try: os.mkdir(LOCK); break
    except FileExistsError:
        if time.time() - t0 > 900: sys.exit("GPU lock busy for 15 min")
        time.sleep(2)
lock_wait = time.time() - t0
try:
    while True:
        tq = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
        if float(tq) < 80: break
        time.sleep(10)
    G32 = cp.asarray(np.ascontiguousarray(A, dtype=np.float32)); G64 = G32.astype(cp.float64)
    AZ = np.arange(8192) * 360 / 8192
    for f in frames:
        x, y = f["e"] - E0, f["n"] - N0
        z = bil(x, y) + f["agl"]
        sky_e = cp.zeros(H * W, cp.uint8)
        tk = time.perf_counter()
        KM(((W + 15) // 16, (H + 15) // 16), (16, 16), (G32, np.int32(NN), np.float64(x), np.float64(y), np.float64(z), np.float64(math.radians(f["yaw_deg"])),
                                                         np.float64(math.radians(f["pitch_deg"])), np.float64(FOV), np.int32(W), np.int32(H), sky_e))
        cp.cuda.Stream.null.synchronize(); ms_e = (time.perf_counter() - tk) * 1000
        tk = time.perf_counter()
        prof = cp.asnumpy(horizon_profile(G64, x, y, z, AZ)); cp.cuda.Stream.null.synchronize(); ms_p = (time.perf_counter() - tk) * 1000
        paz, pel = ray_dirs(math.radians(f["yaw_deg"]), math.radians(f["pitch_deg"]))
        k = paz / (360 / 8192); k0 = np.floor(k).astype(int) % 8192; w_ = k - np.floor(k)
        hz = np.maximum(prof[k0], prof[(k0 + 1) % 8192])      # conservative between neighbouring azimuths
        sky_p = (pel > hz).ravel()
        se = cp.asnumpy(sky_e).astype(bool)
        dis = int(np.count_nonzero(se != sky_p))
        cols = np.unique(np.round(paz[H // 2]).astype(int) % 360)
        f["eye_z_m"] = z
        f["horizon_deg_in_view"] = {"min": float(prof[np.isin(np.round(AZ).astype(int) % 360, cols)].min()), "max": float(prof[np.isin(np.round(AZ).astype(int) % 360, cols)].max())}
        f["sky_fraction"] = {"electron_pixel_march_0p25m": float(se.mean()), "positron_horizon_profile_8192az": float(sky_p.mean()),
                             "furnace_render": f["sky_fraction_furnace"]}
        f["pair_pixels"] = {"what": "pixel is sky: fixed 0.25 m march vs horizon profile", "unit": "pixels", "tolerance": 0, "photons": dis,
                            "photon_share": dis / (W * H)}
        f["pair_furnace"] = pair([f["sky_fraction_furnace"]], [se.mean()], 0.005, "fraction of pixels", "sky fraction: furnace render (mip skip, 0.5 m growing step) vs electron march")
        f["kernel_ms"] = {"electron": ms_e, "positron": ms_p}
    # ground-level horizons against the world's horizon tiles (shade.mjs horizonAt; builder: exact near field, 1 m far)
    hz_rows = []
    for h in WJ["horizon"]:
        if not h.get("cell_e"): continue
        x, y = h["cell_e"] - E0, h["cell_n"] - N0
        z = bil(x, y)
        mine = cp.asnumpy(horizon_profile(G64, x, y, z, np.arange(32) * 11.25))
        wd = np.array([np.nan if v is None else v for v in h["deg"]], float)
        m = np.isfinite(wd)
        # CPU witness on 4 azimuths with numpy, 0.1 m steps all the way
        wz = cp.asnumpy(horizon_profile(np.asarray(A, np.float64), x, y, z, np.array([0.0, 90.0, 180.0, 270.0]), near=32.0, dn=0.1, df=0.25, xp=np)) \
            if len(hz_rows) == 0 else None
        hz_rows.append({"cell_en": [h["cell_e"], h["cell_n"]], "cell_ground_world_m": h["cell_ground_m"], "cell_ground_vram_m": z,
                        "world_deg_32az": [round(v, 2) if np.isfinite(v) else None for v in wd], "cupy_deg_32az": [round(float(v), 3) for v in mine],
                        "pair": pair(wd[m], mine[m], 1.0, "deg", "ground horizon: world .ghz tile (horizonAt) vs this CuPy f64 profile"),
                        "witness_cpu_0_90_180_270": None if wz is None else {"cpu": [float(v) for v in wz], "gpu": [float(mine[0]), float(mine[8]), float(mine[16]), float(mine[24])]}})
    R["horizon_ground_level"] = hz_rows
    R["gpu_lock_wait_s"] = lock_wait
    del G32, G64; cp.get_default_memory_pool().free_all_blocks()
finally:
    os.rmdir(LOCK)
R["horizon_frames"] = frames
R["furnace_at_end"] = {k: status().get(k) for k in ("pid", "alive", "uptime_s", "frame", "fps", "render_ms", "temperature_c", "card_vram_used_mib", "answered")}
R["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
body = json.dumps(R, indent=1, sort_keys=True, default=float)
R_sha = hashlib.sha256(body.encode()).hexdigest()
with open(f"{OUTD}/session.json", "w", encoding="utf-8", newline="\n") as f:
    f.write(body + "\n")
with open(f"{OUTD}/session.json.sha256", "w", encoding="utf-8", newline="\n") as f:
    f.write(f"{R_sha}  session.json\n")
print(R_sha)
