"""p6_sun_shade.py: sun and shade for the resident 20 MW south layout on open-land-01, in VRAM with CuPy.

Row-to-row shading at winter solstice solar noon and annual plane-of-array irradiance per table, with the terrain
horizon taken from the 1 m terrain on the card. Two formulations each (electron / positron), photons = tables that
disagree beyond tolerance, CPU witness on a sample, and comparisons with the world's own sun-mode numbers
(web/world v08 sun.mjs position, the site's 32-sector horizon tiles, and the sun-cell horizon-shaded direct share).
Run under the GPU lock: $GRID_PY $FURNACE_SRC/p6/p6_sun_shade.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import furnace_paths as FP
import hashlib, json, math, os, pathlib, struct, subprocess, sys, time
import numpy as np
import cupy as cp

SITE = "open-land-01"
LID = (FP.LIDAR + f"/{SITE}")
WORLD = FP.WORLD
OUT = (FP.FURNACE + "/out/p6-sun-shade")
WORK = f"{OUT}/work"
LOCK = FP.LOCK
LAT, LON = FP.site_lat_lon()   # the site's latitude and longitude, from the environment
TOFF_H = 0.1834                   # PVGIS instant within each hour
ALBEDO = 0.2
TOL_NOON = 0.05                   # shaded fraction of a table, absolute
TOL_ANNUAL = 0.02                 # annual beam on the plane, relative
t_wall = time.perf_counter()

# ------------------------------------------------------------------ lock and temperature
def temp_c():
    try:
        return float(subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                                    capture_output=True, text=True, timeout=10).stdout.strip().split()[0])
    except Exception:
        return -1.0

for k in range(600):
    try:
        os.mkdir(LOCK); break
    except FileExistsError:
        time.sleep(1)
else:
    sys.exit("gpu lock busy for 10 min")
try:
    while temp_c() >= 80:
        time.sleep(5)
    t_start = temp_c()

    # ------------------------------------------------------------------ world: terrain, layout, TMY
    src = json.load(open(f"{LID}/source.json"))
    E0, N0 = float(src["origin_e_m"]), float(src["origin_n_m"])
    A = np.load(f"{LID}/source.npy", mmap_mode="r")
    NG = A.shape[0]
    g = cp.asarray(np.ascontiguousarray(A, dtype=np.float32))
    GMAX = float(g.max())
    L = json.load(open(f"{WORK}/layout.json"))["layout"]
    Tt = np.array(L["tables"], np.float64).reshape(-1, 12)
    NT = len(Tt)
    tb = L["table"]; BETA = math.radians(tb["tiltDeg"]); SL = tb["depth"] / math.cos(BETA); RISE = tb["rise"]
    LOW = 0.8
    C = Tt[:, :8].reshape(-1, 4, 2) - np.array([E0, N0])          # local metres, corners SW SE NE NW
    Hc = Tt[:, 8:12] + LOW + np.array([0, 0, RISE, RISE])            # corner heights of the table top
    P3 = np.concatenate([C, Hc[:, :, None]], axis=2)                 # (NT,4,3)
    ctr = P3.mean(axis=1)
    raw = open(f"{LID}/sun/tmy-hourly.bin", "rb").read()
    tmy = np.frombuffer(raw[16:], "<u2").reshape(8760, 3).astype(np.float64)   # G(h), Gb(n), Gd(h) W/m2
    GH, BN, DH = tmy[:, 0], tmy[:, 1], tmy[:, 2]

    # ------------------------------------------------------------------ sun: NOAA equations (port of sun.mjs), f64
    def sun_pos(ms):
        jd = ms / 86400000.0 + 2440587.5; T = (jd - 2451545) / 36525
        L0 = np.mod(280.46646 + T * (36000.76983 + T * 0.0003032), 360)
        M = 357.52911 + T * (35999.05029 - 0.0001537 * T); e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
        Mr = np.radians(M)
        Cc = np.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T)) + np.sin(2 * Mr) * (0.019993 - 0.000101 * T) + np.sin(3 * Mr) * 0.000289
        om = np.radians(125.04 - 1934.136 * T); lam = np.radians(L0 + Cc - 0.00569 - 0.00478 * np.sin(om))
        eps0 = 23 + (26 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60) / 60
        eps = np.radians(eps0 + 0.00256 * np.cos(om)); de = np.arcsin(np.sin(eps) * np.sin(lam))
        y = np.tan(eps / 2) ** 2; L0r = np.radians(L0)
        eot = 4 * np.degrees(y * np.sin(2 * L0r) - 2 * e * np.sin(Mr) + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
                             - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))
        minutes = np.mod(ms, 86400000.0) / 60000.0
        tst = np.mod(minutes + eot + 4 * LON, 1440); H = np.radians(tst / 4 - 180); la = math.radians(LAT)
        cz = np.clip(np.sin(la) * np.sin(de) + np.cos(la) * np.cos(de) * np.cos(H), -1, 1)
        el = 90 - np.degrees(np.arccos(cz))
        az = np.mod(np.degrees(np.arctan2(np.sin(H), np.cos(H) * np.sin(la) - np.tan(de) * np.cos(la))) + 180, 360)
        t = np.tan(np.radians(el))
        with np.errstate(all="ignore"):
            r = np.where(el > 5, 58.1 / t - 0.07 / t**3 + 0.000086 / t**5,
                np.where(el > -0.575, 1735 + el * (-518.2 + el * (103.4 + el * (-12.79 + el * 0.711))), -20.772 / t)) / 3600
        r = np.where(el > 85, 0, r)
        return az, el + r
    GAMMA = (LON + 2) * math.sin(math.radians(LAT))   # grid convergence, as sun.mjs
    ms = (np.datetime64("2026-01-01T00:00") - np.datetime64("1970-01-01T00:00")).astype("timedelta64[ms]").astype(np.float64) \
         + (np.arange(8760) + TOFF_H) * 3600000.0
    AZt, EL = sun_pos(ms)
    AZ = AZt - GAMMA                                          # grid azimuth
    up = EL > 0

    # sun pair: this port against the world's own sun.mjs on 48 hours
    pick = np.arange(0, 8760, 8760 // 48)[:48]
    js = ("import { pathToFileURL } from 'node:url'; const S = await import(pathToFileURL('%s/sun.mjs').href);"
          "const t = JSON.parse(process.argv[2]); console.log(JSON.stringify(t.map(m => { const p = S.solarPosition(%r, %r, m); return [p.azimuth, p.elevation]; })));"
          % (WORLD, LAT, LON))
    open(f"{WORK}/sunpair.mjs", "w").write(js)
    wj = np.array(json.loads(subprocess.run(["node", f"{WORK}/sunpair.mjs", json.dumps(ms[pick].tolist())],
                                            capture_output=True, text=True).stdout))
    daz = np.abs(((AZt[pick] - wj[:, 0] + 180) % 360) - 180); del_ = np.abs(EL[pick] - wj[:, 1])
    sun_pair = {"hours": 48, "max_az_diff_deg": float(daz[EL[pick] > 0].max()), "max_el_diff_deg": float(del_.max()),
                "photons": int((del_ > 0.01).sum() + (daz[EL[pick] > 0] > 0.01).sum()), "tolerance_deg": 0.01}
    noon_ms = float(subprocess.run(["node", "-e", "import('%s').then(S=>console.log(S.solarNoon(%r, Date.UTC(2026,11,21))))" % (pathlib.Path(WORLD, "sun.mjs").as_uri(), LON)],
                                   capture_output=True, text=True).stdout)
    az_n, el_n = sun_pos(np.array([noon_ms])); az_n = float(az_n[0] - GAMMA); el_n = float(el_n[0])

    # ------------------------------------------------------------------ kernels
    K = cp.RawModule(code=r"""
extern "C" {
__device__ __forceinline__ float hat(const float* g, int n, float x, float y) {
  if (x < 0.f || y < 0.f || x > n - 1.001f || y > n - 1.001f) return -1e30f;
  int i = (int)x, j = (int)y; float fx = x - i, fy = y - j;
  const float* r0 = g + (size_t)j * n; const float* r1 = r0 + n;
  float a = r0[i] + (r0[i+1] - r0[i]) * fx, b = r1[i] + (r1[i+1] - r1[i]) * fx;
  return a + (b - a) * fy;
}
// electron horizon: from point p, 360 azimuths (1 deg, grid), 1 m bilinear steps to the data edge; degrees
__global__ void horizon(const float* g, int n, const double* p, int np_, float* hor) {
  int id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= np_ * 360) return;
  int q = id / 360, a = id % 360; float az = a * 0.017453292519943295f;
  float x0 = p[3*q], y0 = p[3*q+1], z0 = p[3*q+2], kx = sinf(az), ky = cosf(az), best = -1e9f;
  for (float d = 1.f; d < 4000.f; d += 1.f) { float h = hat(g, n, x0 + kx * d, y0 + ky * d); if (h < -1e29f) break;
    float t = (h - z0) / d; if (t > best) best = t; }
  hor[id] = atanf(best) * 57.29577951308232f;
}
__device__ bool tri(const double* a, const double* b, const double* c, float ox, float oy, float oz, float dx, float dy, float dz) {
  float e1x = b[0]-a[0], e1y = b[1]-a[1], e1z = b[2]-a[2], e2x = c[0]-a[0], e2y = c[1]-a[1], e2z = c[2]-a[2];
  float px = dy*e2z - dz*e2y, py = dz*e2x - dx*e2z, pz = dx*e2y - dy*e2x; float det = e1x*px + e1y*py + e1z*pz;
  if (fabsf(det) < 1e-9f) return false; float inv = 1.f / det;
  float tx = ox-a[0], ty = oy-a[1], tz = oz-a[2]; float u = (tx*px + ty*py + tz*pz) * inv; if (u < 0.f || u > 1.f) return false;
  float qx = ty*e1z - tz*e1y, qy = tz*e1x - tx*e1z, qz = tx*e1y - ty*e1x; float v = (dx*qx + dy*qy + dz*qz) * inv;
  if (v < 0.f || u + v > 1.f) return false; float t = (e2x*qx + e2y*qy + e2z*qz) * inv; return t > 1e-3f;
}
// positron: rays from sample points on tables toward sun directions; tests neighbour table quads (2 triangles)
// then marches the terrain directly. out[pt*nd + k] = 0 lit, 1 table, 2 terrain.
__global__ void rays(const float* g, int n, float gmax, const double* pts, const int* ptab, int npts,
                     const double* quads, const int* nbr, int kn, const float* dirs, int nd, unsigned char* out) {
  long id = (long)blockIdx.x * blockDim.x + threadIdx.x; if (id >= (long)npts * nd) return;
  int pt = id / nd, k = id % nd; float ox = pts[3*pt], oy = pts[3*pt+1], oz = pts[3*pt+2];
  float dx = dirs[3*k], dy = dirs[3*k+1], dz = dirs[3*k+2]; int t = ptab[pt]; unsigned char r = 0;
  for (int m = 0; m < kn && !r; m++) { int o = nbr[t*kn + m]; if (o < 0) break; const double* Q = quads + o*12;
    if (tri(Q, Q+3, Q+6, ox,oy,oz, dx,dy,dz) || tri(Q, Q+6, Q+9, ox,oy,oz, dx,dy,dz)) r = 1; }
  if (!r) { float hz = sqrtf(dx*dx + dy*dy); if (hz > 1e-6f) { float kx = dx/hz, ky = dy/hz, kz = dz/hz;
    for (float d = 1.f; d < 4000.f; d += 1.f) { float z = oz + kz * d; if (z > gmax) break;
      float h = hat(g, n, ox + kx*d, oy + ky*d); if (h < -1e29f) break; if (h > z) { r = 2; break; } } } }
  out[id] = r;
}
}""", options=("--use_fast_math",))
    K_HOR, K_RAYS = K.get_function("horizon"), K.get_function("rays")
    ev0 = cp.cuda.Event(); ev1 = cp.cuda.Event()

    # table geometry: plane normal from corners, front neighbour, neighbour list
    nrm = np.cross(P3[:, 1] - P3[:, 0], P3[:, 3] - P3[:, 0]); nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    vlow = C[:, 0, 1]; ulo, uhi = C[:, 0, 0], C[:, 1, 0]
    front = np.full(NT, -1)
    for i in range(NT):
        dv = vlow[i] - vlow
        ov = np.minimum(uhi[i], uhi) - np.maximum(ulo[i], ulo)
        cand = np.where((dv > 5) & (dv < 20) & (ov > 0.5 * (uhi[i] - ulo[i])))[0]
        if len(cand): front[i] = cand[np.argmin(dv[cand])]
    KN = 32
    dc = np.linalg.norm(ctr[:, None, :2] - ctr[None, :, :2], axis=2); np.fill_diagonal(dc, 1e9)
    nbr = np.argsort(dc, axis=1)[:, :KN].astype(np.int32); nbr[np.take_along_axis(dc, nbr, 1) > 60] = -1
    quads = cp.asarray(P3.reshape(NT, 12)); nbr_d = cp.asarray(np.ascontiguousarray(nbr))

    def sample_points(ns_, nu_):
        s = (np.arange(ns_) + 0.5) / ns_; u = (np.arange(nu_) + 0.5) / nu_
        S_, U_ = np.meshgrid(s, u, indexing="ij")
        w = np.stack([(1 - U_) * (1 - S_), U_ * (1 - S_), U_ * S_, (1 - U_) * S_], -1)       # SW SE NE NW
        pts = np.einsum("ijk,tkc->tijc", w, P3) + nrm[:, None, None, :] * 0.01
        return pts.reshape(-1, 3), np.repeat(np.arange(NT), ns_ * nu_).astype(np.int32), S_

    def run_rays(pts, ptab, dirs):
        npts, nd = len(pts), len(dirs); out = cp.zeros(npts * nd, cp.uint8)
        K_RAYS(((npts * nd + 255) // 256,), (256,), (g, np.int32(NG), np.float32(GMAX), cp.asarray(pts), cp.asarray(ptab), np.int32(npts),
               quads, nbr_d, np.int32(KN), cp.asarray(dirs.astype(np.float32)), np.int32(nd), out))
        return out.reshape(npts, nd)

    def sdir(az, el):
        a, e = np.radians(az), np.radians(el)
        return np.stack([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)], -1)

    # ------------------------------------------------------------------ electron horizon per table (VRAM march)
    ev0.record()
    hor = cp.zeros(NT * 360, cp.float32)
    K_HOR(((NT * 360 + 255) // 256,), (256,), (g, np.int32(NG), cp.asarray(ctr), np.int32(NT), hor))
    ev1.record(); ev1.synchronize(); ms_hor = cp.cuda.get_elapsed_time(ev0, ev1)
    HOR = hor.reshape(NT, 360)

    def hor_at(az):          # (NT, k) linear between 1-degree azimuths
        f = cp.asarray(np.mod(az, 360)); k0 = cp.floor(f).astype(cp.int32) % 360; w = f - cp.floor(f)
        return HOR[:, k0] * (1 - w) + HOR[:, (k0 + 1) % 360] * w

    # ------------------------------------------------------------------ 1) winter solstice solar noon
    d_n = sdir(np.array([az_n]), np.array([el_n]))
    gam = math.radians(az_n - 180); tap = math.tan(math.radians(el_n)) / math.cos(gam)
    zlow = Hc[:, :2].mean(1); vb = vlow
    def row_shade(tap_):     # electron analytic, (NT,) shaded share of slope; tap_ scalar or (k,) -> (NT,k)
        tap_ = np.atleast_1d(tap_)
        f = np.zeros((NT, len(tap_)))
        ok = front >= 0; fi = front[ok]
        P = (vb[ok] - vb[fi])[:, None]; dh = (zlow[fi] - zlow[ok])[:, None]
        s = (dh + SL * math.sin(BETA) - (P - SL * math.cos(BETA)) * tap_) / (math.sin(BETA) + math.cos(BETA) * tap_)
        if NOMINAL:
            f[ok] = np.clip(s / SL, 0, 1)
            return f
        # draped: the world sets each corner at ground + 0.8 (+ rise), so use the real front top edge and back plane
        zft = Hc[fi, 2:4].mean(1)[:, None]; vft = C[fi, 2, 1][:, None]
        zbl = Hc[ok, 0:2].mean(1)[:, None]; dz = (Hc[ok, 2:4].mean(1) - Hc[ok, 0:2].mean(1))[:, None]
        dep = (C[ok, 2, 1] - C[ok, 0, 1])[:, None]
        t = (zft - zbl - (vb[ok][:, None] - vft) * tap_) / (dz + dep * tap_)
        f[ok] = np.clip(t, 0, 1)
        return f
    NOMINAL = True; e_row_nom = row_shade(tap)[:, 0]; NOMINAL = False
    e_row = row_shade(tap)[:, 0]
    e_ter = (cp.asnumpy(hor_at(np.array([az_n]))[:, 0]) > el_n).astype(np.float64)
    NS, NU = 64, 16
    pts, ptab, S_ = sample_points(NS, NU)
    ev0.record(); rn = run_rays(pts, ptab, d_n); ev1.record(); ev1.synchronize(); ms_noon = cp.cuda.get_elapsed_time(ev0, ev1)
    rn = cp.asnumpy(rn[:, 0]).reshape(NT, NS * NU)
    p_row = (rn == 1).mean(1); p_ter = (rn == 2).mean(1); p_any = (rn > 0).mean(1)
    e_any = np.maximum(e_row, e_ter)
    noon_ph_row = int((np.abs(e_row - p_row) > TOL_NOON).sum()); noon_ph_ter = int((np.abs(e_ter - p_ter) > 0.5).sum())

    # ------------------------------------------------------------------ 2) annual, electron: hourly, 8760 x NT on the card
    cb, sb = math.cos(BETA), math.sin(BETA)
    D = sdir(AZ, EL)                                                   # (8760,3)
    cos_i = np.clip(D @ nrm.T, 0, None).T                               # (NT,8760) true corner-plane normal
    ev0.record()
    hA = hor_at(AZ)                                                     # (NT,8760)
    lit_t = (cp.asarray(EL)[None, :] > hA) & cp.asarray(up)[None, :]
    gam_h = np.radians(AZ - 180); cg = np.cos(gam_h)
    tap_h = np.where((cg > 0.01) & up, np.tan(np.radians(np.clip(EL, 0.01, 89))) / np.maximum(cg, 0.01), 1e6)
    fr = cp.asarray(row_shade(tap_h)); fr[:, ~(up & (cg > 0.01))] = 0
    bn = cp.asarray(BN); ci = cp.asarray(cos_i)
    beam_open = (bn[None, :] * ci * cp.asarray(up)[None, :]).sum(1) / 1000
    beam_hor = (bn[None, :] * ci * lit_t).sum(1) / 1000
    beam_e = (bn[None, :] * ci * lit_t * (1 - fr)).sum(1) / 1000
    hz_e = (bn * cp.asarray(np.clip(np.sin(np.radians(EL)), 0, None))[None, :] * lit_t).sum(1) / 1000   # horizontal beam, horizon only
    ev1.record(); ev1.synchronize(); ms_ann_e = cp.cuda.get_elapsed_time(ev0, ev1)
    beam_open, beam_hor, beam_e, hz_e = map(cp.asnumpy, (beam_open, beam_hor, beam_e, hz_e))
    diff_poa = DH.sum() / 1000 * (1 + cb) / 2; refl = ALBEDO * GH.sum() / 1000 * (1 - cb) / 2
    hz_open = float((BN * np.clip(np.sin(np.radians(EL)), 0, None)).sum() / 1000)

    # ------------------------------------------------------------------ annual, positron: 1-degree sky map, 3D rays
    m = up & (BN > 0)
    ka = np.floor(np.mod(AZ[m], 360)).astype(int); ke = np.floor(EL[m]).astype(int); key = ka * 100 + ke
    uk, inv = np.unique(key, return_inverse=True)
    wsum = np.bincount(inv, BN[m]) / 1000                              # kWh/m2 normal, per bin
    dsum = np.stack([np.bincount(inv, BN[m] * D[m][:, c]) for c in range(3)], -1)
    bdir = dsum / np.linalg.norm(dsum, axis=1)[:, None]
    NSa, NUa = 8, 8
    pa, pta, _ = sample_points(NSa, NUa)
    ev0.record(); ra = run_rays(pa, pta, bdir); ev1.record(); ev1.synchronize(); ms_ann_p = cp.cuda.get_elapsed_time(ev0, ev1)
    cos_b = cp.asarray(np.clip(nrm @ bdir.T, 0, None))                 # (NT,B)
    litp = (ra == 0).reshape(NT, NSa * NUa, -1).mean(axis=1, dtype=cp.float32)
    notter = (ra != 2).reshape(NT, NSa * NUa, -1).mean(axis=1, dtype=cp.float32)
    wb = cp.asarray(wsum)
    beam_p = cp.asnumpy((wb[None, :] * cos_b * litp).sum(1))
    beam_hor_p = cp.asnumpy((wb[None, :] * cos_b * notter).sum(1))
    rel = (beam_e - beam_p) / beam_p
    ann_ph = int((np.abs(rel) > TOL_ANNUAL).sum())
    poa_e = beam_e + diff_poa + refl; poa_p = beam_p + diff_poa + refl; poa_open = beam_open + diff_poa + refl

    # ------------------------------------------------------------------ world sun-mode comparisons
    def tile_val(kind, e, n):
        idx = json.load(open(f"{LID}/horizon/horizon-tiles.json")) if kind == "ghz" else \
              json.load(open(f"{WORLD}/sites/{SITE}/sun/cells/sun-cells.json"))
        base = f"{LID}/horizon/" if kind == "ghz" else f"{WORLD}/sites/{SITE}/sun/cells/"
        out, cache = [], {}
        for ee, nn in zip(e, n):
            t = next((t for t in idx["tiles"] if t["e0"] <= ee < t["e0"] + 256 and t["n0"] <= nn < t["n0"] + 256), None)
            if t is None: out.append(None); continue
            if t["file"] not in cache:
                b = open(base + t["file"], "rb").read()
                cache[t["file"]] = (np.frombuffer(b[32:], "<i2").reshape(65, 65, 32) if kind == "ghz"
                                    else np.frombuffer(b[32:], "u1").reshape(65, 65, 12, 4))
            i = int(round((ee - t["e0"]) / 4)); j = int(round((nn - t["n0"]) / 4))
            out.append(cache[t["file"]][j, i])
        return out, idx
    ce_, cn_ = ctr[:, 0] + E0, ctr[:, 1] + N0
    wh, _ = tile_val("ghz", ce_, cn_)
    mine32 = cp.asnumpy(HOR[:, ::1])[:, (np.arange(32) * 11.25).astype(int)]
    mine32 = cp.asnumpy(hor_at(np.arange(32) * 11.25))
    hd = []
    for q in range(NT):
        if wh[q] is None: continue
        w = wh[q].astype(float); ok = w != -32768
        hd.append(mine32[q][ok] - w[ok] / 100)
    hd = np.concatenate(hd) if hd else np.zeros(1)
    wc, cidx = tile_val("gsc", ce_, cn_)
    bq = np.array(cidx["base"]["tmy_direct_kwh_m2"]); world_hz = np.array([np.nan if (v is None or (v[:, 3] == 255).any())
                                                                          else float((v[:, 3] / 250.0 * bq).sum()) for v in wc])
    okw = np.isfinite(world_hz)
    cellrel = (hz_e[okw] - world_hz[okw]) / world_hz[okw]

    # ------------------------------------------------------------------ CPU witness: 6 tables, f64 numpy
    rng = np.random.default_rng(20260927); wt = rng.choice(NT, 6, replace=False)
    def hat64(x, y):
        i = np.floor(x).astype(int); j = np.floor(y).astype(int); ok = (i >= 0) & (j >= 0) & (i < NG - 1) & (j < NG - 1)
        i = np.clip(i, 0, NG - 2); j = np.clip(j, 0, NG - 2); fx, fy = x - i, y - j
        a = A[j, i]; b = A[j, i + 1]; c = A[j + 1, i]; d = A[j + 1, i + 1]
        return np.where(ok, a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy, np.nan)
    wit = []
    dd = np.arange(1, 4000, 1.0)
    for q in wt:
        az = np.radians(np.arange(360.0))
        X = ctr[q, 0] + np.sin(az)[:, None] * dd; Y = ctr[q, 1] + np.cos(az)[:, None] * dd
        Hh = hat64(X, Y); hcpu = np.degrees(np.arctan(np.nanmax((Hh - ctr[q, 2]) / dd, axis=1)))
        hg = cp.asnumpy(HOR[q])
        f = np.mod(AZ, 360); k0 = np.floor(f).astype(int) % 360; w = f - np.floor(f)
        hh = hcpu[k0] * (1 - w) + hcpu[(k0 + 1) % 360] * w
        litc = up & (EL > hh)
        frc = fr[q].get() if hasattr(fr, "get") else fr[q]
        be = float((BN * cos_i[q] * litc * (1 - cp.asnumpy(fr[q]))).sum() / 1000)
        wit.append({"table": int(q), "horizon_max_abs_diff_deg": float(np.abs(hcpu - hg).max()),
                    "beam_cpu_kwh_m2": be, "beam_gpu_kwh_m2": float(beam_e[q]), "rel_diff": float(be / beam_e[q] - 1)})
    wit_ph = int(sum(abs(w["rel_diff"]) > 1e-3 or w["horizon_max_abs_diff_deg"] > 0.05 for w in wit))

    # ------------------------------------------------------------------ results
    pct = lambda a, p: float(np.percentile(a, p))
    def stats(a): return {"mean": float(np.mean(a)), "p5": pct(a, 5), "median": pct(a, 50), "p95": pct(a, 95), "min": float(np.min(a)), "max": float(np.max(a))}
    first_row = front < 0
    res = {
        "what": "p6 sun and shade: 20 MW south fixed-tilt layout (world JS, v08) on open-land-01, 1 m terrain in VRAM",
        "label": L["label"], "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "plant": {"tables": NT, "mwp": L["built"]["mwp"], "tilt_deg": tb["tiltDeg"], "slope_m": SL, "pitch_m": tb["pitch"],
                  "gcr": SL / tb["pitch"], "low_edge_m": LOW, "first_row_tables": int(first_row.sum())},
        "sun_pair_vs_world_sun_mjs": sun_pair,
        "winter_solstice_noon": {
            "utc": np.datetime64(int(noon_ms), "ms").astype(str), "sun_grid_az_deg": az_n, "sun_el_deg_refracted": el_n,
            "profile_angle_deg": math.degrees(math.atan(tap)),
            "no_shade_limit_pitch_m": SL * math.cos(BETA) + SL * math.sin(BETA) / tap,
            "row_shaded_share_of_table": {"electron_analytic_2d": stats(e_row), "positron_raytrace_64x16": stats(p_row),
                                          "electron_nominal_rigid_tilt_first_try": stats(e_row_nom), "rows_behind_only_electron": stats(e_row[~first_row]), "rows_behind_only_positron": stats(p_row[~first_row])},
            "terrain_shaded_share": {"electron_horizon_1deg": float(e_ter.mean()), "positron_raytrace": float(p_ter.mean())},
            "plant_beam_shaded_share": {"electron": float(e_any.mean()), "positron": float(p_any.mean())},
            "pair_row": {"unit": "share of table", "tolerance": TOL_NOON, "photons": noon_ph_row, "photons_first_try_nominal_tilt": int((np.abs(e_row_nom - p_row) > TOL_NOON).sum()), "photons_first_row": int(((np.abs(e_row - p_row) > TOL_NOON) & (front < 0)).sum()), "max_abs_diff": float(np.abs(e_row - p_row).max())},
            "pair_terrain": {"unit": "table in terrain shadow", "tolerance": 0.5, "photons": noon_ph_ter},
            "furnace_resident_shade_op": "open-land-01 whole 2 km site, ground cells in terrain shadow at the same sun: 0.04329 (stride 1 m, 4,194,304 cells), 0.04328 (stride 4 m); 0 photons (answers p6-shade-wsn-s1, -s4)",
            "kernel_ms": ms_noon},
        "annual": {
            "tmy": "PVGIS 5.3 TMY hourly (site file), G(h) %.1f, Gb(n) %.1f, Gd(h) %.1f kWh/m2" % (GH.sum() / 1000, BN.sum() / 1000, DH.sum() / 1000),
            "model": "beam on the corner plane x terrain horizon x row shade; isotropic diffuse x (1+cos b)/2; albedo 0.2",
            "poa_kwh_m2": {"open_no_shade": stats(poa_open), "electron": stats(poa_e), "positron": stats(poa_p)},
            "beam_kwh_m2": {"open": stats(beam_open), "horizon_only_electron": stats(beam_hor), "horizon_only_positron": stats(beam_hor_p),
                            "electron_hourly_analytic_rows": stats(beam_e), "positron_skymap_raytrace": stats(beam_p)},
            "diffuse_poa_kwh_m2": diff_poa, "reflected_poa_kwh_m2": refl,
            "loss_vs_open_pct": {"terrain_electron": float(100 * (1 - beam_hor.sum() / beam_open.sum())),
                                 "terrain_positron": float(100 * (1 - beam_hor_p.sum() / beam_open.sum())),
                                 "rows_and_terrain_electron_beam": float(100 * (1 - beam_e.sum() / beam_open.sum())),
                                 "rows_and_terrain_positron_beam": float(100 * (1 - beam_p.sum() / beam_open.sum())),
                                 "poa_electron": float(100 * (1 - poa_e.sum() / poa_open.sum())),
                                 "poa_positron": float(100 * (1 - poa_p.sum() / poa_open.sum()))},
            "plant_mean_poa_kwh_m2": {"electron": float(poa_e.mean()), "positron": float(poa_p.mean())},
            "pair": {"unit": "annual beam on plane, relative", "tolerance": TOL_ANNUAL, "photons": ann_ph,
                     "max_abs_rel_diff": float(np.abs(rel).max()), "mean_rel_diff": float(rel.mean()),
                     "photons_first_row": int(((np.abs(rel) > TOL_ANNUAL) & first_row).sum())},
            "skymap_bins": int(len(uk)), "rays": int(len(pa) * len(uk)),
            "kernel_ms": {"horizon_360": ms_hor, "electron_hourly": ms_ann_e, "positron_rays": ms_ann_p}},
        "world_sun_mode_compare": {
            "horizon_vs_site_ghz_32_sectors_deg": {"n": int(hd.size), "mean_signed": float(hd.mean()), "p95_abs": pct(np.abs(hd), 95), "max_abs": float(np.abs(hd).max()),
                                                    "note": "mine from table centre (ground + 0.8 m + half rise); world tiles from the 4 m ground node"},
            "open_horizontal_beam_kwh_m2": {"mine_hourly": hz_open, "world_cells_base": float(bq.sum()), "rel_diff": hz_open / float(bq.sum()) - 1},
            "horizon_shaded_horizontal_beam_kwh_m2_at_table_centres": {"tables": int(okw.sum()), "mine_mean": float(hz_e[okw].mean()),
                "world_cells_mean": float(np.nanmean(world_hz)), "mean_rel_diff": float(cellrel.mean()), "max_abs_rel_diff": float(np.abs(cellrel).max()),
                "photons_gt_2pct": int((np.abs(cellrel) > 0.02).sum())}},
        "witness": {"method": "numpy f64: 360-azimuth horizon march and hourly beam sum, same row-shade factor", "tables": wit, "photons": wit_ph},
        "thermal_c": {"start": t_start, "end": temp_c()}, "vram_pool_mib": cp.get_default_memory_pool().total_bytes() / 2**20,
        "wall_s": time.perf_counter() - t_wall,
        "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
        "layout_sha256": hashlib.sha256(open(f"{WORK}/layout.json", "rb").read()).hexdigest(),
        "tmy_sha256": hashlib.sha256(raw).hexdigest(),
    }
    body = json.dumps(res, indent=1, default=float)
    res["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    with open(f"{OUT}/p6-sun-shade.json", "w", newline="\n") as f: json.dump(res, f, indent=1, default=float); f.write("\n")
    with open(f"{OUT}/per-table.csv", "w", newline="\n") as f:
        f.write("table,e,n,front,noon_row_e,noon_row_p,noon_terrain_e,noon_terrain_p,beam_open,beam_e,beam_p,rel,poa_e,poa_p,hz_mine,hz_world\n")
        for q in range(NT):
            f.write(f"{q},{ce_[q]:.1f},{cn_[q]:.1f},{front[q]},{e_row[q]:.4f},{p_row[q]:.4f},{e_ter[q]:.0f},{p_ter[q]:.4f},"
                    f"{beam_open[q]:.2f},{beam_e[q]:.2f},{beam_p[q]:.2f},{rel[q]:.5f},{poa_e[q]:.2f},{poa_p[q]:.2f},{hz_e[q]:.2f},{world_hz[q]:.2f}\n")
    print(json.dumps({k: res[k] for k in ("winter_solstice_noon", "annual", "world_sun_mode_compare", "witness", "sun_pair_vs_world_sun_mjs", "wall_s")}, indent=1, default=float))
finally:
    try: os.rmdir(LOCK)
    except OSError: pass
