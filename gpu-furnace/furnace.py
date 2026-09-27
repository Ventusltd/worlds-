"""furnace.py: ONE resident CuPy process on the Blackwell card.

Keeps the world in VRAM (1 m terrain for every open-land site, its max-mip, the far rings, the plant layout
exported from the world's own JS, cable lines, a camera), renders it constantly (ray cast over the height field,
lines and tables drawn with a depth test), and answers text commands from $FURNACE_ROOT/in/*.cmd with text maths in
$FURNACE_ROOT/out/answers/<id>.json: numbers, units, two formulations (electron / positron), photons (disagreements
beyond tolerance), a CPU witness on a sample, timing, and a sha256 of the answer body.

Run: $GRID_PY $FURNACE_SRC/furnace.py
"""
import furnace_paths as FP
import glob, hashlib, json, math, os, shutil, struct, subprocess, sys, threading, time, traceback, zlib
from datetime import datetime, timezone
import numpy as np
import cupy as cp

ROOT = FP.FURNACE
IN, DONE = f"{ROOT}/in", f"{ROOT}/in/done"
OUT, ANS, FRAMES, WORK = f"{ROOT}/out", f"{ROOT}/out/answers", f"{ROOT}/out/frames", f"{ROOT}/work"
LIDAR = FP.LIDAR
FARRING = FP.FARRING
NODE_EXPORT = f"{ROOT}/src/export_world.mjs"
W, H = 1280, 720
PREVIEW_EVERY = 120          # frames
FPS_CAP = 60.0               # the card is shared: never spin faster than this
TEMP_PAUSE_C = 80
RING = 8                     # frames kept in VRAM
for d in (IN, DONE, ANS, FRAMES, WORK):
    os.makedirs(d, exist_ok=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=float)
        f.write("\n")
    os.replace(tmp, path)


def log(msg):
    with open(f"{OUT}/furnace.log", "a", encoding="utf-8", newline="\n") as f:
        f.write(f"{now_iso()} {msg}\n")


# ------------------------------------------------------------------ kernels
KSRC = r"""
extern "C" {
__device__ __forceinline__ float hat(const float* g, int n, float x, float y) {
  // bilinear, lerp form, x east y north in cells; n = samples per side
  if (x < 0.f || y < 0.f || x > n - 1.001f || y > n - 1.001f) return -1e30f;
  int i = (int)x, j = (int)y; float fx = x - i, fy = y - j;
  const float* r0 = g + (size_t)j * n; const float* r1 = r0 + n;
  float a = r0[i] + (r0[i+1] - r0[i]) * fx, b = r1[i] + (r1[i+1] - r1[i]) * fx;
  return a + (b - a) * fy;
}
// Ray cast: max-mip (8 m blocks) skip + 0.5 m march + 12 bisections. Writes depth (m) and rgba.
__global__ void render(const float* g, const float* mip, int n, int mn, int mb,
                       float ex, float ey, float ez, float yaw, float pitch, float fov,
                       int w, int h, float* depth, unsigned char* rgba, float sunx, float suny, float sunz, float t) {
  int px = blockIdx.x * blockDim.x + threadIdx.x, py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= w || py >= h) return;
  float th = tanf(fov * 0.5f), asp = (float)w / h;
  float sx = (2.f * (px + 0.5f) / w - 1.f) * th * asp, sy = (1.f - 2.f * (py + 0.5f) / h) * th;
  float cy = cosf(yaw), syw = sinf(yaw), cp_ = cosf(pitch), sp = sinf(pitch);
  // forward (yaw 0 = north), right, up
  float fx = syw * cp_, fy = cy * cp_, fz = sp;
  float rx = cy, ry = -syw, rz = 0.f;
  float ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
  float dx = fx + sx * rx + sy * ux, dy = fy + sx * ry + sy * uy, dz = fz + sx * rz + sy * uz;
  float L = rsqrtf(dx*dx + dy*dy + dz*dz); dx *= L; dy *= L; dz *= L;
  float tt = 0.f, tmax = 6000.f, hit = -1.f, prev = 0.f;
  float step = 0.5f;
  for (int it = 0; it < 4000 && tt < tmax; it++) {
    float x = ex + dx * tt, y = ey + dy * tt, z = ez + dz * tt;
    if (x < 0.f || y < 0.f || x > n - 2 || y > n - 2) { if (dz >= 0.f || it > 3900) break; }
    int bi = (int)(x / mb), bj = (int)(y / mb);
    if (bi >= 0 && bj >= 0 && bi < mn && bj < mn && z > mip[bj * mn + bi] + 0.01f) {
      // skip to the block edge
      float bx0 = bi * mb, by0 = bj * mb;
      float tx = dx > 0 ? (bx0 + mb - x) / dx : (dx < 0 ? (bx0 - x) / dx : 1e9f);
      float ty = dy > 0 ? (by0 + mb - y) / dy : (dy < 0 ? (by0 - y) / dy : 1e9f);
      float adv = fminf(tx, ty); prev = tt; tt += fmaxf(adv, 0.f) + 0.05f; continue;
    }
    float gh = hat(g, n, x, y);
    if (gh > -1e29f && z <= gh) {
      float a = prev, b = tt;
      for (int k = 0; k < 12; k++) { float m = 0.5f * (a + b);
        float zm = ez + dz * m, hm = hat(g, n, ex + dx * m, ey + dy * m);
        if (zm <= hm) b = m; else a = m; }
      hit = b; break;
    }
    if (gh < -1e29f && dz >= 0.f) break;
    prev = tt; tt += step * (1.f + tt / 400.f);
  }
  size_t o = (size_t)py * w + px;
  float r, gg, bb;
  if (hit < 0.f) {
    float s = fmaxf(dz, 0.f); r = 0.55f - 0.3f * s; gg = 0.68f - 0.25f * s; bb = 0.88f - 0.1f * s; depth[o] = 1e30f;
  } else {
    float x = ex + dx * hit, y = ey + dy * hit;
    float hx = hat(g, n, x + 1.f, y) - hat(g, n, x - 1.f, y), hy = hat(g, n, x, y + 1.f) - hat(g, n, x, y - 1.f);
    float nx = -hx * 0.5f, ny = -hy * 0.5f, nz = 1.f, nl = rsqrtf(nx*nx + ny*ny + nz*nz);
    float lam = fmaxf(0.f, (nx * sunx + ny * suny + nz * sunz) * nl);
    float slope = 1.f - nz * nl;
    float base_r = 0.30f + 0.25f * slope, base_g = 0.42f - 0.1f * slope, base_b = 0.22f;
    float c = 0.25f + 0.75f * lam;
    // contour every 5 m
    float z = ez + dz * hit; float cz = fabsf(z / 5.f - rintf(z / 5.f));
    if (cz < 0.02f && hit < 800.f) c *= 0.7f;
    float fog = 1.f - expf(-hit / 3500.f);
    r = base_r * c * (1 - fog) + 0.62f * fog; gg = base_g * c * (1 - fog) + 0.72f * fog; bb = base_b * c * (1 - fog) + 0.86f * fog;
    depth[o] = hit;
  }
  rgba[o*4+0] = (unsigned char)(fminf(r, 1.f) * 255); rgba[o*4+1] = (unsigned char)(fminf(gg, 1.f) * 255);
  rgba[o*4+2] = (unsigned char)(fminf(bb, 1.f) * 255); rgba[o*4+3] = 255;
}
// Line primitives: each segment sampled at `per` points, projected, depth tested (1 % slack), painted.
__global__ void lines(const float* seg, const unsigned char* col, int ns, int per,
                      float ex, float ey, float ez, float yaw, float pitch, float fov, int w, int h,
                      const float* depth, unsigned char* rgba) {
  int id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= ns * per) return;
  int s = id / per; float u = (id % per + 0.5f) / per;
  const float* q = seg + s * 6;
  float x = q[0] + (q[3] - q[0]) * u - ex, y = q[1] + (q[4] - q[1]) * u - ey, z = q[2] + (q[5] - q[2]) * u - ez;
  float cy = cosf(yaw), syw = sinf(yaw), cp_ = cosf(pitch), sp = sinf(pitch);
  float fx = syw * cp_, fy = cy * cp_, fz = sp, rx = cy, ry = -syw;
  float ux = ry * fz, uy = -rx * fz, uz = rx * fy - ry * fx;
  float zf = x * fx + y * fy + z * fz; if (zf < 0.5f) return;
  float xr = x * rx + y * ry, yu = x * ux + y * uy + z * uz;
  float th = tanf(fov * 0.5f), asp = (float)w / h;
  int px = (int)((xr / zf / (th * asp) + 1.f) * 0.5f * w), py = (int)((1.f - yu / zf / th) * 0.5f * h);
  if (px < 0 || py < 0 || px >= w || py >= h) return;
  float dist = sqrtf(x*x + y*y + z*z); size_t o = (size_t)py * w + px;
  if (dist > depth[o] * 1.01f + 0.3f) return;
  rgba[o*4+0] = col[s*3+0]; rgba[o*4+1] = col[s*3+1]; rgba[o*4+2] = col[s*3+2];
}
// Sun shadow pair per cell: electron = fixed 1 m march; positron = max-mip skip then 1 m march.
__global__ void shadow_pair(const float* g, const float* mip, int n, int mn, int mb, int stride,
                            float sx, float sy, float sz, unsigned char* e_out, unsigned char* p_out, int cells) {
  int id = blockIdx.x * blockDim.x + threadIdx.x; int m = (n - 1) / stride; if (id >= m * m) return;
  float x0 = (id % m) * stride + 0.5f, y0 = (id / m) * stride + 0.5f;
  float z0 = hat(g, n, x0, y0) + 0.05f;
  float hz = sqrtf(sx*sx + sy*sy); float kx = sx / hz, ky = sy / hz, kz = sz / hz;  // per metre horizontal
  unsigned char e = 0;
  for (float d = 1.f; d < 3000.f; d += 1.f) { float x = x0 + kx * d, y = y0 + ky * d;
    float gh = hat(g, n, x, y); if (gh < -1e29f) break; if (gh > z0 + kz * d) { e = 1; break; } }
  unsigned char p = 0; float d = 1.f;
  for (int it = 0; it < 6000 && d < 3000.f; it++) { float x = x0 + kx * d, y = y0 + ky * d;
    if (x < 0.f || y < 0.f || x > n - 1.001f || y > n - 1.001f) break;
    int bi = (int)(x / mb), bj = (int)(y / mb); float zr = z0 + kz * d;
    if (zr > mip[bj * mn + bi] + 0.01f) {
      float bx0 = bi * mb, by0 = bj * mb;
      float tx = kx > 0 ? (bx0 + mb - x) / kx : (kx < 0 ? (bx0 - x) / kx : 1e9f);
      float ty = ky > 0 ? (by0 + mb - y) / ky : (ky < 0 ? (by0 - y) / ky : 1e9f);
      d += fmaxf(ceilf(fminf(tx, ty)), 1.f); continue; }   // stay on the integer-metre lattice
    if (hat(g, n, x, y) > zr) { p = 1; break; }
    d += 1.f; }
  e_out[id] = e; p_out[id] = p;
}
// Bilinear sample pair at points: electron = lerp form (float32), positron = expanded polynomial (float64).
__global__ void sample_pair(const float* g, int n, const double* pts, int np, double* e_out, double* p_out) {
  int id = blockIdx.x * blockDim.x + threadIdx.x; if (id >= np) return;
  double x = pts[2*id], y = pts[2*id+1];
  e_out[id] = (double)hat(g, n, (float)x, (float)y);
  int i = (int)floor(x), j = (int)floor(y);
  if (i < 0 || j < 0 || i >= n - 1 || j >= n - 1) { p_out[id] = nan(""); return; }
  double fx = x - i, fy = y - j; const float* r0 = g + (size_t)j * n; const float* r1 = r0 + n;
  double a = r0[i], b = r0[i+1], c = r1[i], d = r1[i+1];
  p_out[id] = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}
}
"""
MOD = cp.RawModule(code=KSRC, options=("--use_fast_math",))
K_RENDER, K_LINES = MOD.get_function("render"), MOD.get_function("lines")
K_SHADOW, K_SAMPLE = MOD.get_function("shadow_pair"), MOD.get_function("sample_pair")
MB = 8  # mip block (m)
CARD = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()


# ------------------------------------------------------------------ the resident world
class World:
    def __init__(self):
        self.sites = {}
        idx = json.load(open(f"{LIDAR}/sites-index.json", encoding="utf-8"))
        meta = {s["name"]: s for s in idx.get("sites", [])}
        for d in sorted(glob.glob(f"{LIDAR}/open-land-*")):
            name = os.path.basename(d)
            if not os.path.exists(f"{d}/source.npy") or not name.split("-")[-1].isdigit():
                continue
            src = json.load(open(f"{d}/source.json", encoding="utf-8"))
            a = np.load(f"{d}/source.npy", mmap_mode="r")
            g = cp.asarray(np.ascontiguousarray(a, dtype=np.float32))
            n = g.shape[0]
            mn = (n - 1) // MB
            P = cp.pad(g, ((0, (mn + 1) * MB - n), (0, (mn + 1) * MB - n)), mode="edge")
            A = P.reshape(mn + 1, MB, mn + 1, MB).max(axis=(1, 3))
            m = cp.maximum(cp.maximum(A[:mn, :mn], A[1:, :mn]), cp.maximum(A[:mn, 1:], A[1:, 1:]))
            m = cp.ascontiguousarray(m, dtype=cp.float32)   # conservative: covers every node a block touches
            far = self._far(name)
            self.sites[name] = dict(g=g, mip=m, n=n, mn=mn, e0=float(src["origin_e_m"]), n0=float(src["origin_n_m"]),
                                    far=far, min=float(g.min()), max=float(g.max()), meta=meta.get(name, {}))
        self.plant = {}          # site -> resident plant (tables, lines, electrical) from the world's JS
        self.lines_dev = None
        self.cables = None
        log(f"world resident: {len(self.sites)} sites")

    def _far(self, name):
        out = {}
        tj = f"{FARRING}/{name}/tiles.json"
        if not os.path.exists(tj):
            return out
        meta = json.load(open(tj, encoding="utf-8"))
        for ring in meta["rings"]:
            tiles = [t for t in meta["tiles"] if t["ring"] == ring["ring"]]
            if not tiles:
                continue
            sp, tm = ring["spacing_m"], ring["tile_m"]
            e0, n0 = min(t["e0"] for t in tiles), min(t["n0"] for t in tiles)
            nx = int(round((max(t["e0"] for t in tiles) - e0) / tm)) + 1
            ny = int(round((max(t["n0"] for t in tiles) - n0) / tm)) + 1
            s = tm // sp
            arr = np.full((ny * s + 1, nx * s + 1), np.nan, np.float32)
            for t in tiles:
                b = open(f"{FARRING}/{name}/{t['file']}", "rb").read()
                hdr = struct.unpack("<4sHHHHiiiHHI", b[:32])
                q = np.frombuffer(b, "<u2", offset=32).reshape(hdr[2], hdr[2]).astype(np.float32)
                hgt = np.where(q == 0xFFFF, np.nan, (hdr[7] + q) / 100.0)
                i, j = int(round((t["e0"] - e0) / tm)) * s, int(round((t["n0"] - n0) / tm)) * s
                arr[j:j + hdr[2], i:i + hdr[2]] = hgt
            out[f"r{ring['ring']}"] = dict(g=cp.asarray(arr), e0=e0, n0=n0, spacing=sp)
        return out


# ------------------------------------------------------------------ helpers
def pair_block(electron, positron, tol, unit, what):
    e, p = np.asarray(electron, np.float64), np.asarray(positron, np.float64)
    d = np.abs(e - p)
    ph = int(np.count_nonzero(~(d <= tol)))
    return {"what": what, "unit": unit, "tolerance": tol, "photons": ph,
            "max_abs_diff": float(np.nanmax(d)) if d.size else 0.0}


def bilin_cpu(a, x, y):
    i, j = int(math.floor(x)), int(math.floor(y)); fx, fy = x - i, y - j
    a0 = float(np.float32(a[j, i])); b0 = float(np.float32(a[j, i + 1])); c0 = float(np.float32(a[j + 1, i])); d0 = float(np.float32(a[j + 1, i + 1]))
    return a0 + (b0 - a0) * fx + (c0 - a0) * fy + (a0 - b0 - c0 + d0) * fx * fy


def sample_pair(S, en):
    """en: (k,2) national-grid metres -> electron f32 lerp, positron f64 polynomial."""
    pts = cp.asarray(np.column_stack([en[:, 0] - S["e0"], en[:, 1] - S["n0"]]).astype(np.float64).ravel())
    k = len(en)
    e, p = cp.empty(k, cp.float64), cp.empty(k, cp.float64)
    if k:
        K_SAMPLE(((k + 255) // 256,), (256,), (S["g"], np.int32(S["n"]), pts, np.int32(k), e, p))
    return cp.asnumpy(e), cp.asnumpy(p)


def site_of(F, c):
    name = c.get("site") or F.cam["site"]
    if name not in F.world.sites:
        raise ValueError(f"unknown site {name}; resident: {sorted(F.world.sites)}")
    return name, F.world.sites[name]


def node(req):
    rq = f"{WORK}/req-{os.getpid()}-{time.time_ns()}.json"
    rs = rq.replace("req-", "res-")
    write_json(rq, req)
    t0 = time.perf_counter()
    r = subprocess.run(["node", NODE_EXPORT, rq, rs], capture_output=True, text=True, timeout=300)
    if r.returncode:
        raise RuntimeError("world JS failed: " + (r.stderr or r.stdout)[-800:])
    out = json.load(open(rs, encoding="utf-8"))
    out["node_ms"] = (time.perf_counter() - t0) * 1000
    for f in (rq, rs):
        try: os.remove(f)
        except OSError: pass
    return out


def png_bytes(rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    ch = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ch(b"IDAT", zlib.compress(raw, 6)) + ch(b"IEND", b"")


# ------------------------------------------------------------------ the furnace
class Furnace:
    def __init__(self):
        self.t_start = time.time()
        self.world = World()
        first = sorted(self.world.sites)[0]
        S = self.world.sites[first]
        self.cam = dict(site=first, x=S["n"] * 0.5, y=S["n"] * 0.15, agl=120.0, yaw=0.0, pitch=-0.25, fov=math.radians(70))
        self.fly = None
        self.orbit = True
        self.sun = (math.cos(math.radians(30)) * math.sin(math.radians(200)), math.cos(math.radians(30)) * math.cos(math.radians(200)), math.sin(math.radians(30)))
        self.depth = cp.empty((H, W), cp.float32)
        self.ring = cp.zeros((RING, H, W, 4), cp.uint8)
        self.frame = 0
        self.fps = 0.0
        self.render_ms = 0.0
        self.temp_c = None
        self.gpu_mem_mib = None
        self.gpu_util = None
        self.paused = False
        self.queue_len = 0
        self.answered = 0
        self.last = None
        self.lock = threading.Lock()
        threading.Thread(target=self._smi, daemon=True).start()

    def _smi(self):
        while True:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                t, m, u = [float(v) for v in r.stdout.strip().split(",")]
                self.temp_c, self.gpu_mem_mib, self.gpu_util = t, m, u
            except Exception:
                pass
            time.sleep(3)

    # ---- camera
    def eye(self):
        c, S = self.cam, self.world.sites[self.cam["site"]]
        e, p = sample_pair(S, np.array([[S["e0"] + c["x"], S["n0"] + c["y"]]]))
        gz = e[0] if np.isfinite(e[0]) and e[0] > -1e29 else S["min"]
        return c["x"], c["y"], gz + c["agl"], gz

    def step_camera(self, dt):
        c = self.cam
        if self.fly:
            f = self.fly
            u = min(1.0, (time.time() - f["t0"]) / f["seconds"])
            s = u * u * (3 - 2 * u)
            for k in ("x", "y", "agl", "yaw", "pitch"):
                c[k] = f["from"][k] + (f["to"][k] - f["from"][k]) * s
            if u >= 1:
                self.fly = None
        elif self.orbit:
            S = self.world.sites[c["site"]]
            c["yaw"] = (c["yaw"] + 0.15 * dt) % (2 * math.pi)
            r = S["n"] * 0.35
            c["x"] = S["n"] / 2 - math.sin(c["yaw"]) * r
            c["y"] = S["n"] / 2 - math.cos(c["yaw"]) * r

    def render(self):
        c, S = self.cam, self.world.sites[self.cam["site"]]
        ex, ey, ez, _ = self.eye()
        slot = self.ring[self.frame % RING]
        blk = (16, 16); grd = ((W + 15) // 16, (H + 15) // 16)
        K_RENDER(grd, blk, (S["g"], S["mip"], np.int32(S["n"]), np.int32(S["mn"]), np.int32(MB),
                            np.float32(ex), np.float32(ey), np.float32(ez), np.float32(c["yaw"]), np.float32(c["pitch"]),
                            np.float32(c["fov"]), np.int32(W), np.int32(H), self.depth, slot,
                            np.float32(self.sun[0]), np.float32(self.sun[1]), np.float32(self.sun[2]), np.float32(time.time() - self.t_start)))
        P = self.world.plant.get(c["site"])
        if P is not None and P["segs"] is not None:
            ns, per = P["nseg"], 64
            K_LINES(((ns * per + 255) // 256,), (256,), (P["segs"], P["cols"], np.int32(ns), np.int32(per),
                    np.float32(ex), np.float32(ey), np.float32(ez), np.float32(c["yaw"]), np.float32(c["pitch"]),
                    np.float32(c["fov"]), np.int32(W), np.int32(H), self.depth, slot))
        return slot

    def preview(self, slot):
        img = cp.asnumpy(slot[::2, ::2, :3])
        data = png_bytes(np.ascontiguousarray(img))
        with open(f"{FRAMES}/latest.png.tmp", "wb") as f:
            f.write(data)
        os.replace(f"{FRAMES}/latest.png.tmp", f"{FRAMES}/latest.png")
        keep = f"{FRAMES}/frame-{self.frame:08d}.png"
        shutil.copyfile(f"{FRAMES}/latest.png", keep)
        olds = sorted(glob.glob(f"{FRAMES}/frame-*.png"))
        for o in olds[:-30]:
            try: os.remove(o)
            except OSError: pass

    def status(self):
        pool = cp.get_default_memory_pool()
        write_json(f"{OUT}/STATUS.json", {
            "alive": True, "at": now_iso(), "pid": os.getpid(), "uptime_s": round(time.time() - self.t_start, 1),
            "frame": self.frame, "fps": round(self.fps, 1), "render_ms": round(self.render_ms, 3), "resolution": [W, H],
            "fps_cap": FPS_CAP, "paused_hot": self.paused, "temperature_c": self.temp_c, "gpu_util_pct": self.gpu_util,
            "furnace_vram_mib": round(pool.total_bytes() / 2**20, 1), "card_vram_used_mib": self.gpu_mem_mib,
            "queue_len": self.queue_len, "answered": self.answered, "last_command": self.last,
            "camera": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.cam.items()},
            "orbit": self.orbit, "sites_resident": sorted(self.world.sites),
            "plants_resident": {k: v["summary"] for k, v in self.world.plant.items()},
            "preview": f"{FRAMES}/latest.png"})

    # ---- commands
    def poll(self):
        files = sorted(glob.glob(f"{IN}/*.cmd"))
        cmds = []
        for f in files:
            try:
                txt = open(f, encoding="utf-8").read()
                shutil.move(f, f"{DONE}/{os.path.basename(f)}")
            except OSError:
                continue
            for ln in txt.splitlines():
                if ln.strip():
                    cmds.append(ln)
        self.queue_len = len(cmds)
        for i, ln in enumerate(cmds):
            self.queue_len = len(cmds) - i
            self.answer(ln)
        self.queue_len = 0

    def answer(self, ln):
        t0 = time.perf_counter()
        try:
            c = json.loads(ln)
        except Exception as e:
            c = {"id": f"bad-{time.time_ns()}", "op": "bad"}
            body = {"ok": False, "error": f"not JSON: {e}"}
        else:
            c.setdefault("id", f"cmd-{time.time_ns()}")
            op = c.get("op", "")
            fn = getattr(self, "op_" + op.replace("-", "_"), None)
            try:
                if fn is None:
                    raise ValueError(f"unknown op {op!r}; ops: {sorted(k[3:] for k in dir(self) if k.startswith('op_'))}")
                body = {"ok": True, **fn(c)}
            except Exception as e:
                body = {"ok": False, "error": str(e), "trace": traceback.format_exc(limit=3)}
        cp.cuda.Stream.null.synchronize()
        body = {"id": c.get("id"), "op": c.get("op"), "command": c, **body,
                "timing_ms": round((time.perf_counter() - t0) * 1000, 3), "at": now_iso(),
                "card": CARD + " via CuPy " + cp.__version__}
        canon = json.dumps(body, sort_keys=True, default=float).encode()
        body["sha256"] = hashlib.sha256(canon).hexdigest()
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(c.get("id")))[:80]
        write_json(f"{ANS}/{safe}.json", body)
        self.answered += 1
        self.last = {"id": c.get("id"), "op": c.get("op"), "ok": body.get("ok")}
        log(f"answered {c.get('id')} {c.get('op')} ok={body.get('ok')} {body['timing_ms']} ms")

    def op_ping(self, c):
        name, S = site_of(self, c)
        g = S["g"]
        e = float(cp.sum(g, dtype=cp.float64)) / g.size                                   # electron: f64 reduce
        p = float(cp.sum(cp.sum(g, axis=1, dtype=cp.float32).astype(cp.float64))) / g.size  # positron: f32 rows then f64
        a = np.load(f"{LIDAR}/{name}/source.npy", mmap_mode="r")
        wit = float(np.asarray(a[::64, ::64], np.float32).astype(np.float64).mean())
        return {"pong": True, "site": name, "sites_resident": len(self.world.sites), "frame": self.frame, "fps": round(self.fps, 1),
                "mean_height_m": {"electron": e, "positron": p},
                "pair": pair_block([e], [p], 1e-3, "m", "site mean height, f64 reduce vs f32-rows reduce"),
                "witness": {"cpu_mean_every_64th_node_m": wit, "note": "a 1/4096 sample: agrees to about 0.5 m, not a bit test"}}

    def op_camera(self, c):
        name, S = site_of(self, c)
        cam = dict(self.cam); cam["site"] = name
        if "e" in c: cam["x"] = float(c["e"]) - S["e0"]
        if "n" in c: cam["y"] = float(c["n"]) - S["n0"]
        if "x" in c: cam["x"] = float(c["x"])
        if "y" in c: cam["y"] = float(c["y"])
        for k in ("agl", "yaw", "pitch"):
            if k in c: cam[k] = float(c[k]) if k == "agl" else math.radians(float(c[k]))
        self.orbit = bool(c.get("orbit", False))
        secs = float(c.get("seconds", 0))
        if secs > 0 and name == self.cam["site"]:
            self.fly = {"from": dict(self.cam), "to": cam, "t0": time.time(), "seconds": secs}
        else:
            self.cam = cam
        en = np.array([[S["e0"] + cam["x"], S["n0"] + cam["y"]]])
        e, p = sample_pair(S, en)
        a = np.load(f"{LIDAR}/{name}/source.npy", mmap_mode="r")
        w = bilin_cpu(a, cam["x"], cam["y"])
        return {"site": name, "camera": {"e_m": en[0, 0], "n_m": en[0, 1], "agl_m": cam["agl"], "yaw_deg": math.degrees(cam["yaw"]),
                "pitch_deg": math.degrees(cam["pitch"]), "fly_seconds": secs, "orbit": self.orbit},
                "ground_under_eye_m": {"electron": e[0], "positron": p[0]},
                "pair": pair_block(e, p, 0.005, "m", "bilinear ground: f32 lerp vs f64 polynomial"),
                "witness": {"cpu_f64_polynomial_m": w, "diff_vs_positron_m": abs(w - p[0])}}

    op_fly = op_camera

    def op_orbit(self, c):
        self.orbit = bool(c.get("on", True)); return {"orbit": self.orbit}

    def op_layout(self, c):
        name, S = site_of(self, c)
        n = S["n"]
        if "boundary" in c:
            bnd = [[float(e), float(nn)] for e, nn in c["boundary"]]
        else:
            half = float(c.get("half_m", 400))
            ce, cn = S["e0"] + n / 2, S["n0"] + n / 2
            bnd = [[ce - half, cn - half], [ce + half, cn - half], [ce + half, cn + half], [ce - half, cn + half]]
        es, ns_ = [b[0] for b in bnd], [b[1] for b in bnd]
        i0 = max(0, int(min(es) - S["e0"]) - 40); i1 = min(n, int(max(es) - S["e0"]) + 41)
        j0 = max(0, int(min(ns_) - S["n0"]) - 40); j1 = min(n, int(max(ns_) - S["n0"]) + 41)
        win = cp.asnumpy(S["g"][j0:j1, i0:i1]).astype("<f4")
        gf = f"{WORK}/ground-{name}.f32"; win.tofile(gf)
        req = {"mode": "layout", "boundary": bnd, "targetMW": float(c.get("mw", 50)), "layout": c.get("layout", "south"),
               "options": c.get("options", {}), "grid": c.get("grid"),
               "ground": {"file": gf, "e0": S["e0"] + i0, "n0": S["n0"] + j0, "w": win.shape[1], "h": win.shape[0]}}
        r = node(req)
        L = r["layout"]
        T = np.array(L["tables"], np.float64).reshape(-1, 12) if L["tables"] else np.zeros((0, 12))
        corners = T[:, :8].reshape(-1, 2)
        e, p = sample_pair(S, corners)
        js = T[:, 8:12].ravel()
        # resident plant in VRAM: table top outlines (corners at ground + low edge / + rise) and feeder lines
        low = float(L["table"].get("rise", 1.0))
        segs, cols = [], []
        for t in range(len(T)):
            c4 = T[t, :8].reshape(4, 2); h4 = T[t, 8:12] + 0.8 + np.array([0, 0, low, low]) * (1 if L["layout"] == "south" else 0)
            for k in range(4):
                a_, b_ = k, (k + 1) % 4
                segs.append([c4[a_, 0] - S["e0"], c4[a_, 1] - S["n0"], h4[a_], c4[b_, 0] - S["e0"], c4[b_, 1] - S["n0"], h4[b_]]); cols.append([40, 60, 120])
        comp = L["compound"]["en"]
        feeder_paths = []
        st_at = {s["id"]: s["en"] for s in L["stations"]}
        for f in L["feeders"]:
            pts = [comp] + [st_at[h["to"]] for h in f["hops"] if h["to"] in st_at]
            feeder_paths.append(pts)
            for a_, b_ in zip(pts, pts[1:]):
                segs.append([a_[0] - S["e0"], a_[1] - S["n0"], 0, b_[0] - S["e0"], b_[1] - S["n0"], 0]); cols.append([230, 120, 20])
        segs = np.array(segs, np.float32).reshape(-1, 6)
        # feeder lines drape 0.5 m above ground (positron-sampled)
        fm = segs[:, 2] == 0
        if fm.any():
            z0, _ = sample_pair(S, np.column_stack([segs[fm, 0] + S["e0"], segs[fm, 1] + S["n0"]]))
            z1, _ = sample_pair(S, np.column_stack([segs[fm, 3] + S["e0"], segs[fm, 4] + S["n0"]]))
            segs[fm, 2] = z0 + 0.5; segs[fm, 5] = z1 + 0.5
        # retire any older plant on this site before uploading the new one
        self.world.plant.pop(name, None)
        self.world.plant[name] = {"segs": cp.asarray(segs) if len(segs) else None, "cols": cp.asarray(np.array(cols, np.uint8)),
                                  "nseg": len(segs), "tables": cp.asarray(T), "layout": L, "feeder_paths": feeder_paths,
                                  "summary": {"tables": len(T), "mw": L["built"]["mw"], "mwp": L["built"]["mwp"], "segments": len(segs)}}
        kwp_js = L["built"]["mwp"] * 1000
        kwp_gpu = float(cp.asarray(np.full(len(T), L["table"]["kWp"])).sum())
        ks = min(64, len(corners))
        a = np.load(f"{LIDAR}/{name}/source.npy", mmap_mode="r")
        wit = [bilin_cpu(a, corners[q, 0] - S["e0"], corners[q, 1] - S["n0"]) for q in range(0, len(corners), max(1, len(corners) // ks))][:ks]
        wref = p[::max(1, len(corners) // ks)][:ks]
        return {"site": name, "label": L["label"], "layout": L["layout"], "asked": L["asked"], "built": L["built"], "fits": L["fits"],
                "table": L["table"], "stations": len(L["stations"]), "feeders": len(L["feeders"]), "worst_mv_drop_pct": L["worstDropPct"],
                "cable_m": L["cableM"], "template": L["template"], "notes": L["notes"], "boundary": bnd,
                "table_corner_ground_m": {"electron": "world JS (v08 plant-layout.mjs) bilinear, f64", "positron": "CuPy VRAM bilinear f32 lerp",
                                          "n": int(len(js)), "mean_electron": float(np.mean(js)) if len(js) else None,
                                          "mean_positron": float(np.mean(e)) if len(e) else None},
                "pair": pair_block(js, e, 0.01, "m", "ground at every table corner: JS vs VRAM"),
                "pair_mwp": pair_block([kwp_js], [kwp_gpu], 1.0, "kWp", "DC capacity: JS built.mwp vs VRAM sum of table kWp"),
                "witness": {"cpu_points": len(wit), "max_abs_diff_vs_positron_f64_m": float(np.max(np.abs(np.array(wit) - wref))) if wit else None},
                "node_ms": r["node_ms"], "resident_segments": len(segs)}

    def op_piles(self, c):
        name, S = site_of(self, c)
        P = self.world.plant.get(name)
        if P is None:
            raise ValueError("no resident plant on this site: send a layout command first")
        T = cp.asnumpy(P["tables"]); spacing = float(c.get("spacing_m", 6)); reveal = float(c.get("reveal_m", 1.2))
        a0, a1 = T[:, 0:2], T[:, 2:4]               # low-edge line of each table
        L = np.hypot(*(a1 - a0).T)
        k = np.maximum(2, np.floor(L / spacing).astype(int) + 1)
        K = int(k.max()); s = np.linspace(0, 1, K)
        pts = (a0[:, None, :] + (a1 - a0)[:, None, :] * s[None, :, None]).reshape(-1, 2)
        e, p = sample_pair(S, pts)
        G = cp.asarray(e.reshape(-1, K))
        # top line per table: least squares through ground + reveal. electron closed form, positron lstsq
        x = cp.asarray(s)[None, :] * cp.asarray(L)[:, None]
        xm, gm = x.mean(1, keepdims=True), G.mean(1, keepdims=True)
        b = ((x - xm) * (G - gm)).sum(1) / ((x - xm) ** 2).sum(1)
        top_e = gm + reveal + b[:, None] * (x - xm)
        A = cp.stack([cp.ones_like(x), x], axis=2)                   # (t, K, 2)
        AtA = cp.einsum("tki,tkj->tij", A, A); Atg = cp.einsum("tki,tk->ti", A, G + reveal)
        coef = cp.linalg.solve(AtA, Atg[..., None])[..., 0]
        top_p = coef[:, 0:1] + coef[:, 1:2] * x
        rev = cp.asnumpy(top_e - G).ravel()
        q = np.percentile(rev, [0, 5, 50, 95, 100])
        return {"site": name, "tables": int(len(T)), "piles_per_table_max": K, "piles": int(len(rev)), "spacing_m": spacing,
                "target_reveal_m": reveal, "reveal_m": {"min": q[0], "p5": q[1], "p50": q[2], "p95": q[3], "max": q[4]},
                "slope_along_table_pct": {"max_abs": float(cp.abs(b).max() * 100)},
                "pair": pair_block(cp.asnumpy(top_e).ravel(), cp.asnumpy(top_p).ravel(), 1e-4, "m", "pile top line: closed-form LSQ vs batched normal-equation solve"),
                "pair_ground": pair_block(e, p, 0.005, "m", "ground at piles: f32 lerp vs f64 polynomial"),
                "witness": {"cpu_first_table_slope": float(np.polyfit(cp.asnumpy(x[0]), cp.asnumpy(G[0]), 1)[0]), "gpu_first_table_slope": float(b[0])}}

    def _cables(self, kv=33):
        if self.world.cables is None or self.world.cables[0] != kv:
            self.world.cables = (kv, node({"mode": "cables", "kv": kv})["cables"])
        return {c["area"]: c for c in self.world.cables[1]}

    def op_cable(self, c):
        kv, mva, Lm, pf = float(c.get("kv", 33)), float(c["mva"]), float(c.get("length_m", 1000)), float(c.get("pf", 0.95))
        r = node({"mode": "cables", "kv": kv, "mva": mva, "lengthM": Lm, "pf": pf})
        tab = {q["area"]: q for q in r["cables"]}
        sz = r.get("sizeFor")
        if not sz:
            return {"fits": False, "note": "no single cable carries this load: split the feeder", "cables": r["cables"]}
        z = tab[sz["area"]]
        # positron: dV% = 100 S L (r pf + x sin) / V^2 on the card
        S_, V = cp.float64(mva * 1e6), cp.float64(kv * 1e3)
        dpos = float(100 * S_ * Lm * (z["r_ohm_m"] * pf + z["x_ohm_m"] * math.sqrt(1 - pf * pf)) / V ** 2)
        Iload = mva * 1e6 / (math.sqrt(3) * kv * 1e3)
        return {"kv": kv, "mva": mva, "length_m": Lm, "pf": pf, "area_mm2": sz["area"], "rating_a": sz["amps"], "load_a": Iload,
                "utilisation": sz["utilisation"], "r_ohm_per_km": z["r_ohm_m"] * 1e3, "x_ohm_per_km": z["x_ohm_m"] * 1e3,
                "drop_pct": {"electron": r["dropPct"], "positron": dpos},
                "pair": pair_block([r["dropPct"]], [dpos], 1e-9, "%", "volt drop: world JS dropPercent vs CuPy S*L*(r cos+x sin)/V^2"),
                "witness": {"cpu_load_a": Iload, "js_load_a": sz["loadAmps"]}}

    def op_loadflow(self, c):
        name = c.get("site") or self.cam["site"]
        kv, pf = float(c.get("kv", 33)), float(c.get("pf", 0.95))
        tab = self._cables(kv)
        if "feeders" in c:
            feeders = c["feeders"]
        else:
            P = self.world.plant.get(name)
            if P is None:
                raise ValueError("give feeders:[[{length_m,area_mm2,mva}...]] or run a layout first")
            feeders = [[{"length_m": h["lengthM"], "area_mm2": h["areaMm2"], "mva": h["loadMVA"]} for h in f["hops"]] for f in P["layout"]["feeders"]]
        nf, nh = len(feeders), max(len(f) for f in feeders)
        Z = np.zeros((nf, nh), np.complex128); Sld = np.zeros((nf, nh), np.complex128); hop_mva = np.zeros((nf, nh))
        sinp = math.sqrt(1 - pf * pf)
        for i, f in enumerate(feeders):
            for j, h in enumerate(f):
                q = tab[int(h.get("area_mm2", h.get("areaMm2")))]
                Z[i, j] = (q["r_ohm_m"] + 1j * q["x_ohm_m"]) * float(h.get("length_m", h.get("lengthM")))
                hop_mva[i, j] = float(h.get("mva", h.get("loadMVA")))
        # hop j carries hop_mva[j]; the station at the end of hop j injects hop_mva[j] - hop_mva[j+1]
        inj = hop_mva - np.concatenate([hop_mva[:, 1:], np.zeros((nf, 1))], axis=1)
        Sld = inj * (pf + 1j * sinp) * 1e6 / 3                                   # per phase, VA (generation)
        V0 = kv * 1e3 / math.sqrt(3)
        Zg, Sg = cp.asarray(Z), cp.asarray(Sld)
        # electron: backward/forward sweep (complex, exact AC)
        V = cp.full((nf, nh), V0, cp.complex128)
        for it in range(50):
            I = cp.conj(Sg / V)                                                  # injected currents
            Ib = cp.flip(cp.cumsum(cp.flip(I, 1), 1), 1)                          # branch currents
            Vn = V0 + cp.cumsum(Zg * Ib, 1)                                       # generation raises voltage
            if float(cp.abs(Vn - V).max()) < 1e-9: V = Vn; break
            V = Vn
        rise_e = cp.asnumpy((cp.abs(V[:, -1]) - V0) / V0 * 100) if nh else np.zeros(nf)
        endv = [(abs(complex(cp.asnumpy(V)[i, len(f) - 1])) - V0) / V0 * 100 for i, f in enumerate(feeders)]
        # positron: linear DistFlow sum of hop drops (the world's own rule)
        lin = [sum(float(h.get("mva", h.get("loadMVA"))) * 1e6 * (tab[int(h.get("area_mm2", h.get("areaMm2")))]["r_ohm_m"] * pf + tab[int(h.get("area_mm2", h.get("areaMm2")))]["x_ohm_m"] * sinp)
                   * float(h.get("length_m", h.get("lengthM"))) / (kv * 1e3) ** 2 * 100 for h in f) for f in feeders]
        loss_kw = float(cp.sum(cp.abs(cp.flip(cp.cumsum(cp.flip(cp.conj(Sg / V), 1), 1), 1)) ** 2 * Zg.real) * 3 / 1e3)
        # witness: feeder 0 on the CPU, same sweep in numpy
        z0, s0 = Z[0, :len(feeders[0])], Sld[0, :len(feeders[0])]; v = np.full(len(z0), V0, complex)
        for _ in range(50):
            ib = np.cumsum(np.conj(s0 / v)[::-1])[::-1]; v = V0 + np.cumsum(z0 * ib)
        wit = (abs(v[-1]) - V0) / V0 * 100
        return {"kv": kv, "pf": pf, "feeders": nf, "direction": "export (generation raises voltage toward the far station)",
                "end_voltage_rise_pct": {"electron_ac_sweep": endv, "positron_linear_distflow": lin},
                "losses_kw": loss_kw, "sweep_iterations": it + 1,
                "pair": pair_block(endv, lin, 0.02, "%", "far-end voltage change: exact AC sweep vs linear drop sum"),
                "witness": {"cpu_feeder0_rise_pct": wit, "gpu_feeder0_rise_pct": endv[0]}}

    def op_earthworks(self, c):
        name, S = site_of(self, c)
        e0 = int(round(float(c["e0"]) - S["e0"])); e1 = int(round(float(c["e1"]) - S["e0"]))
        n0 = int(round(float(c["n0"]) - S["n0"])); n1 = int(round(float(c["n1"]) - S["n0"]))
        g = S["g"][n0:n1 + 1, e0:e1 + 1].astype(cp.float64)
        cell = 0.25 * (g[:-1, :-1] + g[1:, :-1] + g[:-1, 1:] + g[1:, 1:])       # electron: cell means
        wt = cp.ones_like(g); wt[0, :] *= 0.5; wt[-1, :] *= 0.5; wt[:, 0] *= 0.5; wt[:, -1] *= 0.5   # positron: trapezoid node weights
        area = float(cell.size)
        bal_p = float((g * wt).sum() / wt.sum())
        lo, hi = float(cell.min()), float(cell.max())
        for _ in range(60):                                                      # electron: bisect cut = fill
            m = 0.5 * (lo + hi); d = float((cell - m).sum())
            lo, hi = (m, hi) if d > 0 else (lo, m)
        bal_e = 0.5 * (lo + hi)
        level = float(c["level"]) if "level" in c else bal_e
        d = cell - level
        cut_e, fill_e = float(cp.clip(d, 0, None).sum()), float(cp.clip(-d, 0, None).sum())
        net_p = float(((g - level) * wt).sum())
        return {"site": name, "pad_m": [e1 - e0, n1 - n0], "area_m2": area, "level_m": level,
                "cut_m3": cut_e, "fill_m3": fill_e, "net_m3": {"electron": cut_e - fill_e, "positron": net_p},
                "balanced_level_m": {"electron_bisect": bal_e, "positron_closed_form": bal_p},
                "pair": pair_block([cut_e - fill_e, bal_e], [net_p, bal_p], 0.01, "m3 | m", "net volume and balance level: cell means vs trapezoid nodes"),
                "method": "cell-mean prism (1 m cells); bulking and slopes not included",
                "witness": self._ew_witness(name, n0, n1, e0, e1, level, cp.asnumpy(wt))}

    def _ew_witness(self, name, n0, n1, e0, e1, level, wt):
        a = np.asarray(np.load(f"{LIDAR}/{name}/source.npy", mmap_mode="r")[n0:n1 + 1, e0:e1 + 1], np.float32).astype(np.float64)
        return {"cpu_trapezoid_net_m3": float(((a - level) * wt).sum())}

    def op_shade(self, c):
        name, S = site_of(self, c)
        az, el = math.radians(float(c.get("sun_az", 180))), math.radians(float(c.get("sun_el", 10)))
        sx, sy, sz = math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)
        stride = int(c.get("stride_m", 8)); m = (S["n"] - 1) // stride
        eo, po = cp.zeros(m * m, cp.uint8), cp.zeros(m * m, cp.uint8)
        t0 = time.perf_counter()
        K_SHADOW(((m * m + 255) // 256,), (256,), (S["g"], S["mip"], np.int32(S["n"]), np.int32(S["mn"]), np.int32(MB), np.int32(stride),
                 np.float32(sx), np.float32(sy), np.float32(sz), eo, po, np.int32(m * m)))
        cp.cuda.Stream.null.synchronize(); ms = (time.perf_counter() - t0) * 1000
        fe, fp = float(eo.mean()), float(po.mean())
        diff = int(cp.count_nonzero(eo != po))
        self.sun = (sx, sy, sz)
        return {"site": name, "sun_az_deg": math.degrees(az), "sun_el_deg": math.degrees(el), "cells": m * m, "cell_stride_m": stride,
                "shaded_fraction": {"electron_fixed_march": fe, "positron_maxmip_march": fp}, "kernel_ms": ms,
                "pair": {"what": "cell in shadow: 1 m march vs max-mip skip + 1 m march", "unit": "cells", "tolerance": 0, "photons": diff},
                "note": "the render's sun now follows this command"}

    def op_ohl(self, c):
        name, S = site_of(self, c)
        A, B = [float(v) for v in c["a"]], [float(v) for v in c["b"]]     # [e, n, attachment height above ground m]
        w_, Ht = float(c.get("w_n_per_m", 9.0)), float(c.get("h_n", 20000.0))
        span = math.hypot(B[0] - A[0], B[1] - A[1]); k = max(2, int(span) + 1)
        s = np.linspace(0, 1, k)
        en = np.column_stack([A[0] + (B[0] - A[0]) * s, A[1] + (B[1] - A[1]) * s])
        ge, gp = sample_pair(S, en)
        za, zb = ge[0] + A[2], ge[-1] + B[2]
        x = cp.asarray(s * span); a = Ht / w_; h = zb - za
        # electron: exact catenary through both ends; positron: parabola
        xm = span / 2 - a * math.asinh(h / (2 * a * math.sinh(span / (2 * a))))
        z_cat = za + a * (cp.cosh((x - xm) / a) - math.cosh(-xm / a))
        z_par = za + h * x / span - w_ * x * (span - x) / (2 * Ht)
        ce, cpz = cp.asnumpy(z_cat) - ge, cp.asnumpy(z_par) - ge
        i = int(np.argmin(ce))
        return {"site": name, "span_m": span, "tension_n": Ht, "weight_n_per_m": w_, "catenary_a_m": a,
                "mid_sag_m": {"electron_catenary": float((za + zb) / 2 - float(z_cat[k // 2])), "positron_parabola": w_ * span ** 2 / (8 * Ht)},
                "min_clearance_m": {"electron": float(ce.min()), "positron": float(cpz.min()), "at_chainage_m": float(s[i] * span)},
                "pair": pair_block(ce, cpz, 0.05, "m", "clearance along the span: catenary vs parabola"),
                "witness": {"cpu_ground_at_min_m": bilin_cpu(np.load(f"{LIDAR}/{name}/source.npy", mmap_mode="r"), en[i, 0] - S["e0"], en[i, 1] - S["n0"]),
                            "gpu_ground_at_min_m": float(gp[i])},
                "note": "Illustrative geometry only; clearances are set by the network operator and GS6."}

    def op_frame(self, c):
        """Write the current VRAM frame as a PNG now; return its path and simple image maths."""
        slot = self.ring[(self.frame - 1) % RING]
        self.preview(slot)
        d = self.depth
        sky = float(cp.mean(d > 1e29))
        return {"png": f"{FRAMES}/latest.png", "frame": self.frame, "sky_fraction": sky,
                "median_ground_distance_m": float(cp.median(d[d < 1e29])) if sky < 1 else None}

    # ---- main loop
    def run(self):
        write_json(f"{ROOT}/furnace.pid.json", {"pid": os.getpid(), "started": now_iso()})
        last_status = 0.0; t_prev = time.perf_counter(); fps_t = t_prev; fps_n = 0; last_poll = 0.0
        while True:
            now = time.perf_counter()
            if now - last_poll > 0.2:
                self.poll(); last_poll = now
            hot = self.temp_c is not None and self.temp_c >= TEMP_PAUSE_C
            self.paused = hot
            if not hot:
                dt = now - t_prev; t_prev = now
                self.step_camera(dt)
                t0 = time.perf_counter()
                slot = self.render()
                cp.cuda.Stream.null.synchronize()
                self.render_ms = (time.perf_counter() - t0) * 1000
                self.frame += 1; fps_n += 1
                if self.frame % PREVIEW_EVERY == 1:
                    self.preview(slot)
            else:
                time.sleep(1.0); t_prev = time.perf_counter()
            if now - fps_t >= 1.0:
                self.fps = fps_n / (now - fps_t); fps_n = 0; fps_t = now
            if now - last_status >= 1.0:
                self.status(); last_status = now
            spare = 1.0 / FPS_CAP - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)


if __name__ == "__main__":
    pool = cp.get_default_memory_pool()
    pool.set_limit(size=12 * 2**30)
    try:
        Furnace().run()
    except Exception:
        log("CRASH " + traceback.format_exc())
        raise
