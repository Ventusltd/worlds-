"""p3_piles.py: automatic piling of every table in VRAM (CuPy, Blackwell), read against the world's own JS (web/world v09).

For a site: lay out a 20 MW south plant with the world's JS (v09 plant-layout), let v09 plant-piles pile it (reference),
then pile the same tables on the card from the table geometry alone, with the v09 rules (pile-rules.mjs):
  lines at 0.2 and 0.8 of the depth, nominal reveal lowEdge + rise f - rails, band nominal -0.4/+0.6, head adjust 0.25,
  twist tolerance 1.0 deg (rigid refit over it), embedment 1.5 m, 5 m spacing, along-row slope limit 0.15.
Two formulations of the top-line fit, on the card, for every line at once:
  electron: v09's candidate set (least-squares slope, both limits, all pairwise slopes), least spread, tie -> nearest LS
  positron: golden-section minimum of the convex spread over the limit interval, then the flat-bottom interval found by
            bisection and the LS slope clipped into it
CPU witness: numpy f64, the v09 sequential loop, on a sample of tables.
usage: python p3_piles.py <site> [<site> ...]      writes $FURNACE_ROOT/out/p3-piles/<site>.txt and .json
"""
import furnace_paths as FP
import hashlib, json, math, os, subprocess, sys, time
import numpy as np
import cupy as cp

LIDAR = FP.LIDAR; OUT = (FP.FURNACE + "/out/p3-piles"); WORK = (FP.FURNACE + "/work")
NODE_JS = (FP.SRC + "/p3_piles_js.mjs")
os.makedirs(OUT, exist_ok=True); os.makedirs(WORK, exist_ok=True)
EPS = 1e-9
R = dict(embed=1.5, below=0.4, above=0.6, adjust=0.25, twist=1.0, rails=0.3, spacing=5.0, grade_w=2.0, lines=(0.2, 0.8), lim=0.15)


def run_js(site, half=350.0, mw=20.0):
    src = json.load(open(f"{LIDAR}/{site}/source.json"))
    a = np.load(f"{LIDAR}/{site}/source.npy", mmap_mode="r"); n = a.shape[0]
    E0, N0 = src["origin_e_m"], src["origin_n_m"]
    ce, cn = E0 + n / 2, N0 + n / 2
    bnd = [[ce - half, cn - half], [ce + half, cn - half], [ce + half, cn + half], [ce - half, cn + half]]
    i0 = max(0, int(ce - half - E0) - 40); i1 = min(n, int(ce + half - E0) + 41)
    j0 = max(0, int(cn - half - N0) - 40); j1 = min(n, int(cn + half - N0) + 41)
    win = np.ascontiguousarray(a[j0:j1, i0:i1], dtype="<f4")
    gf = f"{WORK}/p3-ground-{site}.f32"; win.tofile(gf)
    G = {"file": gf, "e0": E0 + i0, "n0": N0 + j0, "w": win.shape[1], "h": win.shape[0]}
    rq, rs = f"{WORK}/p3-req-{site}.json", f"{WORK}/p3-res-{site}.json"
    json.dump({"boundary": bnd, "targetMW": mw, "layout": "south", "ground": G}, open(rq, "w"))
    t0 = time.perf_counter()
    r = subprocess.run(["node", NODE_JS, rq, rs], capture_output=True, text=True, timeout=600)
    if r.returncode:
        raise RuntimeError(r.stderr[-1500:])
    js = json.load(open(rs, encoding="utf-8")); js["node_wall_s"] = time.perf_counter() - t0
    return js, win, G


def bilin_gpu(g, x, y):                     # f64 bilinear of f32 nodes (as the JS groundAt)
    i = cp.floor(x).astype(cp.int32); j = cp.floor(y).astype(cp.int32); fx, fy = x - i, y - j
    a0 = g[j, i].astype(cp.float64); a1 = g[j, i + 1].astype(cp.float64)
    b0 = g[j + 1, i].astype(cp.float64); b1 = g[j + 1, i + 1].astype(cp.float64)
    h0 = a0 + (a1 - a0) * fx; h1 = b0 + (b1 - b0) * fx
    return h0 + (h1 - h0) * fy


def spread(s, g, b):                        # s (L,k), g (L,k), b (L,m) -> hi, lo (L,m)
    d = g[:, None, :] - b[:, :, None] * s[:, None, :]
    return d.max(2), d.min(2)


def fit_electron(s, g, lim):
    ms, mg = s.mean(1, keepdims=True), g.mean(1, keepdims=True)
    sxx = ((s - ms) ** 2).sum(1); bls = cp.where(sxx > 0, ((s - ms) * (g - mg)).sum(1) / cp.where(sxx > 0, sxx, 1), 0)
    k = s.shape[1]; ii, jj = np.triu_indices(k, 1)
    pw = (g[:, jj] - g[:, ii]) / (s[:, jj] - s[:, ii])
    cand = cp.clip(cp.concatenate([bls[:, None], cp.full((len(s), 1), -lim), cp.full((len(s), 1), lim), pw], 1), -lim, lim)
    hi, lo = spread(s, g, cand); w = hi - lo
    ok = w <= w.min(1, keepdims=True) + 1e-9
    pick = cp.where(ok, cp.abs(cand - bls[:, None]), cp.inf).argmin(1)
    return cand[cp.arange(len(s)), pick], bls


def fit_positron(s, g, lim, bls, it=90):
    W = lambda b: (lambda h, l: (h - l)[:, 0])(*spread(s, g, b[:, None]))
    lo_, hi_ = cp.full(len(s), -lim), cp.full(len(s), lim); r = (math.sqrt(5) - 1) / 2
    for _ in range(it):                     # golden section on the convex spread
        c = hi_ - r * (hi_ - lo_); d = lo_ + r * (hi_ - lo_)
        m = W(c) < W(d); hi_ = cp.where(m, d, hi_); lo_ = cp.where(m, lo_, c)
    bm = (lo_ + hi_) / 2; wm = W(bm)
    # the flat bottom: widest [L, U] around bm with spread <= wm + 1e-9, by bisection each side
    Lx, Li = cp.full(len(s), -lim), bm.copy(); Ux, Ui = cp.full(len(s), lim), bm.copy()
    fl = W(Lx) <= wm + 1e-9; fu = W(Ux) <= wm + 1e-9
    for _ in range(70):
        c = (Lx + Li) / 2; m = W(c) <= wm + 1e-9; Li = cp.where(m, c, Li); Lx = cp.where(m, Lx, c)
        c = (Ux + Ui) / 2; m = W(c) <= wm + 1e-9; Ui = cp.where(m, c, Ui); Ux = cp.where(m, Ux, c)
    Lb = cp.where(fl, -lim, Li); Ub = cp.where(fu, lim, Ui)
    return cp.clip(cp.clip(bls, -lim, lim), Lb, Ub)


def level(s, g, b, band):                   # band (L,3): min, nominal, max
    d = g - b[:, None] * s; hi, lo = d.max(1), d.min(1)
    wlo, whi = hi + band[:, 0], lo + band[:, 2]; anom = d.mean(1) + band[:, 1]
    feas = wlo <= whi + EPS
    return cp.where(feas, cp.clip(anom, wlo, cp.maximum(wlo, whi)), (wlo + whi) / 2)


def piles_from_top(top, g, band):
    rev = top - g
    fill = cp.maximum(rev - band[:, 2:3], 0); cut = cp.maximum(band[:, 0:1] - rev, 0)
    fin = g + fill - cut; frev = top - fin; length = frev + R["embed"]
    fl = (rev > band[:, 2:3] + EPS) * 1 + ((rev < band[:, 0:1] - EPS) & ~(rev > band[:, 2:3] + EPS)) * 2 \
        + (cp.abs(rev - band[:, 1:2]) > R["adjust"] + EPS) * 4
    return dict(rev=rev, cut=cut, fill=fill, fin=fin, frev=frev, length=length, flags=fl.astype(cp.uint8))


def pile_all(uv, tab, gdev, G, fitter):
    """Every table's two lines on the card. Returns per-pile arrays (table, line, k) and per-table twist/grading."""
    nT = len(uv); lenU, depth, rise, low = tab["lenU"], tab["depth"], tab["rise"], tab["lowEdgeM"]
    f = np.array(R["lines"]); nom = low + rise * f - R["rails"]
    k = int(math.floor(lenU / R["spacing"] + EPS)) + 1; over = (lenU - (k - 1) * R["spacing"]) / 2
    s1 = cp.asarray(over + R["spacing"] * np.arange(k))
    uvd = cp.asarray(np.asarray(uv, np.float64))
    e = uvd[:, None, None, 0] + s1[None, None, :]                       # (T, 2 lines, k)
    n = uvd[:, None, None, 1] + cp.asarray(f * depth)[None, :, None] + 0 * s1[None, None, :]
    e, n = cp.broadcast_arrays(e, n); e, n = cp.ascontiguousarray(e), cp.ascontiguousarray(n)
    g = bilin_gpu(gdev, e - G["e0"], n - G["n0"])
    band = cp.asarray(np.stack([nom - R["below"], nom, nom + R["above"]], 1))   # (2,3)
    S = cp.broadcast_to(s1, (nT * 2, k)); Gl = g.reshape(nT * 2, k); B = cp.tile(band, (nT, 1))
    b = fitter(S, Gl); a = level(S, Gl, b, B)
    bT = b.reshape(nT, 2)
    twist = cp.degrees(cp.arctan(cp.abs(bT[:, 0] - bT[:, 1]) * lenU / ((f[1] - f[0]) * depth)))
    twisted = twist > R["twist"] + EPS
    top = (a[:, None] + b[:, None] * S).reshape(nT, 2, k)
    if bool(twisted.any()):                 # rigid frame: one line through both, rear = front + d, front band
        idx = cp.nonzero(twisted)[0]; dnom = nom[1] - nom[0]
        Sr = cp.broadcast_to(cp.concatenate([s1, s1]), (len(idx), 2 * k)); Gr = g[idx].reshape(len(idx), 2 * k)
        br = fitter(Sr, Gr); ar = level(Sr, Gr, br, cp.tile(band[0:1], (len(idx), 1)))
        top[idx, 0] = ar[:, None] + br[:, None] * s1; top[idx, 1] = ar[:, None] + dnom + br[:, None] * s1
    P = piles_from_top(top.reshape(nT * 2, k), Gl, B)
    P = {q: v.reshape(nT, 2 * k) for q, v in P.items()}
    P.update(top=top.reshape(nT, 2 * k), ground=g.reshape(nT, 2 * k), e=e.reshape(nT, 2 * k), n=n.reshape(nT, 2 * k))
    P["twist"], P["twisted"] = twist, twisted
    P["grading"] = ((P["flags"] & 3) > 0).any(1)
    P["cutM3"] = P["cut"].sum(1) * R["spacing"] * R["grade_w"]; P["fillM3"] = P["fill"].sum(1) * R["spacing"] * R["grade_w"]
    P["k"], P["nom"] = k, nom
    return P


# ---------------- CPU witness: the v09 sequential loop in numpy f64 on sampled tables
def cpu_fit(s, g, band, lim):
    n = len(s); ms, mg = s.mean(), g.mean(); sxx = ((s - ms) ** 2).sum(); bls = ((s - ms) * (g - mg)).sum() / sxx if sxx > 0 else 0
    clip = lambda b: max(-lim, min(lim, b)); cand = [clip(bls), -lim, lim]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(s[j] - s[i]) > EPS: cand.append(clip((g[j] - g[i]) / (s[j] - s[i])))
    best = None
    for b in cand:
        d = g - b * s; w = d.max() - d.min()
        if best is None or w < best[1] - 1e-9 or (w <= best[1] + 1e-9 and abs(b - bls) < abs(best[0] - bls)): best = (b, w)
    b = best[0]; d = g - b * s; wlo, whi = d.max() + band[0], d.min() + band[2]; anom = d.mean() + band[1]
    a = max(wlo, min(whi, anom)) if wlo <= whi + EPS else (wlo + whi) / 2
    return a, b


def cpu_table(win, G, u, v, tab, nom, k):
    f = R["lines"]; over = (tab["lenU"] - (k - 1) * R["spacing"]) / 2; s = over + R["spacing"] * np.arange(k)
    def gr(e, n):
        x, y = e - G["e0"], n - G["n0"]; i, j = int(math.floor(x)), int(math.floor(y)); fx, fy = x - i, y - j
        a0, a1, b0, b1 = (float(win[j, i]), float(win[j, i + 1]), float(win[j + 1, i]), float(win[j + 1, i + 1]))
        h0 = a0 + (a1 - a0) * fx; h1 = b0 + (b1 - b0) * fx; return h0 + (h1 - h0) * fy
    gs = [np.array([gr(u + q, v + fl * tab["depth"]) for q in s]) for fl in f]
    bands = [(x - R["below"], x, x + R["above"]) for x in nom]
    fits = [cpu_fit(s, gs[i], bands[i], R["lim"]) for i in range(2)]
    tw = math.degrees(math.atan(abs(fits[0][1] - fits[1][1]) * tab["lenU"] / ((f[1] - f[0]) * tab["depth"])))
    if tw > R["twist"] + EPS:
        a, b = cpu_fit(np.concatenate([s, s]), np.concatenate(gs), bands[0], R["lim"])
        fits = [(a, b), (a + nom[1] - nom[0], b)]
    top = np.concatenate([fits[i][0] + fits[i][1] * s for i in range(2)])
    return top, tw


def run_site(site):
    t_all = time.perf_counter()
    js, win, G = run_js(site)
    tab, uv = js["table"], js["uv"]
    assert not js["frame"]["swap"] and tab["south"]
    gdev = cp.asarray(win)
    cp.cuda.Device().synchronize(); t0 = time.perf_counter()
    PE = pile_all(uv, tab, gdev, G, lambda S, Gl: fit_electron(S, Gl, R["lim"])[0])
    PP = pile_all(uv, tab, gdev, G, lambda S, Gl: fit_positron(S, Gl, R["lim"], fit_electron(S, Gl, R["lim"])[1]))
    cp.cuda.Device().synchronize(); gpu_ms = (time.perf_counter() - t0) * 1000
    k = PE["k"]; nT = len(uv); h = lambda x: cp.asnumpy(x)
    # JS reference per pile
    st = js["stride"]; jt = js["tables"]
    same_n = all(t["n"] == 2 * k for t in jt)
    JA = np.array([t["pa"] for t in jt], np.float64).reshape(nT, 2 * k, st) if same_n else None
    JF = np.array([t["pf"] for t in jt], np.uint8).reshape(nT, 2 * k) if same_n else None
    def pb(x, y, tol, unit, what):
        d = np.abs(np.asarray(x, np.float64) - np.asarray(y, np.float64))
        return {"what": what, "unit": unit, "tolerance": tol, "photons": int(np.count_nonzero(~(d <= tol))), "max_abs_diff": float(np.nanmax(d))}
    pair = {
        "top_e_vs_p": pb(h(PE["top"]), h(PP["top"]), 1e-4, "m", "pile top: electron candidate-set fit vs positron golden+bisection fit (VRAM)"),
        "reveal_e_vs_p": pb(h(PE["rev"]), h(PP["rev"]), 1e-4, "m", "reveal before grading, electron vs positron"),
        "length_e_vs_p": pb(h(PE["length"]), h(PP["length"]), 1e-4, "m", "pile length (finished reveal + embedment)"),
        "flags_e_vs_p": {"photons": int((h(PE["flags"]) != h(PP["flags"])).sum()), "what": "pile flag byte, electron vs positron"},
        "twist_e_vs_p": pb(h(PE["twist"]), h(PP["twist"]), 1e-3, "deg", "table twist"),
    }
    js_cmp = None
    if same_n:
        js_cmp = {
            "position": pb(np.stack([h(PE["e"]), h(PE["n"])], -1), JA[..., 0:2], 1e-6, "m", "pile e,n: VRAM vs v09 JS"),
            "ground": pb(h(PE["ground"]), JA[..., 2], 1e-6, "m", "ground at pile: VRAM f64-of-f32 bilinear vs v09 JS"),
            "top": pb(h(PE["top"]), JA[..., 3], 1e-3, "m", "pile top: VRAM electron vs v09 JS"),
            "length": pb(h(PE["length"]), JA[..., 5], 1e-3, "m", "pile length: VRAM electron vs v09 JS"),
            "reveal": pb(h(PE["rev"]), JA[..., 7], 1e-3, "m", "reveal: VRAM electron vs v09 JS"),
            "finished_reveal": pb(h(PE["frev"]), JA[..., 8], 1e-3, "m", "finished reveal: VRAM vs v09 JS"),
            "flags": {"photons": int((h(PE["flags"]) != JF).sum()), "what": "pile flag byte vs v09 JS"},
            "twist": pb(h(PE["twist"]), [t["twistDeg"] for t in jt], 1e-3, "deg", "table twist vs v09 JS"),
            "twisted": {"photons": int((h(PE["twisted"]) != np.array([t["twisted"] for t in jt])).sum())},
            "grading": {"photons": int((h(PE["grading"]) != np.array([t["grading"] for t in jt])).sum())},
            "cut_fill_m3": pb(h(PE["cutM3"] + PE["fillM3"]), [t["cutM3"] + t["fillM3"] for t in jt], 1e-2, "m3", "grading volume per table"),
        }
        bad = np.nonzero(np.abs(h(PE["top"]) - JA[..., 3]).max(1) > 1e-3)[0]
        js_cmp["tables_top_off"] = [int(x) for x in bad[:20]]
    # CPU witness on a sample
    rng = np.random.default_rng(7); samp = sorted(set(rng.choice(nT, min(40, nT), replace=False).tolist()
                                                  + [int(x) for x in np.nonzero(h(PE["twisted"]))[0][:8]]))
    wd, wt = [], []
    for q in samp:
        top, tw = cpu_table(win, G, uv[q][0], uv[q][1], tab, PE["nom"], k)
        wd.append(np.abs(top - h(PE["top"][q])).max()); wt.append(abs(tw - float(PE["twist"][q])))
    witness = {"tables": len(samp), "incl_twisted": int(sum(bool(PE["twisted"][q]) for q in samp)),
               "max_abs_top_diff_m": float(max(wd)), "max_abs_twist_diff_deg": float(max(wt)),
               "photons_top_1mm": int(sum(x > 1e-3 for x in wd))}
    rev = h(PE["rev"]).ravel(); L = h(PE["length"]).ravel(); fl = h(PE["flags"]).ravel()
    pc = lambda x: {p: round(float(np.percentile(x, p)), 3) for p in (0, 5, 50, 95, 100)}
    grading_tables = int(h(PE["grading"]).sum()); dep = np.maximum(h(PE["cut"]), h(PE["fill"])).max(1)
    ans = {
        "site": site, "method": "VRAM CuPy on RTX 5070 Ti (Blackwell); layout and reference from web/world v09 JS", "rules": R,
        "built": js["built"], "table": tab, "tables": nT, "piles": int(rev.size), "piles_per_table": 2 * k,
        "nominal_reveal_m": {"front_f0.2": round(float(PE["nom"][0]), 3), "rear_f0.8": round(float(PE["nom"][1]), 3)},
        "reveal_m_front": pc(h(PE["rev"])[:, :k]), "reveal_m_rear": pc(h(PE["rev"])[:, k:]),
        "finished_reveal_m": pc(h(PE["frev"]).ravel()),
        "embedment_m": {"each": R["embed"], "total_embedded_m": round(R["embed"] * rev.size, 1), "note": "provisional, pile tests set it"},
        "pile_length_m": {**pc(L), "total": round(float(L.sum()), 1)},
        "flags": {"reveal_high": int(((fl & 1) > 0).sum()), "reveal_low": int(((fl & 2) > 0).sum()), "outside_head_adjust": int(((fl & 4) > 0).sum())},
        "twist_deg": {**pc(h(PE["twist"])), "tables_over_1deg_rigid_refit": int(h(PE["twisted"]).sum())},
        "grading": {"tables": grading_tables, "cut_m3": round(float(h(PE["cutM3"]).sum()), 1), "fill_m3": round(float(h(PE["fillM3"]).sum()), 1),
                    "drop_table_depth_over_0.6m": int((dep > 0.6).sum()), "raise_table_candidates": int(((h(PE["grading"])) & (h(PE["cut"]).max(1) + h(PE["fill"]).max(1) <= 0.25 + 1e-9)).sum())},
        "pair_vram": pair, "vs_v09_js": js_cmp,
        "v09_js_summary": {q: js["summary"].get(q) for q in ("tables", "piles", "totalPileLengthM", "embeddedM", "revealHigh", "revealLow", "revealAdjust",
                                                            "twistTables", "gradingTables", "cutM3", "fillM3", "drop", "raise", "regrade", "revealP", "perMWp")},
        "witness_cpu": witness,
        "timing": {"gpu_ms_both_formulations": round(gpu_ms, 2), "js_layout_ms": round(js["ms"]["layout"], 1), "js_piles_ms": round(js["ms"]["piles"], 1),
                   "wall_s": round(time.perf_counter() - t_all, 2)},
        "vram_pool_mib": round(cp.get_default_memory_pool().total_bytes() / 2**20, 1),
        "label": "Illustrative engineering on generic editable rules; not a design for any site.",
    }
    body = json.dumps(ans, indent=1, default=float)
    ans["sha256"] = hashlib.sha256(body.encode()).hexdigest()
    open(f"{OUT}/{site}.json", "w", newline="\n").write(json.dumps(ans, indent=1, default=float) + "\n")
    return ans


if __name__ == "__main__":
    for s in sys.argv[1:]:
        a = run_site(s)
        print(s, a["tables"], a["piles"], a["timing"], a["pair_vram"]["top_e_vs_p"], (a["vs_v09_js"] or {}).get("top"), a["witness_cpu"])
