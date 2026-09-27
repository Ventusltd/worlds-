"""p2-plant-layout: pair check of furnace layout answers against an independent re-run of the world JS.

electron: the world JS (web/world v08 plant-layout.mjs via export_world.mjs), its own built numbers.
positron: CuPy on the card, from the raw table polygons: count, sum kWp, measured row pitch,
          GCR = collector slope length / measured pitch, table plan area (shoelace) / fenced area,
          plus closed-form table geometry (lenU, depth, pitch, rise) from the options.
witness : numpy on the CPU, a sample of 64 tables (area, pitch).
Run under the GPU lock ($GRID_GPU_LOCK). Writes text to $FURNACE_ROOT/out/p2-plant-layout/.
"""
import furnace_paths as FP
import hashlib, json, math, os, subprocess, sys, time
import numpy as np
import cupy as cp

ANS = (FP.FURNACE + "/out/answers")
OUT = (FP.FURNACE + "/out/p2-plant-layout")
WORK = f"{OUT}/work"
LIDAR = FP.LIDAR
NODE = (FP.SRC + "/export_world.mjs")
DEF = dict(gcr=0.4, ewGapM=3, ridgeGapM=0.3, tiltSouthDeg=25, tiltEwDeg=10, moduleLongM=2.384, moduleShortM=1.134, moduleGapM=0.02)
os.makedirs(WORK, exist_ok=True)


def pair(e, p, tol, unit, what):
    e, p = np.atleast_1d(np.asarray(e, np.float64)), np.atleast_1d(np.asarray(p, np.float64))
    d = np.abs(e - p)
    return {"what": what, "unit": unit, "tolerance": tol, "electron": e.tolist() if e.size < 4 else None,
            "positron": p.tolist() if p.size < 4 else None, "photons": int(np.count_nonzero(~(d <= tol))),
            "max_abs_diff": float(np.nanmax(d)) if d.size else 0.0}


def rerun_js(ans):
    c = ans["command"]; site = ans["site"]
    src = json.load(open(f"{LIDAR}/{site}/source.json", encoding="utf-8"))
    e0s, n0s = float(src["origin_e_m"]), float(src["origin_n_m"])
    a = np.load(f"{LIDAR}/{site}/source.npy", mmap_mode="r"); n = a.shape[0]
    bnd = ans["boundary"]; es = [b[0] for b in bnd]; ns = [b[1] for b in bnd]
    i0 = max(0, int(min(es) - e0s) - 40); i1 = min(n, int(max(es) - e0s) + 41)
    j0 = max(0, int(min(ns) - n0s) - 40); j1 = min(n, int(max(ns) - n0s) + 41)
    win = np.ascontiguousarray(a[j0:j1, i0:i1], dtype=np.float32).astype("<f4")
    gf = f"{WORK}/ground-{c['id']}.f32"; win.tofile(gf)
    req = {"mode": "layout", "boundary": bnd, "targetMW": float(c["mw"]), "layout": c["layout"], "options": c.get("options", {}),
           "grid": None, "ground": {"file": gf, "e0": e0s + i0, "n0": n0s + j0, "w": win.shape[1], "h": win.shape[0]}}
    rq, rs = f"{WORK}/req-{c['id']}.json", f"{WORK}/res-{c['id']}.json"
    json.dump(req, open(rq, "w"))
    t0 = time.perf_counter()
    subprocess.run(["node", NODE, rq, rs], check=True, capture_output=True, text=True, timeout=600)
    L = json.load(open(rs))["layout"]
    os.remove(gf)
    return L, (time.perf_counter() - t0) * 1000


def closed_form(layout, o, mps):
    p = {**DEF, **o}; south = layout == "south"
    tilt = math.radians(p["tiltSouthDeg"] if south else p["tiltEwDeg"])
    slope = 2 * p["moduleLongM"] + p["moduleGapM"] if south else p["moduleLongM"]
    depth = slope * math.cos(tilt) if south else 2 * slope * math.cos(tilt) + p["ridgeGapM"]
    pitch = slope / p["gcr"] if south else depth + p["ewGapM"]
    lenU = mps * (p["moduleShortM"] + p["moduleGapM"]) - p["moduleGapM"]
    collector = slope if south else 2 * slope        # sloped collector length across the row
    return dict(lenU=lenU, depth=depth, pitch=pitch, rise=slope * math.sin(tilt), collector=collector, tiltDeg=math.degrees(tilt))


def measured_pitch(v, lenU_depth_tol=0.05):
    """modal step between distinct row low-edge v values (rows inside a band)."""
    u = np.unique(np.round(v, 3))
    d = np.diff(u); d = d[d > 0.5]
    if not d.size:
        return float("nan")
    r = np.round(d, 2); vals, cnt = np.unique(r, return_counts=True)
    m = vals[np.argmax(cnt)]
    return float(np.mean(d[np.abs(d - m) < 0.01]))


def analyse(ans):
    L, node_ms = rerun_js(ans)
    T = np.array(L["tables"], np.float64).reshape(-1, 12)
    tb = L["table"]; south = L["layout"] == "south"
    mps = ans["template"]["counts"]["modulesPerString"] if ans["built"]["mw"] else None
    # asked template gives module Wp: dcMWp / modules
    cf = closed_form(L["layout"], ans["command"].get("options", {}), int(round((tb["modules"]) / 2)))
    t0 = time.perf_counter()
    G = cp.asarray(T)                                          # tables resident on the card
    xs, ys = G[:, 0:8:2], G[:, 1:8:2]
    area = 0.5 * cp.abs((xs * cp.roll(ys, -1, 1) - cp.roll(xs, -1, 1) * ys).sum(1))
    v = G[:, 1] if south else G[:, 0]                          # across-row axis: north (south layout) / east (EW)
    kwp_gpu = float(cp.full(G.shape[0], tb["kWp"], cp.float64).sum()) if len(T) else 0.0
    area_sum = float(area.sum())
    v_host = cp.asnumpy(v)
    cp.cuda.Device().synchronize(); gpu_ms = (time.perf_counter() - t0) * 1000
    pitch_meas = measured_pitch(v_host)
    gcr_meas = cf["collector"] / pitch_meas if pitch_meas == pitch_meas else float("nan")
    bnd = np.array(ans["boundary"]); fence = 0.5 * abs(np.dot(bnd[:, 0], np.roll(bnd[:, 1], -1)) - np.dot(np.roll(bnd[:, 0], -1), bnd[:, 1]))
    # witness: CPU numpy on 64 tables
    k = min(64, len(T)); idx = np.linspace(0, len(T) - 1, k).astype(int) if k else np.array([], int)
    wa = [0.5 * abs(sum(T[q, 2 * i] * T[q, 2 * ((i + 1) % 4) + 1] - T[q, 2 * ((i + 1) % 4)] * T[q, 2 * i + 1] for i in range(4))) for q in idx]
    ga = cp.asnumpy(area)[idx] if k else []
    b = ans["built"]
    res = {
        "id": ans["id"], "site": ans["site"], "layout": L["layout"], "options": ans["command"].get("options", {}),
        "fence_ha": fence / 1e4, "asked": ans["asked"],
        "built": {"mw": b["mw"], "mwp": b["mwp"], "tables": b["tables"], "stations": b["stations"], "feeders": b["feeders"],
                  "candidates": b["candidates"], "found": b["found"], "roadM": round(b["roadM"], 3)},
        "fits": ans["fits"], "notes": ans["notes"],
        "table_js": tb, "table_closed_form": cf,
        "gcr": {"design": (cf["collector"] / tb["pitch"]), "measured_from_rows": gcr_meas, "unit": "-"},
        "pitch_m": {"js": tb["pitch"], "measured_from_rows": pitch_meas},
        "table_plan_area_m2": area_sum, "land_cover_frac": area_sum / fence,
        "worst_mv_drop_pct": ans["worst_mv_drop_pct"], "cable_m": ans["cable_m"],
        "pairs": {
            "tables_furnace_vs_rerun": pair(b["tables"], len(T), 0, "tables", "furnace answer (VRAM ground) vs JS re-run on disk ground"),
            "mwp_js_vs_gpu": pair(b["mwp"] * 1000, kwp_gpu, 1.0, "kWp", "JS built.mwp vs CuPy sum of table kWp"),
            "pitch_js_vs_measured": pair(tb["pitch"], pitch_meas, 0.01, "m", "JS row pitch vs pitch measured from table rows on GPU"),
            "geometry_js_vs_closed_form": pair([tb["lenU"], tb["depth"], tb["pitch"], tb["rise"], tb["tiltDeg"]],
                                               [cf["lenU"], cf["depth"], cf["pitch"], cf["rise"], cf["tiltDeg"]], 1e-9, "m|deg",
                                               "lenU, depth, pitch, rise, tilt: JS vs Python closed form"),
            "area_js_rect_vs_shoelace": pair(len(T) * tb["lenU"] * tb["depth"], area_sum, 0.5, "m2",
                                             "tables x lenU x depth vs CuPy shoelace sum"),
            "furnace_ground": ans["pair"], "furnace_mwp": ans["pair_mwp"],
        },
        "witness": {"cpu_tables": int(k), "max_abs_area_diff_m2": float(np.max(np.abs(np.array(wa) - ga))) if k else None,
                    "furnace_witness": ans["witness"]},
        "timing_ms": {"furnace": ans["timing_ms"], "rerun_node": node_ms, "gpu": gpu_ms},
        "furnace_sha256": ans["sha256"],
    }
    res["photons_total"] = sum(v["photons"] for v in res["pairs"].values())
    return res


def main(ids):
    rows = []
    for i in ids:
        ans = json.load(open(f"{ANS}/{i}.json"))
        if not ans.get("ok"):
            rows.append({"id": i, "ok": False, "error": ans.get("error")}); continue
        r = analyse(ans); r["ok"] = True
        s = json.dumps(r, indent=1, sort_keys=True, default=float)
        r["sha256"] = hashlib.sha256(s.encode()).hexdigest()
        open(f"{OUT}/{i}.json", "w", newline="\n").write(json.dumps(r, indent=1, sort_keys=True, default=float) + "\n")
        rows.append(r)
        print(i, r["built"]["tables"], r["built"]["mwp"], r["photons_total"], flush=True)
    json.dump(rows, open(f"{WORK}/all.json", "w"), default=float)


if __name__ == "__main__":
    main(sys.argv[1:])
