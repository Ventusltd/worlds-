# p8-modules: module string maths in VRAM (CuPy). Pair method: electron vs positron, CPU witness, sha256.
import furnace_paths as FP
import json, hashlib, time, sys, numpy as np, cupy as cp
SRC = FP.WORLD + "/data/modules.json"
OUT = FP.FURNACE + "/out/p8-modules"
import os; os.makedirs(OUT, exist_ok=True)
cat = json.load(open(SRC, encoding="utf-8"))
VSYS = float(cat["system_voltage_V"]); TCOLD = float(cat["coldest_cell_C"])
T_HOT = 70.0      # assumed hottest cell C for MPPT floor
V_MPPT_MIN = 900.0  # assumed generic 1500 V inverter MPPT floor
RATIOS = [1.20, 1.25, 1.30, 1.35]; AC_MVA = 1.0

rows = []
for c in cat["classes"]:
    for b in c["electrical"]["bins"]:
        rows.append((c["id"], b["pmax"], b["voc"], b["vmp"], b["imp"], b["isc"], c["temp_coeff"]["voc"], c["temp_coeff"]["isc"], c["max_fuse_A"]))
ids = [r[0] for r in rows]
A = np.array([r[1:] for r in rows], dtype=np.float64)  # pmax voc vmp imp isc bvoc bisc fuse
T = np.round(np.arange(-25.0, 5.0001, 0.01), 2)         # cold cell sweep, 3001 temps
t0 = time.perf_counter()
g = cp.asarray(A); gT = cp.asarray(T)
pmax, voc, vmp, imp, isc, bv, bi, fuse = [g[:, k][:, None] for k in range(8)]
# ELECTRON: f64 closed form, n = floor(Vsys / Voc_cold)
voc_c_e = voc * (1.0 + bv / 100.0 * (gT[None, :] - 25.0))
n_e = cp.floor(VSYS / voc_c_e + 1e-9).astype(cp.int32)
# POSITRON: f32, delta form + counting string lengths 1..80 whose voltage fits
v32 = voc.astype(cp.float32); d32 = (v32 * bv.astype(cp.float32) / 100.0) * (gT[None, :].astype(cp.float32) - 25.0)
voc_c_p = v32 + d32
ns = cp.arange(1, 81, dtype=cp.float32)
n_p = (ns[None, None, :] * voc_c_p[:, :, None] <= VSYS * (1 + 1e-6)).sum(axis=2).astype(cp.int32)
cp.cuda.Stream.null.synchronize(); t1 = time.perf_counter()
dv = cp.abs(voc_c_e - voc_c_p.astype(cp.float64)); tolV = 1e-3
ph_v = int((dv > tolV).sum()); ph_n = int((n_e != n_p).sum())
# where n disagrees, report the margin (edge cases)
edge = cp.argwhere(n_e != n_p)
edges = [{"bin": ids[int(i)] + "-" + str(int(A[int(i), 0])), "T_C": float(T[int(j)]), "n_e": int(n_e[i, j]), "n_p": int(n_p[i, j])} for i, j in cp.asnumpy(edge)[:10]]
# CPU witness: 2000 random (bin,T) samples
rng = np.random.default_rng(8); si = rng.integers(0, len(rows), 2000); sj = rng.integers(0, len(T), 2000)
wv = A[si, 1] * (1 + A[si, 5] / 100 * (T[sj] - 25)); wn = np.floor(VSYS / wv + 1e-9).astype(int)
ne_h = cp.asnumpy(n_e); vc_h = cp.asnumpy(voc_c_e)
wit = {"samples": 2000, "n_mismatch": int((ne_h[si, sj] != wn).sum()), "voc_max_abs_diff_V": float(np.abs(vc_h[si, sj] - wv).max())}
jT = int(np.argmin(np.abs(T - TCOLD)))
# hot floor and DC/AC, electron (Pmax label) vs positron (Vmp*Imp)
vmp_hot = vmp * (1 + bv / 100 * (T_HOT - 25)); n_min = cp.ceil(V_MPPT_MIN / vmp_hot).astype(cp.int32)
isc_hot = isc * (1 + bi / 100 * (T_HOT - 25))
nmax_d = n_e[:, jT][:, None]
R = cp.asarray(RATIOS)[None, :]
str_e = cp.ceil(R * AC_MVA * 1e6 / (nmax_d * pmax)); ratio_e = str_e * nmax_d * pmax / (AC_MVA * 1e6)
ratio_p = str_e * nmax_d * (vmp * imp) / (AC_MVA * 1e6)
dr = cp.abs(ratio_e - ratio_p) / ratio_e; tolR = 0.005; ph_r = int((dr > tolR).sum())
H = lambda x: cp.asnumpy(x)
res = []
for k, r in enumerate(rows):
    res.append({"bin": f"{r[0]}-{int(r[1])}W", "voc_V": r[2], "beta_voc_pct_K": r[6],
        "voc_cold_V_electron": round(float(H(voc_c_e)[k, jT]), 4), "voc_cold_V_positron": round(float(H(voc_c_p)[k, jT]), 4),
        "n_max_electron": int(H(n_e)[k, jT]), "n_max_positron": int(H(n_p)[k, jT]),
        "string_voc_cold_V": round(float(H(n_e)[k, jT] * H(voc_c_e)[k, jT]), 1),
        "n_max_at_-20C": int(H(n_e)[k, int(np.argmin(np.abs(T + 20)))]), "n_max_at_0C": int(H(n_e)[k, int(np.argmin(np.abs(T)))]),
        "vmp_hot70_V": round(float(H(vmp_hot)[k, 0]), 2), "n_min_mppt": int(H(n_min)[k, 0]),
        "isc_hot_x1.25_A": round(float(H(isc_hot)[k, 0]) * 1.25, 2), "fuse_A": r[8],
        "string_kWp": round(float(H(n_e)[k, jT] * r[1]) / 1000, 3),
        "dcac": [{"target": RATIOS[m], "strings_per_MVA": int(H(str_e)[k, m]), "ratio_electron_pmax": round(float(H(ratio_e)[k, m]), 4), "ratio_positron_vmpimp": round(float(H(ratio_p)[k, m]), 4)} for m in range(len(RATIOS))]})
# threshold: temperature where n_max drops by one (cold margin)
thr = []
for k, r in enumerate(rows):
    nn = ne_h[k]; ch = np.nonzero(np.diff(nn))[0]
    thr.append({"bin": f"{r[0]}-{int(r[1])}W", "steps_C": [[float(T[c]), int(nn[c]), int(nn[c + 1])] for c in ch]})
ans = {"op": "modules-strings", "source": "web/world/data/modules.json (read-only)",
    "assumptions": {"system_V": VSYS, "coldest_cell_C": TCOLD, "hot_cell_C": T_HOT, "mppt_min_V": V_MPPT_MIN, "ac_block_MVA": AC_MVA, "vmp_coeff": "taken = voc coeff (catalogue has none)"},
    "grid": {"bins": len(rows), "temps": len(T), "T_range_C": [-25, 5], "step_C": 0.01, "cells": len(rows) * len(T)},
    "pair": {"voc_tol_V": tolV, "voc_photons": ph_v, "voc_max_abs_diff_V": float(dv.max()), "n_photons": ph_n, "n_edge_cases": edges,
             "dcac_tol_rel": tolR, "dcac_photons": ph_r, "dcac_max_rel_diff": float(dr.max())},
    "witness": wit, "timing_ms": {"gpu_pair": round((t1 - t0) * 1000, 2)}, "results_at_coldest": res, "n_max_steps": thr,
    "vram_pool_mib": round(cp.get_default_memory_pool().total_bytes() / 2**20, 2)}
s = json.dumps(ans, indent=1, ensure_ascii=True)
ans["sha256"] = hashlib.sha256(s.encode()).hexdigest()
open(OUT + "/answer.json", "w", newline="\n").write(json.dumps(ans, indent=1))
L = [f"p8-modules  string maths in VRAM  sha256 {ans['sha256']}",
     f"system {VSYS:.0f} V, coldest cell {TCOLD} C, hot cell {T_HOT} C, MPPT floor {V_MPPT_MIN} V (assumed); sweep {len(rows)} bins x {len(T)} temps",
     f"pair: Voc photons {ph_v} (max diff {float(dv.max()):.2e} V), n photons {ph_n}, DC/AC photons {ph_r} (max rel diff {float(dr.max()):.4f}, tol {tolR})",
     f"witness: {wit}", "",
     "bin        Voc  Voc(-10C) e/p        nmax e/p  strVoc  n@-20 n@0  Vmp70  nmin  Isc70x1.25/fuse  kWp/str  strings/MVA @1.25  DC/AC e/p"]
for r in res:
    d = r["dcac"][1]
    L.append(f"{r['bin']:9s} {r['voc_V']:5.2f}  {r['voc_cold_V_electron']:7.4f}/{r['voc_cold_V_positron']:7.4f}  {r['n_max_electron']:3d}/{r['n_max_positron']:<3d}  {r['string_voc_cold_V']:6.1f}  {r['n_max_at_-20C']:4d} {r['n_max_at_0C']:4d}  {r['vmp_hot70_V']:5.2f}  {r['n_min_mppt']:3d}   {r['isc_hot_x1.25_A']:5.2f}/{r['fuse_A']}   {r['string_kWp']:7.3f}  {d['strings_per_MVA']:5d}  {d['ratio_electron_pmax']:.4f}/{d['ratio_positron_vmpimp']:.4f}")
L.append(""); L.append("n_max steps (cell C where one module drops):")
for t in thr: L.append(f"  {t['bin']}: {t['steps_C']}")
open(OUT + "/RESULTS.txt", "w", newline="\n").write("\n".join(L) + "\n")
print("\n".join(L))
