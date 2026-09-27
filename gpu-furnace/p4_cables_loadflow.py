# p4-cables-loadflow: electrical checks in VRAM (CuPy, RTX 5070 Ti Blackwell).
# Illustrative engineering only; not a design for any site.
#  A. SLD network graphs ($GRID_LIDAR/sld/graph-*.json, read-only):
#     electron = Newton-Raphson on the nodal admittance matrix Y (dense complex Jacobian, cp.linalg.solve)
#     positron = path-matrix formulation of the radial tree: Zbus = A^T diag(z) A built from topology only,
#                fixed-point current injection V = 1 + Zbus conj(S/V)
#     fault level at every node: electron diag(inv(Y_red)), positron A^T z (path impedance); Ik" = c / |Zth| * Ibase
#     branch currents: electron (Vf - Vt)/z, positron A @ I_inj (downstream injection sums)
#     third check: the premerge-sld results-*.json and js-*.json receipts (V and Zth per node)
#  B. String DC and inverter AC cables (world v08 cable-checks.json schedule):
#     electron = the world's own JS (cable-checks.mjs dcChecks/acChecks via node)
#     positron = CuPy re-derivation: I = P/Vmp (string), S/(sqrt3 V) (AC), drops I L R(theta) / sqrt3 I L (R pf + X sin)
#  CPU witness: plain numpy on a seeded sample. Photons = pairs beyond tolerance.
import furnace_paths as FP
import json, os, sys, time, math, hashlib, subprocess, random, pathlib
import numpy as np
import cupy as cp

SLD = (FP.LIDAR + "/sld")
OUT = (FP.FURNACE + "/out/p4-cables-loadflow")
LOCK = FP.LOCK
V08 = FP.WORLD
os.makedirs(OUT, exist_ok=True)


def temp_c():
    try:
        return int(subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], text=True).split()[0])
    except Exception:
        return -1


def take_lock():
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK); return time.time() - t0
        except FileExistsError:
            if time.time() - t0 > 900: raise SystemExit("gpu lock not free after 15 min")
            time.sleep(2)


def heat_wait():
    w = 0.0
    while temp_c() >= 80:
        time.sleep(5); w += 5
    return w


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def pair(a, b, tol, unit, what):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    d = np.abs(a - b)
    return {"what": what, "unit": unit, "tolerance": tol, "n": int(d.size), "photons": int((d > tol).sum()),
            "max_abs_diff": float(d.max()) if d.size else 0.0}


def sld_case(case):
    gp = f"{SLD}/graph-{case}.json"
    g = json.load(open(gp))
    Sb, cfac = float(g["sBaseMVA"]), float(g["c"])
    ids = [n["id"] for n in g["nodes"]]; ix = {k: i for i, k in enumerate(ids)}
    kv = np.array([n["kv"] for n in g["nodes"]], float)
    kind = [n["kind"] for n in g["nodes"]]
    S = np.array([complex(*n["s"]) for n in g["nodes"]])
    br = [b for b in g["branches"] if not b["open"]]
    f = np.array([ix[b["from"]] for b in br]); t = np.array([ix[b["to"]] for b in br])
    z = np.array([complex(*b["z"]) for b in br]); bkv = np.array([b["kv"] for b in br], float); bkind = [b["kind"] for b in br]
    n, m = len(ids), len(br)
    s0 = ix["GRID"]
    assert m == n - 1, "closed network is not a tree"
    # ---------- electron: Newton-Raphson on Y ----------
    t0 = time.perf_counter()
    y = cp.asarray(1 / z)
    Y = cp.zeros((n, n), cp.complex128)
    fg, tg = cp.asarray(f), cp.asarray(t)
    Yr_, Yi_ = cp.zeros((n, n)), cp.zeros((n, n))
    for (a_, b_, sgn) in ((fg, fg, 1), (tg, tg, 1), (fg, tg, -1), (tg, fg, -1)):
        cp.add.at(Yr_, (a_, b_), sgn * y.real); cp.add.at(Yi_, (a_, b_), sgn * y.imag)
    Y = Yr_ + 1j * Yi_; del Yr_, Yi_
    pq = np.array([i for i in range(n) if i != s0]); pqg = cp.asarray(pq)
    Sg = cp.asarray(S)
    V = cp.ones(n, cp.complex128)
    for it in range(30):
        I = Y @ V
        mis = V * cp.conj(I) - Sg
        F = cp.concatenate([mis.real[pqg], mis.imag[pqg]])
        if float(cp.abs(F).max()) < 1e-13: break
        Vn = V / cp.abs(V)
        Yp = Y[pqg][:, pqg]; Vq, Iq, Nq = V[pqg], I[pqg], Vn[pqg]
        dS_dth = 1j * Vq[:, None] * cp.conj(-Yp * Vq[None, :]); dS_dth[cp.arange(len(pq)), cp.arange(len(pq))] += 1j * Vq * cp.conj(Iq)
        dS_dvm = Vq[:, None] * cp.conj(Yp * Nq[None, :]); dS_dvm[cp.arange(len(pq)), cp.arange(len(pq))] += cp.conj(Iq) * Nq
        J = cp.concatenate([cp.concatenate([dS_dth.real, dS_dvm.real], 1), cp.concatenate([dS_dth.imag, dS_dvm.imag], 1)], 0)
        del dS_dth, dS_dvm, Yp
        dx = cp.linalg.solve(J, -F)
        k = len(pq)
        th = cp.angle(V); vm = cp.abs(V)
        th[pqg] += dx[:k]; vm[pqg] += dx[k:]
        V = vm * cp.exp(1j * th)
    nr_it = it + 1
    Yr = Y[pqg][:, pqg]
    Zth_e = cp.zeros(n, cp.complex128); Zth_e[pqg] = cp.diag(cp.linalg.inv(Yr))
    Ib_e = (V[fg] - V[tg]) * y
    cp.cuda.Stream.null.synchronize(); te = time.perf_counter() - t0
    Ve = cp.asnumpy(V); Zte = cp.asnumpy(Zth_e); Ibe = cp.asnumpy(Ib_e)
    # ---------- positron: path matrix from topology ----------
    t0 = time.perf_counter()
    adj = {i: [] for i in range(n)}
    for b in range(m): adj[f[b]].append((t[b], b)); adj[t[b]].append((f[b], b))
    parent = np.full(n, -1); pbr = np.full(n, -1); orient = np.zeros(n); depth = np.zeros(n, int)
    order = [s0]; seen = {s0}
    for u in order:
        for v_, b in adj[u]:
            if v_ not in seen:
                seen.add(v_); parent[v_] = u; pbr[v_] = b; depth[v_] = depth[u] + 1
                orient[v_] = 1.0 if f[b] == u else -1.0; order.append(v_)
    A = cp.zeros((m, n), cp.float64)
    for d in range(1, depth.max() + 1):
        lv = np.where(depth == d)[0]
        A[:, cp.asarray(lv)] = A[:, cp.asarray(parent[lv])]
        A[cp.asarray(pbr[lv]), cp.asarray(lv)] = 1.0
    zg = cp.asarray(z)
    Zbus = (A.T * zg) @ A  # (n,m)*(m,) -> A^T diag(z) A
    Vp = cp.ones(n, cp.complex128)
    for pit in range(200):
        Iinj = cp.conj(Sg / Vp); Iinj[s0] = 0
        Vn_ = 1 + Zbus @ Iinj
        if float(cp.abs(Vn_ - Vp).max()) < 1e-14: Vp = Vn_; break
        Vp = Vn_
    Zth_p = A.T @ zg
    Iinj = cp.conj(Sg / Vp); Iinj[s0] = 0
    Ib_tree = A @ Iinj  # current flowing away from GRID along each branch
    # orient to from->to convention: branch b is the parent branch of node child(b)
    child = np.zeros(m, int); child[pbr[pbr >= 0]] = np.where(pbr >= 0)[0]
    Ib_p = -Ib_tree * cp.asarray(orient[child])  # generation: current flows toward GRID
    cp.cuda.Stream.null.synchronize(); tp = time.perf_counter() - t0
    Vp_ = cp.asnumpy(Vp); Ztp = cp.asnumpy(Zth_p); Ibp = cp.asnumpy(Ib_p)
    # ---------- CPU witness: plain python sweep on a seeded sample ----------
    rng = random.Random(20260927)
    ch = {i: [] for i in range(n)}
    for v_ in range(n):
        if parent[v_] >= 0: ch[parent[v_]].append(v_)
    Vw = np.ones(n, complex)
    post = order[::-1]
    for _ in range(100):
        Ic = np.conj(S / Vw); Ic[s0] = 0; Isub = Ic.copy()
        for u in post:
            if parent[u] >= 0: Isub[parent[u]] += Isub[u]
        Vnw = Vw.copy()
        for u in order[1:]: Vnw[u] = Vnw[parent[u]] + z[pbr[u]] * Isub[u]
        if np.abs(Vnw - Vw).max() < 1e-14: Vw = Vnw; break
        Vw = Vnw
    samp = sorted(rng.sample(range(n), min(64, n)))
    wit = pair(np.abs(Ve[samp]), np.abs(Vw[samp]), 1e-9, "pu", "|V|: GPU Newton vs CPU python sweep, 64 seeded nodes")
    # ---------- receipts from premerge-sld ----------
    rp = f"{SLD}/results-{case}.json"; jp = f"{SLD}/js-{case}.json"
    rec = {}
    for tag, p in (("results", rp), ("js", jp)):
        if os.path.exists(p):
            r = json.load(open(p)); rows = {q[0]: q for q in r["nodes"]}
            common = [i for i, k in enumerate(ids) if k in rows]
            vm_r = np.array([rows[ids[i]][1] for i in common]); zr = np.array([complex(rows[ids[i]][3], rows[ids[i]][4]) for i in common])
            rec[tag] = {"file_sha256": fsha(p), "nodes": len(common),
                        "V": pair(np.abs(Ve[common]), vm_r, 1e-8, "pu", f"|V| electron vs {tag} receipt"),
                        "Zth": pair(np.abs(Zte[common]), np.abs(zr), 1e-8, "pu", f"|Zth| electron vs {tag} receipt")}
    # ---------- engineering numbers ----------
    Ibase = Sb * 1e6 / (math.sqrt(3) * kv * 1e3)  # A per node kv
    Ik_e = np.where(np.arange(n) == s0, np.nan, cfac / np.maximum(np.abs(Zte), 1e-30) * Ibase / 1e3)
    Ik_p = np.where(np.arange(n) == s0, np.nan, cfac / np.maximum(np.abs(Ztp), 1e-30) * Ibase / 1e3)
    land = ix["LAND"]
    rise_e = (np.abs(Ve) - abs(Ve[land])) / abs(Ve[land]) * 100
    rise_p = (np.abs(Vp_) - abs(Vp_[land])) / abs(Vp_[land]) * 100
    Ibb = Sb * 1e6 / (math.sqrt(3) * bkv * 1e3)
    IA_e = np.abs(Ibe) * Ibb; IA_p = np.abs(Ibp) * Ibb
    by = {}
    for kd in sorted(set(bkind)):
        sel = np.array([k == kd for k in bkind])
        j = int(np.argmax(IA_e * sel))
        by[kd] = {"count": int(sel.sum()), "kv": float(bkv[j]), "max_A_electron": float(IA_e[j]), "max_A_positron": float(IA_p[j]),
                  "median_A": float(np.median(IA_e[sel]))}
    boards = [i for i in range(n) if kind[i] == "mv-board"]
    lvb = [i for i in range(n) if kind[i] == "lv-board"]
    loss = float(np.sum(np.abs(Ibe) ** 2 * z.real) * Sb)
    gen = float(S.real.sum() * Sb)
    out = {"case": case, "label": "Illustrative, not a design for any site.", "graph_sha256": fsha(gp),
           "nodes": n, "closed_branches": m, "open_branches": len(g["branches"]) - m, "gen_MW": gen,
           "electron": {"method": "Newton-Raphson, dense complex Jacobian on Y (CuPy)", "iterations": nr_it, "seconds": te},
           "positron": {"method": "path matrix A from topology, Zbus = A^T diag(z) A, fixed-point injection (CuPy)", "iterations": pit + 1, "seconds": tp},
           "pairs": {"V": pair(np.abs(Ve), np.abs(Vp_), 1e-9, "pu", "|V| NR vs path fixed point"),
                     "angle": pair(np.angle(Ve), np.angle(Vp_), 1e-9, "rad", "angle NR vs path fixed point"),
                     "Zth": pair(np.abs(Zte), np.abs(Ztp), 1e-9, "pu", "|Zth| diag inv(Y) vs path sum"),
                     "Ibranch": pair(IA_e, IA_p, 1e-6, "A", "branch current (Vf-Vt)/z vs downstream injection sum"),
                     "fault_kA": pair(Ik_e[np.isfinite(Ik_e)], Ik_p[np.isfinite(Ik_p)], 1e-6, "kA", "Ik'' both")},
           "witness": wit, "receipts": rec,
           "worst": {"rise_to_landing_pct": {"electron": float(rise_e.max()), "positron": float(rise_p.max()), "at": ids[int(np.argmax(rise_e))]},
                     "max_V_pu": float(np.abs(Ve).max()),
                     "landing_V_pu_400kV": float(abs(Ve[land])),
                     "mv_board_fault_kA": [round(float(Ik_e[i]), 3) for i in boards],
                     "lv_board_fault_kA_min_max": [float(np.nanmin(Ik_e[lvb])), float(np.nanmax(Ik_e[lvb]))] if lvb else None,
                     "landing_fault_kA_400kV": float(Ik_e[land]),
                     "losses_MW": loss, "losses_pct_of_gen": 100 * loss / gen if gen else None},
           "branch_currents_by_kind": by}
    photons = sum(p["photons"] for p in out["pairs"].values()) + wit["photons"]
    out["photons"] = photons
    del Y, J, A, Zbus; cp.get_default_memory_pool().free_all_blocks()
    return out


def cables():
    cc = f"{V08}/sites/open-land-01/cables/cable-checks.json"
    d = json.load(open(cc))
    js = ("import {parseChecks,defaultChoice,acChecks,dcChecks} from '" + pathlib.Path(V08, "cable-checks.mjs").as_uri() + "';"
          "import fs from 'fs';const c=parseChecks(JSON.parse(fs.readFileSync('" + cc + "','utf8')));"
          "const ch=defaultChoice(c);const A=acChecks(c,ch),D=dcChecks(c);"
          "console.log(JSON.stringify({choice:ch,ac:A.circuits.map(x=>x.vd_pct),acload:A.circuits.map(x=>x.loading_pct),rating:A.rating_A,"
          "dc:D.circuits.map(x=>x.vd_pct),dcload:D.circuits.map(x=>x.loading_pct),acsum:A.summary,dcsum:D.summary}));")
    t0 = time.perf_counter()
    r = subprocess.run(["node", "--input-type=module", "-e", js], capture_output=True, text=True)
    if r.returncode: raise RuntimeError(r.stderr[-800:])
    J = json.loads(r.stdout); tjs = time.perf_counter() - t0
    a = d["assumed"]; dcc = d["dc"]["columns"]; acc = d["ac"]["columns"]
    DC = np.array([[row[dcc.index(k)] for k in ("length_m", "current_A", "design_current_A", "rating_A")] for row in d["dc"]["rows"]], float)
    AC = np.array([[row[acc.index(k)] for k in ("length_m", "current_A", "design_current_A")] for row in d["ac"]["rows"]], float)
    t0 = time.perf_counter()
    g = cp.asarray(DC)
    Rdc = a["dc"]["r20_ohm_km"] * (1 + a["dc"]["alpha"] * (a["dc"]["theta_C"] - 20))
    vd_dc = g[:, 1] * g[:, 0] / 1e3 * Rdc / a["string_vmp_V"] * 100
    ld_dc = g[:, 2] / g[:, 3] * 100
    # string current from first principles: P_string / Vmp, P from 30 x 660 W (world explorer rule)
    I_str = cp.float64(30 * 660.0) / a["string_vmp_V"]
    ch = J["choice"]; mat, size = ch["mat"], str(ch["size"]); pf = min(1, max(0.5, ch["pf"])); sn = math.sqrt(1 - pf * pf)
    Rac = a["ac"]["r20_ohm_km"][mat][size] * (1 + a["ac"]["alpha"][mat] * (a["ac"]["theta_C"] - 20))
    h = cp.asarray(AC)
    vd_ac = math.sqrt(3) * h[:, 1] * h[:, 0] / 1e3 * (Rac * pf + a["ac"]["x_ohm_km"] * sn) / a["ac"]["volts"] * 100
    I_ac = cp.float64(a["ac"]["kva"] * 1e3) / (math.sqrt(3) * a["ac"]["volts"])
    cp.cuda.Stream.null.synchronize(); tg = time.perf_counter() - t0
    vdc, ldc, vac = cp.asnumpy(vd_dc), cp.asnumpy(ld_dc), cp.asnumpy(vd_ac)
    per = {}
    for row, v in zip(d["dc"]["rows"], vdc): per[(row[1], row[2])] = per.get((row[1], row[2]), 0) + v
    wi = random.Random(20260927).sample(range(len(DC)), 64)
    wit = pair([DC[i, 1] * DC[i, 0] / 1e3 * Rdc / a["string_vmp_V"] * 100 for i in wi], vdc[wi], 1e-12, "%", "DC drop: CPU python vs GPU, 64 seeded circuits")
    ld_ac = np.array(J["acload"], float)
    return {"label": "Illustrative check, assumptions editable; not a design.", "schedule_sha256": fsha(cc),
            "assumed": a, "choice": ch,
            "string_dc": {"circuits": len(DC), "strings": len(per),
                          "current_A": {"scheduled": float(DC[:, 1].max()), "positron_P_over_Vmp": float(I_str)},
                          "design_current_A_max": float(DC[:, 2].max()),
                          "vd_pct_conductor": {"min": float(vdc.min()), "max": float(vdc.max())},
                          "vd_pct_string_pos+neg": {"min": min(per.values()), "max": max(per.values())},
                          "loading_pct": {"min": float(ldc.min()), "max": float(ldc.max()), "above100": int((ldc > 100).sum())},
                          "pair_vd": pair(J["dc"], vdc, 1e-9, "%", "DC drop: world JS dcChecks vs CuPy I L R(theta)/Vmp"),
                          "pair_load": pair(J["dcload"], ldc, 1e-9, "%", "DC loading: JS vs CuPy"),
                          "pair_current": pair([DC[:, 1].max()], [float(I_str)], 0.01, "A", "string current: schedule vs 30x660W/Vmp")},
            "inverter_ac": {"circuits": len(AC), "kv": a["ac"]["volts"] / 1e3,
                            "current_A": {"scheduled": float(AC[:, 1].max()), "positron_S_over_sqrt3V": float(I_ac)},
                            "vd_pct": {"min": float(vac.min()), "max": float(vac.max())},
                            "rating_A": J["rating"], "loading_pct_max": float(np.nanmax(ld_ac)) if np.isfinite(ld_ac).any() else None,
                            "pair_vd": pair(J["ac"], vac, 1e-9, "%", "AC drop: world JS acChecks vs CuPy sqrt3 I L (R pf + X sin)/V"),
                            "pair_current": pair([AC[:, 1].max()], [float(I_ac)], 0.01, "A", "AC current: schedule vs S/(sqrt3 V)")},
            "witness": wit, "seconds": {"js": tjs, "gpu": tg}}


def main():
    wait = take_lock()
    try:
        hw = heat_wait()
        cases = [c for c in ["p10", "p50", "p100", "p250", "p500", "p1000", "block4"] if os.path.exists(f"{SLD}/graph-{c}.json")]
        res = []
        for c in cases:
            hw += heat_wait()
            r = sld_case(c); r["temp_c_after"] = temp_c(); res.append(r)
            print(c, r["photons"], r["worst"]["rise_to_landing_pct"], r["worst"]["mv_board_fault_kA"][:3], flush=True)
        cb = cables()
        peak = cp.get_default_memory_pool().total_bytes()
    finally:
        os.rmdir(LOCK)
    summary = {"schema": "p4-cables-loadflow/1", "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               "cupy": cp.__version__, "lock_wait_s": wait, "heat_pause_s": hw, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "sld": res, "cables": cb,
               "photons_total": sum(r["photons"] for r in res) + cb["string_dc"]["pair_vd"]["photons"] + cb["string_dc"]["pair_load"]["photons"]
               + cb["inverter_ac"]["pair_vd"]["photons"] + cb["witness"]["photons"]}
    summary["sha256"] = sha(summary)
    with open(f"{OUT}/results.json", "w", newline="\n") as fh: json.dump(summary, fh, indent=1)
    print("photons_total", summary["photons_total"], "sha", summary["sha256"])


if __name__ == "__main__":
    main()
