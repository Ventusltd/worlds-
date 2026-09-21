"""500 MW down to one substation, every loop enumerated.

ANONYMISED. The hierarchy is ordinary practice, taken as a shape and nothing
else: module -> string -> inverter at 0.8 kV -> block -> one 33 kV collection
substation -> 400 kV interface. No project, client, site, maker or model
appears here or in any output. 500 MW DC is not the size of anything real.

WHAT IS ENUMERATED, AND WHY IT IS THE LOOP
Every string is a loop. Its inductance is DERIVED from geometry - conductor
separation, conductor radius, and the R6 length that is ours and proven -
never swept as an invented quantity. That loop decides three things at once:

  the cold-voltage verdict        does the string breach the 1500 V rating
  the DC:AC verdict               against the AMBIENT-DERATED inverter rating
  the transient at disconnection  V = L*I/t, which the arc suppressor must eat

Those have always been three separate arguments. They are one geometry.

    python substation.py            the space
    python substation.py --run      enumerate it on the card
"""
import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np

MODULES = r"E:\swarm\feed\MODULES.json"
MU0 = 4e-7 * math.pi

# ---- the hierarchy, from the drawing's shape only
TARGET_DC_MW = 500.0
BLOCK_MW = 9.856
AC_V, COLLECT_KV, GRID_KV = 800.0, 33.0, 400.0
BLOCKS_PER_SUBSTATION = [4, 6, 8]          # one 33 kV substation serves these

# ---- axes
SERIES = list(range(20, 35))                          # 15
STRINGS = [24, 28, 32]                                # 3 machine variants
INV_KVA = [352.0, 320.0, 295.0]                       # 30 / 40 / 50 C derated
CSA = [4.0, 6.0, 10.0]                                # 3
RAD_M = [math.sqrt(c / 1e6 / math.pi) for c in CSA]
N_SEP, SEP_LO, SEP_HI = 41, 0.02, 2.00                # separation, m
N_HOME, HOME_LO, HOME_HI = 41, 5.0, 150.0             # one-way home run, m
N_BRK, BRK_LO, BRK_HI = 51, 0.1e-6, 10.0e-6           # break time, s
N_CUR, CUR_LO, CUR_HI = 41, 5.0, 45.0                 # interrupted current, A
TMIN = [-14.0, -12.0, -11.0, -10.0, -8.0, -6.0]       # 6
ROUTING = [0, 1]                                      # 0 turned, 1 sequential
PITCH = 1.303

MAX_V = 1500.0
DCAC_LO, DCAC_HI = 1.15, 1.40     # OURS. No maker and no contract publishes one.
WITH_HI = 3000.0                  # series pair of 1700 V devices

OK, DISP, COLD, BAND, TRANS = 0, 1, 2, 3, 4
LABEL = ["clears every test", "disputed: the clause decides it",
         "breaches the 1500 V rating", "DC:AC outside our chosen band",
         "transient beyond the suppressor withstand"]


def bins():
    if not os.path.isfile(MODULES):
        raise SystemExit("FAIL: %s missing. Not enumerating on remembered "
                         "numbers." % MODULES)
    d = json.load(io.open(MODULES, encoding="utf-8"))
    e = d["classes"] if isinstance(d, dict) and "classes" in d else d
    if isinstance(e, dict):
        e = list(e.values())
    out = []
    for m in e:
        def g(k):
            v = m.get(k)
            return v.get("v") if isinstance(v, dict) else v
        voc, beta, wp, isc = g("voc"), g("beta_pct_per_K"), g("wp"), g("isc")
        if None in (voc, beta, wp, isc):
            continue
        out.append((float(voc), float(beta), float(wp), float(isc)))
    if not out:
        raise SystemExit("FAIL: no usable bins.")
    return out


B = bins()
RAD = [len(B), len(SERIES), len(STRINGS), len(INV_KVA), len(CSA), N_SEP,
       N_HOME, N_BRK, N_CUR, len(TMIN), len(ROUTING)]
SIZE = int(np.prod([float(x) for x in RAD]))

SRC = """
long long r = idx;
int bi = (int)(r %% %(R0)d); r /= %(R0)d;
int si = (int)(r %% %(R1)d); r /= %(R1)d;
int ni = (int)(r %% %(R2)d); r /= %(R2)d;
int ii = (int)(r %% %(R3)d); r /= %(R3)d;
int ci = (int)(r %% %(R4)d); r /= %(R4)d;
int pi = (int)(r %% %(R5)d); r /= %(R5)d;
int hi = (int)(r %% %(R6)d); r /= %(R6)d;
int qi = (int)(r %% %(R7)d); r /= %(R7)d;
int ai = (int)(r %% %(R8)d); r /= %(R8)d;
int ti = (int)(r %% %(R9)d); r /= %(R9)d;
int ro = (int)(r %% %(R10)d);

double voc = tvoc[bi], bet = tbeta[bi], w = twp[bi];
double se = ser[si], sn = strs[ni], kva = inv[ii];
double rad = trad[ci], tm = tmin[ti];
double d    = %(SLO).17g + (%(SHI).17g - %(SLO).17g) * (double)pi / (%(R5)d - 1.0);
double home = %(HLO).17g + (%(HHI).17g - %(HLO).17g) * (double)hi / (%(R6)d - 1.0);
double brk  = %(BLO).17g + (%(BHI).17g - %(BLO).17g) * (double)qi / (%(R7)d - 1.0);
double cur  = %(CLO).17g + (%(CHI).17g - %(CLO).17g) * (double)ai / (%(R8)d - 1.0);

/* R6, ours: metres of conductor in the loop */
double len = (ro == 0) ? (2.006 * se + 2.0 * home)
                       : (0.63 * se + (se - 2.0) * %(PITCH).17g + 0.848 + 2.0 * home);
/* inductance DERIVED from geometry, never swept */
double L = (%(MU0).17g / 3.14159265358979) * (log(d / rad) + 0.25) * len;

double voc_cold = voc * (1.0 + bet / 100.0 * (tm - 25.0));
double v_str    = se * voc_cold;
double v_str_a  = se * voc * (1.0 + bet / 100.0 * (tm + 10.0 - 25.0));
double dcac     = (se * w / 1000.0) * sn / kva;
double v_kick   = L * cur / brk;

if (v_str_a > %(MAXV).17g)                       code = %(COLD)d;
else if (v_str > %(MAXV).17g)                    code = %(DISP)d;
else if (dcac < %(LO).17g || dcac > %(HI).17g)   code = %(BAND)d;
else if (v_str + v_kick > %(WHI).17g)            code = %(TRANS)d;
else                                             code = %(OK)d;
""" % {"R0": RAD[0], "R1": RAD[1], "R2": RAD[2], "R3": RAD[3], "R4": RAD[4],
       "R5": RAD[5], "R6": RAD[6], "R7": RAD[7], "R8": RAD[8], "R9": RAD[9],
       "R10": RAD[10], "SLO": SEP_LO, "SHI": SEP_HI, "HLO": HOME_LO,
       "HHI": HOME_HI, "BLO": BRK_LO, "BHI": BRK_HI, "CLO": CUR_LO,
       "CHI": CUR_HI, "PITCH": PITCH, "MU0": MU0, "MAXV": MAX_V,
       "LO": DCAC_LO, "HI": DCAC_HI, "WHI": WITH_HI, "COLD": COLD,
       "DISP": DISP, "BAND": BAND, "TRANS": TRANS, "OK": OK}


def reference(idx):
    out = np.zeros(len(idx), np.int64)
    for j, i in enumerate(idx):
        r = int(i); v = []
        for q in RAD:
            v.append(r % q); r //= q
        bi, si, ni, ii, ci, pi_, hi, qi, ai, ti, ro = v
        voc, bet, w, _isc = B[bi]
        se, sn, kva = SERIES[si], STRINGS[ni], INV_KVA[ii]
        rad, tm = RAD_M[ci], TMIN[ti]
        d = SEP_LO + (SEP_HI - SEP_LO) * pi_ / (N_SEP - 1.0)
        home = HOME_LO + (HOME_HI - HOME_LO) * hi / (N_HOME - 1.0)
        brk = BRK_LO + (BRK_HI - BRK_LO) * qi / (N_BRK - 1.0)
        cur = CUR_LO + (CUR_HI - CUR_LO) * ai / (N_CUR - 1.0)
        ln = (2.006 * se + 2 * home) if ro == 0 else \
             (0.63 * se + (se - 2) * PITCH + 0.848 + 2 * home)
        L = (MU0 / math.pi) * (math.log(d / rad) + 0.25) * ln
        vs = se * voc * (1 + bet / 100 * (tm - 25))
        va = se * voc * (1 + bet / 100 * (tm + 10 - 25))
        dcac = (se * w / 1000.0) * sn / kva
        vk = L * cur / brk
        if va > MAX_V: out[j] = COLD
        elif vs > MAX_V: out[j] = DISP
        elif dcac < DCAC_LO or dcac > DCAC_HI: out[j] = BAND
        elif vs + vk > WITH_HI: out[j] = TRANS
        else: out[j] = OK
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    names = ["module bin", "modules in series", "strings per inverter",
             "ambient-derated rating", "conductor mm2", "conductor separation",
             "home run length", "break time", "interrupted current",
             "site min temp", "routing"]
    print("500 MW DC DOWN TO ONE SUBSTATION - anonymised, every loop")
    print("  module -> string -> inverter at %.1f kV -> block of %g MW ->"
          % (AC_V / 1000, BLOCK_MW))
    print("  one %g kV collection substation -> %g kV interface"
          % (COLLECT_KV, GRID_KV))
    print("  blocks in the plant: %d    one substation serves %s"
          % (math.ceil(TARGET_DC_MW / BLOCK_MW), BLOCKS_PER_SUBSTATION))
    for nm, n in zip(names, RAD):
        print("    %-24s %6d" % (nm, n))
    print("    %-24s %6s" % ("", "------"))
    print("    %-24s %s cases" % ("", "{:,}".format(SIZE)))
    if not a.run:
        print("\n  --run to enumerate on the card.")
        return 0

    import cupy as cp
    K = cp.ElementwiseKernel(
        "int64 idx, raw float64 tvoc, raw float64 tbeta, raw float64 twp, "
        "raw float64 ser, raw float64 strs, raw float64 inv, raw float64 trad, "
        "raw float64 tmin", "int64 code", SRC, "ventus_substation")
    T = (cp.asarray(np.array([b[0] for b in B])),
         cp.asarray(np.array([b[1] for b in B])),
         cp.asarray(np.array([b[2] for b in B])),
         cp.asarray(np.array(SERIES, float)),
         cp.asarray(np.array(STRINGS, float)),
         cp.asarray(np.array(INV_KVA, float)),
         cp.asarray(np.array(RAD_M)),
         cp.asarray(np.array(TMIN, float)))

    stride = 1000003
    while np.gcd(stride, SIZE) != 1:
        stride += 2
    probe = (np.arange(200000, dtype=np.int64) * stride) % SIZE
    differ = int((cp.asnumpy(K(cp.asarray(probe), *T)) != reference(probe)).sum())
    print("\n  verified: processor vs card on 200,000 cases, %d differ" % differ)
    if differ:
        print("  FAIL: nothing reported for arithmetic that differs."); return 1

    free, _ = cp.cuda.Device(0).mem_info
    batch = max(1 << 16, min(1 << 23, int(free * 0.25 / 16)))
    tot = cp.zeros(5, dtype=cp.int64)
    done, t0 = 0, time.perf_counter()
    while done < SIZE:
        n = min(batch, SIZE - done)
        tot += cp.bincount(K(cp.arange(done, done + n, dtype=cp.int64), *T),
                           minlength=5)[:5]
        done += n
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    got = [int(x) for x in cp.asnumpy(tot)]
    if sum(got) != SIZE:
        print("  FAIL: reduction lost cases."); return 1
    print("  ENUMERATED %s in %.1f s (%s loops/s)"
          % ("{:,}".format(SIZE), sec, "{:,.0f}".format(SIZE / sec)))
    for k in range(5):
        print("    %-44s %16s %7.3f%%"
              % (LABEL[k], "{:,}".format(got[k]), 100.0 * got[k] / SIZE))

    blocks = math.ceil(TARGET_DC_MW / BLOCK_MW)
    print("\n  THE SUBSTATION, at 30 in series, 24 strings, 660 Wp:")
    for bps in BLOCKS_PER_SUBSTATION:
        mw = bps * BLOCK_MW
        inv = math.ceil(mw * 1000 / 475.2)
        print("    %d blocks = %6.2f MW DC, %3d inverters, %6s strings, %7s modules"
              % (bps, mw, inv, "{:,}".format(inv * 24), "{:,}".format(inv * 24 * 30)))
    print("    plant total: %d blocks, %d substations at 8 blocks each"
          % (blocks, math.ceil(blocks / 8)))

    out = {"cases": SIZE, "seconds": round(sec, 2), "verified_differ": differ,
           "axes": dict(zip(names, RAD)), "verdicts": dict(zip(LABEL, got)),
           "hierarchy": {"target_dc_mw": TARGET_DC_MW, "block_mw": BLOCK_MW,
                         "blocks": blocks, "ac_v": AC_V,
                         "collect_kv": COLLECT_KV, "grid_kv": GRID_KV,
                         "blocks_per_substation": BLOCKS_PER_SUBSTATION},
           "inductance": "DERIVED from conductor separation, conductor radius "
                         "and R6 length. Never swept as an invented quantity.",
           "dcac_band": "1.15-1.40 is OURS. No manufacturer and no contract "
                        "read this night publishes one.",
           "what_this_is_not": "DC sizing and a first-order transient screen. "
                               "Not a design, not a study, not a connection "
                               "assessment. No AC analysis, no protection "
                               "coordination, no clamping or snubber modelled.",
           "anonymised": "Architecture only. No project, client, site, maker "
                         "or model. 500 MW DC is not the size of anything real."}
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "night-results", "substation.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("\n  wrote night-results/substation.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
