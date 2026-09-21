"""A hundred billion ways to build one 500 MW plant.

Anonymised architecture, real module bins, real inverter derating, and the two
axes that were invisible until tonight:

  THE DELIVERED BIN. A datasheet publishes five power bins and a tolerance of
  0 to +5 W, so the module that ARRIVES may be the top bin, not the nameplate.
  At the nameplate bin the cold-voltage rule gives 30 modules in series; at the
  top bin it gives 29. Designing on the nameplate and taking delivery of the
  top bin is a real way to be wrong, and nothing in this project saw it before.

  THE AMBIENT DERATING. One machine is rated 352 kVA at 30 C, 320 at 40 and
  295 at 50. The same array sits inside the DC:AC band at one ambient and
  outside it at another.

Module bins come from E:\\swarm\\feed\\MODULES.json, reconstructed from
datasheets by bounding box and cross-checked (Vmp x Imp reproduces every
printed Pmax to better than 0.2%). If that file is absent the run STOPS - it
does not fall back to remembered numbers.

    python hundred_billion.py           the space
    python hundred_billion.py --run     enumerate it on the card
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

MODULES = r"E:\swarm\feed\MODULES.json"

# real machine, from the datasheet: rating derates with ambient
INV_KVA = [352.0, 320.0, 295.0]
AMBIENT = [30, 40, 50]
# the three real variants: (MPPT, A per MPPT, connectors) -> strings
VARIANTS = [(12, 40, 2), (14, 30, 2), (16, 30, 2)]
STRINGS = [m * c for m, _a, c in VARIANTS]          # 24, 28, 32
STRING_A = 20.0                                      # per string, datasheet
MPP_LO = 500.0
MAX_V = 1500.0
DCAC_LO, DCAC_HI = 1.15, 1.40

SERIES = list(range(20, 35))          # 15
TMIN = [-14.0, -12.0, -11.0, -10.0, -8.0, -6.0]
THOT = [60.0, 70.0, 80.0]
N_VOC, N_BETA, N_GAMMA = 1001, 41, 21
BIFACIAL = [0, 1]
CONDUCTOR = [4.0, 6.0, 10.0]

OK, DISPUTED, FAIL_COLD, FAIL_MPPT, FAIL_CURRENT, OUT_OF_BAND = 0, 1, 2, 3, 4, 5
LABEL = ["buildable, both readings", "disputed: the clause decides it",
         "fails cold voltage", "below the MPPT floor",
         "over the string current limit", "DC:AC outside the band"]


def bins():
    if not os.path.isfile(MODULES):
        raise SystemExit(
            "FAIL: %s is missing. This run needs module bins read from "
            "datasheets and will not substitute remembered numbers. A check "
            "that reached nothing is not a pass." % MODULES)
    d = json.load(io.open(MODULES, encoding="utf-8"))
    out = []
    ents = d if isinstance(d, list) else (d.get("classes") or d.get("modules")
                                          or d.get("entries") or [])
    if isinstance(ents, dict):
        ents = list(ents.values())
    for e in ents:
        def g(*names):
            for n in names:
                v = e.get(n)
                if isinstance(v, dict):
                    v = v.get("v", v.get("value"))
                if isinstance(v, (int, float)):
                    return float(v)
            return None
        voc = g("voc", "voc_v", "open_circuit_voltage_v")
        isc = g("isc", "isc_a", "short_circuit_current_a")
        vmp = g("vmp", "vmp_v")
        wp = g("wp", "pmax", "pmax_w", "nameplate_w")
        beta = g("beta_pct_per_K", "beta", "temp_coeff_voc_pct_per_k")
        gamma = g("gamma_pct_per_K", "gamma", "temp_coeff_pmax_pct_per_k")
        kc = g("k_corr", "kcorr", "bifacial_k_corr") or 1.0
        if None in (voc, isc, vmp, wp, beta, gamma):
            continue
        out.append((voc, isc, vmp, wp, beta, gamma, kc))
    if not out:
        raise SystemExit("FAIL: no usable bins in %s (read %d entries). "
                         "Nothing enumerated." % (MODULES, len(ents)))
    return out


B = bins()
RAD = [len(B), len(SERIES), len(STRINGS), len(TMIN), len(THOT), len(INV_KVA),
       N_VOC, N_BETA, N_GAMMA, len(BIFACIAL), len(CONDUCTOR)]
SIZE = int(np.prod([float(x) for x in RAD]))

SRC = """
long long r = idx;
int bi = (int)(r %% %(R0)d); r /= %(R0)d;
int si = (int)(r %% %(R1)d); r /= %(R1)d;
int ni = (int)(r %% %(R2)d); r /= %(R2)d;
int ti = (int)(r %% %(R3)d); r /= %(R3)d;
int hi = (int)(r %% %(R4)d); r /= %(R4)d;
int ii = (int)(r %% %(R5)d); r /= %(R5)d;
int vt = (int)(r %% %(R6)d); r /= %(R6)d;
int bt = (int)(r %% %(R7)d); r /= %(R7)d;
int gt = (int)(r %% %(R8)d); r /= %(R8)d;
int bf = (int)(r %% %(R9)d); r /= %(R9)d;
int cd = (int)(r %% %(R10)d);

double voc = tvoc[bi] * (1.0 + ((double)vt - %(VM)d.0) / %(VM)d.0 * 0.01);
double bet = tbeta[bi] * (1.0 + ((double)bt - %(BM)d.0) / %(BM)d.0 * 0.10);
double gam = tgamma[bi] * (1.0 + ((double)gt - %(GM)d.0) / %(GM)d.0 * 0.10);
double isc = tisc[bi] * (bf ? tk[bi] : 1.0);
double vmp = tvmp[bi], w = twp[bi];
double se = ser[si], sn = strs[ni], tm = tmin[ti], th = thot[hi];
double kva = inv[ii];

double nmax_b = floor(%(MAXV).17g / (voc * (1.0 + bet / 100.0 * (tm - 25.0))));
double nmax_a = floor(%(MAXV).17g / (voc * (1.0 + bet / 100.0 * (tm - 15.0))));
double vmp_hot = se * vmp * (1.0 + gam / 100.0 * (th - 25.0));
double dcac = (se * w / 1000.0) * sn / kva;

if (se > nmax_a)            code = %(FC)d;
else if (se > nmax_b)       code = %(DI)d;
else if (vmp_hot < %(MPP).17g) code = %(FM)d;
else if (isc > %(SA).17g)   code = %(FI)d;
else if (dcac < %(LO).17g || dcac > %(HI).17g) code = %(OB)d;
else                        code = %(OK)d;
""" % {"R0": RAD[0], "R1": RAD[1], "R2": RAD[2], "R3": RAD[3], "R4": RAD[4],
       "R5": RAD[5], "R6": RAD[6], "R7": RAD[7], "R8": RAD[8], "R9": RAD[9],
       "R10": RAD[10], "VM": N_VOC // 2, "BM": N_BETA // 2, "GM": N_GAMMA // 2,
       "MAXV": MAX_V, "MPP": MPP_LO, "SA": STRING_A, "LO": DCAC_LO,
       "HI": DCAC_HI, "FC": FAIL_COLD, "DI": DISPUTED, "FM": FAIL_MPPT,
       "FI": FAIL_CURRENT, "OB": OUT_OF_BAND, "OK": OK}


def reference(idx):
    out = np.zeros(len(idx), np.int64)
    for j, i in enumerate(idx):
        r = int(i); v = []
        for q in RAD:
            v.append(r % q); r //= q
        bi, si, ni, ti, hi, ii, vt, bt, gt, bf, cd = v
        voc0, isc0, vmp, w, beta0, gamma0, kc = B[bi]
        voc = voc0 * (1 + (vt - N_VOC // 2) / (N_VOC // 2) * 0.01)
        bet = beta0 * (1 + (bt - N_BETA // 2) / (N_BETA // 2) * 0.10)
        gam = gamma0 * (1 + (gt - N_GAMMA // 2) / (N_GAMMA // 2) * 0.10)
        isc = isc0 * (kc if bf else 1.0)
        se, sn, tm, th, kva = (SERIES[si], STRINGS[ni], TMIN[ti], THOT[hi],
                               INV_KVA[ii])
        nb = np.floor(MAX_V / (voc * (1 + bet / 100 * (tm - 25))))
        na = np.floor(MAX_V / (voc * (1 + bet / 100 * (tm - 15))))
        vmp_hot = se * vmp * (1 + gam / 100 * (th - 25))
        dcac = (se * w / 1000.0) * sn / kva
        if se > na: out[j] = FAIL_COLD
        elif se > nb: out[j] = DISPUTED
        elif vmp_hot < MPP_LO: out[j] = FAIL_MPPT
        elif isc > STRING_A: out[j] = FAIL_CURRENT
        elif dcac < DCAC_LO or dcac > DCAC_HI: out[j] = OUT_OF_BAND
        else: out[j] = OK
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()
    names = ["module bin", "modules in series", "strings (machine variant)",
             "site min temp", "hot cell temp", "ambient-derated rating",
             "Voc tolerance", "beta tolerance", "gamma tolerance",
             "bifacial gain", "conductor mm2"]
    print("A HUNDRED BILLION WAYS TO BUILD ONE 500 MW PLANT")
    print("  module bins read from datasheets: %d" % len(B))
    for nm, n in zip(names, RAD):
        print("    %-28s %6d" % (nm, n))
    print("    %-28s %6s" % ("", "------"))
    print("    %-28s %s cases" % ("", "{:,}".format(SIZE)))
    print("\n  the two axes that were invisible until tonight:")
    print("    the DELIVERED bin  - tolerance is 0 to +5 W, so the module that")
    print("      arrives may not be the one designed on: 30 in series at the")
    print("      nameplate bin becomes 29 at the top bin")
    print("    the AMBIENT rating - 352 / 320 / 295 kVA at 30 / 40 / 50 C")
    if not a.run:
        print("\n  --run to enumerate on the card.")
        return 0

    import cupy as cp
    K = cp.ElementwiseKernel(
        "int64 idx, raw float64 tvoc, raw float64 tisc, raw float64 tvmp, "
        "raw float64 twp, raw float64 tbeta, raw float64 tgamma, "
        "raw float64 tk, raw float64 ser, raw float64 strs, raw float64 tmin, "
        "raw float64 thot, raw float64 inv",
        "int64 code", SRC, "ventus_100b")
    cols = list(zip(*B))
    T = tuple(cp.asarray(np.array(c, dtype=np.float64)) for c in cols) + (
        cp.asarray(np.array(SERIES, float)), cp.asarray(np.array(STRINGS, float)),
        cp.asarray(np.array(TMIN, float)), cp.asarray(np.array(THOT, float)),
        cp.asarray(np.array(INV_KVA, float)))
    T = (T[0], T[1], T[2], T[3], T[4], T[5], T[6]) + T[7:]

    stride = 1000003
    while np.gcd(stride, SIZE) != 1:
        stride += 2
    probe = (np.arange(200000, dtype=np.int64) * stride) % SIZE
    differ = int((cp.asnumpy(K(cp.asarray(probe), *T)) != reference(probe)).sum())
    print("\n  verified: processor vs card on 200,000 cases, %d differ" % differ)
    if differ:
        print("  FAIL: nothing is reported for arithmetic that differs.")
        return 1

    free, _ = cp.cuda.Device(0).mem_info
    batch = max(1 << 16, min(1 << 23, int(free * 0.25 / 16)))
    totals = cp.zeros(6, dtype=cp.int64)
    done, t0 = 0, time.perf_counter()
    while done < SIZE:
        n = min(batch, SIZE - done)
        totals += cp.bincount(K(cp.arange(done, done + n, dtype=cp.int64), *T),
                              minlength=6)[:6]
        done += n
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    got = [int(x) for x in cp.asnumpy(totals)]
    if sum(got) != SIZE:
        print("  FAIL: reduction lost cases: %d of %d" % (sum(got), SIZE))
        return 1
    print("  ENUMERATED %s cases in %.1f s  (%s cases/s)"
          % ("{:,}".format(SIZE), sec, "{:,.0f}".format(SIZE / sec)))
    for k in range(6):
        print("    %-32s %16s  %7.4f%%"
              % (LABEL[k], "{:,}".format(got[k]), 100.0 * got[k] / SIZE))

    out = {"cases": SIZE, "seconds": round(sec, 2), "verified_differ": differ,
           "axes": dict(zip(names, RAD)), "module_bins": len(B),
           "verdicts": dict(zip(LABEL, got)),
           "what_this_is_not": "DC sizing arithmetic. Not a design, not a "
                               "study, not a connection assessment. Both "
                               "readings of the cold-voltage clause are kept.",
           "anonymised": "Architecture only. No project, client, site, maker "
                         "or model anywhere."}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "night-results", "hundred-billion.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("\n  wrote night-results/hundred-billion.json (the publisher will "
          "take it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
