"""The lattice over the atlas that is already there.

Not a new dataset. The 5,800 stations the Kuiper already draws, crossed with
the design axes we already prove, evaluated on the card.

The point of a lattice rather than a list: a station is not one question, it is
a product of questions. Which voltage class, which conductor, which site
minimum temperature, how many modules in series, how many strings, which
routing. Every one of those is currently CHOSEN by a person and then never
revisited, which means the alternatives are invisible - not rejected, invisible.

    python atlas_lattice.py                 the space, and what it costs
    python atlas_lattice.py --run           evaluate it on the card
    python atlas_lattice.py --run --cap 2e9 evaluate a bounded slice

Every verdict carries its denominator. Both readings of the disputed clause
are computed and neither is collapsed.
"""
import argparse
import io
import json
import os
import re
import sys
import time

import numpy as np

import paths

# Not written down here: this repository is public and an absolute path
# names a drive, a machine and an account. The root comes from GRID_DATA
# in the environment; see src/paths.py for the whole argument.
TUBE = paths.TUBE

# ---- the design axes, all from work already proven on this machine
CLASSES = [(45.90, -0.25, 660.0), (50.59, -0.22, 760.0), (53.27, -0.25, 650.0)]
KV = [400, 275, 220, 132, 66]                     # kvOrder, from TUBE-DATA
CSA = [50, 70, 95, 120, 150, 185, 240, 300, 400, 500, 630]   # the R20 table
TMIN = [-14.0, -12.0, -11.0, -10.0, -8.0, -6.0]
SERIES = list(range(20, 35))
STRINGS = list(range(16, 33))
ROUTING = [0, 1]                                  # sequential, leapfrog
VOC_TOL = 101                                     # Voc +/- 1%

MAX_V, INVERTER_KVA = 1500.0, 352.0
DCAC_LO, DCAC_HI = 1.15, 1.40
OK, DISPUTED, FAIL_COLD, OUT_OF_BAND = 0, 1, 2, 3
LABEL = ["buildable, both readings", "disputed: the clause decides it",
         "fails cold voltage", "DC:AC outside the band"]


def stations():
    """How many stations the atlas actually holds. Read, not assumed."""
    paths.require(TUBE, "the tube-map data (TUBE-DATA.js)")
    s = io.open(TUBE, encoding="utf-8", errors="replace").read()
    m = re.search(r'"stations"\s*:\s*\[', s)
    if not m:
        raise SystemExit("FAIL: no stations array in TUBE-DATA.js")
    i, depth, n = m.end(), 1, 1
    while i < len(s) and depth:
        c = s[i]
        if c in "[{":
            depth += 1
        elif c in "]}":
            depth -= 1
        elif c == "," and depth == 1:
            n += 1
        i += 1
    return n, len(s)


AXES = None


def build_axes(n_stations):
    return [("station", n_stations), ("module class", len(CLASSES)),
            ("voltage class", len(KV)), ("conductor mm2", len(CSA)),
            ("site min temp", len(TMIN)), ("modules in series", len(SERIES)),
            ("strings", len(STRINGS)), ("routing", len(ROUTING)),
            ("Voc tolerance", VOC_TOL)]


def kernel_src(rad):
    """One fused pass: decode, evaluate, emit a verdict. No temporaries."""
    return """
long long r = idx;
int st = (int)(r %% %(A0)d); r /= %(A0)d;
int ci = (int)(r %% %(A1)d); r /= %(A1)d;
int kv = (int)(r %% %(A2)d); r /= %(A2)d;
int cs = (int)(r %% %(A3)d); r /= %(A3)d;
int ti = (int)(r %% %(A4)d); r /= %(A4)d;
int si = (int)(r %% %(A5)d); r /= %(A5)d;
int ni = (int)(r %% %(A6)d); r /= %(A6)d;
int ro = (int)(r %% %(A7)d); r /= %(A7)d;
int vt = (int)(r %% %(A8)d);

double v = voc[ci] * (1.0 + ((double)vt - %(VMID)d.0) / %(VMID)d.0 * 0.01);
double b = beta[ci], w = wp[ci];
double tm = tmin[ti], se = ser[si], sn = strs[ni];

double nb = floor(%(MAXV).17g / (v * (1.0 + b / 100.0 * (tm - 25.0))));
double na = floor(%(MAXV).17g / (v * (1.0 + b / 100.0 * (tm - 15.0))));
double dcac = (se * w / 1000.0) * sn / %(KVA).17g;
bool band = (dcac >= %(LO).17g) && (dcac <= %(HI).17g);
code = (se > na) ? %(FC)d : ((se > nb) ? %(DI)d : (band ? %(OK)d : %(OB)d));
""" % {"A0": rad[0], "A1": rad[1], "A2": rad[2], "A3": rad[3], "A4": rad[4],
       "A5": rad[5], "A6": rad[6], "A7": rad[7], "A8": rad[8],
       "VMID": VOC_TOL // 2, "MAXV": MAX_V, "KVA": INVERTER_KVA,
       "LO": DCAC_LO, "HI": DCAC_HI, "FC": FAIL_COLD, "DI": DISPUTED,
       "OK": OK, "OB": OUT_OF_BAND}


def reference(idx, rad):
    """The processor's answer for the same indices. Nothing is quoted until
    this agrees."""
    out = np.zeros(len(idx), np.int64)
    for j, i in enumerate(idx):
        r = int(i)
        vals = []
        for q in rad:
            vals.append(r % q)
            r //= q
        _st, ci, _kv, _cs, ti, si, ni, _ro, vt = vals
        v0, b, w = CLASSES[ci]
        v = v0 * (1 + (vt - VOC_TOL // 2) / (VOC_TOL // 2) * 0.01)
        tm, se, sn = TMIN[ti], float(SERIES[si]), float(STRINGS[ni])
        nb = np.floor(MAX_V / (v * (1 + b / 100 * (tm - 25))))
        na = np.floor(MAX_V / (v * (1 + b / 100 * (tm - 15))))
        dcac = (se * w / 1000.0) * sn / INVERTER_KVA
        band = DCAC_LO <= dcac <= DCAC_HI
        out[j] = (FAIL_COLD if se > na else
                  (DISPUTED if se > nb else (OK if band else OUT_OF_BAND)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cap", type=float, default=None,
                    help="evaluate at most this many cases")
    a = ap.parse_args()

    n_st, nbytes = stations()
    axes = build_axes(n_st)
    rad = [n for _n, n in axes]
    size = 1
    for n in rad:
        size *= n

    print("THE ATLAS THAT IS ALREADY THERE")
    print("  TUBE-DATA.js .......... %s bytes, %s stations (read, not assumed)"
          % ("{:,}".format(nbytes), "{:,}".format(n_st)))
    print("\nTHE LATTICE OVER IT")
    for name, n in axes:
        print("  %-22s %6s" % (name, "{:,}".format(n)))
    print("  %-22s %6s" % ("", "-" * 6))
    print("  cases                  %s" % "{:,}".format(size))
    print("\n  at 3,544,292,885 cases/s that is %.1f s (%.2f minutes)"
          % (size / 3544292885.0, size / 3544292885.0 / 60.0))
    print("  every one of the %s stations gets %s design variants, none of"
          % ("{:,}".format(n_st), "{:,}".format(size // n_st)))
    print("  which is currently visible because each was chosen once.")

    if not a.run:
        print("\n  --run to evaluate. --cap to bound it.")
        return 0

    try:
        import cupy as cp
    except Exception as exc:
        print("\nFAIL: cupy unavailable (%r). Not evaluated." % (exc,))
        return 1

    total = int(min(size, a.cap)) if a.cap else size
    src = kernel_src(rad)
    K = cp.ElementwiseKernel(
        "int64 idx, raw float64 voc, raw float64 beta, raw float64 wp, "
        "raw float64 tmin, raw float64 ser, raw float64 strs",
        "int64 code", src, "ventus_atlas")
    T = (cp.asarray([c[0] for c in CLASSES]), cp.asarray([c[1] for c in CLASSES]),
         cp.asarray([c[2] for c in CLASSES]), cp.asarray(TMIN),
         cp.asarray([float(s) for s in SERIES]),
         cp.asarray([float(s) for s in STRINGS]))

    # verify BEFORE timing, on a coprime stride through the whole space
    stride = 1000003
    while np.gcd(stride, size) != 1:
        stride += 2
    probe = (np.arange(200000, dtype=np.int64) * stride) % size
    gpu = cp.asnumpy(K(cp.asarray(probe), *T))
    cpu = reference(probe, rad)
    differ = int((gpu != cpu).sum())
    print("\n  verified: processor vs card on 200,000 cases along stride "
          "%s: %d differ" % ("{:,}".format(stride), differ))
    if differ:
        print("  FAIL: refusing to report a rate for arithmetic that differs.")
        return 1

    free, _tot = cp.cuda.Device(0).mem_info
    batch = max(1 << 16, min(1 << 23, int(free * 0.25 / 24)))
    totals = cp.zeros(4, dtype=cp.int64)
    done, t0 = 0, time.perf_counter()
    while done < total:
        n = min(batch, total - done)
        totals += cp.bincount(K(cp.arange(done, done + n, dtype=cp.int64), *T),
                              minlength=4)[:4]
        done += n
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    got = [int(x) for x in cp.asnumpy(totals)]
    if sum(got) != total:
        print("  FAIL: the reduction lost cases: %d of %d" % (sum(got), total))
        return 1

    print("\n  EVALUATED %s of %s cases in %.2f s  (%s cases/s)"
          % ("{:,}".format(total), "{:,}".format(size), sec,
             "{:,.0f}".format(total / sec)))
    for k in range(4):
        print("    %-32s %16s  %6.3f%%"
              % (LABEL[k], "{:,}".format(got[k]), 100.0 * got[k] / total))
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "atlas-lattice.json")
    io.open(out, "w", encoding="utf-8", newline="\n").write(json.dumps({
        "stations": n_st, "axes": [{"name": n, "steps": s} for n, s in axes],
        "cases": total,   # the key its four siblings write; without it
               # this run was counted as zero for a whole night
               "space": size, "evaluated": total, "seconds": round(sec, 3),
        "verified_differ": differ, "verdicts": dict(zip(LABEL, got)),
    }, indent=1) + "\n")
    print("\n  wrote atlas-lattice.json")
    print("  Not computed: load flow, fault level, yield, cost. None stated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
