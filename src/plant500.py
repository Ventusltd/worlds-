"""A 500 MW DC plant, anonymised, with the module rating as a swept axis.

THE PROVENANCE, STATED PLAINLY
The ARCHITECTURE is taken from an operational project's single line diagram:
module -> string -> inverter at 0.8 kV -> collection at 33 kV -> grid at
400 kV, gathered into blocks of roughly 9.9 MW. That shape is ordinary
engineering practice and is what is reused here.

NOTHING IDENTIFYING IS REUSED. No project, client, consultant, site, drawing
number, equipment maker or model appears in this file, its output, or anything
derived from it. The plant modelled here is 500 MW DC, which is not the size of
the source, and every quantity is recomputed from the axes below rather than
copied. The source document does not leave this machine.

THE AXES - all of them the client's to change
  module rating   250 to 800 Wp, which is the whole point: a plant designed
                  around 660 Wp modules is a different plant at 250 and at 800,
                  and nobody sees the difference because the number is chosen
                  once at the start and never revisited
  modules in series, strings per inverter, site minimum temperature,
  inverter class, module class

    python plant500.py                 the space and what it costs
    python plant500.py --run           enumerate it on the card
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np

# ---- the anonymised architecture, from the SLD's structure only
TARGET_DC_MW = 500.0
AC_VOLTS = 800.0            # inverter terminal, 0.8 kV
COLLECT_KV = 33.0
GRID_KV = 400.0
BLOCK_MW = 9.856            # the repeated block size in the source topology

# ---- module classes we own, plus the swept rating
CLASSES = [(45.90, -0.25), (50.59, -0.22), (53.27, -0.25)]   # Voc, beta %/K
WP_LO, WP_HI, WP_STEP = 250, 800, 10                          # 56 values
TMIN = [-14.0, -12.0, -11.0, -10.0, -8.0, -6.0]
SERIES = list(range(20, 35))          # 15
STRINGS = list(range(16, 37))         # 21
# FROM THE DATASHEET, read 2026-09-21. The AC rating DERATES with ambient,
# which the band analysis must use: same array, different verdict.
INVERTER_KVA = [352.0, 320.0, 295.0]      # @30C, @40C, @50C - one machine
AMBIENT = [30, 40, 50]
# variants: (MPPT count, max A per MPPT, connectors per MPPT) -> strings
VARIANTS = [(12, 40, 2), (14, 30, 2), (16, 30, 2)]   # 24 / 28 / 32 strings
STRING_A_LIMIT = 20.0                      # 20 A per string, datasheet
MPP_LO, MPP_HI = 500.0, 1500.0             # MPP window, datasheet

MAX_V = 1500.0
DCAC_LO, DCAC_HI = 1.15, 1.40
OK, DISPUTED, FAIL_COLD, OUT_OF_BAND = 0, 1, 2, 3
LABEL = ["buildable, both readings", "disputed: the clause decides it",
         "fails cold voltage", "DC:AC outside the band"]

WP = list(range(WP_LO, WP_HI + 1, WP_STEP))
RAD = [len(CLASSES), len(WP), len(TMIN), len(SERIES), len(STRINGS),
       len(INVERTER_KVA)]
SIZE = int(np.prod(RAD))

SRC = """
long long r = idx;
int ci = (int)(r %% %(A0)d); r /= %(A0)d;
int wi = (int)(r %% %(A1)d); r /= %(A1)d;
int ti = (int)(r %% %(A2)d); r /= %(A2)d;
int si = (int)(r %% %(A3)d); r /= %(A3)d;
int ni = (int)(r %% %(A4)d); r /= %(A4)d;
int ii = (int)(r %% %(A5)d);

double v = voc[ci], b = beta[ci];
double w = wp[wi], tm = tmin[ti], se = ser[si], sn = strs[ni], kva = inv[ii];

double nb = floor(%(MAXV).17g / (v * (1.0 + b / 100.0 * (tm - 25.0))));
double na = floor(%(MAXV).17g / (v * (1.0 + b / 100.0 * (tm - 15.0))));
double kw_inv = se * w / 1000.0 * sn;
double dcac = kw_inv / kva;
bool band = (dcac >= %(LO).17g) && (dcac <= %(HI).17g);
code = (se > na) ? %(FC)d : ((se > nb) ? %(DI)d : (band ? %(OK)d : %(OB)d));
""" % {"A0": RAD[0], "A1": RAD[1], "A2": RAD[2], "A3": RAD[3], "A4": RAD[4],
       "A5": RAD[5], "MAXV": MAX_V, "LO": DCAC_LO, "HI": DCAC_HI,
       "FC": FAIL_COLD, "DI": DISPUTED, "OK": OK, "OB": OUT_OF_BAND}


def decode(i):
    out, r = [], int(i)
    for q in RAD:
        out.append(r % q)
        r //= q
    return out


def plant(ci, wi, ti, si, ni, ii):
    """Everything the client actually asks for, recomputed from the axes."""
    w, se, sn, kva = WP[wi], SERIES[si], STRINGS[ni], INVERTER_KVA[ii]
    kw_string = se * w / 1000.0
    kw_inv = kw_string * sn
    inverters = int(np.ceil(TARGET_DC_MW * 1000.0 / kw_inv))
    return {
        "module_wp": w, "series": se, "strings_per_inverter": sn,
        "inverter_kva": kva,
        "kw_dc_string": round(kw_string, 3),
        "kw_dc_inverter": round(kw_inv, 2),
        "dc_ac": round(kw_inv / kva, 4),
        "inverters_for_500MW": inverters,
        "modules_total": inverters * sn * se,
        "strings_total": inverters * sn,
        "mated_pairs_total": inverters * sn * 2,
        "mva_ac_total": round(inverters * kva / 1000.0, 1),
        "blocks_at_%.3f_MW" % BLOCK_MW: int(np.ceil(TARGET_DC_MW / BLOCK_MW)),
    }


def reference(idx):
    out = np.zeros(len(idx), np.int64)
    for j, i in enumerate(idx):
        ci, wi, ti, si, ni, ii = decode(i)
        v, b = CLASSES[ci]
        tm, se, sn, w, kva = TMIN[ti], SERIES[si], STRINGS[ni], WP[wi], INVERTER_KVA[ii]
        nb = np.floor(MAX_V / (v * (1 + b / 100 * (tm - 25))))
        na = np.floor(MAX_V / (v * (1 + b / 100 * (tm - 15))))
        dcac = (se * w / 1000.0) * sn / kva
        band = DCAC_LO <= dcac <= DCAC_HI
        out[j] = (FAIL_COLD if se > na else
                  (DISPUTED if se > nb else (OK if band else OUT_OF_BAND)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    print("A 500 MW DC PLANT - anonymised architecture, swept design")
    print("  topology reused  module -> string -> inverter at %.1f kV -> "
          "collection at %g kV -> grid at %g kV" % (AC_VOLTS / 1000, COLLECT_KV,
                                                    GRID_KV))
    print("  blocks           %g MW, as the source groups them" % BLOCK_MW)
    print("  NOTHING IDENTIFYING is reused: no project, client, site, drawing,")
    print("  maker or model, here or in any output.")
    print()
    print("  THE AXES, all of them the client's to change")
    for nm, n, rng in (("module class", RAD[0], "3 owned classes"),
                       ("module rating", RAD[1], "%d-%d Wp in %d Wp steps"
                        % (WP_LO, WP_HI, WP_STEP)),
                       ("site min temp", RAD[2], "%g to %g C" % (TMIN[0], TMIN[-1])),
                       ("modules in series", RAD[3], "%d-%d" % (SERIES[0], SERIES[-1])),
                       ("strings per inverter", RAD[4], "%d-%d" % (STRINGS[0], STRINGS[-1])),
                       ("inverter class", RAD[5], "352 / 465 / 500")):
        print("    %-22s %5d   %s" % (nm, n, rng))
    print("    %-22s %5s" % ("", "-----"))
    print("    %-22s %s cases" % ("", "{:,}".format(SIZE)))

    # the reference design, at the source's own module rating
    ref = plant(0, WP.index(660), 3, SERIES.index(30), STRINGS.index(24), 0)
    print("\n  AT 660 Wp, 30 IN SERIES, 24 STRINGS, 352 kVA CLASS")
    for k, v in ref.items():
        print("    %-24s %s" % (k, "{:,}".format(v) if isinstance(v, int) else v))

    print("\n  THE SAME PLANT AT OTHER MODULE RATINGS (30 series, 24 strings)")
    print("    %5s %10s %12s %12s %10s" % ("Wp", "kW/string", "inverters",
                                           "modules", "DC:AC"))
    for w in (250, 400, 550, 660, 700, 800):
        p = plant(0, WP.index(w), 3, SERIES.index(30), STRINGS.index(24), 0)
        print("    %5d %10.2f %12s %12s %10.3f"
              % (w, p["kw_dc_string"], "{:,}".format(p["inverters_for_500MW"]),
                 "{:,}".format(p["modules_total"]), p["dc_ac"]))
    print("    -> the plant needs %s modules at 250 Wp and %s at 800 Wp for the"
          % ("{:,}".format(plant(0, WP.index(250), 3, SERIES.index(30),
                                 STRINGS.index(24), 0)["modules_total"]),
             "{:,}".format(plant(0, WP.index(800), 3, SERIES.index(30),
                                 STRINGS.index(24), 0)["modules_total"])))
    print("       same 500 MW DC. That ratio is chosen once and never revisited.")

    if not a.run:
        print("\n  --run to enumerate on the card.")
        return 0

    try:
        import cupy as cp
    except Exception as exc:
        print("\nFAIL: cupy unavailable (%r). Not evaluated." % (exc,))
        return 1

    K = cp.ElementwiseKernel(
        "int64 idx, raw float64 voc, raw float64 beta, raw float64 wp, "
        "raw float64 tmin, raw float64 ser, raw float64 strs, raw float64 inv",
        "int64 code", SRC, "ventus_plant500")
    T = (cp.asarray([c[0] for c in CLASSES]), cp.asarray([c[1] for c in CLASSES]),
         cp.asarray([float(x) for x in WP]), cp.asarray(TMIN),
         cp.asarray([float(s) for s in SERIES]),
         cp.asarray([float(s) for s in STRINGS]),
         cp.asarray(INVERTER_KVA))

    stride = 1000003
    while np.gcd(stride, SIZE) != 1:
        stride += 2
    probe = (np.arange(min(200000, SIZE), dtype=np.int64) * stride) % SIZE
    differ = int((cp.asnumpy(K(cp.asarray(probe), *T)) != reference(probe)).sum())
    print("\n  verified: processor vs card on %s cases, %d differ"
          % ("{:,}".format(len(probe)), differ))
    if differ:
        print("  FAIL: refusing to report anything for arithmetic that differs.")
        return 1

    t0 = time.perf_counter()
    totals = cp.bincount(K(cp.arange(SIZE, dtype=cp.int64), *T), minlength=4)[:4]
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    got = [int(x) for x in cp.asnumpy(totals)]
    if sum(got) != SIZE:
        print("  FAIL: reduction lost cases: %d of %d" % (sum(got), SIZE))
        return 1
    print("  ENUMERATED %s cases in %.4f s" % ("{:,}".format(SIZE), sec))
    for k in range(4):
        print("    %-32s %10s  %6.2f%%"
              % (LABEL[k], "{:,}".format(got[k]), 100.0 * got[k] / SIZE))

    out = {"target_dc_mw": TARGET_DC_MW, "axes": dict(zip(
        ["module_class", "module_wp", "site_tmin", "series", "strings",
         "inverter_kva"], RAD)), "cases": SIZE, "seconds": round(sec, 4),
        "verified_differ": differ, "verdicts": dict(zip(LABEL, got)),
        "reference_design_660wp": ref,
        "anonymised": "Architecture only. No project, client, site, drawing, "
                      "maker or model is recorded. 500 MW DC is not the size "
                      "of the source.",
        "what_this_is_not": "Not a design, not a study, not a connection "
                            "assessment. DC sizing arithmetic under both "
                            "readings of the cold-voltage clause."}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "plant500.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("\n  wrote plant500.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
