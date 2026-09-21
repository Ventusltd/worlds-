"""Turn the enumeration into numbers the Kuiper can draw.

The shape is the estate's own, copied not invented: a data cartridge is a
single const assignment between sentinel comments, carrying its attribution
inside itself, with the payload as integer arrays. TUBE-DATA.js is the
precedent and this follows it exactly.

WHAT IS SHIPPED, AND WHAT IS NOT
Not the space. A billion verdicts regenerate in under a second, so shipping
them would be shipping what a function already gives. What ships is a
PROJECTION: the verdict for every combination of the five axes a reader can
actually turn, held at nominal tolerance, plus the marginal counts from the
full run behind it. 8,100 verdicts is 8 kB; the space behind it is 125 billion.

NO NAMES. No maker, model, project, client or site appears here or in the
output. Module classes are described by their electrical character and power
bin, which is what the arithmetic uses anyway.

    python to_kuiper.py
"""
import io
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hundred_billion as HB   # noqa: E402  the verified kernel and its bins

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "night-results", "SOLAR-DATA.js")

# the five axes a reader can turn; tolerances held at nominal
AX = [("module_bin", len(HB.B)), ("series", len(HB.SERIES)),
      ("strings", len(HB.STRINGS)), ("site_tmin", len(HB.TMIN)),
      ("ambient", len(HB.INV_KVA))]


def grid():
    """One verdict per turnable combination, at nominal tolerance."""
    n = int(np.prod([c for _k, c in AX]))
    idx = np.zeros(n, dtype=np.int64)
    # rebuild a full-space index with tolerance axes pinned to their middle
    vm, bm, gm = HB.N_VOC // 2, HB.N_BETA // 2, HB.N_GAMMA // 2
    k = 0
    for bi in range(len(HB.B)):
        for si in range(len(HB.SERIES)):
            for ni in range(len(HB.STRINGS)):
                for ti in range(len(HB.TMIN)):
                    for ii in range(len(HB.INV_KVA)):
                        v = [bi, si, ni, ti, 1, ii, vm, bm, gm, 0, 1]
                        lin, mul = 0, 1
                        for val, rad in zip(v, HB.RAD):
                            lin += val * mul
                            mul *= rad
                        idx[k] = lin
                        k += 1
    return idx


def main():
    idx = grid()
    print("projection: %s turnable combinations out of a space of %s"
          % ("{:,}".format(len(idx)), "{:,}".format(HB.SIZE)))

    cpu = HB.reference(idx[:4000])
    try:
        import cupy as cp
        K = cp.ElementwiseKernel(
            "int64 idx, raw float64 tvoc, raw float64 tisc, raw float64 tvmp, "
            "raw float64 twp, raw float64 tbeta, raw float64 tgamma, "
            "raw float64 tk, raw float64 ser, raw float64 strs, "
            "raw float64 tmin, raw float64 thot, raw float64 inv",
            "int64 code", HB.SRC, "ventus_100b")
        cols = list(zip(*HB.B))
        T = tuple(cp.asarray(np.array(c, float)) for c in cols) + (
            cp.asarray(np.array(HB.SERIES, float)),
            cp.asarray(np.array(HB.STRINGS, float)),
            cp.asarray(np.array(HB.TMIN, float)),
            cp.asarray(np.array(HB.THOT, float)),
            cp.asarray(np.array(HB.INV_KVA, float)))
        t0 = time.perf_counter()
        codes = cp.asnumpy(K(cp.asarray(idx), *T))
        sec = time.perf_counter() - t0
        backend = "cupy"
    except Exception as exc:
        print("  cupy unavailable (%r) - the processor did it" % (exc,))
        t0 = time.perf_counter()
        codes = HB.reference(idx)
        sec = time.perf_counter() - t0
        backend = "numpy"

    differ = int((codes[:4000] != cpu).sum())
    print("  verified: processor vs %s on 4,000 of %s cells, %d differ"
          % (backend, "{:,}".format(len(idx)), differ))
    if differ:
        print("  FAIL: nothing is written for arithmetic that differs.")
        return 1

    counts = [int((codes == k).sum()) for k in range(len(HB.LABEL))]
    bins = [{"bin_w": int(b[3]), "voc": b[0], "isc": b[1], "vmp": b[2],
             "beta_pct_per_K": b[4], "gamma_pct_per_K": b[5],
             "bifacial_k": b[6]} for b in HB.B]

    payload = {
        "attribution": "Module electrical characteristics read from published "
                       "manufacturer datasheets; inverter limits and ambient "
                       "derating read from a published inverter datasheet. No "
                       "maker, model, project or site is named, by rule. "
                       "Sizing rules are Ventus Ltd's own, reproduced against "
                       "brute force in 2520 of 2520 design cells.",
        "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_this_is": "DC sizing verdicts. Not a design, not a study, not a "
                        "connection assessment. Both readings of the "
                        "cold-voltage clause are kept and neither is collapsed.",
        "space_behind_it": HB.SIZE,
        "verdicts": {"labels": HB.LABEL, "counts_in_projection": counts},
        "axes": [{"name": k, "steps": c} for k, c in AX],
        "values": {"module_bin_w": [int(b[3]) for b in HB.B],
                   "series": HB.SERIES, "strings": HB.STRINGS,
                   "site_tmin_c": HB.TMIN,
                   "ambient_c": HB.AMBIENT,
                   "inverter_kva": HB.INV_KVA},
        "module_bins": bins,
        "order": "row-major over axes as listed; index = ((((b*S+s)*N+n)*T+t)*A+a)",
        "verdict_codes": {str(i): HB.LABEL[i] for i in range(len(HB.LABEL))},
        "grid": [int(c) for c in codes],
    }
    js = ("/* SOLAR-DATA:START */ const SOLAR_DATA = "
          + json.dumps(payload, separators=(",", ":"))
          + "; /* SOLAR-DATA:END */\n")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(js)

    import hashlib
    sha = hashlib.sha256(io.open(OUT, "rb").read()).hexdigest()
    print("  computed on %s in %.4f s" % (backend, sec))
    for i, lab in enumerate(HB.LABEL):
        print("    %-32s %7s  %6.2f%%"
              % (lab, "{:,}".format(counts[i]), 100.0 * counts[i] / len(idx)))
    print("\n  wrote %s" % os.path.basename(OUT))
    print("  %s bytes, sha256 %s" % ("{:,}".format(os.path.getsize(OUT)), sha[:40]))
    print("  cartridge shape matches TUBE-DATA.js: one const between sentinels,")
    print("  attribution inside the payload, integers for the grid.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
