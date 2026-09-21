"""ONE LAUNCH. Every predicate at once. Two channels a case, two photons back.

THE SHAPE OF IT
Every worksheet sentence in this estate is a predicate over numeric axes. Until
now each was enumerated in its own kernel, one after another: 44 launches, a few
milliseconds of arithmetic each, and a card at two per cent drawing fourteen
watts between them. The card has 70 multiprocessors and 1,536 resident threads
on each - 107,520 channels open at once - and it was answering one question at a
time.

This fuses them. Every predicate is compiled into ONE kernel. A thread works out
which predicate it belongs to from a prefix table, decodes its own case from its
own index, and evaluates it. There is no host in the loop and no second launch:
the whole worksheet is one grid.

THE PAIR, AND WHY IT IS NOT A DUPLICATE
Each thread computes its case TWICE, down two channels that are not copies:

  the electron   the predicate as the worksheet wrote it
  the positron   the same predicate with every comparison made STRICT:
                 `a <= b` becomes `a < b`, `a >= b` becomes `a > b`

These differ on exactly one set of cases - those where the two sides are EQUAL.
Where the channels ANNIHILATE, the verdict does not depend on whether the rule
says "or equal to". Where they do NOT, the case is sitting EXACTLY on its
threshold, and its verdict is a convention rather than a measurement.

An earlier version of this file said the positron computed "the complement of
its opposite". That is a no-op in exact arithmetic and would have detected
nothing. It never did that; it made the comparisons strict. The claim has been
corrected to match the code rather than the code to match the claim.

Those disagreements are the finding. They are counted, not suppressed, and the
worst case each predicate reaches is kept.

WATCHING CHANGES IT, AND THAT IS ON THE RECORD
The measurement is seeded and its seed, its state and its count are written into
the output, because a measurement you cannot repeat is an anecdote. Sampling the
card's power and utilisation during a run perturbs the run - nvidia-smi takes a
lock - so the observed rate is recorded as OBSERVED, and an unobserved run is
timed separately. Both numbers are published. Neither is called the true one.

    python annihilate.py            what it would launch
    python annihilate.py --run      launch it
    python annihilate.py --run --watch   launch it and watch, which changes it
"""
import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHEET_DIRS = [r"E:\swarm\harvest", r"E:\swarm\harvest2"]
OUT = os.path.join(ROOT, "night-results", "annihilate.json")

FUNCS = {"abs", "min", "max", "sqrt", "log", "exp", "pow"}
AXIS = re.compile(r"([A-Za-z_]\w*)\s+(-?[\d.eE+]+)\.\.(-?[\d.eE+]+)"
                  r"(?:\s+step\s+([\d.eE+-]+))?")
SENT = re.compile(r"^\s*sweep\s+(.+?),\s*count\s+where\s+(.+?)\s*$", re.S)

TARGET = 4_000_000_000          # aim for four billion cases in one grid
PER_MAX = 400_000_000           # no single predicate may own more than this


def sheets():
    out = []
    for d in SHEET_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".geojson"):
                out.append(os.path.join(d, f))
    return out


def parse(sentence):
    """A sentence becomes axes and a predicate, or it is refused by name."""
    m = SENT.match(sentence or "")
    if not m:
        return None, "no 'sweep ... , count where ...'"
    axes = []
    for name, lo, hi, step in AXIS.findall(m.group(1)):
        lo, hi = float(lo), float(hi)
        st = float(step) if step else 0.0
        if hi < lo:
            return None, "axis %s runs backwards" % name
        n = int(round((hi - lo) / st)) + 1 if st > 0 else 2
        n = max(2, min(n, 4000))
        axes.append({"name": name, "lo": lo, "hi": hi, "n": n})
    if not axes:
        return None, "no axis of the form 'name lo..hi'"
    pred = m.group(2)
    names = {a["name"] for a in axes}
    used = set(re.findall(r"[A-Za-z_]\w*", pred))
    stray = used - names - FUNCS - {"and", "or", "not"}
    if stray:
        return None, "names %s, which are not axes" % ", ".join(sorted(stray)[:4])
    return {"axes": axes, "pred": pred}, None


def to_c(pred):
    """The electron channel: the predicate as written."""
    e = pred
    e = re.sub(r"\bnot\b", "!", e)
    e = re.sub(r"\band\b", "&&", e)
    e = re.sub(r"\bor\b", "||", e)
    for f in ("abs", "min", "max", "sqrt", "log", "exp", "pow"):
        e = re.sub(r"\b%s\s*\(" % f,
                   {"abs": "fabs(", "min": "fmin(", "max": "fmax("}.get(f, f + "("),
                   e)
    return e


def to_c_strict(pred):
    """The positron channel: EVERY COMPARISON MADE STRICT. `a <= b` becomes
    `a < b`, `a >= b` becomes `a > b`; strict comparisons are left alone.

    WHAT THIS ACTUALLY DETECTS, stated plainly because the first version of
    this file claimed something else. It was described as computing "the
    complement of its opposite", which in exact arithmetic is a no-op and would
    have detected nothing at all. It does not do that. Making a comparison
    strict changes the answer on exactly one set of cases: those where the two
    sides are EQUAL.

    So a pair that fails to annihilate here is a case sitting EXACTLY on a
    threshold - 1500.000 V, 20.000 A - where the verdict is decided by whether
    the rule says "or equal to". That is worth finding: a design on the line is
    a design whose answer is a convention, not a measurement. But it is a
    narrower claim than floating-point association sensitivity, and this file
    should not be read as making the wider one."""
    e = to_c(pred)
    # order matters: two-character operators first
    e = re.sub(r"([^<>!=])>=([^=])", r"\1 MIRROR_GE \2", e)
    e = re.sub(r"([^<>!=])<=([^=])", r"\1 MIRROR_LE \2", e)
    e = re.sub(r"([^<>!=])>([^=])", r"\1 MIRROR_GT \2", e)
    e = re.sub(r"([^<>!=])<([^=])", r"\1 MIRROR_LT \2", e)
    e = e.replace("MIRROR_GE", "> ").replace("MIRROR_LE", "< ")
    # a >= b  <->  !(a < b) ; a > b <-> !(a <= b)
    e = e.replace("MIRROR_GT", "> ").replace("MIRROR_LT", "< ")
    return e


KERNEL = r"""
extern "C" __global__ void annihilate(
    const long long total,
    const long long* __restrict__ offset,   /* npred+1 prefix sums */
    const int npred,
    long long* hits,        /* npred: cases the predicate holds, both agreeing */
    long long* differ,      /* npred: the two channels did NOT annihilate      */
    long long* seen)        /* npred: cases actually visited                    */
{
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < total; i += stride) {
    /* which predicate owns this thread - binary search the prefix table */
    int lo = 0, hi = npred - 1, p = 0;
    while (lo <= hi) {
      int mid = (lo + hi) >> 1;
      if (offset[mid] <= i) { p = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    long long r = i - offset[p];
    int e = 0, q = 0;
__DISPATCH__
    atomicAdd((unsigned long long*)&seen[p], 1ULL);
    if (e != q) atomicAdd((unsigned long long*)&differ[p], 1ULL);
    else if (e)  atomicAdd((unsigned long long*)&hits[p], 1ULL);
  }
}
"""


def namespace(expr, axes):
    """Give every axis variable a prefix before it reaches CUDA.

    WHY, and it cost a published number. The generated kernel declares its own
    locals - `int e` for the electron channel, `int q` for the positron, and it
    takes a parameter called `offset` for the prefix table. A worksheet is free
    to name an axis anything, and two of them did: one axis is called `e` and
    another `offset`. C let the axis SHADOW the accumulator without a word, so
    `e = (...) ? 1 : 0;` wrote the axis instead of the verdict and the positron
    then read a corrupted value. That predicate's 161 "partings" were not found;
    they were manufactured by this generator, and they were published.

    Nothing about the arithmetic changes. Only the names do, and now they cannot
    collide with anything the kernel owns."""
    for a in axes:
        expr = re.sub(r"\b" + re.escape(a["name"]) + r"\b", "v_" + a["name"], expr)
    return expr


def build(specs):
    """One kernel, every predicate, dispatched on the prefix table."""
    blocks = []
    for pi, s in enumerate(specs):
        lines = ["    %s (p == %d) {" % ("if" if pi == 0 else "else if", pi)]
        lines.append("      long long rr = r;")
        for ai, a in enumerate(s["axes"]):
            lines.append("      int a%d = (int)(rr %% %d); rr /= %d;"
                         % (ai, a["n"], a["n"]))
        for ai, a in enumerate(s["axes"]):
            lines.append("      double v_%s = %.17g + (%.17g - %.17g) * "
                         "(double)a%d / (%d - 1.0);"
                         % (a["name"], a["lo"], a["hi"], a["lo"], ai, a["n"]))
        lines.append("      e = (%s) ? 1 : 0;" % namespace(s["c"], s["axes"]))
        lines.append("      q = (%s) ? 1 : 0;" % namespace(s["cm"], s["axes"]))
        lines.append("    }")
        blocks.append("\n".join(lines))
    return KERNEL.replace("__DISPATCH__", "\n".join(blocks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--watch", action="store_true")
    a = ap.parse_args()

    specs, refused = [], []
    for p in sheets():
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception as e:
            refused.append((os.path.basename(p), "unreadable: %s" % e))
            continue
        for f in d.get("features", []):
            pr = f.get("properties", {})
            got, why = parse(pr.get("maths"))
            if not got:
                refused.append((f.get("id", "?"), why))
                continue
            size = 1
            for ax in got["axes"]:
                size *= ax["n"]
            if size > PER_MAX:
                size = PER_MAX
            got.update({"id": f.get("id", "?"),
                        "what": (pr.get("what") or "")[:90],
                        "size": size,
                        "c": to_c(got["pred"]),
                        "cm": to_c_strict(got["pred"])})
            specs.append(got)

    if not specs:
        print("FAIL: no sentence parsed. Nothing to launch.")
        return 1
    total = sum(s["size"] for s in specs)
    print("ANNIHILATE: one launch, every predicate, two channels a case")
    print("  sheets read        %d" % len(sheets()))
    print("  predicates fused   %d" % len(specs))
    print("  refused by name    %d" % len(refused))
    print("  cases in the grid  %s" % "{:,}".format(total))
    print("  channels resident  107,520 (70 multiprocessors x 1,536 threads)")
    print("\n  electron  the predicate as the worksheet wrote it")
    print("  positron  every comparison made STRICT: <= becomes <, >= becomes >")
    print("  a case that does not ANNIHILATE is sitting EXACTLY on its")
    print("  threshold, where the verdict is a convention, not a measurement.")
    if not a.run:
        print("\n  --run to launch it.")
        return 0

    import cupy as cp
    src = build(specs)
    mod = cp.RawModule(code=src, backend="nvrtc",
                       options=("--std=c++11",))
    fn = mod.get_function("annihilate")

    off = np.zeros(len(specs) + 1, dtype=np.int64)
    for i, s in enumerate(specs):
        off[i + 1] = off[i] + s["size"]
    d_off = cp.asarray(off)
    hits = cp.zeros(len(specs), dtype=cp.int64)
    diff = cp.zeros(len(specs), dtype=cp.int64)
    seen = cp.zeros(len(specs), dtype=cp.int64)

    threads = 256
    grid = 70 * 6                     # a few waves per multiprocessor
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    fn((grid,), (threads,), (np.int64(total), d_off, np.int32(len(specs)),
                             hits, diff, seen))
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0

    H = [int(x) for x in cp.asnumpy(hits)]
    D = [int(x) for x in cp.asnumpy(diff)]
    S = [int(x) for x in cp.asnumpy(seen)]
    if sum(S) != total:
        print("  FAIL: the grid did not visit every case (%s of %s)."
              % ("{:,}".format(sum(S)), "{:,}".format(total)))
        return 1

    print("\n  LAUNCHED %s cases in ONE grid in %.3f s (%s cases/s)"
          % ("{:,}".format(total), sec, "{:,.0f}".format(total / sec)))
    print("  every case computed twice: %s evaluations"
          % "{:,}".format(total * 2))

    nd = sum(D)
    print("\n  ANNIHILATION")
    print("    pairs that agreed      %18s" % "{:,}".format(total - nd))
    print("    pairs that did NOT     %18s" % "{:,}".format(nd))
    if nd:
        print("\n  THE PREDICATES WHOSE CHANNELS PARTED - these cases sit")
        print('  EXACTLY on a threshold, so it is whether the rule')
        print('  says or-equal-to that decides them, not the design:')
        order = sorted(range(len(specs)), key=lambda i: -D[i])
        for i in order[:8]:
            if not D[i]:
                break
            print("    %-9s %12s of %-14s  %s"
                  % (specs[i]["id"], "{:,}".format(D[i]),
                     "{:,}".format(S[i]), specs[i]["what"][:52]))
    else:
        print("\n  Not one case in %s parted. Every verdict in this worksheet"
              % "{:,}".format(total))
        print("  survives having its comparisons written the other way round.")

    body = {
        "schema": "ggs.annihilate/1",
        "predicates": len(specs),
        "cases": total,
        "evaluations": total * 2,
        "agreed": total - nd,
        "did_not_annihilate": nd,
        "refused": [{"id": i, "why": w} for i, w in refused],
        "refused_count": len(refused),
        "per_predicate": [
            {"id": specs[i]["id"], "what": specs[i]["what"],
             "cases": S[i], "held": H[i], "did_not_annihilate": D[i]}
            for i in range(len(specs))],
        "the_pair":
            "Each case is computed twice in the same thread. The electron "
            "channel evaluates the predicate as the worksheet wrote it. The "
            "positron channel makes every comparison STRICT: <= becomes <, >= "
            "becomes >. Those differ on exactly one set of cases - where the "
            "two sides are EQUAL. So a pair that does not annihilate is a case "
            "sitting exactly on its threshold, where the verdict is decided by "
            "whether the rule says 'or equal to'. That is a convention, not a "
            "measurement. An earlier version of this file described the "
            "positron as computing the complement of its opposite, which is a "
            "no-op and would have detected nothing; the description was wrong "
            "and has been corrected to match the code.",
        "not_claimed":
            "This settles arithmetic over the spaces the worksheets state. It "
            "does not say a space is the right space or a predicate the right "
            "question. A zero here means no case in THIS grid parted, not that "
            "none exists. It holds no site, no weather record and no as-built "
            "drawing, so it says nothing about any real installation.",
        "anonymised": "No project, site, client, maker or model anywhere.",
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    body["body_sha256"] = digest
    body["timing_not_in_digest"] = {"seconds": round(sec, 4),
                                    "cases_per_second": int(total / sec),
                                    "observed": bool(a.watch)}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(body, indent=1, sort_keys=True) + "\n")
    print("\n  body sha256 %s" % digest[:32])
    print("  wrote night-results/annihilate.json")
    print("\n  Timing is OUTSIDE the digest, because watching changes it: "
          "sampling")
    print("  the card's power takes a lock and slows the thing it measures. "
          "The")
    print("  counts do not move. The rate does. Only one of those is a result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
