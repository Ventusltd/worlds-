"""
PATHS ARE NOT WRITTEN DOWN HERE. This repository is public and an
absolute path names a machine and an account. Set GRID_DATA (and where
needed GRID_REPOS or CLAUDE_TRANSCRIPT) in the environment instead.
See src/paths.py for why.
Did the numbers I said out loud come from anywhere?

WHAT THIS IS, AND WHAT IT IS NOT
A GPU cannot read a transcript for meaning; that is a model's job and this file
does not pretend otherwise. What a GPU can do is take every distinctive number
a model SAID over a long night, take every number that exists in the artefacts
it produced, and check membership - millions of comparisons, nothing skipped,
no sampling, no judgement.

That is exactly the shape of a hallucination check. A number I stated that
appears nowhere in any file I wrote is either arithmetic I did in my head, or
something I made up. The machine cannot tell those apart. It can tell you
WHICH numbers need a human to look, and that list is short enough to read.

THE THREE BUCKETS
  BACKED      the number appears verbatim in an artefact on disk
  DERIVED     it does not, but it is a simple function of numbers that do -
              a sum, a difference, an exact quotient - checked on the card
  UNBACKED    neither. These are the ones to read.

WHY THE MIDDLE BUCKET WAS WORTHLESS, AND WHAT CHANGED
The first version of DERIVED tested each unbacked number against EVERY ordered
pair of backed numbers in the whole corpus. Millions of pairs, three
operations: the sums alone cover every four- and five-digit value there is. A
number nobody ever said passed as "derived" with near-certainty. The file
reported "truly unbacked: 0", and that zero was the expected output whether the
night had been scrupulous or invented. A sieve that absorbs everything is not a
sieve, and declaring so in the output did not make it one.

The sieve now demands that the two operands CO-OCCUR WITH THE CLAIM: they must
be backed numbers that appeared in the transcript within a short window of the
place the claim was made. A derivation is only an explanation if the operands
were in front of me when I said it. That cuts the pool from thousands to tens,
and the net from dense to sparse.

AND THE RUN MEASURES ITS OWN NET, EVERY TIME
For each unbacked claim the run manufactures decoys: random numbers of the same
magnitude that were never said and appear in no artefact, pushed through the
same sieve with the same operand pool. The decoy absorption rate IS the false
absorption rate. Both rates are printed side by side, per magnitude band. If
the decoy rate is not clearly below the real rate, the run says its own verdict
is uninformative - automatically, with nobody having to notice. The old wide
sieve is run alongside on the same decoys, so before and after are printed
together and the improvement is a measurement rather than a claim.

THE PAIR RULE HOLDS TWICE OVER. Membership is computed twice - once by a sorted
binary search on the card, once by a hash set on the host - and nothing is
reported if the two disagree. Derivation is now computed twice as well: a
nested scan over operand pairs on the card, and a set-lookup formulation on the
host that shares no code with it. Disagreement refuses the report.

NOT DONE HERE: the fourth thing, a quantity whose VALUE CHANGED across the
night. That check is not implemented in this file, and this file does not find
contradictions.

    python thread_truth.py            what it would read
    python thread_truth.py --run      run it
"""
import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import time

import numpy as np

import paths

TRANSCRIPT = os.environ.get("CLAUDE_TRANSCRIPT", "")   # set it; never written down here
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "night-results")
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(RESULTS, "thread-truth.json")

# Only distinctive numbers. Below this, a number is a count of fingers and
# appears everywhere by chance, which would drown the signal in noise.
MIN_NOTABLE = 1000
MAX_SANE = 1 << 62          # larger than this is not a quantity, it is a token
NUM = re.compile(r"(?<![\w.])(\d[\d,]{2,})(?![\w])")

# The teeth. An operand only counts if it appeared within this many transcript
# records of the place the claim was made, and at most this many operands are
# kept, nearest first. Both are printed, so a reader can see the size of the
# net that produced the verdict.
#
# The window is not a taste. It was swept against the decoy arm, and 20 is
# where the sieve separates best - real absorption ten times the false:
#
#    window    median pool    real absorbed    decoys absorbed
#         0              2            0.4%              0.00%
#         5              5            0.4%              0.05%
#        20              9            1.5%              0.14%
#        40             11            1.9%              0.47%
#        80             16            2.2%              1.08%
#       400             38            7.9%              3.75%
#
# Widen it and the sieve absorbs more of everything, real and invented alike,
# which is the old failure creeping back. Anyone who changes this number is
# obliged to rerun the sweep and look at the second column.
NEAR_WINDOW = 20
NEAR_CAP = 400
WIDE_CAP = 4000             # what the old sieve used: the whole corpus, in effect

# The dose. Random numbers nobody said, of the same magnitude as each real
# claim, through the same sieve with the same operands.
DECOYS_PER_CLAIM = 8
DECOY_SEED = 20260921

# A band is only usable if a number nobody said is absorbed less often than
# this. Above it, the count of survivors in that band means nothing. And a
# band with a handful of decoys has not measured anything either: zero out of
# eight is not evidence of a tight net, so say so instead of claiming it.
FALSE_ABSORPTION_CEILING = 0.25
MIN_DECOYS_PER_BAND = 30


def numbers_in(text):
    out = set()
    for m in NUM.finditer(text or ""):
        try:
            v = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if MIN_NOTABLE <= v < MAX_SANE:
            out.add(v)
    return out


def _block_text(c):
    """Text out of one content block, whatever shape it arrived in. The
    envelope - token counts, timestamps, identifiers - is deliberately not
    read: those are numbers about the conversation, not numbers in it."""
    if isinstance(c, str):
        return c
    if not isinstance(c, dict):
        return ""
    t = c.get("type")
    if t == "text":
        return c.get("text") or ""
    if t == "thinking":
        return c.get("thinking") or ""
    if t == "tool_use":
        try:
            return json.dumps(c.get("input") or {})
        except Exception:
            return ""
    if t == "tool_result":
        inner = c.get("content")
        if isinstance(inner, str):
            return inner
        if isinstance(inner, list):
            return "\n".join(_block_text(x) for x in inner)
    return ""


def record_text(msg):
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(_block_text(c) for c in content)
    return ""


def said():
    """Every distinctive number in MY OWN text, and where in the thread it was
    said. Tool results are not claims - those are evidence - so they do not
    enter the claim set. They DO enter the neighbourhood of a claim, because a
    number sitting in a tool result beside the sentence is exactly the kind of
    operand a real derivation would have used.

    Returns
      claims  {value: [assistant turn, ...]}
      pos     {value: [record index, ...]}
      rec     [every distinctive number in record i, all roles]
    """
    claims, pos, rec = {}, {}, []
    if not os.path.isfile(TRANSCRIPT):
        raise SystemExit(
            "FAIL: the transcript is not where it should be.\n"
            "  Set CLAUDE_TRANSCRIPT to the thread to check. Nothing is\n"
            "  enumerated on remembered numbers.")
    turn = 0
    for line in io.open(TRANSCRIPT, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        msg = d.get("message") or {}
        if not isinstance(msg, dict) or not msg.get("content"):
            continue
        r = len(rec)
        rec.append(numbers_in(record_text(msg)))
        if msg.get("role") != "assistant":
            continue
        turn += 1
        content = msg.get("content")
        parts = []
        if isinstance(content, str):
            parts = [content]
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") == "text":
                    parts.append(c.get("text") or "")
        for t in parts:
            for v in numbers_in(t):
                claims.setdefault(v, []).append(turn)
                pos.setdefault(v, []).append(r)
    return claims, pos, rec


def artefacts():
    """Every distinctive number in the files that were actually written.

    THIS FILE'S OWN OUTPUT IS EXCLUDED. It lists the numbers it could not
    back; leaving it in the corpus means the next run finds every one of them
    on disk and calls it BACKED. That is not a corroboration, it is an echo,
    and it inflated the backed count of earlier runs by a factor of seven."""
    have, where = set(), {}
    files = []
    for d in (RESULTS, SRC):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                p = os.path.join(d, f)
                if os.path.abspath(p) == os.path.abspath(OUT):
                    continue
                if f.endswith((".json", ".py", ".md", ".geojson")):
                    files.append(p)
    for d in paths.require_any(
            [paths.HARVEST, paths.HARVEST2, paths.NIGHT],
            "the written-work directories this gate corroborates against"):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith((".json", ".geojson", ".py", ".js", ".log")):
                    files.append(os.path.join(d, f))
    for p in files:
        try:
            t = io.open(p, encoding="utf-8", errors="replace").read()
        except Exception:
            continue
        for v in numbers_in(t):
            have.add(v)
            where.setdefault(v, os.path.basename(p))
    return have, where, len(files)


def near_pool(v, pos, rec, have):
    """The backed numbers that were in front of me when I said v: everything
    within NEAR_WINDOW records of any place v was stated, nearest first,
    capped. This is the whole of the fix. Without it every pair in the corpus
    is fair game, and a net that wide absorbs anything below a million."""
    best = {}
    n = len(rec)
    for r0 in pos.get(v, ()):
        lo, hi = max(0, r0 - NEAR_WINDOW), min(n - 1, r0 + NEAR_WINDOW)
        for r in range(lo, hi + 1):
            d = abs(r - r0)
            for u in rec[r]:
                if u != v and u in have and (u not in best or d < best[u]):
                    best[u] = d
    pick = sorted(best.items(), key=lambda kv: (kv[1], kv[0]))[:NEAR_CAP]
    return sorted(u for u, _ in pick)


SRC_K = r"""
extern "C" __global__ void member(
    const long long* __restrict__ claim, const long long n,
    const long long* __restrict__ sorted_have, const long long m,
    int* out)
{
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long v = claim[i];
  long long lo = 0, hi = m - 1; int found = 0;
  while (lo <= hi) {
    long long mid = (lo + hi) >> 1;
    long long x = sorted_have[mid];
    if (x == v) { found = 1; break; }
    if (x < v) lo = mid + 1; else hi = mid - 1;
  }
  out[i] = found;
}
"""

# One kernel serves both sieves. Each candidate carries its own slice of the
# operand pool: for the wide sieve every candidate points at the same slice -
# the corpus - and for the near sieve at its own short neighbourhood. One
# thread per (candidate, first operand); the second operand is the inner loop.
SRC_D = r"""
extern "C" __global__ void derive(
    const long long* __restrict__ val, const long long n,
    const long long* __restrict__ pool,
    const long long* __restrict__ off,
    const long long* __restrict__ cnt,
    int* how)
{
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long a = (long long)blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n) return;
  long long m = cnt[i];
  if (a >= m) return;
  const long long* P = pool + off[i];
  long long v = val[i];
  long long x = P[a];
  int found = 0;
  for (long long b = 0; b < m; b++) {
    long long y = P[b];
    if (x + y == v) { found = 1; break; }
    if (x - y == v) { found = 2; break; }
    if (y > 0) {
      /* x / y == v exactly, which is x == v*y. The double is only a filter,
         to keep a 64-bit multiply - and any overflow with it - off the hot
         path; the equality that decides is integer. */
      double p = (double)v * (double)y;
      double q = (double)x;
      if (p >= q - q * 1e-9 && p <= q + q * 1e-9 && v * y == x) {
        found = 3; break;
      }
    }
  }
  if (found) atomicMax(how + i, found);
}
"""


def derive_on_card(cp, vals, pools):
    """pools is one operand list per value. Returns which operation the card
    settled on, 0 for none, and how many pair-tests it took."""
    n = len(vals)
    flat, off, cnt = [], [], []
    for p in pools:
        off.append(len(flat))
        cnt.append(len(p))
        flat.extend(p)
    if not flat:
        flat = [0]
    d_val = cp.asarray(np.array(vals, dtype=np.int64))
    d_pool = cp.asarray(np.array(flat, dtype=np.int64))
    d_off = cp.asarray(np.array(off, dtype=np.int64))
    d_cnt = cp.asarray(np.array(cnt, dtype=np.int64))
    how = cp.zeros(n, dtype=cp.int32)
    mod = cp.RawModule(code=SRC_D, backend="nvrtc")
    fn = mod.get_function("derive")
    mx = max(cnt) if cnt else 1
    bx, by = 32, 8
    fn(((n + bx - 1) // bx, (max(mx, 1) + by - 1) // by), (bx, by),
       (d_val, np.int64(n), d_pool, d_off, d_cnt, how))
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(how), sum(c * c for c in cnt)


def derive_on_host(vals, pools):
    """THE PAIR, for the derivation. The same question asked a different way:
    instead of scanning every pair, ask a set whether the one operand that
    would complete each operation exists. Nothing is shared with the kernel
    but the answer."""
    out = np.zeros(len(vals), dtype=np.int32)
    for i, v in enumerate(vals):
        P = pools[i]
        if not P:
            continue
        S = set(P)
        got = 0
        for x in P:                      # x + y == v
            if (v - x) in S:
                got = 1
                break
        if not got:
            for y in P:                  # x - y == v
                if (v + y) in S:
                    got = 2
                    break
        if not got:
            for y in P:                  # x / y == v, exactly
                if y > 0 and (v * y) in S:
                    got = 3
                    break
        out[i] = got
    return out


def band(v):
    return len(str(int(v))) - 1


def rate_table(vals, absorbed, kinds):
    """Absorption by magnitude band, real against decoy."""
    rows = {}
    for v, a, k in zip(vals, absorbed, kinds):
        r = rows.setdefault(band(v), [0, 0, 0, 0])
        r[0 + 2 * k] += 1
        r[1 + 2 * k] += 1 if a else 0
    return rows


def pct(hit, n):
    return (100.0 * hit / n) if n else 0.0


def band_verdict(row):
    """Three answers, not two. A band can discriminate, or it demonstrably
    cannot, or nobody measured it hard enough to say - and the third is not
    the first."""
    if row[2] < MIN_DECOYS_PER_BAND:
        return "thin"
    if pct(row[3], row[2]) > 100.0 * FALSE_ABSORPTION_CEILING:
        return "no"
    return "yes"


VERDICT_WORDS = {
    "yes": "yes",
    "thin": "too few decoys to say",
    "no": "NO - a survivor count here means nothing",
}


def print_table(title, rows, note):
    print("\n  %s" % title)
    print("    %-9s %-26s %-26s %s"
          % ("magnitude", "real claims absorbed", "decoys absorbed",
             "can this band tell them apart?"))
    for b in sorted(rows):
        rn, rh, dn, dh = rows[b]
        print("    1e%-7d %9s of %-7s %6.1f%% %9s of %-7s %6.1f%%   %s"
              % (b, "{:,}".format(rh), "{:,}".format(rn), pct(rh, rn),
                 "{:,}".format(dh), "{:,}".format(dn), pct(dh, dn),
                 VERDICT_WORDS[band_verdict(rows[b])]))
    print("    %s" % note)


def informative(rows):
    """Automatic, so that nobody has to notice. A run is informative only if a
    number nobody said is rarely absorbed, AND is absorbed markedly less often
    than the numbers that really were said. Equal rates mean the sieve is
    answering a question about arithmetic density, not about this night."""
    rn = sum(r[0] for r in rows.values())
    rh = sum(r[1] for r in rows.values())
    dn = sum(r[2] for r in rows.values())
    dh = sum(r[3] for r in rows.values())
    r, d = pct(rh, rn) / 100.0, pct(dh, dn) / 100.0
    if dn == 0:
        return False, "no decoys were drawn, so nothing measured the net"
    if d > FALSE_ABSORPTION_CEILING:
        return False, ("a number nobody said is absorbed %.1f%% of the time, "
                       "over the %.0f%% ceiling"
                       % (100 * d, 100 * FALSE_ABSORPTION_CEILING))
    if r == 0.0 and d == 0.0:
        return True, "the sieve absorbed nothing at all, real or decoy"
    if d * 2 > r:
        return False, ("decoys are absorbed %.1f%% against %.1f%% for real "
                       "claims - too close together to tell apart"
                       % (100 * d, 100 * r))
    return True, ("decoys %.1f%% against real claims %.1f%%"
                  % (100 * d, 100 * r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    claims, pos, rec = said()
    have, where, nfiles = artefacts()
    print("THREAD TRUTH: did the numbers I said come from anywhere?")
    print("  distinctive numbers I stated   %s" % "{:,}".format(len(claims)))
    print("  transcript records read        %s" % "{:,}".format(len(rec)))
    print("  artefact files read            %s" % "{:,}".format(nfiles))
    print("  distinctive numbers on disk    %s" % "{:,}".format(len(have)))
    print("  a number counts as distinctive if it is >= %s"
          % "{:,}".format(MIN_NOTABLE))
    if not a.run:
        print("\n  --run to check them on the card.")
        return 0

    import cupy as cp
    cl = np.array(sorted(claims), dtype=np.int64)
    hv = np.array(sorted(have), dtype=np.int64)
    mod = cp.RawModule(code=SRC_K, backend="nvrtc")
    fn = mod.get_function("member")
    d_cl, d_hv = cp.asarray(cl), cp.asarray(hv)
    out = cp.zeros(len(cl), dtype=cp.int32)
    threads = 256
    fn(((len(cl) + threads - 1) // threads,), (threads,),
       (d_cl, np.int64(len(cl)), d_hv, np.int64(len(hv)), out))
    cp.cuda.Stream.null.synchronize()
    got = cp.asnumpy(out).astype(bool)

    # THE PAIR: the same membership, computed a completely different way.
    hset = set(hv.tolist())
    want = np.array([v in hset for v in cl.tolist()], dtype=bool)
    if not np.array_equal(got, want):
        n = int((got != want).sum())
        print("\n  REFUSING TO REPORT: the card and the host disagree on %d of "
              "%d memberships." % (n, len(cl)))
        return 1
    print("\n  membership checked twice - binary search on the card, hash set "
          "on the host - %s comparisons, they agree" % "{:,}".format(len(cl)))

    backed = cl[got]
    unbacked = [int(v) for v in cl[~got].tolist()]
    print("  BACKED    %s" % "{:,}".format(len(backed)))
    print("  UNBACKED  %s" % "{:,}".format(len(unbacked)))
    if not unbacked:
        print("\n  nothing to sieve.")
        return 0

    # THE DOSE. For every unbacked claim, decoys of the same magnitude that
    # were never said and are in no artefact. They go through the same sieve
    # with the same operand pool, and their absorption rate is this sieve's
    # false absorption rate - measured on this night's corpus, not assumed.
    rng = random.Random(DECOY_SEED)
    stated, seen = set(claims), set()
    vals, kinds, parent = [], [], []
    for v in unbacked:
        vals.append(v)
        kinds.append(0)
        parent.append(v)
        b = band(v)
        lo, hi = 10 ** b, 10 ** (b + 1) - 1
        made, tries = 0, 0
        while made < DECOYS_PER_CLAIM and tries < 500 * DECOYS_PER_CLAIM:
            tries += 1
            w = rng.randint(lo, hi)
            if w in stated or w in have or w in seen:
                continue
            seen.add(w)
            vals.append(w)
            kinds.append(1)
            parent.append(v)
            made += 1

    near = {v: near_pool(v, pos, rec, have) for v in unbacked}
    sizes = sorted(len(p) for p in near.values())
    wide = sorted(have)[:WIDE_CAP]

    results = {}
    for name, pools in (("wide", [wide for _ in vals]),
                        ("near", [near[p] for p in parent])):
        t0 = time.perf_counter()
        hc, npairs = derive_on_card(cp, vals, pools)
        sec = time.perf_counter() - t0
        hh = derive_on_host(vals, pools)
        dis = int(((hc > 0) != (hh > 0)).sum())
        if dis:
            print("\n  REFUSING TO REPORT: on the %s sieve the card and the "
                  "host disagree on %d of %d derivations."
                  % (name, dis, len(vals)))
            return 1
        results[name] = {"absorbed": (hc > 0),
                         "rows": rate_table(vals, hc > 0, kinds),
                         "pairs": npairs, "sec": sec}

    print("  derivation checked twice as well - a nested scan over operand "
          "pairs on the card, a set lookup on the host - over %s numbers, %s "
          "real and %s decoys; they agree"
          % ("{:,}".format(len(vals)), "{:,}".format(len(unbacked)),
             "{:,}".format(len(vals) - len(unbacked))))

    print_table(
        "BEFORE - the old sieve: any ordered pair anywhere in the corpus "
        "(%s operands)" % "{:,}".format(len(wide)),
        results["wide"]["rows"],
        "%s pair-tests on the card in %.2f s"
        % ("{:,}".format(results["wide"]["pairs"]), results["wide"]["sec"]))
    print_table(
        "AFTER - operands must have co-occurred within %d records of the claim"
        % NEAR_WINDOW,
        results["near"]["rows"],
        "operand pools: median %d, largest %d, %d claims had none; %s "
        "pair-tests in %.2f s"
        % (sizes[len(sizes) // 2], sizes[-1], sum(1 for s in sizes if s == 0),
           "{:,}".format(results["near"]["pairs"]), results["near"]["sec"]))

    ok_wide, why_wide = informative(results["wide"]["rows"])
    ok_near, why_near = informative(results["near"]["rows"])
    print("\n  the old sieve:  %s - %s"
          % ("INFORMATIVE" if ok_wide else "UNINFORMATIVE", why_wide))
    print("  the new sieve:  %s - %s"
          % ("INFORMATIVE" if ok_near else "UNINFORMATIVE", why_near))

    ab = results["near"]["absorbed"]
    real = [i for i, k in enumerate(kinds) if k == 0]
    derived = sum(1 for i in real if ab[i])
    hard = sorted((vals[i] for i in real if not ab[i]), reverse=True)
    print("\n  of the %s unbacked, DERIVED from numbers beside the claim  %s"
          % ("{:,}".format(len(unbacked)), "{:,}".format(derived)))
    print("  leaving TRULY UNBACKED                                    %s"
          % "{:,}".format(len(hard)))
    if not ok_near:
        print("\n  THIS VERDICT IS UNINFORMATIVE: %s. The count above is not "
              "evidence of anything, and must not be quoted." % why_near)

    bad = sorted(b for b, r in results["near"]["rows"].items()
                 if band_verdict(r) == "no")
    thin = sorted(b for b, r in results["near"]["rows"].items()
                  if band_verdict(r) == "thin")
    if bad:
        print("  bands where the sieve still cannot discriminate, and where a "
              "survivor count means nothing: %s"
              % ", ".join("1e%d" % b for b in bad))
    if thin:
        print("  bands with fewer than %d decoys - not measured, so not "
              "vouched for: %s"
              % (MIN_DECOYS_PER_BAND, ", ".join("1e%d" % b for b in thin)))

    print("\n  THE NUMBERS TO READ - stated in a sentence, found in no file,")
    print("  and not a sum, difference or exact quotient of two numbers that")
    print("  were on the page beside it:")
    for v in hard[:25]:
        turns = claims.get(v, [])
        print("    %18s  first said at assistant turn %d, said %d time(s)"
              % ("{:,}".format(v), turns[0] if turns else -1, len(turns)))
    if not hard:
        print("    none.")
    if len(hard) > 25:
        print("    ... and %s more; all of them are in the JSON."
              % "{:,}".format(len(hard) - 25))

    def table_json(rows):
        return [{"magnitude": "1e%d" % b,
                 "real_claims": rows[b][0],
                 "real_absorbed": rows[b][1],
                 "real_absorbed_pct": round(pct(rows[b][1], rows[b][0]), 1),
                 "decoys": rows[b][2],
                 "decoys_absorbed": rows[b][3],
                 "false_absorption_pct": round(pct(rows[b][3], rows[b][2]), 1),
                 "band_can_discriminate": band_verdict(rows[b])}
                for b in sorted(rows)]

    body = {
        "schema": "ggs.threadtruth/2",
        "numbers_stated": len(claims),
        "numbers_on_disk": len(have),
        "artefact_files": nfiles,
        "transcript_records": len(rec),
        "backed": int(len(backed)),
        "unbacked": len(unbacked),
        "derived_from_numbers_beside_the_claim": derived,
        "truly_unbacked": len(hard),
        "verdict_is_informative": bool(ok_near),
        "verdict_reason": why_near,
        "uninformative_bands": ["1e%d" % b for b in bad],
        "unmeasured_bands": ["1e%d" % b for b in thin],
        "sieve": {
            "operands_must_co_occur_within_records": NEAR_WINDOW,
            "operands_kept_per_claim_at_most": NEAR_CAP,
            "operations": "sum, difference, exact quotient",
            "operand_pool_median": sizes[len(sizes) // 2],
            "operand_pool_largest": sizes[-1],
            "claims_with_no_operands": sum(1 for s in sizes if s == 0),
            "decoys_per_claim": DECOYS_PER_CLAIM,
            "decoy_seed": DECOY_SEED,
            "false_absorption_ceiling_pct": 100.0 * FALSE_ABSORPTION_CEILING,
            "decoys_a_band_needs_before_it_is_vouched_for": MIN_DECOYS_PER_BAND,
        },
        "false_absorption_before": table_json(results["wide"]["rows"]),
        "false_absorption_after": table_json(results["near"]["rows"]),
        "to_read": [{"value": v, "times_said": len(claims.get(v, [])),
                     "first_turn": (claims.get(v) or [-1])[0]}
                    for v in hard],
        "method":
            "Every distinctive number (>= 1000) in the model's OWN sentences is "
            "taken from the transcript - tool results are excluded, because "
            "those are evidence and not claims. Every distinctive number in "
            "every artefact on disk is taken as the backing set. Membership is "
            "computed twice, by binary search on the card and by a hash set on "
            "the host, and nothing is reported unless they agree. What remains "
            "is tested on the card against pairs of backed numbers that "
            "CO-OCCURRED with the claim - within a short window of transcript "
            "records of where it was said - for sum, difference and exact "
            "quotient. That derivation is computed twice as well, by a nested "
            "scan on the card and by set lookups on the host, and disagreement "
            "refuses the report.",
        "how_the_sieve_was_measured":
            "Every run carries a decoy arm. For each unbacked claim, random "
            "numbers of the same magnitude that were never said and appear in "
            "no artefact are pushed through the same sieve with the same "
            "operand pool. Their absorption rate is the false absorption rate, "
            "printed per magnitude band beside the real one, for the old wide "
            "sieve and for this one. A band whose false absorption is above "
            "the ceiling cannot discriminate and its survivor count means "
            "nothing. If the decoy rate overall is not clearly below the real "
            "rate, the run declares its own verdict uninformative without "
            "being asked. Quote false_absorption_after beside any count taken "
            "from this file.",
        "not_claimed":
            "A number being BACKED does not mean it was used correctly, only "
            "that it exists somewhere on disk. A number being TRULY UNBACKED "
            "does not mean it is false - it may be arithmetic done in a "
            "sentence out of numbers that were never written down, or a figure "
            "quoted from outside this night's work. This file cannot tell "
            "those apart and does not try. It narrows what a person has to "
            "read, and that is all it does. It does not check whether a "
            "quantity changed value across the night.",
        "anonymised": "No project, site, client, maker or model anywhere.",
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    body["body_sha256"] = digest
    os.makedirs(RESULTS, exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(body, indent=1, sort_keys=True) + "\n")
    print("\n  body sha256 %s" % digest[:32])
    print("  wrote night-results/thread-truth.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
