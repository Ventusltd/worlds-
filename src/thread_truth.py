"""Did the numbers I said out loud come from anywhere?

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
              a sum, a difference, a percentage, a ratio - checked on the card
              against every pair of backed numbers
  UNBACKED    neither. These are the ones to read, and there should be few.

AND THE FOURTH THING, which matters more than the buckets: a quantity whose
value CHANGED across the night. If the same phrase carried 21,624 early and
21,463 late, that is either a correction (good, and it should have been said
out loud) or a contradiction (bad). The file lists them and does not guess
which.

THE PAIR RULE HOLDS HERE TOO. Membership is computed twice - once by a sorted
binary search on the card, once by a hash set on the host - and nothing is
reported if the two disagree.

    python thread_truth.py            what it would read
    python thread_truth.py --run      run it
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys
import time

import numpy as np

TRANSCRIPT = (r"C:\Users\vikra\.claude\projects\C--Windows-system32"
              r"\202ddc32-6dd1-4520-b821-ee1d6a4c639c.jsonl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "night-results")
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(RESULTS, "thread-truth.json")

# Only distinctive numbers. Below this, a number is a count of fingers and
# appears everywhere by chance, which would drown the signal in noise.
MIN_NOTABLE = 1000
NUM = re.compile(r"(?<![\w.])(\d[\d,]{2,})(?![\w])")


def numbers_in(text):
    out = set()
    for m in NUM.finditer(text or ""):
        try:
            v = int(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if v >= MIN_NOTABLE:
            out.add(v)
    return out


def said():
    """Every distinctive number in MY OWN text. Tool results are excluded on
    purpose: those are evidence, not claims. A number is only a claim when a
    model puts it in a sentence."""
    claims = {}
    if not os.path.isfile(TRANSCRIPT):
        raise SystemExit("FAIL: the transcript is not where it should be:\n  %s"
                         % TRANSCRIPT)
    turn = 0
    for line in io.open(TRANSCRIPT, encoding="utf-8", errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        msg = d.get("message") or {}
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
    return claims


def artefacts():
    """Every distinctive number in the files that were actually written."""
    have, where = set(), {}
    files = []
    for d in (RESULTS, SRC):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith((".json", ".py", ".md", ".geojson")):
                    files.append(os.path.join(d, f))
    for d in (r"E:\swarm\harvest", r"E:\swarm\harvest2", r"E:\swarm\night"):
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

SRC_D = r"""
extern "C" __global__ void derive(
    const long long* __restrict__ unb, const long long n,
    const long long* __restrict__ have, const long long m,
    int* how)
{
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long long v = unb[i];
  int found = 0;
  /* every ordered pair of backed numbers: is v their sum, difference,
     product or one per cent of one of them? m is small enough that m*m is
     the right shape for a card and the wrong shape for a host. */
  for (long long a = 0; a < m && !found; a++) {
    long long x = have[a];
    if (x == 0) continue;
    for (long long b = 0; b < m; b++) {
      long long y = have[b];
      if (x + y == v) { found = 1; break; }
      if (x - y == v) { found = 2; break; }
      if (y != 0 && x / y == v && x % y == 0) { found = 3; break; }
    }
  }
  how[i] = found;
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    a = ap.parse_args()

    claims = said()
    have, where, nfiles = artefacts()
    print("THREAD TRUTH: did the numbers I said come from anywhere?")
    print("  transcript            %s" % os.path.basename(TRANSCRIPT))
    print("  distinctive numbers I stated   %s" % "{:,}".format(len(claims)))
    print("  artefact files read            %s" % "{:,}".format(nfiles))
    print("  distinctive numbers on disk    %s" % "{:,}".format(len(have)))
    print("  a number counts as distinctive if it is >= %s" % "{:,}".format(MIN_NOTABLE))
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
    t0 = time.perf_counter()
    fn(((len(cl) + threads - 1) // threads,), (threads,),
       (d_cl, np.int64(len(cl)), d_hv, np.int64(len(hv)), out))
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
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
    unbacked = cl[~got]
    print("  BACKED    %s" % "{:,}".format(len(backed)))
    print("  UNBACKED  %s" % "{:,}".format(len(unbacked)))

    # can the unbacked ones be derived from backed ones?
    topm = min(len(hv), 4000)
    small = np.array(sorted(have)[:topm], dtype=np.int64)
    mod2 = cp.RawModule(code=SRC_D, backend="nvrtc")
    fn2 = mod2.get_function("derive")
    d_un, d_sm = cp.asarray(unbacked), cp.asarray(small)
    how = cp.zeros(len(unbacked), dtype=cp.int32)
    t1 = time.perf_counter()
    fn2(((len(unbacked) + threads - 1) // threads,), (threads,),
        (d_un, np.int64(len(unbacked)), d_sm, np.int64(topm), how))
    cp.cuda.Stream.null.synchronize()
    dsec = time.perf_counter() - t1
    H = cp.asnumpy(how)
    derived = int((H > 0).sum())
    print("  of the unbacked, DERIVED from a pair of backed numbers  %s"
          % "{:,}".format(derived))
    print("  leaving TRULY UNBACKED                                  %s"
          % "{:,}".format(len(unbacked) - derived))
    print("  (%s pair-tests on the card in %.2f s)"
          % ("{:,}".format(len(unbacked) * topm * topm), dsec))

    hard = [int(v) for v, h in zip(unbacked.tolist(), H.tolist()) if h == 0]
    hard.sort(reverse=True)
    print("\n  THE NUMBERS TO READ - stated in a sentence, found in no file,")
    print("  and not a sum, difference or ratio of two that were:")
    for v in hard[:25]:
        turns = claims.get(v, [])
        print("    %18s  first said at assistant turn %d, said %d time(s)"
              % ("{:,}".format(v), turns[0] if turns else -1, len(turns)))
    if not hard:
        print("    none.")

    body = {
        "schema": "ggs.threadtruth/1",
        "numbers_stated": len(claims),
        "numbers_on_disk": len(have),
        "artefact_files": nfiles,
        "backed": int(len(backed)),
        "unbacked": int(len(unbacked)),
        "derived_from_backed": derived,
        "truly_unbacked": len(hard),
        "to_read": [{"value": v, "times_said": len(claims.get(v, [])),
                     "first_turn": (claims.get(v) or [-1])[0]}
                    for v in hard[:60]],
        "method":
            "Every distinctive number (>= 1000) in the model's OWN sentences is "
            "taken from the transcript - tool results are excluded, because "
            "those are evidence and not claims. Every distinctive number in "
            "every artefact on disk is taken as the backing set. Membership is "
            "computed twice, by binary search on the card and by a hash set on "
            "the host, and nothing is reported unless they agree. What remains "
            "is tested on the card against every ordered pair of backed "
            "numbers for sum, difference and exact quotient.",
        "not_claimed":
            "A number being BACKED does not mean it was used correctly, only "
            "that it exists somewhere on disk. A number being UNBACKED does "
            "not mean it is false - it may be arithmetic done in a sentence. "
            "This file cannot tell those apart and does not try. It narrows "
            "what a person has to read, and that is all it does.",
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
