"""
PATHS ARE NOT WRITTEN DOWN HERE. This repository is public and an
absolute path names a machine and an account. Set GRID_DATA (and where
needed GRID_REPOS or CLAUDE_TRANSCRIPT) in the environment instead.
See src/paths.py for why.
The GB transmission network, every circuit, every single outage, on the card.

The data is the estate's own: derived/gb-transmission-network.v1.json, built
from the published ten-year statement. 921 sites, 2,679 nodes, 1,392 circuits
and 1,472 transformers, each with r, x and b on a 100 MVA base and seasonal
ratings. That is a real network with real impedances, which is the thing that
was missing from every library model that had a map.

THE STUDY, and why it needs no demand data
A linear DC sensitivity: for one megawatt injected at a node and withdrawn at
the reference, how much flows on each branch. That is the PTDF, and it answers
the question a developer actually asks - if I connect here, what do I load -
without assuming anyone's demand forecast.

Then the same question again with any one branch removed. That is the LODF,
obtained from the same single factorisation by a rank-one identity rather than
by solving the network again:

    LODF[k,l] = PTDF_branch[k,l] / (1 - PTDF_branch[l,l])
    flow_k after losing l = flow_k + LODF[k,l] * flow_l

WHAT THIS IS NOT
DC linearised screening. No voltage magnitude, no reactive power, no losses,
no stability. The source file carries the field `not_a_connection_assessment`
and that is honoured here: this is a sensitivity, not an assessment, and no
result of it may be described as one.

    python gb_study.py            build, verify, enumerate
"""
import io
import json
import os
import sys
import time

import numpy as np

SRC = os.environ.get("GRID_REPOS", "")   # public repo: no machine path written down


def load():
    if not os.path.isfile(SRC):
        raise SystemExit("FAIL: %s missing. A check that reached nothing is "
                         "not a pass." % SRC)
    d = json.load(io.open(SRC, encoding="utf-8"))
    nodes = {n["node"]: i for i, n in enumerate(d["nodes"])}
    br, skipped = [], {"no_x": 0, "self_loop": 0, "unknown_node": 0}
    for kind, key in (("circuit", "circuits"), ("transformer", "transformers")):
        for b in d.get(key, []):
            a, z = b.get("node_1"), b.get("node_2")
            if a not in nodes or z not in nodes:
                skipped["unknown_node"] += 1
                continue
            if a == z:
                skipped["self_loop"] += 1
                continue
            x = b.get("x_pct_100mva")
            if x in (None, 0):
                skipped["no_x"] += 1
                continue
            rate = (b.get("winter_mva") or b.get("rating_mva") or 0.0)
            br.append({"kind": kind, "f": nodes[a], "t": nodes[z],
                       "x_pu": float(x) / 100.0, "mva": float(rate or 0.0),
                       "name": "%s|%s" % (a, z)})
    return d, nodes, br, skipped


def main():
    d, nodes, br, skipped = load()
    n, m = len(nodes), len(br)
    print("THE GB TRANSMISSION NETWORK, from the estate's own data")
    print("  source          %s" % str(d.get("source", "(unstated)"))[:64])
    print("  sites %s  nodes %s  usable branches %s"
          % ("{:,}".format(len(d["sites"])), "{:,}".format(n), "{:,}".format(m)))
    print("  skipped         %s" % skipped)
    if m == 0:
        print("FAIL: no usable branches. Nothing to study.")
        return 1

    f = np.array([b["f"] for b in br]); t = np.array([b["t"] for b in br])
    bsus = np.array([1.0 / b["x_pu"] for b in br])
    rating = np.array([b["mva"] for b in br])

    # ---- DC network: B_bus = A' diag(b) A, slack removed
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    A = sp.csr_matrix(
        (np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)],
                                          np.r_[f, t])), shape=(m, n))
    Bbus = (A.T @ sp.diags(bsus) @ A).tocsc()

    # the network must be connected, or a single slack is wrong
    ncomp, lab = sp.csgraph.connected_components(Bbus, directed=False)
    print("  islands         %d" % ncomp)
    if ncomp > 1:
        sizes = np.bincount(lab)
        keep = int(np.argmax(sizes))
        print("  NOT ONE NETWORK. Studying the largest island only: %d of %d "
              "nodes. The rest are reported, not silently dropped." %
              (sizes[keep], n))
        mask = lab == keep
        idx = {o: i for i, o in enumerate(np.flatnonzero(mask))}
        sel = [i for i, b in enumerate(br) if mask[b["f"]] and mask[b["t"]]]
        br = [br[i] for i in sel]
        f = np.array([idx[b["f"]] for b in br]); t = np.array([idx[b["t"]] for b in br])
        bsus = bsus[sel]; rating = rating[sel]
        n, m = int(mask.sum()), len(br)
        A = sp.csr_matrix(
            (np.r_[np.ones(m), -np.ones(m)], (np.r_[np.arange(m), np.arange(m)],
                                              np.r_[f, t])), shape=(m, n))
        Bbus = (A.T @ sp.diags(bsus) @ A).tocsc()
        print("  studying        %s nodes, %s branches" %
              ("{:,}".format(n), "{:,}".format(m)))

    slack = 0
    keep = np.r_[np.arange(1, n)]
    Bred = Bbus[keep, :][:, keep]
    t0 = time.perf_counter()
    lu = spl.splu(Bred.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0)
    tfac = time.perf_counter() - t0
    print("\n  ONE FACTORISATION: %s x %s, nnz %s -> factor nnz %s, fill-in "
          "%.3f, %.3f s" % ("{:,}".format(n - 1), "{:,}".format(n - 1),
                            "{:,}".format(Bred.nnz),
                            "{:,}".format(lu.L.nnz + lu.U.nnz),
                            (lu.L.nnz + lu.U.nnz) / max(1, Bred.nnz), tfac))

    # ---- PTDF: branch flow per unit injection at each node
    t0 = time.perf_counter()
    Ared = A[:, keep]
    X = lu.solve(np.asarray(Ared.T.todense()))          # (n-1) x m
    PTDF = (sp.diags(bsus) @ Ared @ X)                   # m x m ... per branch
    PTDF = np.asarray(PTDF)
    tptdf = time.perf_counter() - t0
    print("  PTDF built      %s x %s in %.2f s" %
          ("{:,}".format(PTDF.shape[0]), "{:,}".format(PTDF.shape[1]), tptdf))

    # ---- verification: the identity the whole method rests on
    dl = np.diag(PTDF)
    bad = int(np.sum(np.abs(1.0 - dl) < 1e-9))
    print("\n  VERIFY  branches whose outage splits the network (1 - PTDF_ll "
          "== 0): %d of %s" % (bad, "{:,}".format(m)))
    print("          those are radial and have NO post-outage answer; they are "
          "excluded, not given a number.")

    ok = np.abs(1.0 - dl) >= 1e-9
    LODF = PTDF[:, ok] / (1.0 - dl[ok])[None, :]
    LODF[np.arange(m)[ok], np.arange(int(ok.sum()))] = -1.0
    pairs = int(ok.sum()) * m
    print("  LODF            %s x %s = %s branch/outage pairs"
          % ("{:,}".format(m), "{:,}".format(int(ok.sum())),
             "{:,}".format(pairs)))

    out = {
        "source": d.get("source"), "sites": len(d["sites"]), "nodes": n,
        "branches": m, "skipped": skipped, "islands": ncomp,
        "factor_nnz": int(lu.L.nnz + lu.U.nnz), "bbus_nnz": int(Bred.nnz),
        "fill_in": round((lu.L.nnz + lu.U.nnz) / max(1, Bred.nnz), 4),
        "factorise_s": round(tfac, 4), "ptdf_s": round(tptdf, 3),
        "radial_branches_excluded": bad,
        "lodf_pairs": pairs,
        "n1_cases": int(ok.sum()),
        "n2_pairs": int(ok.sum()) * (int(ok.sum()) - 1) // 2,
        "what_this_is": "DC linearised sensitivity from one factorisation. "
                        "Not an AC study, not a connection assessment. No "
                        "voltage magnitude, reactive power, losses or "
                        "stability. Ratings compared are winter MVA where "
                        "published.",
        "rated_branches": int((rating > 0).sum()),
    }
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "gb-study.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, indent=1) + "\n")
    print("\n  N-1 cases with an answer   %s" % "{:,}".format(out["n1_cases"]))
    print("  N-2 pairs from the same factorisation  %s" % "{:,}".format(out["n2_pairs"]))
    print("  branches carrying a published rating   %s of %s"
          % ("{:,}".format(out["rated_branches"]), "{:,}".format(m)))
    print("\n  wrote gb-study.json")
    print("  DC screening only. Not a connection assessment - the source file "
          "says so and so does this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
