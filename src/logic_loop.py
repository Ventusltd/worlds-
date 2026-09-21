"""logic_loop.py - the logic-checking loop.

Every gate in this estate answers ONE question about ONE predicate.
This asks what the four predicates say about EACH OTHER: implication,
exclusion, dead rules, equivalence, and which rule binds first.

Pair rule: every count is produced twice - once by a CUDA kernel, once by a
CuPy array path - and nothing is reported unless they agree exactly.

Run with the CuPy interpreter named by GRID_PY: $GRID_PY src/logic_loop.py
"""
import json
import os
import sys

import cupy as cp

import paths

# Not written down here: this repository is public and an absolute path
# names a drive, a machine and an account. The root comes from GRID_DATA
# in the environment; see src/paths.py for the whole argument.
MODULES = paths.MODULES
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "night-results", "logic-loop.json")

RULES = ["over_volts", "over_string", "over_mppt", "over_fuse"]

# ---- the stated space -------------------------------------------------
S_LO, S_HI = 20, 34          # series modules a string, inclusive
NS = S_HI - S_LO + 1         # 15
NR = 820                     # rear gain steps, 0 .. 0.30
NT = 820                     # cell temperature steps, -20 .. +25 C
REAR_STEP = 0.30 / (NR - 1)
T_LO = -20.0
T_STEP = 45.0 / (NT - 1)

V_LIMIT = 1500.0             # sourced: printed "1500V DC" on both documents
I_STRING_LIMIT = 20.0        # NO SOURCE
I_MPPT_LIMIT = 40.0          # NO SOURCE


# ---- bins -------------------------------------------------------------
def load_bins():
    paths.require(MODULES, "the module electrical data (MODULES.json)")
    d = json.load(open(MODULES, "r", encoding="utf-8"))
    bins = []
    for c in d["classes"]:
        bins.append(dict(
            class_id=c["class_id"],
            voc=float(c["voc"]["v"]),
            isc=float(c["isc"]["v"]),
            imp=float(c["imp"]["v"]),
            beta=float(c["beta_pct_per_K"]["v"]),
            alpha=float(c["alpha"]["v"]),
            phi=(float(c["k_corr"]["v"]) - 1.0) / 0.135,
            fuse=float(c["max_series_fuse_a"]["v"]),
        ))
    bins.sort(key=lambda b: b["class_id"])
    return bins


# ---- CUDA kernels -----------------------------------------------------
SRC = r'''
extern "C" {

__device__ __forceinline__ void verdicts(
    double voc, double isc, double imp, double beta, double alpha,
    double phi, double fuse, int series, double rear, double T,
    bool *v, bool *s, bool *m, bool *f)
{
    double dT = T - 25.0;
    double voc_T = voc * (1.0 + beta / 100.0 * dT);
    double g = (1.0 + phi * rear) * (1.0 + alpha / 100.0 * dT);
    double isc_T = isc * g;
    double imp_T = imp * g;
    *v = !((double)series * voc_T <= VLIM);
    *s = !(imp_T <= SLIM);
    *m = !(isc_T * 2.0 <= MLIM);
    *f = !(1.25 * isc_T <= fuse);
}

// one thread per case; histogram of the 16 joint truth patterns
__global__ void hist_kernel(const double *p, int nbins,
                            double rstep, double tlo, double tstep,
                            unsigned long long *out)
{
    __shared__ unsigned long long sh[16];
    if (threadIdx.x < 16) sh[threadIdx.x] = 0ULL;
    __syncthreads();

    long long total = (long long)nbins * NS * NR * NT;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        long long t = i;
        int ti = (int)(t % NT); t /= NT;
        int ri = (int)(t % NR); t /= NR;
        int si = (int)(t % NS); t /= NS;
        int bi = (int)t;
        const double *b = p + bi * 7;
        bool v, s, m, f;
        verdicts(b[0], b[1], b[2], b[3], b[4], b[5], b[6],
                 20 + si, ri * rstep, tlo + ti * tstep, &v, &s, &m, &f);
        int code = (v ? 1 : 0) | (s ? 2 : 0) | (m ? 4 : 0) | (f ? 8 : 0);
        atomicAdd(&sh[code], 1ULL);
    }
    __syncthreads();
    if (threadIdx.x < 16) atomicAdd(&out[threadIdx.x], sh[threadIdx.x]);
}

// one thread per line; walks one axis safe -> unsafe, finds the first rule to fail
// axis 0 = series 20->34, axis 1 = rear 0->30%, axis 2 = T +25 down to -20 C
__global__ void order_kernel(const double *p, int nbins, int axis,
                             double rstep, double tlo, double tstep,
                             unsigned long long *out)
{
    int steps = (axis == 0) ? NS : ((axis == 1) ? NR : NT);
    long long nlines = (long long)nbins * NS * NR * NT / steps;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < nlines; i += stride) {
        long long t = i;
        int a, b_, bi;
        if (axis == 0) { a = (int)(t % NT); t /= NT; b_ = (int)(t % NR); t /= NR; }
        else if (axis == 1) { a = (int)(t % NT); t /= NT; b_ = (int)(t % NS); t /= NS; }
        else { a = (int)(t % NR); t /= NR; b_ = (int)(t % NS); t /= NS; }
        bi = (int)t;
        const double *bp = p + bi * 7;
        int first[4] = {-1, -1, -1, -1};
        for (int k = 0; k < steps; ++k) {
            int series; double rear, T;
            if (axis == 0) { series = 20 + k; rear = b_ * rstep; T = tlo + a * tstep; }
            else if (axis == 1) { series = 20 + b_; rear = k * rstep; T = tlo + a * tstep; }
            else { series = 20 + b_; rear = a * rstep; T = tlo + (NT - 1 - k) * tstep; }
            bool v, s, m, f;
            verdicts(bp[0], bp[1], bp[2], bp[3], bp[4], bp[5], bp[6],
                     series, rear, T, &v, &s, &m, &f);
            bool r[4] = {v, s, m, f};
            for (int j = 0; j < 4; ++j)
                if (r[j] && first[j] < 0) first[j] = k;
            if (first[0] >= 0 && first[1] >= 0 && first[2] >= 0 && first[3] >= 0) break;
        }
        int best = 1 << 30;
        for (int j = 0; j < 4; ++j)
            if (first[j] >= 0 && first[j] < best) best = first[j];
        if (best == (1 << 30)) { atomicAdd(&out[4], 1ULL); continue; }
        atomicAdd(&out[5], 1ULL);
        for (int j = 0; j < 4; ++j)
            if (first[j] == best) atomicAdd(&out[j], 1ULL);
    }
}
}
'''


def build(nbins):
    opts = ("-DNS=%d" % NS, "-DNR=%d" % NR, "-DNT=%d" % NT,
            "-DVLIM=%r" % V_LIMIT, "-DSLIM=%r" % I_STRING_LIMIT,
            "-DMLIM=%r" % I_MPPT_LIMIT)
    return cp.RawModule(code=SRC, options=opts)


# ---- CuPy array path (independent implementation) ---------------------
def cupy_hist(bins):
    series = cp.arange(S_LO, S_HI + 1, dtype=cp.float64)
    rear = cp.arange(NR, dtype=cp.float64) * REAR_STEP
    T = T_LO + cp.arange(NT, dtype=cp.float64) * T_STEP
    hist = cp.zeros(16, dtype=cp.uint64)
    for b in bins:
        dT = T - 25.0
        voc_T = b["voc"] * (1.0 + b["beta"] / 100.0 * dT)           # (NT,)
        g = (1.0 + b["phi"] * rear)[:, None] * (1.0 + b["alpha"] / 100.0 * dT)[None, :]
        isc_T = b["isc"] * g                                        # (NR,NT)
        imp_T = b["imp"] * g
        v = ~(series[:, None, None] * voc_T[None, None, :] <= V_LIMIT)
        s = ~(imp_T <= I_STRING_LIMIT)
        m = ~(isc_T * 2.0 <= I_MPPT_LIMIT)
        f = ~(1.25 * isc_T <= b["fuse"])
        rest = (s.astype(cp.int32) * 2) | (m.astype(cp.int32) * 4) | (f.astype(cp.int32) * 8)
        code = v.astype(cp.int32) | rest[None, :, :]
        hist += cp.bincount(code.ravel(), minlength=16).astype(cp.uint64)
    return hist


def cupy_order(bins, axis):
    series = cp.arange(S_LO, S_HI + 1, dtype=cp.float64)
    rear = cp.arange(NR, dtype=cp.float64) * REAR_STEP
    T = T_LO + cp.arange(NT, dtype=cp.float64) * T_STEP
    out = cp.zeros(6, dtype=cp.uint64)
    for b in bins:
        dT = T - 25.0
        voc_T = b["voc"] * (1.0 + b["beta"] / 100.0 * dT)
        g = (1.0 + b["phi"] * rear)[:, None] * (1.0 + b["alpha"] / 100.0 * dT)[None, :]
        isc_T = b["isc"] * g
        imp_T = b["imp"] * g
        shape = (NS, NR, NT)
        v = cp.broadcast_to(~(series[:, None, None] * voc_T[None, None, :] <= V_LIMIT), shape)
        s = cp.broadcast_to(~(imp_T <= I_STRING_LIMIT)[None, :, :], shape)
        m = cp.broadcast_to(~(isc_T * 2.0 <= I_MPPT_LIMIT)[None, :, :], shape)
        f = cp.broadcast_to(~(1.25 * isc_T <= b["fuse"])[None, :, :], shape)
        arrs = [v, s, m, f]
        if axis == 0:      # walk series, lines indexed (rear, T)
            arrs = [cp.ascontiguousarray(a.transpose(1, 2, 0)).reshape(-1, NS) for a in arrs]
        elif axis == 1:    # walk rear, lines indexed (series, T)
            arrs = [cp.ascontiguousarray(a.transpose(0, 2, 1)).reshape(-1, NR) for a in arrs]
        else:              # walk T from +25 down, lines indexed (series, rear)
            arrs = [cp.ascontiguousarray(a[:, :, ::-1]).reshape(-1, NT) for a in arrs]
        steps = arrs[0].shape[1]
        firsts = []
        for a in arrs:
            any_ = a.any(axis=1)
            idx = cp.argmax(a, axis=1)
            firsts.append(cp.where(any_, idx, steps + 1))
        F = cp.stack(firsts, axis=0)                 # (4, lines)
        best = F.min(axis=0)
        fail = best <= steps
        out[5] += cp.uint64(int(fail.sum()))
        out[4] += cp.uint64(int((~fail).sum()))
        for j in range(4):
            out[j] += cp.uint64(int(((F[j] == best) & fail).sum()))
    return out


# ---- derive the logic from the joint histogram ------------------------
def counts_from_hist(h):
    h = [int(x) for x in h]
    single = [sum(h[c] for c in range(16) if (c >> j) & 1) for j in range(4)]
    both = [[sum(h[c] for c in range(16) if ((c >> i) & 1) and ((c >> j) & 1))
             for j in range(4)] for i in range(4)]
    a_not_b = [[single[i] - both[i][j] for j in range(4)] for i in range(4)]
    return single, both, a_not_b


def pattern_name(c):
    parts = [RULES[j] for j in range(4) if (c >> j) & 1]
    return "+".join(parts) if parts else "none"


def main():
    bins = load_bins()
    nb = len(bins)
    total = nb * NS * NR * NT
    params = cp.asarray([[b["voc"], b["isc"], b["imp"], b["beta"],
                          b["alpha"], b["phi"], b["fuse"]] for b in bins],
                        dtype=cp.float64).ravel()
    mod = build(nb)

    # ---- pair 1: joint truth-table histogram
    hk = mod.get_function("hist_kernel")
    h_gpu = cp.zeros(16, dtype=cp.uint64)
    hk((2048,), (256,), (params, cp.int32(nb), cp.float64(REAR_STEP),
                         cp.float64(T_LO), cp.float64(T_STEP), h_gpu))
    cp.cuda.Stream.null.synchronize()
    h_cupy = cupy_hist(bins)
    if not bool((h_gpu == h_cupy).all()):
        sys.stderr.write("PAIR RULE BROKEN: histogram mismatch\n%r\n%r\n"
                         % (h_gpu.tolist(), h_cupy.tolist()))
        return 2
    if int(h_gpu.sum()) != total:
        sys.stderr.write("PAIR RULE BROKEN: histogram total %d != %d\n"
                         % (int(h_gpu.sum()), total))
        return 2

    # ---- pair 2: binding order, per axis
    ok = mod.get_function("order_kernel")
    axes = ["series_20_to_34", "rear_0_to_30pct", "T_plus25_down_to_minus20C"]
    order = {}
    for ax in range(3):
        o_gpu = cp.zeros(6, dtype=cp.uint64)
        ok((2048,), (256,), (params, cp.int32(nb), cp.int32(ax),
                             cp.float64(REAR_STEP), cp.float64(T_LO),
                             cp.float64(T_STEP), o_gpu))
        cp.cuda.Stream.null.synchronize()
        o_cupy = cupy_order(bins, ax)
        if not bool((o_gpu == o_cupy).all()):
            sys.stderr.write("PAIR RULE BROKEN: order axis %d\n%r\n%r\n"
                             % (ax, o_gpu.tolist(), o_cupy.tolist()))
            return 2
        o = [int(x) for x in o_gpu.tolist()]
        order[axes[ax]] = {
            "lines_total": o[4] + o[5],
            "lines_with_no_failure": o[4],
            "lines_with_a_failure": o[5],
            "first_rule_to_fail": {RULES[j]: o[j] for j in range(4)},
        }

    single, both, a_not_b = counts_from_hist(h_gpu)

    implications, exclusions, equivalences, dead = [], [], [], []
    for i in range(4):
        if single[i] == 0:
            dead.append(RULES[i])
        for j in range(4):
            if i == j:
                continue
            if a_not_b[i][j] == 0 and single[i] > 0:
                implications.append({"antecedent": RULES[i], "consequent": RULES[j],
                                     "counterexamples": 0,
                                     "cases_where_antecedent_true": single[i]})
            if i < j:
                if both[i][j] == 0:
                    exclusions.append({"a": RULES[i], "b": RULES[j],
                                       "cases_true_together": 0})
                if a_not_b[i][j] == 0 and a_not_b[j][i] == 0 and single[i] > 0:
                    equivalences.append({"a": RULES[i], "b": RULES[j],
                                         "cases": single[i]})
    implications.sort(key=lambda d: (d["antecedent"], d["consequent"]))
    exclusions.sort(key=lambda d: (d["a"], d["b"]))
    equivalences.sort(key=lambda d: (d["a"], d["b"]))
    dead.sort()

    fail_any = total - int(h_gpu[0])
    doc = {
        "space": {
            "bins": [b["class_id"] for b in bins],
            "series": {"from": S_LO, "to": S_HI, "steps": NS},
            "rear_gain_fraction": {"from": 0.0, "to": 0.30, "steps": NR},
            "cell_temperature_C": {"from": T_LO, "to": 25.0, "steps": NT},
            "total_cases": total,
        },
        "limits": {
            "max_system_volts": {"value": V_LIMIT, "source":
                "FROM-DATASHEET: both source documents print 1500 V DC"},
            "string_current_amps": {"value": I_STRING_LIMIT, "source":
                "NO SOURCE: this limit appears nowhere in the evidence base"},
            "mppt_input_amps": {"value": I_MPPT_LIMIT, "source":
                "NO SOURCE: recorded absent from both source documents"},
            "series_fuse_amps": {"value": 35.0, "source":
                "FROM-DATASHEET: max_series_fuse_a printed on both documents"},
        },
        "pair_rule": {
            "implementations": ["cuda_raw_kernel", "cupy_array_path"],
            "agreed_exactly": True,
        },
        "joint_truth_table": {pattern_name(c): int(h_gpu[c]) for c in range(16)},
        "rule_region_sizes": {RULES[j]: single[j] for j in range(4)},
        "cases_failing_at_least_one_rule": fail_any,
        "q1_implications": implications,
        "q2_exclusions": exclusions,
        "q3_dead_rules": dead,
        "q4_equivalences": equivalences,
        "q5_binding_order": order,
        "what_this_does_not_prove": [
            "This is logic over a stated space at a stated resolution. Outside that box, or between its grid points, nothing here holds.",
            "Two of the four limits have NO SOURCE. The 20 A string limit appears nowhere in the evidence base; the 40 A MPPT limit is recorded absent from both source documents.",
            "Redundancy between two invented constants is not a fact about the world. An implication resting on an unsourced number is a fact about the number, not about any hardware.",
            "No diode, cable, inverter, tracker or wiring loss is modelled. Voc, Isc and Imp are datasheet values scaled by printed coefficients only.",
            "Rear gain is applied as a linear phi from a bin's k_corr; one source document does not define the rear-irradiance condition its k_corr comes from.",
        ],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("cases enumerated: %d   (bins %d x series %d x rear %d x T %d)"
          % (total, nb, NS, NR, NT))
    print("pair rule: CUDA kernel and CuPy array path agree exactly on every count.")
    print("")
    print("REGION SIZES")
    for j in range(4):
        print("  %-12s true in %12d of %d cases" % (RULES[j], single[j], total))
    print("")
    print("IMPLICATIONS (zero counterexamples over the whole space)")
    if not implications:
        print("  none.")
    for d in implications:
        print("  whenever %s is true, %s is true as well." % (d["antecedent"], d["consequent"]))
        print("      so %s can never be the binding constraint on its own; it never "
              "changes an answer %s has not already changed."
              % (d["antecedent"], d["consequent"]))
    print("")
    print("DEAD RULES (never true anywhere in the space)")
    if not dead:
        print("  none - every rule fires somewhere.")
    for r in dead:
        print("  %s never fires. In a verdict list it is decoration." % r)
    print("")
    print("EXCLUSIONS (never true together)")
    if not exclusions:
        print("  none - every pair of rules overlaps somewhere.")
    for d in exclusions:
        print("  %s and %s describe disjoint regions; no design fails both." % (d["a"], d["b"]))
    print("")
    print("EQUIVALENCES (true on exactly the same set)")
    if not equivalences:
        print("  none - no two rules are one rule written twice.")
    for d in equivalences:
        print("  %s and %s are one rule written twice." % (d["a"], d["b"]))
    print("")
    print("BINDING ORDER - walking each axis from safe to unsafe")
    for ax in axes:
        o = order[ax]
        print("  %s: %d of %d lines reach a failure" %
              (ax, o["lines_with_a_failure"], o["lines_total"]))
        for r in RULES:
            print("      %-12s fails first on %12d lines" % (r, o["first_rule_to_fail"][r]))
    print("")
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
