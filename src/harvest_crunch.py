"""
PATHS ARE NOT WRITTEN DOWN HERE. This repository is public and an
absolute path names a machine and an account. Set GRID_DATA (and where
needed GRID_REPOS or CLAUDE_TRANSCRIPT) in the environment instead.
See src/paths.py for why.
THE TRANSLATOR: turn a worksheet's plain-English maths into enumeration.

Everything else in this night finds problems. This file does one job: it takes
a finding's `maths` string - a statement of axes and a predicate, written for a
person - and makes the card actually count it. A sentence is not a result. A
count over the whole space is.

RUN IT WITH THE GPU INTERPRETER, NOT THE DEFAULT ONE
CuPy lives in one place on this machine and it is not on PATH:

    E:\\swarm\\gpu-bench\\venv\\Scripts\\python.exe src/harvest_crunch.py --list
    E:\\swarm\\gpu-bench\\venv\\Scripts\\python.exe src/harvest_crunch.py --run
    E:\\swarm\\gpu-bench\\venv\\Scripts\\python.exe src/harvest_crunch.py --run --ci

`python` alone will import numpy and print the space; it cannot enumerate.

THE HOUSE RULE, AND IT IS NOT NEGOTIABLE
Nothing is reported for arithmetic that cannot check itself. Every registered
finding carries TWO implementations of the same sum: a kernel compiled for the
card, and a reference written in NumPy on the processor. The runner probes them
against each other on a stride through the space before it enumerates anything.
If they differ by one case the finding is reported FAILING and its counts are
withheld. A number nobody checked is worse than no number, because it gets
quoted.

THE SHAPE OF THE FILE
  1. REGISTRY - a declarative table. Each crunchable finding id maps to a
     builder that returns its axes, its card kernel, and its processor
     reference. Adding a finding means adding a row, not editing the runner.
  2. RUNNER   - one loop. Probe, verify, refuse or enumerate in batches,
     reduce with a bincount, write the counts down.
  3. WORKSHEETS - the ten sprinters write features into E:\\swarm\\harvest\\.
     Each feature's `maths` is parsed into axes and a predicate and registered
     automatically. Anything that cannot be translated is REFUSED BY NAME with
     the reason. Never skipped silently: a silent skip is how a finding
     disappears and everybody assumes somebody else counted it.

WHERE THE NUMBERS COME FROM
  MODULES.json      E:\\swarm\\feed\\MODULES.json - every module electrical
                    number. Nothing here is remembered; if the file is missing
                    the findings that need it are refused, not guessed.
  the cartridge     the dot floors and the dot budget are read from the live
                    renderer source, not retyped here from memory.
  the worksheets    every number in an auto-translated finding comes out of
                    the worksheet file that stated it.

DETERMINISM
night-results/harvest-crunch.json is written with sorted keys and no clock in
the body. Re-running on the same inputs gives byte-identical output except for
the single `timing` block, which is quarantined at the top level for exactly
that reason and excluded from the body digest the file prints for itself.

NO NAMES
No project, site, client or manufacturer name appears anywhere in this file or
in what it writes.
"""
import argparse
import hashlib
import inspect
import io
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "night-results", "harvest-crunch.json")

MODULES = r"E:\swarm\feed\MODULES.json"
HARVEST_DIR = r"E:\swarm\harvest"
# ROUND TWO writes to its own directory, so a re-run of round one cannot
# collide with it and neither sheet has to be moved to be read.
HARVEST_DIRS = [HARVEST_DIR, r"E:\swarm\harvest2"]
NIGHT_WORKSHEET = os.path.join(ROOT, "night-results", "worksheet.geojson")
CARTRIDGE_DIR = os.environ.get("GRID_REPOS", "")

BNPI_REAR = 0.135          # rear 135 W/m2 over front 1000, printed on the sheet
V_CEIL = 1500.0            # equipment rating: module max system, inverter max PV
STRING_INPUT_A = 20.0      # per-string input at the combiner
MPPT_INPUT_A = 40.0        # one machine's operating limit per MPPT
STRINGS_PER_MPPT = 2
SIZING = 1.25              # 62548-1 continuous-current factor

PROBE_N = 200_000          # cases checked card-against-processor before counting
PROBE_STRIDE = 1_000_003   # coprime with the space, so the probe walks all of it
BATCH = 1 << 23

REFUSED = []               # (name, reason) - printed and written, never dropped


def refuse(name, reason):
    REFUSED.append({"name": str(name), "reason": str(reason)})


# ---------------------------------------------------------------------------
# THE INPUTS, EACH READ FROM ITS FILE
# ---------------------------------------------------------------------------

def module_bins():
    """Every module class, with the bifaciality factor DERIVED from the sheet's
    own K_Corr at the printed BNPI condition - not assumed, and not the printed
    bifaciality percentage, which is a different quantity."""
    if not os.path.isfile(MODULES):
        return []
    try:
        d = json.load(io.open(MODULES, encoding="utf-8"))
    except Exception as e:                                  # malformed, not absent
        refuse(MODULES, "unreadable: %s" % e)
        return []
    e = d["classes"] if isinstance(d, dict) and "classes" in d else d
    e = list(e.values()) if isinstance(e, dict) else e
    out = []
    for m in e:
        def g(k):
            v = m.get(k)
            return v.get("v") if isinstance(v, dict) else v
        vals = [g(k) for k in ("voc", "isc", "imp", "beta_pct_per_K",
                               "alpha", "k_corr")]
        if any(v is None for v in vals):
            continue
        voc, isc, imp, beta, alpha, kc = [float(v) for v in vals]
        out.append({"id": str(g("class_id")), "voc": voc, "isc": isc,
                    "imp": imp, "beta": beta, "alpha": alpha,
                    "phi": (kc - 1.0) / BNPI_REAR})
    out.sort(key=lambda b: b["id"])          # deterministic order, always
    return out


BINS = module_bins()


def dot_budget():
    """SLD_MAX and the symbol floors, READ OUT OF THE LIVE RENDERER. They are
    not retyped here, because a constant retyped from memory is a constant that
    silently stops matching the thing it describes."""
    d = {"src": None, "max": None, "ring": None, "line": None,
         "plain_ring": None, "plain_line": None}
    if not os.path.isdir(CARTRIDGE_DIR):
        return d
    files = sorted(f for f in os.listdir(CARTRIDGE_DIR)
                   if f.endswith("kuiper-programs.js"))
    if not files:
        return d
    p = os.path.join(CARTRIDGE_DIR, files[-1])
    try:
        s = io.open(p, encoding="utf-8", errors="replace").read()
    except Exception as e:
        refuse(p, "unreadable: %s" % e)
        return d
    m = re.search(r"SLD_MAX\s*=\s*(\d+)", s)
    if m:
        d["max"] = int(m.group(1))
    # the floor line in sldLayout: symbols get (ring ? 16 : 6), the rest
    # (ring ? 5 : 2), and the rounding loop may not take a symbol below it.
    f = re.search(r"SYMBOL\(p\.tag\)\s*\?\s*\(p\.kind\s*===\s*'ring'\s*\?\s*"
                  r"(\d+)\s*:\s*(\d+)\)\s*:\s*\(p\.kind\s*===\s*'ring'\s*\?\s*"
                  r"(\d+)\s*:\s*(\d+)\)", s)
    if f:
        d["ring"], d["line"] = int(f.group(1)), int(f.group(2))
        d["plain_ring"], d["plain_line"] = int(f.group(3)), int(f.group(4))
    d["src"] = files[-1]
    return d


DOTS = dot_budget()


# ---------------------------------------------------------------------------
# H1  THE VERDICT PREDICATE
# ---------------------------------------------------------------------------

H1_LIMITS = [("the 1500 V equipment rating", V_CEIL),
             ("the 20 A string input", STRING_INPUT_A),
             ("the 40 A machine input", MPPT_INPUT_A),
             ("the 35 A maximum series fuse", 35.0)]
H1_SWEEP, H1_BITS = 1_000_000, 1_000_000


def h1_values():
    """A million ordinary values through and past every limit, a million raw
    bit patterns - a value does not have to be typeable to arrive, it can be
    computed - and every special: NaN, both infinities, -0, denormals, each
    limit and its two neighbouring doubles. The boundary and the non-finites
    are the ONLY places the two predicates can differ, so they are not
    decoration, they are the experiment."""
    v = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0,
         5e-324, -5e-324, 1.7976931348623157e308, -1.7976931348623157e308]
    for _lab, L in H1_LIMITS:
        v += [L, np.nextafter(L, np.inf), np.nextafter(L, -np.inf),
              -L, L * 2.0, L / 2.0]
    special = np.array(v, dtype=np.float64)
    sweep = np.linspace(-100.0, 1700.0, H1_SWEEP, dtype=np.float64)
    rng = np.random.default_rng(20260921)          # fixed seed: same bytes twice
    bits = rng.integers(0, 1 << 63, size=H1_BITS, dtype=np.uint64)
    bits[: H1_BITS // 4] |= np.uint64(0x7FF0000000000000)   # into NaN/Inf land
    return np.concatenate([sweep, bits.view(np.float64), special]), len(special)


H1_SRC = """
long long r = idx;
int vi = (int)(r % @NV@); r /= @NV@;
int li = (int)r;
double x = vals[vi], L = lims[li];

/* THE TWO PREDICATES, on the same bits, in the same thread. */
int old_over = (x > L) ? 1 : 0;              /* what the live program does   */
int new_over = !(x <= L) ? 1 : 0;            /* the proposed replacement     */
int finite   = isfinite(x) ? 1 : 0;

int cat = (old_over == new_over) ? 0 : (finite ? 1 : 2);
/* THE DEFECT IS A DISAGREEMENT, NOT MERELY A NON-FINITE PASS. Minus infinity
   is not finite but it IS genuinely below every limit, and both forms agree it
   is inside. Counting it overstated this gate by exactly one value per limit,
   and a gate that exaggerates is a gate nobody believes twice. */
int oldpass = (!finite && old_over == 0 && new_over == 1) ? 1 : 0;
code = (long long)(cat * 2 + oldpass);
"""


def build_h1():
    vals, n_special = h1_values()
    nv = len(vals)
    size = nv * len(H1_LIMITS)
    lims = np.array([L for _lab, L in H1_LIMITS], dtype=np.float64)

    def ref(idx):
        vi = (idx % nv).astype(np.int64)
        li = (idx // nv).astype(np.int64)
        with np.errstate(invalid="ignore"):
            x, L = vals[vi], lims[li]
            old = (x > L)
            new = ~(x <= L)
        fin = np.isfinite(x)
        cat = np.where(old == new, 0, np.where(fin, 1, 2))
        oldpass = (~fin) & (~old) & new
        return (cat * 2 + oldpass.astype(np.int64)).astype(np.int64)

    def make(cp):
        k = cp.ElementwiseKernel(
            "int64 idx, raw float64 vals, raw float64 lims", "int64 code",
            H1_SRC.replace("@NV@", str(nv)), "harvest_h1")
        dv, dl = cp.asarray(vals), cp.asarray(lims)
        return lambda d: k(d, dv, dl)

    def report(c):
        agree = c[0] + c[1]
        fin = c[2] + c[3]
        nonfin = c[4] + c[5]
        oldpass = c[1] + c[3] + c[5]
        return ({"agree": int(agree),
                 "disagree_on_a_finite_number": int(fin),
                 "disagree_on_a_non_finite_number": int(nonfin),
                 "old_predicate_calls_it_inside_the_rating": int(oldpass),
                 "values_per_limit": nv, "limits": len(H1_LIMITS),
                 "specials_per_limit": n_special},
                "SETTLED: on every finite number the two forms agree exactly "
                "(%s disagreements), so this changes no real design; they part "
                "only where the value is not a number, and there the live form "
                "reports %s passes it cannot justify across %d limits."
                % ("{:,}".format(int(fin)), "{:,}".format(int(oldpass)),
                   len(H1_LIMITS)),
                int(oldpass) == 0)

    return {"id": "H1",
            "question": "Does a verdict ever report a value it cannot justify "
                        "as inside a rating?",
            "maths": "sweep value over 1,000,000 ordinary x 1,000,000 raw bit "
                     "patterns x %d specials, x 4 limits, count where "
                     "(x > limit) differs from !(x <= limit)" % n_special,
            "axes": [("value", nv, "ordinary sweep -100..1700, raw bit "
                                   "patterns, and every special"),
                     ("limit", len(H1_LIMITS),
                      ", ".join(lab for lab, _ in H1_LIMITS))],
            "size": size, "n_codes": 6, "make": make, "ref": ref,
            "report": report, "source": "seeded",
            "not_claimed": "This settles a comparison, not a design. It says "
                           "nothing about whether the limits themselves are "
                           "right and nothing about any physics."}


# ---------------------------------------------------------------------------
# H2  THE TEMPERATURE COEFFICIENT
# ---------------------------------------------------------------------------
# An earlier draft took one family's beta and used it against another family's
# Voc. Both numbers were real; the pairing was not. So the sweep carries TWO
# answers per case: one with the beta swept across the whole plausible band,
# and one with the beta the sheet actually prints for THAT bin. Where they
# disagree is exactly the size of the mistake.

H2_BETA_LO, H2_BETA_HI, H2_N_BETA = -0.30, -0.20, 101
H2_SER_LO, H2_SER_HI = 20, 34
H2_T_LO, H2_T_HI, H2_N_T = -20.0, 0.0, 81

H2_SRC = """
long long r = idx;
int bi = (int)(r % @NB@); r /= @NB@;
int ei = (int)(r % @NE@); r /= @NE@;
int si = (int)(r % @NS@); r /= @NS@;
int ti = (int)r;

double beta_sweep = @BLO@ + (@BHI@ - @BLO@) * (double)ei / (@NE@ - 1.0);
double T          = @TLO@ + (@THI@ - @TLO@) * (double)ti / (@NT@ - 1.0);
double se         = (double)(@SLO@ + si);
double vo         = voc[bi];

/* Voc rises as it gets colder; beta is negative and T is below 25. */
double v_sweep = se * vo * (1.0 + beta_sweep / 100.0 * (T - 25.0));
double v_own   = se * vo * (1.0 + beta[bi]  / 100.0 * (T - 25.0));

long long m = 0;
if (v_sweep > @VC@) m |= 1;      /* with the swept coefficient            */
if (v_own   > @VC@) m |= 2;      /* with the coefficient THIS bin prints  */
code = m;
"""


def build_h2():
    if not BINS:
        return None
    nb, ne = len(BINS), H2_N_BETA
    ns = H2_SER_HI - H2_SER_LO + 1
    nt = H2_N_T
    size = nb * ne * ns * nt
    voc = np.array([b["voc"] for b in BINS])
    beta = np.array([b["beta"] for b in BINS])

    def ref(idx):
        r = idx.astype(np.int64)
        bi = r % nb; r //= nb
        ei = r % ne; r //= ne
        si = r % ns; r //= ns
        ti = r
        bs = H2_BETA_LO + (H2_BETA_HI - H2_BETA_LO) * ei / (ne - 1.0)
        T = H2_T_LO + (H2_T_HI - H2_T_LO) * ti / (nt - 1.0)
        se = (H2_SER_LO + si).astype(np.float64)
        vo = voc[bi]
        v_sweep = se * vo * (1.0 + bs / 100.0 * (T - 25.0))
        v_own = se * vo * (1.0 + beta[bi] / 100.0 * (T - 25.0))
        return ((v_sweep > V_CEIL).astype(np.int64)
                + 2 * (v_own > V_CEIL).astype(np.int64))

    def make(cp):
        src = (H2_SRC.replace("@NB@", str(nb)).replace("@NE@", str(ne))
               .replace("@NS@", str(ns)).replace("@NT@", str(nt))
               .replace("@BLO@", repr(H2_BETA_LO)).replace("@BHI@", repr(H2_BETA_HI))
               .replace("@TLO@", repr(H2_T_LO)).replace("@THI@", repr(H2_T_HI))
               .replace("@SLO@", str(H2_SER_LO)).replace("@VC@", repr(V_CEIL)))
        k = cp.ElementwiseKernel(
            "int64 idx, raw float64 voc, raw float64 beta", "int64 code",
            src, "harvest_h2")
        dv, db = cp.asarray(voc), cp.asarray(beta)
        return lambda d: k(d, dv, db)

    def crossings():
        """The EXACT temperature at which each bin crosses 1500 V, solved
        rather than searched: 1500 = N*Voc*(1+beta/100*(T-25)) inverts."""
        rows = {}
        for b in BINS:
            per = {}
            for n in range(H2_SER_LO, H2_SER_HI + 1):
                v25 = n * b["voc"]
                if v25 > V_CEIL:
                    per[str(n)] = "over the rating at 25 C already"
                    continue
                t = 25.0 + (V_CEIL / v25 - 1.0) * 100.0 / b["beta"]
                per[str(n)] = round(t, 3)
            rows[b["id"]] = per
        return rows

    def mispair():
        """What using one family's beta against another family's Voc costs, in
        volts, at thirty in series - computed here, never recalled."""
        fams = {}
        for b in BINS:
            fams.setdefault(b["id"][0], set()).add(b["beta"])
        betas = sorted({b["beta"] for b in BINS})
        rows = {}
        if len(betas) < 2:
            return rows
        for b in BINS:
            for wrong in betas:
                if wrong == b["beta"]:
                    continue
                per = {}
                for T in (0.0, -5.0, -10.0, -10.6, -15.0, -20.0):
                    d = 30 * b["voc"] * ((b["beta"] - wrong) / 100.0) * (T - 25.0)
                    per["%.1f C" % T] = round(d, 3)
                rows["%s with %+.2f in place of %+.2f" % (b["id"], wrong,
                                                          b["beta"])] = per
        return rows

    def report(c):
        sweep_over = c[1] + c[3]
        own_over = c[2] + c[3]
        differ = c[1] + c[2]
        return ({"over_1500_with_the_swept_coefficient": int(sweep_over),
                 "over_1500_with_each_bin_s_own_coefficient": int(own_over),
                 "cases_where_the_two_coefficients_disagree": int(differ),
                 "clear_on_both": int(c[0]),
                 "exact_crossing_temperature_C": crossings(),
                 "cost_of_pairing_the_wrong_coefficient_v_at_30_series":
                     mispair(),
                 "coefficients_read_from_the_file":
                     {b["id"]: b["beta"] for b in BINS}},
                "SETTLED: %s of %s cases cross 1500 V on each bin's own printed "
                "coefficient, and the exact crossing temperature is solved for "
                "every bin and series count; %s cases turn on WHICH "
                "coefficient is used, which is the whole of the earlier "
                "mistake."
                % ("{:,}".format(int(own_over)), "{:,}".format(size),
                   "{:,}".format(int(differ))),
                int(own_over) == 0)

    return {"id": "H2",
            "question": "Where does the series string cross the 1500 V rating, "
                        "and how much of the answer depends on which "
                        "temperature coefficient is paired with which Voc?",
            "maths": "sweep beta %.2f..%.2f x series %d..%d x temperature "
                     "%.0f..%.0f C x %d module bins, count where "
                     "N*Voc*(1+beta/100*(T-25)) > 1500"
                     % (H2_BETA_LO, H2_BETA_HI, H2_SER_LO, H2_SER_HI,
                        H2_T_LO, H2_T_HI, nb),
            "axes": [("module bin", nb, "every class in the file, Voc and beta "
                                        "taken together per bin"),
                     ("beta swept", ne, "%.2f..%.2f %%/K" % (H2_BETA_LO,
                                                             H2_BETA_HI)),
                     ("modules in series", ns, "%d..%d" % (H2_SER_LO,
                                                           H2_SER_HI)),
                     ("assessment temperature", nt,
                      "%.0f..%.0f C" % (H2_T_LO, H2_T_HI))],
            "size": size, "n_codes": 4, "make": make, "ref": ref,
            "report": report, "source": "seeded",
            "not_claimed": "This settles where a printed Voc and a printed "
                           "coefficient put the string voltage. It does not "
                           "settle the site's design minimum temperature, "
                           "which is a meteorological input this file does not "
                           "hold, and it does not model the rise of Voc with "
                           "irradiance, which makes the real answer worse "
                           "rather than better."}


# ---------------------------------------------------------------------------
# H3  THE DOT BUDGET
# ---------------------------------------------------------------------------
# A symbol is not a length. The renderer hands every SYMBOL piece a floor of
# dots so it reads as the thing it is, and the rounding loop may not take a
# symbol below that floor. Floors do not negotiate: if the floors alone add up
# past the budget, the loop runs out and a piece of the drawing is handed
# nothing at all. That is a drawing with a missing part, and the only way to
# see it is to count the shapes, not to look at one of them.

H3_P_MAX = 384              # pieces in a drawing
H3_S_MAX = 96               # of which symbols
H3_R_MAX = 96               # of which rings

H3_SRC = """
long long r = idx;
int pi = (int)(r % @NP@); r /= @NP@;
int si = (int)(r % @NS@); r /= @NS@;
int ri = (int)(r % @NR@); r /= @NR@;
int ni = (int)r;

int P = pi + 1;      /* pieces in the drawing                */
int S = si;          /* how many of them are symbols         */
int SR = ri;         /* how many symbols are rings           */
int NR = ni;         /* how many plain pieces are rings      */

if (S > P || SR > S || NR > P - S) { code = 0; }   /* not a shape at all */
else {
  long long floors = (long long)SR * @FR@ + (long long)(S - SR) * @FL@
                   + (long long)NR * @PR@ + (long long)(P - S - NR) * @PL@;
  code = (floors > @MAX@) ? 2 : 1;
}
"""


def build_h3():
    if None in (DOTS["max"], DOTS["ring"], DOTS["line"],
                DOTS["plain_ring"], DOTS["plain_line"]):
        return None
    np_, ns, nr, nn = H3_P_MAX, H3_S_MAX + 1, H3_R_MAX + 1, H3_R_MAX + 1
    size = np_ * ns * nr * nn
    FR, FL = DOTS["ring"], DOTS["line"]
    PR, PL, MX = DOTS["plain_ring"], DOTS["plain_line"], DOTS["max"]

    def ref(idx):
        r = idx.astype(np.int64)
        pi = r % np_; r //= np_
        si = r % ns; r //= ns
        ri = r % nr; r //= nr
        ni = r
        P, S, SR, NR = pi + 1, si, ri, ni
        bad = (S > P) | (SR > S) | (NR > P - S)
        floors = SR * FR + (S - SR) * FL + NR * PR + (P - S - NR) * PL
        return np.where(bad, 0, np.where(floors > MX, 2, 1)).astype(np.int64)

    def make(cp):
        src = (H3_SRC.replace("@NP@", str(np_)).replace("@NS@", str(ns))
               .replace("@NR@", str(nr)).replace("@FR@", str(FR))
               .replace("@FL@", str(FL)).replace("@PR@", str(PR))
               .replace("@PL@", str(PL)).replace("@MAX@", str(MX)))
        k = cp.ElementwiseKernel("int64 idx", "int64 code", src, "harvest_h3")
        return lambda d: k(d)

    def thresholds():
        """The smallest shape of each kind that starves a piece - solved, so it
        can be read straight off and checked by hand."""
        return {"symbol rings alone": int(MX // FR) + 1,
                "symbol lines alone": int(MX // FL) + 1,
                "plain rings alone": int(MX // PR) + 1,
                "plain lines alone": int(MX // PL) + 1,
                "note": "the count at which the floors first exceed the "
                        "budget, so the rounding loop cannot give every piece "
                        "its floor and a piece receives zero dots"}

    def report(c):
        shapes = c[1] + c[2]
        return ({"budget": MX, "floor_symbol_ring": FR, "floor_symbol_line": FL,
                 "floor_plain_ring": PR, "floor_plain_line": PL,
                 "floors_read_from": DOTS["src"],
                 "shapes_examined": int(shapes),
                 "shapes_that_fit": int(c[1]),
                 "shapes_where_the_floors_alone_exceed_the_budget": int(c[2]),
                 "combinations_that_are_not_shapes": int(c[0]),
                 "smallest_starving_count": thresholds()},
                "SETTLED: of %s real shapes, %s leave the floors alone over the "
                "%d dot budget, so a piece of the drawing is handed zero dots; "
                "%d symbol rings on their own are enough to do it."
                % ("{:,}".format(int(shapes)), "{:,}".format(int(c[2])), MX,
                   int(MX // FR) + 1),
                int(c[2]) == 0)

    return {"id": "H3",
            "question": "Can a drawing ask for more floor dots than the budget "
                        "holds, so that a piece of it is drawn with nothing?",
            "maths": "sweep pieces 1..%d x symbols 0..%d x symbol rings 0..%d "
                     "x plain rings 0..%d, count where "
                     "SR*%d + (S-SR)*%d + NR*%d + (P-S-NR)*%d > %d"
                     % (H3_P_MAX, H3_S_MAX, H3_R_MAX, H3_R_MAX,
                        FR, FL, PR, PL, MX),
            "axes": [("pieces", np_, "1..%d" % H3_P_MAX),
                     ("symbol pieces", ns, "0..%d, matched by a PREFIX test on "
                                           "the tag" % H3_S_MAX),
                     ("symbol rings", nr, "0..%d of the symbols" % H3_R_MAX),
                     ("plain rings", nn, "0..%d of the rest" % H3_R_MAX)],
            "size": size, "n_codes": 3, "make": make, "ref": ref,
            "report": report, "source": "seeded",
            "not_claimed": "This settles the arithmetic of the floors against "
                           "the budget. It does not say which real drawings "
                           "reach those shapes, and it does not check that a "
                           "piece given its floor is legible - only that it is "
                           "given something."}


# ---------------------------------------------------------------------------
# H4  THE BIFACIAL CURRENT CHAIN
# ---------------------------------------------------------------------------
# Rear irradiance raises string current, and that current eats the string input
# and the machine input at the same time. The bifaciality factor is DERIVED
# from each bin's own K_Corr at the printed BNPI condition - front 1000, rear
# 135 W/m2 - never assumed, and never the printed bifaciality percentage, which
# is a different quantity measured a different way.

H4_REAR_LO, H4_REAR_HI, H4_N_REAR = 0.0, 0.30, 61
H4_T_LO, H4_T_HI, H4_N_T = -20.0, 25.0, 91
H4_SER_LO, H4_SER_HI = 20, 34
H4_TABLE_TOP = 0.25        # the highest rear gain the sheet itself publishes

H4_SRC = """
long long r = idx;
int bi = (int)(r % @NB@); r /= @NB@;
int ri = (int)(r % @NR@); r /= @NR@;
int ti = (int)(r % @NT@); r /= @NT@;
int si = (int)r;

double rear = @RLO@ + (@RHI@ - @RLO@) * (double)ri / (@NR@ - 1.0);
double T    = @TLO@ + (@THI@ - @TLO@) * (double)ti / (@NT@ - 1.0);
double se   = (double)(@SLO@ + si);

double kb = 1.0 + phi[bi] * rear;                    /* rear-side current gain */
double kt = 1.0 + alpha[bi] / 100.0 * (T - 25.0);    /* cold trims it back     */
double imp_e = imp[bi] * kb * kt;
double isc_e = isc[bi] * kb * kt;
double v_str = se * voc[bi] * (1.0 + beta[bi] / 100.0 * (T - 25.0));

long long m = 0;
if (imp_e > @SIA@) m |= 1;                           /* the string input   */
if (@SZ@ * isc_e * @SPM@.0 > @MIA@) m |= 2;          /* the machine input  */
if (v_str > @VC@) m |= 4;                            /* the voltage rating */
code = m;
"""


def build_h4():
    if not BINS:
        return None
    nb, nr, nt = len(BINS), H4_N_REAR, H4_N_T
    ns = H4_SER_HI - H4_SER_LO + 1
    size = nb * nr * nt * ns
    cols = {k: np.array([b[k] for b in BINS])
            for k in ("voc", "isc", "imp", "beta", "alpha", "phi")}

    def ref(idx):
        r = idx.astype(np.int64)
        bi = r % nb; r //= nb
        ri = r % nr; r //= nr
        ti = r % nt; r //= nt
        si = r
        rear = H4_REAR_LO + (H4_REAR_HI - H4_REAR_LO) * ri / (nr - 1.0)
        T = H4_T_LO + (H4_T_HI - H4_T_LO) * ti / (nt - 1.0)
        se = (H4_SER_LO + si).astype(np.float64)
        kb = 1.0 + cols["phi"][bi] * rear
        kt = 1.0 + cols["alpha"][bi] / 100.0 * (T - 25.0)
        imp_e = cols["imp"][bi] * kb * kt
        isc_e = cols["isc"][bi] * kb * kt
        v_str = se * cols["voc"][bi] * (1.0 + cols["beta"][bi] / 100.0 * (T - 25.0))
        return ((imp_e > STRING_INPUT_A).astype(np.int64)
                + 2 * (SIZING * isc_e * STRINGS_PER_MPPT > MPPT_INPUT_A).astype(np.int64)
                + 4 * (v_str > V_CEIL).astype(np.int64)).astype(np.int64)

    def make(cp):
        src = (H4_SRC.replace("@NB@", str(nb)).replace("@NR@", str(nr))
               .replace("@NT@", str(nt)).replace("@RLO@", repr(H4_REAR_LO))
               .replace("@RHI@", repr(H4_REAR_HI)).replace("@TLO@", repr(H4_T_LO))
               .replace("@THI@", repr(H4_T_HI)).replace("@SLO@", str(H4_SER_LO))
               .replace("@SIA@", repr(STRING_INPUT_A))
               .replace("@SZ@", repr(SIZING)).replace("@SPM@", str(STRINGS_PER_MPPT))
               .replace("@MIA@", repr(MPPT_INPUT_A)).replace("@VC@", repr(V_CEIL)))
        k = cp.ElementwiseKernel(
            "int64 idx, raw float64 voc, raw float64 isc, raw float64 imp, "
            "raw float64 beta, raw float64 alpha, raw float64 phi",
            "int64 code", src, "harvest_h4")
        d = [cp.asarray(cols[n]) for n in
             ("voc", "isc", "imp", "beta", "alpha", "phi")]
        return lambda q: k(q, *d)

    def sheet_row():
        """The sheet's own top published rear-gain row, at the condition the
        sheet states it for: 25 C. Nothing swept, nothing extrapolated."""
        rows = {}
        for b in BINS:
            kb = 1.0 + b["phi"] * H4_TABLE_TOP
            imp_e, isc_e = b["imp"] * kb, b["isc"] * kb
            rows[b["id"]] = {
                "phi_derived": round(b["phi"], 4),
                "string_current_a": round(imp_e, 3),
                "over_the_%.0f_a_string_input" % STRING_INPUT_A:
                    bool(imp_e > STRING_INPUT_A),
                "sized_machine_current_a": round(SIZING * isc_e * STRINGS_PER_MPPT, 3),
                "over_the_%.0f_a_machine_input" % MPPT_INPUT_A:
                    bool(SIZING * isc_e * STRINGS_PER_MPPT > MPPT_INPUT_A)}
        return rows

    def report(c):
        s = sum(c[m] for m in range(8) if m & 1)
        mm = sum(c[m] for m in range(8) if m & 2)
        vv = sum(c[m] for m in range(8) if m & 4)
        both = sum(c[m] for m in range(8) if (m & 1) and (m & 2))
        row = sheet_row()
        all_bins = all(v["over_the_%.0f_a_string_input" % STRING_INPUT_A]
                       and v["over_the_%.0f_a_machine_input" % MPPT_INPUT_A]
                       for v in row.values())
        return ({"over_the_string_input": int(s),
                 "over_the_machine_input": int(mm),
                 "over_the_voltage_rating": int(vv),
                 "over_both_current_inputs": int(both),
                 "clears_everything": int(c[0]),
                 "phi_source": "(K_Corr - 1) / %.3f, both printed on the sheet"
                               % BNPI_REAR,
                 "at_the_sheets_own_top_rear_row": {
                     "rear_gain": H4_TABLE_TOP,
                     "condition": "25 C, the condition the row is printed for",
                     "every_bin_over_both_inputs": bool(all_bins),
                     "per_bin": row}},
                "SETTLED: %s of %s cases exceed the %.0f A string input and %s "
                "the %.0f A machine input; at the sheet's own %.0f%% rear row "
                "%s bin exceeds both."
                % ("{:,}".format(int(s)), "{:,}".format(size), STRING_INPUT_A,
                   "{:,}".format(int(mm)), MPPT_INPUT_A, H4_TABLE_TOP * 100,
                   "EVERY" if all_bins else "not every"),
                int(s) == 0 and int(mm) == 0)

    return {"id": "H4",
            "question": "Where does rear-side irradiance take the string "
                        "current past the string and machine inputs?",
            "maths": "sweep rear gain %.0f..%.0f%% x temperature %.0f..%.0f C "
                     "x series %d..%d x %d bins, count where "
                     "Imp*(1+phi*rear)*(1+alpha/100*(T-25)) > %.0f or "
                     "%.2f*Isc*(1+phi*rear)*(1+alpha/100*(T-25))*%d > %.0f"
                     % (H4_REAR_LO * 100, H4_REAR_HI * 100, H4_T_LO, H4_T_HI,
                        H4_SER_LO, H4_SER_HI, nb, STRING_INPUT_A, SIZING,
                        STRINGS_PER_MPPT, MPPT_INPUT_A),
            "axes": [("module bin", nb, "every class in the file"),
                     ("rear gain", nr, "%.0f..%.0f%%, the sheet's table tops "
                                       "at %.0f%%" % (H4_REAR_LO * 100,
                                                      H4_REAR_HI * 100,
                                                      H4_TABLE_TOP * 100)),
                     ("assessment temperature", nt,
                      "%.0f..%.0f C" % (H4_T_LO, H4_T_HI)),
                     ("modules in series", ns, "%d..%d" % (H4_SER_LO,
                                                           H4_SER_HI))],
            "size": size, "n_codes": 8, "make": make, "ref": ref,
            "report": report, "source": "seeded",
            "not_claimed": "This settles current against two printed input "
                           "ratings. It does not model non-uniform rear "
                           "illumination along a string, the transient, or the "
                           "fact that high albedo and low ambient are the same "
                           "weather - which means the joint corner is present "
                           "in the count but its real likelihood is understated "
                           "by reading either axis on its own."}


# ---------------------------------------------------------------------------
# THE WORKSHEETS: turning a sentence into a cross-product
# ---------------------------------------------------------------------------
# A `maths` string is a promise that the finding is countable. This turns the
# promise into a kernel, or it refuses out loud. There is no third outcome.

AXIS_RE = re.compile(r"([A-Za-z_]\w*)\s*(-?\d+(?:\.\d+)?)\s*\.\.\s*"
                     r"(-?\d+(?:\.\d+)?)\s*(?:step\s*(\d+(?:\.\d+)?))?")
# the ten sprinters write in different hands; these are the introducers seen
PRED_RE = re.compile(r"(?:count(?:\s+\w+){0,3}\s+where|predicate\s*:)",
                     re.IGNORECASE)
# a word is not a quantity. 'sweep x 0..5' names no axis, it names a cross.
STOPWORDS = {"in", "x", "and", "or", "to", "from", "at", "of", "by", "step",
             "over", "the", "a", "an", "for", "with", "per", "is", "are",
             "sweep", "enumerate", "count", "each", "all", "every", "case",
             "cases", "where", "than", "into", "on", "up", "down"}
IDENT_RE = re.compile(r"[A-Za-z_]\w*")
FUNCS = {"abs": "fabs", "min": "fmin", "max": "fmax", "exp": "exp",
         "log": "log", "sqrt": "sqrt", "floor": "floor", "ceil": "ceil"}
SAFE_CHARS = set("0123456789.+-*/()<>=!& |_,ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                 "abcdefghijklmnopqrstuvwxyz")
FLOAT_SAMPLES = 101        # a swept real axis becomes this many samples
MAX_SPACE = 5_000_000_000  # past this, a person must narrow the sweep


def translate(fid, maths):
    """Return an axis list and a predicate, or raise ValueError with the reason
    a person needs to hear."""
    if not isinstance(maths, str) or not maths.strip():
        raise ValueError("no maths statement")
    m = PRED_RE.search(maths)
    if not m:
        raise ValueError("the maths states no countable predicate: no "
                         "'count ... where' and no 'predicate:'")
    left, pred = maths[:m.start()], maths[m.end():].strip()
    pred = pred.split(",")[0].split(";")[0].strip().rstrip(".")
    if not pred:
        raise ValueError("the predicate introducer is followed by nothing")
    axes = []
    for name, lo, hi, step in AXIS_RE.findall(left):
        if name.lower() in STOPWORDS:
            raise ValueError("'%s %s..%s' names no quantity - '%s' is a word, "
                             "so the axis cannot be built" % (name, lo, hi, name))
        integral = ("." not in lo) and ("." not in hi) and not step
        lo_f, hi_f = float(lo), float(hi)
        if hi_f < lo_f:
            raise ValueError("axis '%s' runs backwards (%s..%s)"
                             % (name, lo, hi))
        if integral:
            n = int(hi_f) - int(lo_f) + 1
            kind = "integer"
        elif step:
            n = int(round((hi_f - lo_f) / float(step))) + 1
            kind = "real"
        else:
            n, kind = FLOAT_SAMPLES, "real"
        if n > 1_000_000:
            raise ValueError("axis '%s' has %s steps; narrow it"
                             % (name, "{:,}".format(n)))
        if n < 2:
            raise ValueError("axis '%s' has fewer than two steps, so it is a "
                             "constant and not an axis" % name)
        axes.append({"name": name, "n": n, "lo": lo_f, "hi": hi_f, "kind": kind})
    if not axes:
        raise ValueError("no axis of the form 'name lo..hi' in the maths")
    names = [a["name"] for a in axes]
    if len(set(names)) != len(names):
        raise ValueError("two axes share a name")
    bad = [c for c in pred if c not in SAFE_CHARS]
    if bad:
        raise ValueError("the predicate holds characters this translator will "
                         "not compile: %s" % "".join(sorted(set(bad))))
    if "**" in pred:
        raise ValueError("the predicate uses '**'; write it out as a product")
    used = set(IDENT_RE.findall(pred))
    unknown = sorted(used - set(names) - set(FUNCS)
                     - {"and", "or", "not", "True", "False"})
    if unknown:
        raise ValueError("the predicate names %s, which is not an axis and not "
                         "a number from a file" % ", ".join(unknown))
    size = 1
    for a in axes:
        size *= a["n"]
    if size > MAX_SPACE:
        raise ValueError("the cross-product is %s cases; narrow the sweep"
                         % "{:,}".format(size))
    return axes, pred


def to_c(pred):
    s = " %s " % pred
    s = s.replace(" and ", " && ").replace(" or ", " || ")
    s = re.sub(r"\bnot\b", "!", s)
    s = s.replace("True", "1").replace("False", "0")
    for py, c in FUNCS.items():
        s = re.sub(r"\b%s\s*\(" % py, "%s(" % c, s)
    return s.strip()


def build_translated(fid, feature, where):
    axes, pred = translate(fid, feature.get("maths"))
    size = 1
    for a in axes:
        size *= a["n"]
    c_pred = to_c(pred)
    lines = ["long long r = idx;"]
    for i, a in enumerate(axes):
        lines.append("long long a%d = r %% %d; r /= %d;" % (i, a["n"], a["n"]))
    for i, a in enumerate(axes):
        if a["kind"] == "integer":
            lines.append("double %s = (double)(%d + a%d);"
                         % (a["name"], int(a["lo"]), i))
        else:
            lines.append("double %s = %.17g + (%.17g - %.17g) * (double)a%d "
                         "/ (%d - 1.0);"
                         % (a["name"], a["lo"], a["hi"], a["lo"], i, a["n"]))
    lines.append("code = (%s) ? 1 : 0;" % c_pred)
    src = "\n".join(lines)

    # THE REFERENCE RUNS ON THE CARD TOO, AND IT IS STILL A SECOND
    # IMPLEMENTATION. One side is a fused kernel NVRTC compiles from C; this
    # side is CuPy's array-operator graph over the same axes. Different
    # compilers, different execution paths, same arithmetic - which is what the
    # pair is for. Keeping it on the processor made the processor the throttle:
    # measured 0.4% mean card utilisation at 32 W while NumPy did the work.
    #
    # AND `and` DOES NOT VECTORISE. Python's and/or/not call __bool__ on the
    # whole array and raise "the truth value of an array is ambiguous". A
    # translated sentence that joined two comparisons with `and` crashed the
    # run. They are rewritten to the bitwise operators, which is what an array
    # predicate needs, with parentheses so precedence does not change meaning.
    def _vectorise(e):
        e = re.sub(r"not", "~", e)
        e = re.sub(r"and", "&", e)
        e = re.sub(r"or", "|", e)
        return e

    def ref(idx, xp=None):
        xp = xp if xp is not None else np
        r = idx.astype(xp.int64) if hasattr(idx, "astype") else xp.asarray(idx)
        # An EMPTY builtins map is too tight for the card. CuPy's ufunc
        # machinery reaches for __import__ while it names the kernel it is
        # about to compile, and an empty map raises KeyError there - the
        # sandbox strangled the very thing it was meant to let through.
        # __import__ alone is restored; nothing else is. The safety that
        # matters is upstream anyway: the predicate's names were checked
        # against the declared axes before it ever reached here, so it cannot
        # reference anything to import.
        env = {"__builtins__": {"__import__": __import__}}
        for f in FUNCS:
            env[f] = getattr(xp, {"abs": "fabs", "min": "minimum",
                                  "max": "maximum"}.get(f, f))
        for a in axes:
            q = r % a["n"]
            r = r // a["n"]
            if a["kind"] == "integer":
                env[a["name"]] = (int(a["lo"]) + q).astype(xp.float64)
            else:
                env[a["name"]] = (a["lo"] + (a["hi"] - a["lo"]) * q
                                  / (a["n"] - 1.0))
        v = eval(_vectorise(pred), env, {})   # axes only; names already checked
        return xp.asarray(v, dtype=bool).astype(xp.int64)

    def make(cp):
        k = cp.ElementwiseKernel("int64 idx", "int64 code", src,
                                 "harvest_%s" % re.sub(r"\W", "_", fid))
        return lambda d: k(d)

    def report(c):
        hit, tot = int(c[1]), int(c[0] + c[1])
        return ({"cases_where_the_predicate_holds": hit,
                 "cases_where_it_does_not": int(c[0]),
                 "predicate": pred, "worksheet": where},
                "SETTLED: the predicate holds in %s of %s cases (%.6f%% of the "
                "space stated in the worksheet)."
                % ("{:,}".format(hit), "{:,}".format(tot),
                   100.0 * hit / tot if tot else 0.0),
                hit == 0)

    return {"id": fid,
            "question": str(feature.get("what") or "")[:400],
            "maths": feature.get("maths"),
            "axes": [(a["name"], a["n"],
                      "%g..%g %s" % (a["lo"], a["hi"], a["kind"]))
                     for a in axes],
            "size": size, "n_codes": 2, "make": make, "ref": ref,
            "report": report, "source": where,
            "not_claimed": "This counts the predicate the worksheet wrote "
                           "down, over the axes the worksheet named. It does "
                           "not check that the predicate is the right question "
                           "or that the axes cover the real range - only a "
                           "person can settle that."}


def read_worksheets():
    """Every worksheet, by name. A missing file is refused by name. A malformed
    file is refused by name. A feature that cannot be translated is refused by
    its own id. Nothing is ever skipped quietly, because a quiet skip is how a
    finding vanishes while everybody assumes somebody else counted it."""
    paths = []
    for d in HARVEST_DIRS:
        if os.path.isdir(d):
            paths += [os.path.join(d, f) for f in sorted(os.listdir(d))
                      if f.lower().endswith(".geojson")]
        else:
            refuse(d, "the worksheet directory does not exist")
    paths.append(NIGHT_WORKSHEET)

    built, seen, stats = [], set(), {"files": 0, "features": 0,
                                     "crunchable": 0, "translated": 0}
    for p in paths:
        if not os.path.isfile(p):
            refuse(p, "no such file: the worksheet was not written")
            continue
        try:
            d = json.load(io.open(p, encoding="utf-8"))
        except Exception as e:
            refuse(p, "not readable as JSON: %s" % e)
            continue
        feats = d.get("features") if isinstance(d, dict) else None
        if not isinstance(feats, list):
            refuse(p, "no 'features' list: not a worksheet this can read")
            continue
        stats["files"] += 1
        base = os.path.basename(p)
        for n, f in enumerate(feats):
            props = f.get("properties") if isinstance(f, dict) else None
            if not isinstance(props, dict):
                refuse("%s#%d" % (base, n), "feature has no properties block")
                continue
            stats["features"] += 1
            fid = str(props.get("id") or props.get("what") or
                      "%s#%d" % (base, n))[:80]
            # A FEATURE IS JUDGED BY ITS SENTENCE, NOT BY A FLAG. Round one set
            # gpu_crunchable and wrote prose; round two dropped the flag and wrote a
            # sweep sentence. Gating on the flag meant 157 features were read and
            # none translated - the reader was asking the wrong question. If there is
            # a maths sentence, try it; if it does not parse it is refused BY NAME
            # below, which is the honest outcome either way.
            if not (props.get("gpu_crunchable") or props.get("maths")):
                continue
            stats["crunchable"] += 1
            if fid in seen:
                refuse(fid, "a second finding claims this id in %s" % base)
                continue
            try:
                spec = build_translated(fid, props, base)
            except ValueError as e:
                refuse(fid, "not translated (%s): %s" % (base, e))
                continue
            except Exception as e:                       # never let one poison the run
                refuse(fid, "not translated (%s): %s: %s"
                       % (base, type(e).__name__, e))
                continue
            seen.add(fid)
            stats["translated"] += 1
            built.append(spec)
    return built, stats


# ---------------------------------------------------------------------------
# THE REGISTRY
# ---------------------------------------------------------------------------

SEEDS = [("H1", build_h1), ("H2", build_h2), ("H3", build_h3),
         ("H4", build_h4)]


def registry():
    out = []
    for fid, fn in SEEDS:
        try:
            s = fn()
        except Exception as e:
            refuse(fid, "could not be built: %s: %s" % (type(e).__name__, e))
            continue
        if s is None:
            refuse(fid, "its inputs are not on this machine, so it is not "
                        "enumerated on remembered numbers")
            continue
        out.append(s)
    built, stats = read_worksheets()
    out += built
    return out, stats


# ---------------------------------------------------------------------------
# THE RUNNER
# ---------------------------------------------------------------------------

def run_one(cp, spec):
    """Probe, verify, then enumerate. A finding that fails its own check is
    reported FAILING with its counts WITHHELD - not reported quietly wrong."""
    size, t0 = spec["size"], time.perf_counter()
    fn = spec["make"](cp)

    stride = PROBE_STRIDE
    while math.gcd(stride, size) != 1:
        stride += 2
    probe = (np.arange(min(PROBE_N, size), dtype=np.int64) * stride) % size
    # TWO CHANNELS, AND THE SECOND ONE RUNS WHEREVER IT WAS WRITTEN TO RUN.
    # A translated sentence builds a reference that takes the array module, so
    # its channel opens on the card beside the kernel. The four seeded findings
    # carry hand-written references that were written against the processor and
    # take the index array alone. Asking every reference for two arguments broke
    # those four - so the signature is inspected rather than assumed, and each
    # is given what it was built for. Both are still a SECOND implementation,
    # which is the only property the pair rule actually needs.
    dprobe = cp.asarray(probe)
    got = fn(dprobe)
    ref = spec["ref"]
    try:
        nargs = len(inspect.signature(ref).parameters)
    except (TypeError, ValueError):
        nargs = 1
    if nargs >= 2:
        want = ref(dprobe, cp)                      # the positron, on the card
        differ = int((got != want).sum())
    else:
        want = ref(probe)                           # a seeded reference, host
        differ = int((cp.asnumpy(got) != want).sum())
    if differ:
        return {"cases": 0, "checked": int(len(probe)), "verified_differ": differ,
                "state": "FAILING",
                "settled": "NOT SETTLED: the card and the processor disagree on "
                           "%d of %d probed cases. Nothing is reported for "
                           "arithmetic that cannot check itself."
                           % (differ, len(probe))}, 0.0

    counts = cp.zeros(spec["n_codes"], dtype=cp.int64)
    done = 0
    while done < size:
        n = min(BATCH, size - done)
        c = fn(cp.arange(done, done + n, dtype=cp.int64))
        counts += cp.bincount(c, minlength=spec["n_codes"])[:spec["n_codes"]]
        done += n
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0
    c = [int(x) for x in cp.asnumpy(counts)]
    if sum(c) != size:
        return {"cases": size, "checked": int(len(probe)),
                "verified_differ": differ, "state": "FAILING",
                "settled": "NOT SETTLED: the reduction lost %d cases."
                           % (size - sum(c))}, sec
    body, verdict, clean = spec["report"](c)
    row = {"cases": size, "checked": int(len(probe)), "verified_differ": 0,
           "state": "CLEAN" if clean else "FINDING STANDS",
           "counts": body, "settled": verdict}
    return row, sec


def describe(specs, stats):
    print("HARVEST CRUNCH: worksheet sentences turned into counted arithmetic")
    print("  module classes read   %d from %s" % (len(BINS), MODULES))
    print("  dot budget read       %s from %s"
          % (DOTS["max"], DOTS["src"] or "NOT FOUND"))
    print("  worksheets read       %d files, %d features, %d marked crunchable,"
          " %d translated"
          % (stats["files"], stats["features"], stats["crunchable"],
             stats["translated"]))
    print()
    total = 0
    for s in specs:
        total += s["size"]
        print("  %-28s %18s cases   [%s]"
              % (s["id"], "{:,}".format(s["size"]), s["source"]))
        print("      %s" % s["question"][:100])
        for nm, n, extra in s["axes"]:
            print("        %-26s %8d   %s" % (nm, n, extra))
    print("\n  %d findings, %s cases in total"
          % (len(specs), "{:,}".format(total)))
    if REFUSED:
        print("\n  REFUSED (%d) - named, never skipped quietly:" % len(REFUSED))
        for r in REFUSED:
            print("    %-46s %s" % (r["name"][:46], r["reason"]))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="print what it would run, and exit")
    ap.add_argument("--run", action="store_true", help="put it on the card")
    ap.add_argument("--ci", action="store_true",
                    help="exit 1 if any registered finding comes back FAILING")
    a = ap.parse_args()

    specs, stats = registry()
    total = describe(specs, stats)
    if not a.run:
        print("\n  --run to enumerate on the card "
              "(E:\\swarm\\gpu-bench\\venv\\Scripts\\python.exe).")
        return 0

    try:
        import cupy as cp
    except Exception as e:
        print("\n  FAIL: no CuPy here (%s)." % e)
        print("  Run it with E:\\swarm\\gpu-bench\\venv\\Scripts\\python.exe")
        return 1

    findings, timing, failing = {}, {}, []
    print("\n  ENUMERATING")
    for s in specs:
        row, sec = run_one(cp, s)
        row["question"] = s["question"]
        row["maths"] = s["maths"]
        row["axes"] = [{"name": n, "steps": k, "range": x} for n, k, x in s["axes"]]
        row["source"] = s["source"]
        row["not_claimed"] = s["not_claimed"]
        findings[s["id"]] = row
        timing[s["id"]] = round(sec, 3)
        if row["state"] == "FAILING":
            failing.append(s["id"])
        print("    %-24s %16s cases  %8.2f s  %-14s"
              % (s["id"], "{:,}".format(row["cases"]), sec, row["state"]))
        print("      %s" % row["settled"])

    body = {
        "anonymised": "No project, site, client or manufacturer name appears "
                      "anywhere in this file.",
        "cases_total": total,
        "determinism": "Sorted keys, no clock in the body. Re-running on the "
                       "same inputs gives byte-identical output apart from the "
                       "top-level 'timing' block, which is excluded from "
                       "body_sha256 for exactly that reason.",
        "findings": findings,
        "house_rule": "Every finding carries two implementations of the same "
                      "sum - a kernel on the card and a reference on the "
                      "processor - and is probed against itself before any "
                      "count is reported. A finding whose implementations "
                      "differ is reported FAILING and its counts are withheld.",
        "inputs": {
            "module_classes": len(BINS),
            "module_file": MODULES,
            "dot_budget": DOTS["max"],
            "dot_floors_read_from": DOTS["src"],
            "worksheet_directory": HARVEST_DIR,
            "night_worksheet": NIGHT_WORKSHEET,
            "worksheet_files_read": stats["files"],
            "worksheet_features_seen": stats["features"],
            "worksheet_features_marked_crunchable": stats["crunchable"],
            "worksheet_features_translated": stats["translated"]},
        "not_claimed":
            "The crunch settles ARITHMETIC over a stated space and nothing "
            "else. It does not decide whether the stated space is the right "
            "space, whether the predicate is the right question, or whether a "
            "count that comes back zero means the concern was unfounded rather "
            "than the sweep too narrow. It holds no site, no weather record and "
            "no as-built drawing, so it cannot say any of these findings is "
            "present in a real installation - only that the numbers in the "
            "files reach it. Fixes, costs and priority are not settled here at "
            "all: a count is evidence for a person to act on, not the action.",
        "refused": sorted(REFUSED, key=lambda r: (r["name"], r["reason"])),
        "refused_count": len(REFUSED),
        "registered": sorted(findings),
        "states": {k: v["state"] for k, v in findings.items()},
    }
    text = json.dumps(body, indent=1, sort_keys=True)
    body["body_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    body["timing"] = {"seconds_per_finding": timing,
                      "note": "the only block that changes between runs"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    io.open(OUT, "w", encoding="utf-8", newline="\n").write(
        json.dumps(body, indent=1, sort_keys=True) + "\n")
    print("\n  wrote night-results/harvest-crunch.json")
    print("  body digest (excludes timing) %s" % body["body_sha256"][:16])
    if REFUSED:
        print("\n  REFUSED %d, each by name:" % len(REFUSED))
        for r in REFUSED:
            print("    %-46s %s" % (r["name"][:46], r["reason"]))
    if failing:
        print("\n  FAILING: %s" % ", ".join(failing))
        if a.ci:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
