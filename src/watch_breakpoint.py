"""
PATHS ARE NOT WRITTEN DOWN HERE. This repository is public and an
absolute path names a machine and an account. Set GRID_DATA (and where
needed GRID_REPOS or CLAUDE_TRANSCRIPT) in the environment instead.
See src/paths.py for why.

watch_breakpoint.py -- the photon that did not return, turned into code.

WHAT THIS IS
  Predicate K05-05 ("the 640/641 step: one CSS pixel changes which furniture the
  layout stacks") was one of 18,231 cases that failed to annihilate when a fused
  5.3-billion-case grid computed every predicate twice, once as written and once
  with every comparison made STRICT. A predicate that survives strictness is a
  predicate sitting exactly on a boundary. This file finds that boundary, on the
  GPU, at 1 px resolution, and measures it.

THE MODEL (read from source, not invented)
  shell  ...\\i0094\\releases\\003100000000-kuiper-shell\\index.html
    :125  @media (max-width:640px){ ... }
    :529  const narrow = () => innerWidth <= 640;
    :540  layout()   narrow -> HUD pushed under the header; card lifted off the foot
    :546  safeBox()  x0=8, x1=innerWidth-8, y0=top+8, y1=max(top+24, bot-8)
    :561  homeCamera()  R = min(s.x1-s.x0, s.y1-s.y0)/2*0.80 ; z = R/sqrt(SPACE)
    :328  gl_PointSize = clamp(u_zoom*0.34, 1.0, 9.0)
  cartridge ...\\i0094\\cartridges\\009400000000-kuiper-programs.js
    :581, :2819  innerWidth >= 500 && innerWidth < 700   -> 'micro' vs 'macro'
    :1999, :2586 @media (max-width:700px)
  So the shell restacks at 640 and the cartridge restacks at 700. 641..700 px is a
  band 60 px wide in which the two files disagree about what kind of screen this is.

  Derived clear height (this is exactly the K05-05 `maths` sentence):
    wide   (w  > brk):  H = max(16, h - hdr - foot - 16)                HUD beside header, card floats
    narrow (w <= brk):  H = max(16, h - hdr - hud - foot - cardh - 26)  HUD under header, card on the foot
    fitted disc diameter on screen  D = 0.80 * min(w - 16, H)
  The dot is clamp(z*0.34, 1, 9) px across; the sunflower nearest-neighbour gap is
  ~1 wafer unit, i.e. z px on screen, so the "three dot radii" rule is z >= 1.5*dot,
  which binds only while the 1 px floor is active, i.e. it fails exactly when z < 1.5.

  ONE CuPy ElementwiseKernel and ONE CuPy array-expression path compute every
  number independently. Nothing is reported unless the two agree exactly.

PASTE-READY EDITS THAT CLOSE THE BAND
  (1) FILE  ...\\i0094\\releases\\003100000000-kuiper-shell\\index.html   LINE 529
      FROM  const narrow = () => innerWidth <= 640;                // one number, used by the layout and by the test
      TO    const narrow = () => innerWidth <= 700;                // one number, used by the layout and by the test
  (2) FILE  ...\\i0094\\releases\\003100000000-kuiper-shell\\index.html   LINE 125
      FROM  @media (max-width:640px){
      TO    @media (max-width:700px){
      (the CSS breakpoint must move with narrow(), or the strip restyle and the
       camera's idea of the furniture part company and you have made a SECOND band)
  The cartridge already uses 700 in all four places (:581, :1999, :2586, :2819), so
  it needs no edit. After (1)+(2) both files agree on one number.

  HONEST COST: this does NOT delete the step, it MOVES it from 640/641 to 700/701.
  The step is not caused by the number; it is caused by the two layouts carrying
  different furniture. Any single number leaves a 1 px cliff somewhere. And it is
  not free: every width in 641..700 -- screens that are fine today -- switches to
  the narrow branch and the fitted disc there SHRINKS by the measured ratio. The
  numbers printed below say by how much, for how many, and how many of those newly
  narrow cases fall under the three-radii rule.

WHAT THIS DOES NOT PROVE
  It models a camera read from source. It renders nothing: no WebGL context, no
  shader, no real getBoundingClientRect -- the furniture heights are swept, not
  measured. It holds no real device, so it cannot tell you what any phone shows.
  It is arithmetic about arithmetic, and it is only as true as the lines above.
"""
import json
import os

import cupy as cp

OUT = os.environ.get("GRID_REPOS", "")
W0, W1 = 600, 760                 # the band under test, 1 px resolution
SH_BRK, CT_BRK = 640, 700         # shell narrow(), cartridge media query

H = cp.arange(320, 2161, 8, dtype=cp.int32)
HD = cp.arange(24, 121, 8, dtype=cp.int32)     # header
HU = cp.arange(24, 161, 8, dtype=cp.int32)     # hud
FT = cp.arange(24, 121, 8, dtype=cp.int32)     # foot
CD = [int(v) for v in range(0, 261, 20)]       # short card
SPACE = cp.asarray([1000, 4000, 16000, 65536, 262144, 1048576, 4000000],
                   dtype=cp.float64)

_SHAPE = (H.size, HD.size, HU.size, FT.size)
_h = cp.broadcast_to(H[:, None, None, None], _SHAPE).astype(cp.int32)
_hd = cp.broadcast_to(HD[None, :, None, None], _SHAPE).astype(cp.int32)
_hu = cp.broadcast_to(HU[None, None, :, None], _SHAPE).astype(cp.int32)
_ft = cp.broadcast_to(FT[None, None, None, :], _SHAPE).astype(cp.int32)

_kern = cp.ElementwiseKernel(
    'int32 w, int32 h, int32 hd, int32 hu, int32 ft, int32 cd, int32 brk',
    'float64 d',
    '''
    int HH;
    if (w <= brk) { HH = h - hd - hu - ft - cd - 26; }
    else          { HH = h - hd - ft - 16; }
    if (HH < 16) HH = 16;
    int wa = w - 16;
    int m = (wa < HH) ? wa : HH;
    d = 0.80 * (double)m;
    ''', 'kuiper_diam')


def _diam_array(w, cd, brk):
    """Path A: pure CuPy array expressions."""
    if w <= brk:
        clear = cp.maximum(16, _h - _hd - _hu - _ft - cd - 26)
    else:
        clear = cp.maximum(16, _h - _hd - _ft - 16)
    return 0.80 * cp.minimum(w - 16, clear).astype(cp.float64)


def _diam_kernel(w, cd, brk):
    """Path B: one CUDA kernel, branch inside the kernel."""
    return _kern(cp.int32(w), _h, _hd, _hu, _ft, cp.int32(cd), cp.int32(brk))


def diam(w, cd, brk):
    a = _diam_array(w, cd, brk)
    b = _diam_kernel(w, cd, brk)
    if not bool(cp.all(a == b)):
        raise SystemExit("DISAGREEMENT array vs kernel at w=%d cd=%d -- nothing reported" % (w, cd))
    return a


def agree(x, y, tag):
    if abs(float(x) - float(y)) > 1e-12:
        raise SystemExit("DISAGREEMENT (%s): %r vs %r -- nothing reported" % (tag, x, y))
    return float(x)


def profile(brk):
    """Mean fitted diameter at every width in the sweep, for one breakpoint."""
    out = {}
    for w in range(W0, W1 + 1):
        tot = 0.0
        n = 0
        for cd in CD:
            d = diam(w, cd, brk)
            tot += float(d.sum())
            n += d.size
        out[w] = tot / n
    return out


def biggest_steps(prof):
    return sorted(((prof[w + 1] - prof[w], w) for w in range(W0, W1)),
                  key=lambda t: -abs(t[0]))


# ---- 1. locate the discontinuity, 1 px resolution, as shipped -------------
prof640 = profile(SH_BRK)
steps = biggest_steps(prof640)
jump_delta, jump_w = steps[0]
runner_delta, runner_w = steps[1]

# ---- 2. size of the jump: ratio of disc diameter either side -------------
parts = []
for cd in CD:
    lo = diam(jump_w, cd, SH_BRK)          # narrow branch
    hi = diam(jump_w + 1, cd, SH_BRK)      # wide branch
    parts.append((hi / lo).ravel())
RAT = cp.concatenate(parts)
ratio_mean = agree(float(RAT.mean()), float(RAT.sum()) / RAT.size, "ratio_mean")
srt = cp.sort(RAT)
ratio_max = agree(float(RAT.max()), float(srt[-1]), "ratio_max")
ratio_min = agree(float(RAT.min()), float(srt[0]), "ratio_min")
ratio_med = float(cp.median(RAT))
frac_step = agree(float((RAT != 1.0).mean()),
                  float((RAT != 1.0).sum()) / RAT.size, "frac_step")

# ---- 3. combinations inside the disagreement band ------------------------
band_lo, band_hi = SH_BRK + 1, CT_BRK      # shell says wide, cartridge styles a phone
sqrt_space = cp.sqrt(SPACE)[None, :]
in_band = 0
bad_today = 0
bad_after = 0
shrink_sum = 0.0
shrink_n = 0
for w in range(band_lo, band_hi + 1):
    for cd in CD:
        dw = diam(w, cd, SH_BRK).ravel()   # what ships today (wide branch)
        dn = diam(w, cd, CT_BRK).ravel()   # what the fix gives (narrow branch)
        shrink_sum += float((dn / dw).sum())
        shrink_n += dn.size
        zw = (dw / 2.0)[:, None] / sqrt_space
        zn = (dn / 2.0)[:, None] / sqrt_space
        in_band += zw.size
        bad_today += int(agree(int(cp.count_nonzero(zw < 1.5)),
                               int((zw < 1.5).sum()), "bad_today"))
        bad_after += int(agree(int(cp.count_nonzero(zn < 1.5)),
                               int((zn < 1.5).sum()), "bad_after"))
mean_shrink = shrink_sum / shrink_n

# ---- 4. does unifying on 700 remove the step or move it? -----------------
prof700 = profile(CT_BRK)
j700_delta, j700_w = biggest_steps(prof700)[0]

res = {
    "band": {
        "cartridge_breakpoint_px": CT_BRK,
        "disagreement_band_px": [band_lo, band_hi],
        "disagreement_band_width_px": band_hi - band_lo + 1,
        "shell_breakpoint_px": SH_BRK,
    },
    "cases": {
        "combos_in_band": in_band,
        "combos_in_band_failing_three_radii_after_fix": bad_after,
        "combos_in_band_failing_three_radii_today": bad_today,
        "furniture_grid_size": int(H.size * HD.size * HU.size * FT.size * len(CD)),
        "space_values": [int(v) for v in SPACE.get()],
        "widths_swept": [W0, W1],
    },
    "jump": {
        "discontinuity_between_px": [int(jump_w), int(jump_w) + 1],
        "mean_diameter_delta_px": round(float(jump_delta), 9),
        "ratio_max": round(ratio_max, 9),
        "ratio_mean": round(ratio_mean, 9),
        "ratio_median": round(ratio_med, 9),
        "ratio_min": round(ratio_min, 9),
        "runner_up_step_at_px": int(runner_w),
        "runner_up_step_delta_px": round(float(runner_delta), 9),
        "share_of_cases_that_step": round(frac_step, 9),
    },
    "unify_on_700": {
        "cost_mean_diameter_ratio_in_band": round(mean_shrink, 9),
        "largest_step_after_delta_px": round(float(j700_delta), 9),
        "largest_step_after_px": [int(j700_w), int(j700_w) + 1],
        "removes_step": False,
        "verdict": "moves, does not remove",
    },
    "verification": {
        "method": "cupy ElementwiseKernel vs cupy array expressions, exact equality",
        "status": "agreed",
    },
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, sort_keys=True)
    f.write("\n")

print()
print("THE JUMP IS BETWEEN %d px AND %d px." % (jump_w, jump_w + 1))
print("  At %d px the shell restacks: HUD under the header, card on the foot." % jump_w)
print("  Crossing one CSS pixel changes the fitted disc diameter by a factor of")
print("    %.4f on average, %.4f at worst, %.4f at best; %.1f%% of all cases step."
      % (ratio_mean, ratio_max, ratio_min, 100.0 * frac_step))
print("  Mean fitted diameter moves %.2f px across that single pixel." % jump_delta)
print("  The next largest step anywhere in %d..%d is only %.4f px (at %d)."
      % (W0, W1, runner_delta, runner_w))
print()
print("THE DISAGREEMENT BAND is %d..%d px, %d px wide: the shell calls it wide, the"
      % (band_lo, band_hi, band_hi - band_lo + 1))
print("  cartridge still styles it as a phone. %d viewport x furniture x SPACE" % in_band)
print("  combinations sit inside it. %d already break the three-radii rule;" % bad_today)
print("  after unifying on 700 that becomes %d." % bad_after)
print()
print("DOES 700 FIX IT?  No -- it MOVES it. With both files on 700 the largest step")
print("  in the sweep is %.2f px between %d and %d px: the same cliff, %d px right."
      % (j700_delta, j700_w, j700_w + 1, j700_w - jump_w))
print("  And it is not free: across %d..%d the fitted disc becomes %.4f of what it"
      % (band_lo, band_hi, mean_shrink))
print("  is today -- a real shrink on screens that are fine right now.")
print()
print("NOT PROVEN: a camera read from source; rendered nothing; held no device.")
print("wrote", OUT)
