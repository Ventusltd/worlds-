"""Worlds: the same points, rearranged, on the card.

From the Kuiper to the wafer. The Kuiper arranges work by ORDER - oldest at
the middle, newest at the rim, on a golden spiral that never has to move
anything already placed. A design sweep needs the opposite: neighbouring cases
adjacent, so the boundary between one verdict and another shows as an edge
rather than as dust.

So this holds both arrangements and the journey between them. Nothing is
created or destroyed when a world changes shape; the same points move. That is
what makes the sweep legible as a descendant of the archive rather than a
different object entirely.

Everything is generated from an index on the card. No geometry is stored: the
world is a function, which is why eighteen quintillion planets fit in a few
hundred megabytes and why a billion designs fit in none at all.

    python worlds.py                 generate, verify, write worlds.html

Verified before anything is drawn: the card's geometry must equal the
processor's, point for point.
"""
import io
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from placement import GOLDEN, hilbert, hilbert_order, lattice, morph  # noqa: E402

try:
    import cupy as cp
except Exception as exc:                                   # pragma: no cover
    print("cupy unavailable: %r" % (exc,))
    sys.exit(1)

# ---- the space being arranged: a solar string design, as we already prove it
CLASSES = [(45.90, -0.25, 660.0), (50.59, -0.22, 760.0), (53.27, -0.25, 650.0)]
TMIN = [-14.0, -12.0, -11.0, -10.0, -8.0, -6.0]
SERIES = [float(s) for s in range(20, 35)]
STRINGS = [float(n) for n in range(16, 33)]
NC, NT, NS, NN = len(CLASSES), len(TMIN), len(SERIES), len(STRINGS)
RADICES = [NC, NT, NS, NN]
SIZE = NC * NT * NS * NN
ORDER = hilbert_order(SIZE)

INVERTER_KVA, MAX_V = 352.0, 1500.0
DCAC_LO, DCAC_HI = 1.15, 1.40
OK, DISPUTED, FAIL_COLD, OUT_OF_BAND = 0, 1, 2, 3
LABEL = ["buildable, both readings", "disputed: the clause decides it",
         "fails cold voltage", "DC:AC outside the band"]
COLOUR = ["#4ade80", "#f2b05e", "#5ec8f2", "#38415a"]

_SRC = """
long long r = idx;
int ci = (int)(r %% %(NC)d); r /= %(NC)d;
int ti = (int)(r %% %(NT)d); r /= %(NT)d;
int si = (int)(r %% %(NS)d); r /= %(NS)d;
int ni = (int)(r %% %(NN)d);

double v = voc[ci], b = beta[ci], w = wp[ci];
double tm = tmin[ti], se = ser[si], st = strs[ni];

double nb = floor(%(MAXV).1f / (v * (1.0 + b / 100.0 * (tm - 25.0))));
double na = floor(%(MAXV).1f / (v * (1.0 + b / 100.0 * (tm - 15.0))));
double dcac = (se * w / 1000.0) * st / %(KVA).1f;
bool band = (dcac >= %(LO).2f) && (dcac <= %(HI).2f);
code = (se > na) ? %(FC)d : ((se > nb) ? %(DI)d : (band ? %(OK)d : %(OB)d));

/* ARRANGEMENT ONE - the archive. Adjacency is order, oldest at the middle.
   Place j never moves when j+1 arrives, and the angle never aligns. */
double rad = %(SPACING).17g * sqrt((double)idx);
/* REDUCE BEFORE THE TRIG. By index 4,589 the accumulated angle is ~11,012
   rad, and CUDA and the host reduce arguments that large by different
   routes - a difference that sqrt(idx) then multiplies. fmod is exact on
   doubles, so both sides must do this identically or they disagree. */
double th  = fmod((double)idx * %(GOLDEN).17f, 6.283185307179586477);
gx = (float)(rad * cos(th));
gy = (float)(rad * sin(th));

/* ARRANGEMENT TWO - the swept space. Adjacency is one parameter, one step,
   so a verdict boundary is an edge. Hilbert curve, side 2^order. */
long long t = idx; long long hx = 0, hy = 0;
for (long long s = 1; s < (1LL << %(ORDER)d); s <<= 1) {
    long long rx = ((t / 2) %% 2) ? 1 : 0;
    long long ry = ((t ^ rx) %% 2) ? 1 : 0;
    if (ry == 0) {
        if (rx == 1) { hx = s - 1 - hx; hy = s - 1 - hy; }
        long long tmp = hx; hx = hy; hy = tmp;
    }
    hx += s * rx; hy += s * ry; t /= 4;
}
/* .17g, not .6f: a constant baked into kernel source at six
   decimals differs from the host's double by ~5e-7 relative,
   which an offset of up to 64 turns into 3e-5 absolute. That is
   how the first run failed its own 0-differ check. */
px = (float)(((double)hx - %(HALF).17g) * %(HSCALE).17g);
py = (float)(((double)hy - %(HALF).17g) * %(HSCALE).17g);
""" % {"NC": NC, "NT": NT, "NS": NS, "NN": NN, "MAXV": MAX_V,
       "KVA": INVERTER_KVA, "LO": DCAC_LO, "HI": DCAC_HI, "FC": FAIL_COLD,
       "DI": DISPUTED, "OK": OK, "OB": OUT_OF_BAND, "ORDER": ORDER,
       "SPACING": 1.0, "GOLDEN": GOLDEN, "HALF": (1 << ORDER) / 2.0,
       "HSCALE": 2.0 * np.sqrt(SIZE) / (1 << ORDER)}

WORLD = cp.ElementwiseKernel(
    "int64 idx, raw float64 voc, raw float64 beta, raw float64 wp, "
    "raw float64 tmin, raw float64 ser, raw float64 strs",
    "int64 code, float32 gx, float32 gy, float32 px, float32 py",
    _SRC, "ventus_world")


def tables(xp):
    return (xp.asarray([c[0] for c in CLASSES]),
            xp.asarray([c[1] for c in CLASSES]),
            xp.asarray([c[2] for c in CLASSES]),
            xp.asarray(TMIN), xp.asarray(SERIES), xp.asarray(STRINGS))


def reference(n):
    """The same worlds on the processor, using the canonical placement."""
    gx = np.zeros(n, np.float32); gy = np.zeros(n, np.float32)
    px = np.zeros(n, np.float32); py = np.zeros(n, np.float32)
    code = np.zeros(n, np.int64)
    half, hscale = (1 << ORDER) / 2.0, 2.0 * np.sqrt(SIZE) / (1 << ORDER)
    for i in range(n):
        ci, ti, si, ni = lattice(i, RADICES)
        v, b, w = CLASSES[ci]
        tm, se, st = TMIN[ti], SERIES[si], STRINGS[ni]
        nb = np.floor(MAX_V / (v * (1 + b / 100 * (tm - 25))))
        na = np.floor(MAX_V / (v * (1 + b / 100 * (tm - 15))))
        dcac = (se * w / 1000.0) * st / INVERTER_KVA
        band = DCAC_LO <= dcac <= DCAC_HI
        code[i] = (FAIL_COLD if se > na else
                   (DISPUTED if se > nb else (OK if band else OUT_OF_BAND)))
        th = np.fmod(i * GOLDEN, 2.0 * np.pi)
        g = (np.sqrt(i) * np.cos(th), np.sqrt(i) * np.sin(th))
        gx[i], gy[i] = np.float32(g[0]), np.float32(g[1])
        hx, hy = hilbert(i, ORDER)
        px[i] = np.float32((hx - half) * hscale)
        py[i] = np.float32((hy - half) * hscale)
    return code, gx, gy, px, py


def page(code, gx, gy, px, py, path):
    """One plane, zoomable, and a handle that moves the ancestor into the
    descendant. The points never change; only the arrangement does."""
    pts = ",".join("[%d,%.4f,%.4f,%.4f,%.4f]" % (c, a, b, d, e)
                   for c, a, b, d, e in zip(code, gx, gy, px, py))
    counts = [int((code == k).sum()) for k in range(4)]
    legend = "".join(
        '<span class="k"><i style="background:%s"></i>%s <b>%s</b></span>'
        % (COLOUR[k], LABEL[k], "{:,}".format(counts[k])) for k in range(4))
    html = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Worlds</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}
body{margin:0;background:#07090d;color:#cfe3f2;font:14px/1.5 ui-monospace,Menlo,Consolas,monospace;overflow:hidden}
canvas{display:block;width:100vw;height:100vh;cursor:crosshair}
.bar{position:fixed;left:0;right:0;bottom:0;padding:10px 14px;background:#0b0e14e6;border-top:1px solid #1b2030}
.k{margin-right:14px;font-size:11.5px;white-space:nowrap}
.k i{display:inline-block;width:9px;height:9px;margin-right:5px;border-radius:2px;vertical-align:middle}
.t{position:fixed;top:0;left:0;right:0;padding:10px 14px;background:#0b0e14e6;border-bottom:1px solid #1b2030;font-size:12px}
.t b{color:#fff}.t span{color:#8b93a7}
input[type=range]{width:min(320px,45vw);vertical-align:middle}
#card{position:fixed;right:14px;top:56px;max-width:290px;padding:10px 12px;background:#11151f;border:1px solid #1b2030;border-radius:8px;font-size:11.5px;display:none}
</style></head><body>
<div class="t"><b>Worlds</b> &#8212; %(n)s designs, every point generated on the card
<span>&#183; drag to pan, wheel to zoom, tap a point</span><br>
<span>ancestor</span> <input id="m" type="range" min="0" max="1000" value="1000"> <span>descendant</span>
<span id="ml">&#183; the Kuiper arrangement &#8594; the swept plane</span></div>
<canvas id="c"></canvas>
<div id="card"></div>
<div class="bar">%(legend)s</div>
<script>
const P=[%(pts)s], COL=%(col)s, LAB=%(lab)s;
const c=document.getElementById("c"),g=c.getContext("2d"),card=document.getElementById("card");
let z=1,ox=0,oy=0,t=1,drag=null;
function fit(){c.width=innerWidth*devicePixelRatio;c.height=innerHeight*devicePixelRatio;draw();}
function pos(p){const k=t*t*(3-2*t);return [p[1]+(p[3]-p[1])*k, p[2]+(p[4]-p[2])*k];}
function draw(){
 g.setTransform(1,0,0,1,0,0);g.fillStyle="#07090d";g.fillRect(0,0,c.width,c.height);
 const s=Math.min(c.width,c.height)/(2.6*Math.sqrt(P.length))*z;
 const cx=c.width/2+ox*devicePixelRatio, cy=c.height/2+oy*devicePixelRatio;
 const r=Math.max(0.7,Math.min(4.2,s*0.42))*devicePixelRatio;
 for(const p of P){const [x,y]=pos(p);
  g.fillStyle=COL[p[0]];g.beginPath();g.arc(cx+x*s,cy-y*s,r,0,6.2832);g.fill();}
}
addEventListener("resize",fit);
c.addEventListener("wheel",e=>{e.preventDefault();const k=Math.exp(-e.deltaY*0.0012);z=Math.max(0.25,Math.min(90,z*k));draw();},{passive:false});
c.addEventListener("pointerdown",e=>{drag=[e.clientX,e.clientY,ox,oy];c.setPointerCapture(e.pointerId);});
c.addEventListener("pointermove",e=>{if(!drag)return;ox=drag[2]+(e.clientX-drag[0]);oy=drag[3]+(e.clientY-drag[1]);draw();});
c.addEventListener("pointerup",e=>{
 if(drag&&Math.abs(e.clientX-drag[0])<3&&Math.abs(e.clientY-drag[1])<3){
  const s=Math.min(c.width,c.height)/(2.6*Math.sqrt(P.length))*z;
  const cx=c.width/2+ox*devicePixelRatio, cy=c.height/2+oy*devicePixelRatio;
  let best=-1,bd=1e18;
  P.forEach((p,i)=>{const [x,y]=pos(p);const dx=cx+x*s-e.clientX*devicePixelRatio,dy=cy-y*s-e.clientY*devicePixelRatio;const d=dx*dx+dy*dy;if(d<bd){bd=d;best=i;}});
  if(best>=0&&bd<(28*devicePixelRatio)**2){
   const i=best,rad=%(rad)s;let r=i,ax=[];for(const q of rad){ax.push(r%%q);r=Math.floor(r/q);}
   card.style.display="block";
   card.innerHTML="<b>index "+i+"</b><br>"+LAB[P[i][0]]+
    "<br><span style='color:#8b93a7'>class "+ax[0]+" &#183; site min "+%(tmin)s[ax[1]]+" &#176;C &#183; "+
    %(ser)s[ax[2]]+" in series &#183; "+%(strs)s[ax[3]]+" strings</span>";
  } else card.style.display="none";
 }
 drag=null;});
document.getElementById("m").addEventListener("input",e=>{t=e.target.value/1000;
 document.getElementById("ml").textContent=t<0.02?"\\u00b7 the Kuiper arrangement: adjacency is order":t>0.98?"\\u00b7 the swept plane: adjacency is one parameter, one step":"\\u00b7 the same points, moving";draw();});
fit();
</script></body></html>
""" % {"n": "{:,}".format(len(code)), "pts": pts, "legend": legend,
       "col": str(COLOUR).replace("'", '"'), "lab": str(LABEL).replace("'", '"'),
       "rad": str(RADICES), "tmin": str(TMIN), "ser": str([int(s) for s in SERIES]),
       "strs": str([int(s) for s in STRINGS])}
    io.open(path, "w", encoding="utf-8", newline="\n").write(html)


def main():
    print("world: %s designs over %d axes %s, hilbert order %d (side %d)"
          % ("{:,}".format(SIZE), len(RADICES), RADICES, ORDER, 1 << ORDER))
    t = tables(cp)
    idx = cp.arange(SIZE, dtype=cp.int64)
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    code, gx, gy, px, py = WORLD(idx, *t)
    cp.cuda.Stream.null.synchronize()
    sec = time.perf_counter() - t0

    h = [cp.asnumpy(a) for a in (code, gx, gy, px, py)]
    r = reference(SIZE)
    worst = max(float(np.abs(h[i].astype(np.float64)
                             - r[i].astype(np.float64)).max()) for i in range(5))
    print("verified against the processor, all %s worlds: worst difference "
          "%.3e" % ("{:,}".format(SIZE), worst))
    if worst > 0:
        print("FAIL: the card and the processor drew different worlds")
        return 1
    if SIZE < 1 << 20:
        print("the card generated them in %.4f s - NOT a throughput, the "
              "space is small" % sec)
    else:
        print("%s designs/s" % "{:,.0f}".format(SIZE / sec))

    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(os.path.dirname(here), "worlds.html")
    page(*h, path=out)
    counts = [int((h[0] == k).sum()) for k in range(4)]
    print("\nwrote %s" % os.path.basename(out))
    for k in range(4):
        print("  %-32s %8s  %5.2f%%"
              % (LABEL[k], "{:,}".format(counts[k]), 100.0 * counts[k] / SIZE))
    print("\nThe slider moves the SAME points from the archive arrangement to")
    print("the swept one. Nothing is created or destroyed between them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
