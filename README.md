# worlds-
From the Kuiper to the Wafer let's do GPU maths through geometry and more

---

A world here is not stored. It is **generated from an index on the card** — the
same index that yields the engineering verdict also yields the picture of it.
Same index, same world, every time, on any machine, with nothing kept. That is
why eighteen quintillion procedural planets fit in a few hundred megabytes, and
why a few billion designs fit in none at all: **a world is a function, not a
database.**

Open `worlds.html`. Drag to pan, wheel to zoom, tap a point for its card. Then
move the slider at the top.

## The slider is the point

The Kuiper arranges work by **order** — oldest at the middle, newest at the
rim, on a golden spiral. It is an archive, and its geometry is a promise: place
*j* never moves when place *j+1* arrives, and the angle never aligns, so no
rows, spokes or gaps appear however much is added. A stone board written as
mathematics.

A design sweep needs the opposite. There, "near" has to mean **one parameter
differing by one step**, so the boundary between *buildable* and *disputed*
reads as an edge rather than scattered dust. That is a Hilbert curve, and it
depends on exactly the property the golden spiral is built to destroy.

So the page holds both, and the slider moves the **same points** from one to
the other. Nothing is created and nothing is destroyed in between — which is
what makes the sweep legible as a *descendant* of the archive rather than a
different object that happens to share a colour scheme.

The wafer is the ancestor. The arrangement is the animation.

## What is drawn

4,590 string designs over four axes — module class, site minimum temperature,
modules in series, strings per inverter — each coloured by its verdict:

| | |
|---|---|
| buildable on both readings of the clause | 684 · 14.90% |
| **disputed: the reading alone decides it** | **204 · 4.44%** |
| fails cold voltage | 1,853 · 40.37% |
| DC:AC outside the band | 1,849 · 40.28% |

The disputed band is the reason for all of this. Those designs are buildable
under one reading of a temperature clause and not under the other, and no
sampling strategy finds them, because they are not representative cases — they
are the boundary. Both readings are computed and printed everywhere in this
work and never collapsed into the one you would prefer.

## Verified before it is drawn

The card's geometry must equal the processor's, point for point, or the script
refuses to write the page. Currently **worst difference 0.000e+00 across all
4,590 worlds**.

That check earned its keep twice in one sitting:

- the first run failed at **3.052e-05**, because a scale constant was baked
  into the kernel source with `%.6f`. Six decimals is a relative error of about
  5e-7, which an offset of up to 64 turns into 3e-5 absolute. Formatting a
  constant into source is a precision decision, and it looked like a rounding
  detail right up until it wasn't;
- the accumulated golden angle reaches ~11,012 radians by index 4,589, so both
  sides now reduce modulo 2π **before** the trig call rather than trusting two
  libraries to agree on argument reduction that far out.

Neither would have been caught by looking at the picture. Both pictures looked
right.

## The maths lives elsewhere

`src/placement.py` is a **vendored copy** of the canonical implementation in
[grid-distance-maths](https://github.com/Ventusltd/grid-distance-maths)
(`src/placement.py`, commit `8e91b07`), which holds a Python and a JavaScript
half to each other so every tool answers the same question with the same
number — 4,096 rows, 11 columns, worst difference 1.4e-14.

It is copied rather than imported so this repository runs standalone. If the
two ever drift, the canonical one wins.

## Run it

```
python src/worlds.py
```

Needs `numpy` and `cupy`. Without a card it says so and stops, rather than
quietly drawing something unverified.

## What this does not claim

Nothing here is a design, a study, or a cable schedule. No yield, energy,
export, length or cost figure is computed and none is stated. Every verdict
carries its denominator and both readings of the disputed clause. A qualified
engineer should be consulted before any commitment.

Code: Apache-2.0. No warranty is given.
