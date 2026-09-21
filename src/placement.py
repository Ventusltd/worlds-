"""Placement: where a thing goes on a plane, and what "near" means there.

The companion to geodesy.py. Geodesy answers where something IS; placement
answers where to PUT it when you are drawing a space that has no map.

Three arrangements, because three different questions need three different
meanings of "near". Choosing an arrangement is choosing what adjacency means,
and getting that wrong makes a boundary invisible.

  golden(key)   an archive. Adjacency is ORDER. r = spacing*sqrt(key),
                theta = key*GOLDEN. Area grows linearly with count, so density
                is constant forever; the golden angle never aligns, so no rows,
                spokes or gaps appear however many are added; and place j never
                moves when place j+1 arrives. This is the Kuiper arrangement.

  hilbert(d)    a swept space. Adjacency is ONE PARAMETER DIFFERING BY ONE
                STEP. Consecutive indices land adjacent at every zoom, so a
                boundary between two verdicts appears as a coherent edge
                rather than scattered dust. The golden spiral deliberately
                destroys this locality, which is exactly why it is wrong here.

  lattice(i)    the raw mixed-radix decode: index -> one value per axis. The
                thing both of the above are arranging.

And morph(), because an arrangement is not a still: the same points moved from
one arrangement to another is how one becomes legible as the descendant of the
other.

Held to placement.mjs by test/verify_placement.mjs. Neither is authoritative;
they must agree.
"""
import math

__all__ = ["GOLDEN", "golden", "hilbert", "lattice", "morph", "hilbert_order"]

# the golden angle, pi*(3-sqrt(5)) = 2.39996... rad. The most irrational angle
# there is, which is the whole point: a rational one eventually makes spokes.
GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


def golden(key, spacing=1.0):
    """Archive placement. Point `key` of an unbounded, append-only sequence."""
    if key < 0:
        raise ValueError("key must be >= 0")
    r = spacing * math.sqrt(key)
    t = key * GOLDEN
    return (r * math.cos(t), r * math.sin(t))


def hilbert_order(n):
    """The smallest order whose curve holds n points: side 2**order."""
    if n <= 0:
        raise ValueError("n must be > 0")
    order = 0
    while (1 << order) * (1 << order) < n:
        order += 1
    return order


def hilbert(d, order):
    """Locality-preserving placement. Distance `d` along a Hilbert curve of
    side 2**order, as integer (x, y). Consecutive d are always adjacent."""
    side = 1 << order
    if d < 0 or d >= side * side:
        raise ValueError("d out of range for order %d" % order)
    x = y = 0
    t = d
    s = 1
    while s < side:
        rx = 1 if (t // 2) % 2 else 0
        ry = 1 if (t ^ rx) % 2 else 0
        # rotate the quadrant so the curve stays continuous
        if ry == 0:
            if rx == 1:
                x, y = s - 1 - x, s - 1 - y
            x, y = y, x
        x += s * rx
        y += s * ry
        t //= 4
        s <<= 1
    return (x, y)


def lattice(index, radices):
    """Mixed-radix decode: a linear index to one value per axis, least
    significant axis first. The space every arrangement is arranging."""
    if index < 0:
        raise ValueError("index must be >= 0")
    out, r = [], index
    for radix in radices:
        if radix < 1:
            raise ValueError("every radix must be >= 1")
        out.append(r % radix)
        r //= radix
    return out


def morph(a, b, t, ease=True):
    """One point on its way from arrangement a to arrangement b.

    An arrangement is not a still. Moving the SAME points from one to another
    is what shows that one descends from the other: nothing is created or
    destroyed, only rearranged.
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError("t must be in [0, 1]")
    k = t * t * (3.0 - 2.0 * t) if ease else t      # smoothstep
    return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)
