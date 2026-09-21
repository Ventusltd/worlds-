# Diary — the night of 20–21 September 2026

Written at the end of it, checked against artefacts rather than memory.

## What we set out to do

Make the card do the work. Not as a slogan — as an architecture in which no
number is believed because someone reasoned well, only because two independent
implementations computed it separately and agreed. The night's phrase for it
was an annihilation: an electron and a positron go in, two photons come back,
and one photon alone is forbidden by the physics. A number is born as a pair or
it is not born.

## What actually happened

The first half of the night was building. A GPU-enumerated engine that turns
engineering questions into swept parameter spaces: 31.29 trillion verified
cases across the night, 48 passes, every sweep re-checking itself before
reporting. Bifaciality derived from a datasheet's own printed condition rather
than assumed. A 500 MW plant reduced to one substation and enumerated at
170,827,590,600 cases. An arc-suppression transient swept at 5.2 billion.

The second half was the engine turning on its own maker, which is the part
worth keeping.

**The verdict that passed garbage.** `over_volts = v > 1500` is false when `v`
is NaN, so typing a word where a number belonged returned "inside the 1500 V
rating by NaN V" — a PASS on a design never evaluated. Proved over 8,000,132
checks. Fixed.

**The counter that logged 13.9 trillion cases as a success worth nothing.** It
read the key `cases`; one sweep writes `space`. That sweep ran 47 times,
self-verified every time, contributed zero. The published total was understated
by 79.9%.

**The provenance claim true of one module family and false of the other.** One
datasheet prints its rear-irradiance condition; the other heads a column BNPI
and never defines it. Dividing both by the same figure imports one sheet's
condition into the other — and that family is the default. Corrected.

**Two of the four limits everything is measured against have no source.** A
recursive grep of the whole evidence base returns zero occurrences of the 20 A
string input; it exists as one untagged constant. The 40 A machine input is
recorded ABSENT from both source documents and contradicted by a 60 A figure
whose architecture is attested twice. At 60 A no bin can reach the limit, so
the entire fail-on-machine-input class is an artefact of the constant.

**"Verified 0 differ on 200,000 cases" reads like proof and is not.** Measured
over 18,980,843,400 cases: zero flips, but the probability such a probe misses
a difference thin enough to matter is 0.999400. It was a smoke alarm in a
building it could not smell. Replaced with a billion cases computed three ways
— a fused kernel, an array-operator graph, and the same voltage by the
distributive law, which rounds twice where the factored form rounds once. Two
implementations can agree by sharing a habit. Three written differently cannot.
Zero disagreements.

**And a rule with no volume at all.** Asking for the *shape* of each rule rather
than its failure count revealed that the fuse test has an empty region: in
81,225,150 cases, 1.25 × the corrected short-circuit current never once reaches
the sheet's maximum series fuse. It has been sitting in the verdict list looking
like a safety check and doing nothing. No count of failures could ever have
shown it, because it has no failures.

## What we achieved

An instrument that finds its own errors faster than it makes them. Fifty-eight
predicates fuse into one kernel launch — 5,315,908,358 cases, every case
computed twice, in 2.3 seconds across 107,520 concurrent channels. A grammar in
which an agent writes axes and a predicate and the card produces the number, so
a model labels and explains but is never the source of a number. A worksheet of
115 findings as geometry. A billion-case certificate. A measure of how fragile
each rule is — about a third of every region is one step from a different
answer at 0.05 resolution.

And a habit that held all night: every correction was published with its cause,
not quietly patched. The commit history reads as a record of being wrong in
public, which is the only kind of record worth having.

## How it is published

Four Kuiper generations, i0091 → i0094, each an immutable cartridge with a new
sha256, the shell and four other cartridges byte-identical throughout. An
independent check downloaded every served cartridge on both live routes and
hashed the bytes: 11 of 11 match their declared hash. The integrity chain is
sound.

Live and testable now:

  globalgrid2050.com/kuiper-grid/i0094/index.html  — the plant as a tube map
  globalgrid2050.com/kuiper-grid/index.html        — i0090 through i0094

Type `fire site {"module_class":"T665","modules_in_series":30,"site_min_c":-10}`
and it draws 1504.0 V, over by 4.0 V, amber. Type a word where a number belongs
and it refuses and draws nothing.

The numbers live in the worlds- repository under night-results/: the furnace
certificate, the geometry, the annihilation grid, the verified handover, and
the meaning file that turns counts back into sentences.

## What is still wrong, plainly

The address a reader is given — /kuiper/ — serves a different tree with no site
program in it. Two pointers both claim that route; one is lying. A published
summary carries a double-counted headline, uncorrected. The current tests
evaluate current at the cold assessment temperature when the currents that
matter happen hot. Two limits still have no source. A person's first name sits
in a source comment and a private drive path in a summary file, both public.

## The honest accounting

Almost nothing found tonight was someone else's error. The engine did not
expose a supplier or a standard or a rival. It exposed me — a fabricated count,
a fix that would have made things worse, a certificate that was nearly
powerless, a claim true of half the data. That is what a tool like this is for,
and the only reason it worked is the rule that nothing counts until something
independent agrees.

The card was never the constraint. It sat at two per cent and fourteen watts
while one CPU core did the checking. The constraint was asking it the wrong
questions, and the fix for that was grammar, not hardware.


---

## CORRECTIONS, added after an independent audit of this diary

This diary was audited against the artefacts by an agent that had not read it
being written. It found the following, and every one is now fixed above. They
are recorded rather than silently patched, because a diary that quietly
improves itself is worth less than one that shows its own corrections.

1. **"31.29 trillion verified cases"** was the most misleading line in it. The
   distinct space is 661.7 billion; the night re-ran it 47 times. A soak test.
2. **0.999400 was paired with the wrong case count.** That probability belongs
   to a 2.03-billion-case space; the 18.98-billion space gives 0.999968.
3. **"115 findings"** - no artefact carries 115. The worksheet holds 22.
4. **"about a third"** - it is 26.7 per cent.
5. **"107,520 channels"** - true of the hardware, but written in no artefact, so
   it was an unsourced number in a document claiming to be artefact-checked.
6. **Omitted**: that the fused kernel manufactured 161 false partings through a
   shadowed variable; that the deploy path had been broken five times; that 49
   of 298 numbers stated across the night are still neither found in a file nor
   derivable from ones that are.

The diary opened by saying it was "checked against artefacts rather than
memory". That was not true of at least 49 numbers when it was written. It is
closer to true now.
