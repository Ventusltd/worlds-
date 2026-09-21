# The night of 20–21 September, verified against artefacts

Written to survive compaction. Every claim below was checked by an independent
agent reading primary evidence — git log, published JSON, source files, live
HTTP — not by reading my account of the night. Claims that failed verification
are kept and marked, because deleting them would make this look more reliable
than it is.

---

## THE 1000-WORD SUMMARY

**What was built.** A GPU-enumerated engine that turns engineering questions
into swept parameter spaces, verified arithmetic, and a drawing in the Kuiper's
visual grammar. It now runs end to end: an agent writes axes and a predicate in
a fixed grammar, a translator compiles it to CUDA, the card enumerates it with a
second independent implementation beside it, and nothing is reported unless the
two agree. Fifty-eight predicates now fuse into **one kernel launch** —
5,315,908,358 cases, every case computed twice, 10.6 billion evaluations, in
about 2.3 seconds on 107,520 concurrent channels.

**What was published.** Four Kuiper generations, i0091 → i0094, each an
immutable cartridge with a new sha256, the shell and four other cartridges
byte-identical throughout. An independent verifier downloaded every served
cartridge on both live routes and hashed the bytes: **11 of 11 match their
declared sha256.** The integrity chain is sound. Also published: a public
language gate, a verdict gate, a layout gate, the worksheets, the crunch
registry, and the fused annihilation grid.

**What the engine found.** Almost everything it found was wrong with *our own
work*, which is the only kind of thing a tool like this can honestly find.

1. **A verdict passed garbage.** `over_volts = v > 1500` is false when `v` is
   NaN, so typing a word where a number belonged returned "inside the 1500 V
   rating by NaN V" — a PASS on a design never evaluated. Proved over 8,000,132
   checks: zero disagreements on any finite number, 1,001,580 where the old form
   passes what it cannot justify. Fixed in i0093 with a guard that refuses in
   words and four predicates flipped to `!(x <= limit)`.

2. **A counter logged 13.9 trillion cases as a success worth nothing.** It read
   the key `cases`; one sweep writes `space`. That sweep ran 47 times,
   self-verified every time, and contributed zero. The published overnight total
   was understated by 79.9%. Corrected 17,391,591,847,800 → 31,292,801,587,800,
   and the arithmetic closes exactly from the artefacts alone.

3. **A provenance claim was true of one module family and false of the other.**
   φ = (K_Corr − 1)/0.135 is genuinely derived where the sheet prints its BNPI
   rear-irradiance condition. The other sheet heads a column BNPI and **never
   defines the condition** — so dividing by 0.135 imports the first sheet's
   condition into the second. That family is the default. Corrected in i0094;
   each bin now carries `bnpi_stated` and the card says which it is.

4. **Two of the four limits everything is measured against have no source.** A
   recursive grep of the entire evidence base returns **zero occurrences** of
   the 20 A string input — it exists as one untagged constant. The 40 A machine
   input is recorded ABSENT from both source documents and contradicted by a
   60 A figure whose architecture is attested twice. At 60 A no bin can reach
   the limit, so **the entire fail-on-machine-input class is an artefact of the
   constant, not of the designs.** 1500 V and 35 A trace cleanly to datasheets.

5. **"Verified 0 differ on 200,000 cases" reads like proof and is not.**
   Measured properly over 18,980,843,400 cases: zero flips across either
   threshold, but the probability such a probe would MISS a difference that thin
   is **0.999968**. The certificate was nearly powerless. It had nothing to
   catch, which is luck, not rigour.

6. **My own fused kernel manufactured 161 partings.** A worksheet named an axis
   `e`; the kernel's electron accumulator is also `e`; C let the axis shadow it
   silently. Caught only because a watcher re-evaluated on a different compiler.
   Correction landed at exactly the predicted 161: 21,624 → 21,463.

7. **A fix I was about to ship would have made things worse.** Unifying the
   640/700 breakpoints does not remove the one-pixel cliff — it moves it 60 px
   and shrinks the fitted disc to 0.8153 of today's on screens that are fine.

**What the gates caught about themselves.** The public language gate had been
failing for days on a missing file: all twelve self-hosted runner workspaces
carried `core.sparseCheckout=true`, so `scripts/` never landed. It failed
loudly — red, not a false green — which is why nothing bad shipped. Clearing the
config alone did not fix it; three things make a tree sparse and the index's
skip-worktree bit is the one that hides files. And the deploy path allow-list
did not list `kuiper-grid/**`, so i0092 was pushed, gated, hashed and merged
while the live site served i0091 — nothing red anywhere. That file's own
comments were already counting: third time, fourth time. This was the fifth.

**What is still wrong.** `/kuiper/` — the address a reader is given — serves a
different tree, generation 202609200300, which contains **no site program at
all**. Two live pointers both declare `"live_route": "/kuiper/"`; one is lying.
`SUMMARY.md` publishes a headline that double-counts (17,562,419,519,040) and a
row that under-counts, both uncorrected. The current tests evaluate Isc and Imp
at the *cold* assessment temperature where alpha is positive — voltage cold,
current hot, and I had it half right.

**The honest accounting.** Nearly every finding is a defect in my own work. The
engine did not expose anyone else's errors; it exposed mine. That is what it is
for, and the only reason it worked is the rule that no number counts until
something independent agrees with it.

---

## TOP TEN PRIORITIES

1. **Quarantine every current-test verdict.** Any published pass/fail rate using
   the string-input or machine-input predicate is resting on two unsourced
   constants. The voltage and fuse counts can stand.
2. **Make the machine input an axis, not a constant.** The evidence base already
   orders this twice and defines the band {40, 50, 60, 80}.
3. **Source or delete the 20 A**, and state why the string test judges on Imp
   while the machine test judges on Isc. The 1.25 asymmetry is reasoned in the
   code; the Imp/Isc asymmetry is not.
4. **Fix current-at-temperature.** Voltage cold, current hot. Live in i0094 and
   non-conservative.
5. **Decide what `/kuiper/` serves.** Promote i0094 or say plainly at that
   address that it is not current. One of the two `live_route` fields is false.
6. **Rewrite "verified 0 differ" everywhere** to state the probe size and its
   coverage fraction. It touches every number this estate has published.
7. **Correct or withdraw `SUMMARY.md`'s double-counted headline.** A wrong total
   published beside a carefully corrected one undoes the correction's credit.
8. **Add a categorical axis to the grammar.** Zero of 93 refusals were parser
   defects; most want categories, not intervals. One feature, largest unlock.
9. **Make PUBLISHED mean served bytes** — a post-deploy step that fetches each
   route, hashes each cartridge, compares to the pointer. Catches the i0092
   class of failure in a minute.
10. **Invert the deploy allow-list to `paths-ignore`**, or add a check that
    fails when a folder containing `current.json` is not covered.

---

## WHAT FAILED VERIFICATION, KEPT ON PURPOSE

- "48 passes, 0 failures" — really 47 full passes plus a partial, and the log
  contains 3 genuinely failed runs from an earlier aborted session.
- Per-sweep JSON timings are faster than every logged pass, so those files came
  from a different, unlogged run than the one the total counts.
- `annihilate.json` and its commit message publish two different runs' numbers.
- Of 298 distinctive numbers I stated across 1,622 turns, 91 appear verbatim in
  an artefact, 158 are derivable from ones that do, and **49 are neither** —
  narrowed by machine, still unread by a person.
