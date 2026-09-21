# The return half: what the night's numbers mean

Every number below is quoted from one of: night-results/harvest-crunch.json,
annihilate.json, collapse.json, ulp.json, breakpoint.json,
enumerated-total-corrected.json. Nothing is computed here.

Note before the ten: the brief said harvest-crunch.json holds 48 findings and
476,426,722 cases. The file on disk says `"registered": ["H1","H2","H3","H4"]`,
`"refused_count": 59` and `"cases_total": 360526364`. I report what the file says.

---

## 1. The comparison that was never broken on a real number

THE SENTENCE  "sweep value over 1,000,000 ordinary x 1,000,000 raw bit patterns x 33 specials, x 4 limits, count where (x > limit) differs from !(x <= limit)"
THE COUNT     "disagree_on_a_finite_number": 0, out of "cases": 8000132.
WHAT IT MEANS On every ordinary number, the two ways of asking "is this over the
              limit?" gave the same answer every single time. Nothing changes.
              Nobody has to revisit a design because of this.
WHAT IT DOES NOT MEAN It does not mean the rewrite was pointless (see 2), and it
              does not mean the limits themselves are right. The file says so
              itself: "It says nothing about whether the limits themselves are
              right and nothing about any physics."

## 2. The million passes that cannot be justified

THE SENTENCE  same sentence as 1.
THE COUNT     "disagree_on_a_non_finite_number": 1001580; "old_predicate_calls_it_inside_the_rating": 1001580, across "limits": 4, out of 8,000,132 cases.
WHAT IT MEANS When the input is not a number at all - a blank, a broken reading,
              a divide by zero - the old test quietly answered "that is fine, it
              is inside the rating". A person should make missing or broken input
              stop the check and say so, instead of passing.
WHAT IT DOES NOT MEAN It does not mean a million real designs were wrongly passed.
              A million is the size of the sweep of junk values, not a count of
              anything that has happened. The right reading is "the failure mode
              exists and is silent", not "it has occurred a million times".

## 3. Where a string crosses 1500 V

THE SENTENCE  "sweep beta -0.30..-0.20 x series 20..34 x temperature -20..0 C x 10 module bins, count where N*Voc*(1+beta/100*(T-25)) > 1500"
THE COUNT     "over_1500_with_each_bin_s_own_coefficient": 460863 out of "cases": 1227150.
WHAT IT MEANS Using each module class's own printed numbers, a large share of the
              combinations of string length and cold temperature put the string
              over its 1500 V rating - and the file solves the exact temperature
              at which each length crosses. A person should pick string length
              against that crossing temperature, not against a rule of thumb.
WHAT IT DOES NOT MEAN It is NOT "a third of designs are unsafe". The sweep includes
              string lengths and temperatures nobody would choose; the fraction is
              a property of the box that was swept, not of the world. And the file
              warns the cold design temperature "is a meteorological input this
              file does not hold", so which side of the crossing a real site sits
              on is not settled here.

## 4. The part of the answer that turns on which coefficient you paired

THE SENTENCE  same sentence as 3.
THE COUNT     "cases_where_the_two_coefficients_disagree": 23929 out of 1,227,150. The file calls it "the whole of the earlier mistake".
WHAT IT MEANS Only a small slice of the answers change depending on whether the
              right temperature coefficient was paired with the right module class
              - but inside that slice the verdict flips outright. A person should
              treat the pairing as a data-entry check, not a modelling debate.
WHAT IT DOES NOT MEAN It does not mean the pairing error is harmless because the
              slice is small. The file also prints the voltage cost of the wrong
              pairing at 30 in series (for example "T645 with -0.22 in place of
              -0.25" at "-10.0 C": 14.269), and that is a real margin, not a
              rounding.

## 5. Current: nothing clears

THE SENTENCE  "sweep rear gain 0..30% x temperature -20..25 C x series 20..34 x 10 bins, count where Imp*(1+phi*rear)*(1+alpha/100*(T-25)) > 20 or 1.25*Isc*(1+phi*rear)*(1+alpha/100*(T-25))*2 > 40"
THE COUNT     "over_the_machine_input": 832650 out of "cases": 832650, and "clears_everything": 0.
WHAT IT MEANS Every single combination swept went over the machine's 40 A input.
              Not most - all of them. Either the rating being compared against is
              the wrong rating, or the way the current is being worked out is
              wrong. A person should stop and establish which, before quoting any
              of the other current numbers.
WHAT IT DOES NOT MEAN It does NOT mean every real machine is overloaded. A count
              that comes back "all of them" is far more often a sign that the test
              is mis-stated than that the world has failed everywhere. This is the
              clearest case in the night of a big number that is a question, not a
              finding.

## 6. Drawings that run out of dots

THE SENTENCE  "sweep pieces 1..384 x symbols 0..96 x symbol rings 0..96 x plain rings 0..96, count where SR*16 + (S-SR)*6 + NR*5 + (P-S-NR)*2 > 896"
THE COUNT     "shapes_where_the_floors_alone_exceed_the_budget": 98231558 of "shapes_examined": 125864192; "shapes_that_fit": 27632634; and "57 symbol rings on their own are enough to do it".
WHAT IT MEANS The drawing gives each piece a minimum number of dots. Add up enough
              pieces and the minimums alone exceed the whole budget, and then some
              piece of the picture is drawn with nothing at all. The number to
              carry is 57: past about that many ringed symbols, the picture starts
              losing parts silently. A person should make the drawing say "too
              much asked for" rather than drop a piece.
WHAT IT DOES NOT MEAN 98 million is not 98 million broken drawings. It is the count
              of shapes in a swept grid, most of which no real drawing asks for.
              The file is explicit: "It does not say which real drawings reach
              those shapes."

## 7. Five billion pairs, and the 21,463 that sat on the line

THE SENTENCE  Each case run twice: the rule as written, and the same rule with every "or equal to" made strict. The two differ on exactly one set of cases - where the two sides are equal.
THE COUNT     "did_not_annihilate": 21463 out of "cases": 5315908358.
WHAT IT MEANS Out of five billion checks, about twenty thousand landed exactly on
              their threshold, where pass or fail is decided by whether the rule
              says "or equal to". That is a wording choice, not a measurement. A
              person should write down, once, which convention is used - and not
              argue any individual case on its merits.
WHAT IT DOES NOT MEAN It does not mean 21,463 designs are wrong. The file refuses
              this directly: it proves only that these cases "land exactly on a
              threshold, so their verdict is decided by whether the rule says 'or
              equal to' rather than by the design".

## 8. Where the pairs actually parted - and where they did not

THE SENTENCE  The same pair test, recorded with its axis values.
THE COUNT     collapse.json: "partings_seen": 21463, "partings_stored": 138, "host_agreed": 138, "host_disagreed": 0. The biggest single source is K05-05 with "partings_seen": 18231 of "cases": 400000000; the string-voltage rule K01-01 has "partings_seen": 0 of 46965.
WHAT IT MEANS Almost all of the on-the-line cases come from one screen-layout rule
              about window height, not from anything electrical - and the
              electrical string-voltage rule produced none at all. Every stored
              case was re-checked on a second machine and agreed, none disagreed.
              A person should read this as: the threshold ambiguity is a
              screen-layout housekeeping matter, and the card's arithmetic is
              trustworthy.
WHAT IT DOES NOT MEAN The zero for the voltage rule does not mean the voltage rule
              is safe. It means no case in THIS grid sat exactly on the line - the
              file says "A zero here means no case in THIS grid parted, not that
              none exists". And only 138 of 21,463 were kept (a buffer of 40 per
              predicate), so the stored set is a sample, not the whole.

## 9. Zero last-bit flips in nineteen billion

THE SENTENCE  Run the same physics two ways that differ only in the last bit of arithmetic - one divides by a pi cut to 15 digits, one by full pi; one multiplies in a different order - and count where the verdict flips.
THE COUNT     "v1500_transient_space": "cases": 18980843400, "flips": 0. And "p_probe_sees_nothing_at_that_upper_bound": 0.999968.
WHAT IT MEANS Not one case in nineteen billion changed its answer because of the
              last digit of the arithmetic. The zero rules out the worry that
              published verdicts sit on knife-edges of rounding. Nothing changes:
              the file states "no published total moves".
WHAT IT DOES NOT MEAN It does not mean the probe proved anything. The file is
              unusually honest here: a probe of 200,000 would have seen nothing
              anyway with probability 0.999968 even if flips existed at the 95 per
              cent upper bound. The full enumeration carries the result; the probe
              alone would have been worthless. And it does not say which side of a
              threshold is correct - "both forms are valid doubles".

## 10. The one-pixel cliff

THE SENTENCE  Sweep window widths 600..760 against a grid of furniture sizes, and measure what changes between one pixel and the next.
THE COUNT     "combos_in_band": 4131887760; "combos_in_band_failing_three_radii_today": 2490173280; the jump sits at "discontinuity_between_px": [640, 641] with "share_of_cases_that_step": 1.0, "mean_diameter_delta_px": 57.195331165 and "ratio_max": 27.5.
WHAT IT MEANS There is a single pixel of window width where the layout jumps, and
              it jumps for every case, not some of them, because two rules
              disagree about where "narrow" starts - 640 in one place, 700 in the
              other. On a real phone, one pixel of width changes the picture
              dramatically. A person should make the two rules use one number.
WHAT IT DOES NOT MEAN Using one number does not fix it. The file tested exactly
              that: "unify_on_700": "removes_step": false, verdict "moves, does not
              remove", with the largest step afterwards still 63.193599563 px
              between 700 and 701. The cliff is in the layout itself, not in the
              disagreement about where it sits. Anyone who reads this as "align
              the breakpoints" has misread the number.

---

## A number I could not give a meaning to, and say so

enumerated-total-corrected.json reports "corrected_total_cases": 31292801587800
against "as_first_published": 17391591847800 - "understated_by_pct": 79.9,
because "The overnight counter read one key ... One sweep writes 'space' and
'evaluated' instead." I can say what went wrong with the bookkeeping. I cannot
say what the total number of enumerated cases MEANS, because nobody has
established what a case is worth. Thirty-one trillion cases is a statement about
how finely the axes were sliced and about nothing else, until someone writes down
which of those cases a decision depends on. Until that is written down, the
headline total should not be quoted at all.

## Where zeroes are results

- "disagree_on_a_finite_number": 0 rules out that the comparison rewrite changes
  any real verdict.
- "flips": 0 in 18,980,843,400 rules out last-bit rounding as an explanation for
  any published verdict.
- "host_disagreed": 0 rules out the card and the processor disagreeing on the
  cases that were stored.
- "clears_everything": 0 in the current sweep rules nothing out - it flags the
  test as mis-stated.

## Where a large count is large only because the axis was swept finely

Items 3, 6 and 10. 460,863 crossings, 98,231,558 over-budget shapes and
2,490,173,280 failing combinations are all counts of grid points, and the grid
was chosen by whoever wrote the sweep. None is a count of anything that has
happened. The honest way to quote each is as its threshold - the crossing
temperature, the 57 rings, the 640/641 pixel - never as its total.

---

# THE THREE THAT WOULD CHANGE A DECISION

1. "over_the_machine_input": 832650 out of "cases": 832650, and "clears_everything": 0.
   DECISION: stop quoting any current result and settle whether the rating or the
   formula is wrong, before a single string sizing is issued on these numbers.

2. "unify_on_700": "removes_step": false - "moves, does not remove", with the
   largest step after the change still 63.193599563 px between 700 and 701.
   DECISION: do not spend the sprint aligning the two breakpoints; the layout
   itself has to stop jumping, which is a different and larger piece of work.

3. "shapes_where_the_floors_alone_exceed_the_budget": 98231558 of 125,864,192,
   and "57 symbol rings on their own are enough to do it".
   DECISION: cap the drawing where the minimums exceed the 896 budget and make it
   refuse out loud, rather than letting a piece of the picture be drawn with
   nothing and shipped looking complete.
