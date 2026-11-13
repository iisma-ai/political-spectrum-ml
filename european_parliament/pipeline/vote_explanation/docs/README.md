# Vote explanation: decisions and documentation

## Question

When do European Parliament members vote against their political group, and do they follow their group or their national party when the two differ? Which groups, countries, national parties, subjects and kinds of roll call go with defection?

Same problem name as the [USA](../../../../usa/pipeline/vote_explanation/docs/README.md), so results can be compared.

Type: analytical, using notebooks.

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-13 | Report results as associations | Observational data on requested roll calls doesn't support causal claims |
| 2026-11-13 | Define the group line as the group's most common of For, Against and Abstention with the member included, with no line on a tie or under 5 active members | Leaving the member out makes every member of an evenly split group a defector. The USA defection rows also include the member. The leave-one-out line stays the baseline for prediction (step 4) |
| 2026-11-13 | Count abstaining against a For or Against line as defection, and report the For/Against-only rate alongside | Abstention is an active choice and often a group's line; the entity decisions name the For/Against version as secondary |
| 2026-11-13 | Define the national party line as the position all the member's other active national party colleagues took, needing at least two of them | National parties are small: a plurality would depend on the member's own vote, and leaving the member out turns even splits into mutual defection |
| 2026-11-13 | Treat the export's "Independent" national party entries as no party | They group unrelated independents by country; counting them as parties made Greens/EFA follow its "national party" on only 68% of conflicts |
| 2026-11-13 | Compare country defection with the rate expected from members' groups | Countries are spread unevenly across groups |
| 2026-11-13 | Use the responsible committee as the subject, with joint committees and roll calls without a committee as their own categories | One label per roll call (the OEIL subjects have about two per roll call); covers 17,660 of 25,688 roll calls |
| 2026-11-13 | Estimate odds ratios separately in each term on 300,000 sampled positions, with errors clustered by member and by roll call | Groups differ between terms; subject and roll-call type are shared by everyone voting on a roll call |
| 2026-11-13 | Show models with and without the national colleagues' votes on the same roll call, and label the second as describing the vote | That information is only known after the vote and nearly restates defection |
| 2026-11-13 | Predict defection with a time split: train on the 9th term from 2020, test on the 10th, against a constant rate and the member's own rate in their last earlier half-year | Project rule on time splits; group identity isn't a feature because groups changed between terms |

## Notebooks

- [01_defection_patterns](../notebooks/01_defection_patterns.ipynb): defection by group, country (against the rate expected from groups), kind of roll call and half-year; concentration among members; cutoff and abstention checks.
- [02_group_or_national_party](../notebooks/02_group_or_national_party.ipynb): how often group and national party lines conflict, whom members follow, where group defections come from, by delegation size and kind of roll call, and the national parties that conflict most.
- [03_subjects_and_model](../notebooks/03_subjects_and_model.ipynb): defection by responsible committee and a shift-share between terms, odds ratios by term, and time-split prediction with a bootstrap over roll calls.

Shared code is in `explanation.py`, tested by `test_explanation.py`.

## Open questions

- Whether national parties in government at home behave differently. Government membership isn't in the export and would need an outside source, such as ParlGov.
- Whether following the national party reflects instructions or shared views. A revealed line from votes can't tell them apart.
- Defection by OEIL subject, which is finer than committees but has several labels per roll call.
- Replacement and conversion in the change between terms, using the 341 re-elected members.

## Documentation

- [Entity decisions](../../../docs/README.md)
- [Polarization clustering](../../polarization_clustering/docs/README.md)
