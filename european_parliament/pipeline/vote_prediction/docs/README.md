# Vote prediction: decisions and documentation

## Question

How well can a European Parliament member's next position be predicted, especially when they break from their political group's line, and does a sequence model of each member's record help?

Same problem name and models as the [USA](../../../../usa/pipeline/vote_prediction/docs/README.md), so results can be compared.

Type: predictive, using `run_model.py`, with notebooks for evaluation.

## Setup

- **Target:** whether a group member's active position (For, Against or Abstention) differs from their group's line on the roll call.
- **Group line:** the group's most common of the three positions, leaving the member out, as the entity decisions set for prediction. It is the realised line from the final vote, standing in for the groups' voting lists published before the vote, which aren't in the data. So every model, including the group-line baseline, is conditional on knowing how every group voted.
- **Rows kept:**
  - roll calls that aren't near-unanimous (smaller of For and Against at least 2.5%), as in the other steps. This share comes from the final vote, like the group lines;
  - group members only: non-attached members have no group line;
  - a group is dropped for a roll call when its two most common positions, with the member, are 0 or 1 vote apart (0.44% of positions). Deciding this without the member would keep or drop rows depending on the member's own vote;
  - positions with a group line (at least 5 other active members, no tie).
- **Two kinds of roll call,** the counterpart of the USA's party-opposed roll calls: whether the member's group line matched the line of most other groups (71% of positions) or not.
- **Member history:** one sequence per member across both terms, ordered by time, with flags for a change of group or term. History starts with the data in July 2019; roll calls from 2019 only supply history.
- **Periods:** train on January 2020 to June 2023, choose the number of trees and GRU epochs on July 2023 to 15 July 2024 (end of the 9th term), and test on the 10th term (16 July 2024 to October 2026).

## Models

| Model | Inputs |
|---|---|
| Group line | Everyone votes with their group's line; the reference |
| Own past rate | The member's earlier defection rate on the same kind of roll call (with most groups or not), no fitting |
| Logistic regression | Nine history rates (the member's, their national party's and their group's, from earlier roll calls only), roll-call features (kind of vote, own group line, share of other groups with the same line, responsible committee, group's seat share), member features (country, age, national party seats in the group, no national party, changed group), and history times the kind of roll call |
| Gradient boosting | The same features, without the interactions |
| GRU | The member's whole sequence of positions: each roll call's features plus whether they defected on the previous one. Three seeds |
| Gradient boosting + split | Adds how split the member's group was (with the member) and what their national colleagues in the group did, known only after the vote: an upper reference, not a forecast |

Group identity isn't a feature, because groups changed between terms (Identity and Democracy ended; Patriots for Europe and Europe of Sovereign Nations are new in the test term). Group and national party history come from earlier roll calls only, so colleagues' positions on the same roll call never enter.

## Main results

Pending method review. Test term: the 10th (16 July 2024 to October 2026), 3.7 million positions, 8.1% defections ([notebook 01](../notebooks/01_model_comparison.ipynb)).

| Model | Line accuracy | Defection AUC | Average precision |
|---|---|---|---|
| Group line | 91.9% | 0.500 | 0.081 |
| Own past rate | 91.9% | 0.726 | 0.202 |
| Logistic regression | 92.0% | 0.799 | 0.325 |
| Gradient boosting | 92.1% | 0.802 | 0.332 |
| GRU, three seeds | 92.6% | 0.834 to 0.839 | 0.415 to 0.425 |
| Boosting + split (after the vote) | 96.0% | 0.967 | 0.827 |

- Every step except boosting over logistic regression is clearly larger than chance in a two-way bootstrap (roll calls and members), on all positions and on both kinds of roll call.
- The GRU's gain over boosting comes entirely from earlier positions on the same sitting day: on a member's first position of the day the two are equal (AUC 0.696 and 0.703, average precision 0.183 for both).
- No decay across the term change, and the GRU beats boosting in every group.

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-13 | Use the group line without the member as the target's reference | Entity decision for prediction; the member's own vote mustn't define what they are compared with |
| 2026-11-13 | Give every model the realised lines of all groups | Counterpart of the USA's known party positions; voting lists are published before votes but aren't in the data |
| 2026-11-13 | Split results by whether the member's group took the line of most other groups | Counterpart of party-opposed and bipartisan roll calls; the analysis rules ask for contested votes separately |
| 2026-11-13 | Decide the close-split filter on the whole group, including the member | Same fix as the USA's: otherwise the member's own vote decides whether the row is kept |
| 2026-11-13 | Compute group and national party history over earlier roll calls only | Colleagues on the same roll call vote at the same time; tested in `test_features.py` |
| 2026-11-13 | Train on the 9th term and test on the 10th, with July 2023 to July 2024 for model choices | Project rule on time splits; the 10th term is the latest and has new groups |
| 2026-11-13 | Leave group identity out of the features | Groups changed between terms, so group dummies wouldn't carry over to the test term |

## Running

```
python run_model.py            # reuses saved predictions and weights in outputs/
python run_model.py --refit    # refits every model
python run_model.py --rebuild  # also rebuilds the feature table
```

Outputs in `outputs/` aren't committed.

## Documentation

- [USA vote prediction](../../../../usa/pipeline/vote_prediction/docs/README.md)
- [Vote explanation](../../vote_explanation/docs/README.md)
