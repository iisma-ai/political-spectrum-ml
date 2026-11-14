# Vote prediction: decisions and documentation

## Question

Given how every political group voted on a roll call, how well can it be predicted which European Parliament members break from their group's line, and does a sequence model of each member's record help? Asked in two settings: one roll call ahead during a sitting, and before the sitting day starts.

Same problem name and models as the [USA](../../../../usa/pipeline/vote_prediction/docs/README.md), so results can be compared.

Type: predictive, using `run_model.py`, with notebooks for evaluation.

## Setup

- **Target:** whether a group member's active position (For, Against or Abstention) differs from their group's line on the roll call.
- **Group line:** the group's most common of the three positions, leaving the member out, as the entity decisions set for prediction. It is the realised line from the final vote, standing in for the groups' voting lists published before the vote, which aren't in the data. So every model, including the group-line baseline, is conditional on knowing how every group voted. Being the realised plurality of colleagues, it absorbs any revolt within the group; how often it differs from the voting list is unknown.
- **Rows kept:**
  - roll calls that aren't near-unanimous (smaller of For and Against at least 2.5%), as in the other steps. This share comes from the final vote, like the group lines;
  - group members only: non-attached members have no group line;
  - a group is dropped for a roll call when its two most common positions, with the member, are 0 or 1 vote apart (0.24% to 0.39% of positions with a group line, per period; notebook 01), so all its members are kept or dropped together. Given the other members' votes, this drops defectors and keeps loyal members where the others split by 1 or 2 votes; notebook 01 counts the defections it removes. A filter on the others' margin would instead keep only the minority side of every 1-vote split, all as defectors (the tie artefact the explanation step avoided). Neither is clean; this one matches the USA;
  - positions with a group line (at least 5 other active members, no tie).
- **Two kinds of roll call,** the counterpart of the USA's party-opposed roll calls: whether the member's group line matched the line of most other groups or not. The share falls from 74% of positions in training to 63% in the test term.
- **Two settings:**
  - **one roll call ahead:** history includes everything before the roll call, including the member's positions on the same sitting day, usually seconds earlier. Earlier results of a sitting are known during it, so this isn't leakage;
  - **day-ahead:** history stops at the end of the previous sitting day, as if all of a day's predictions were made before the sitting. The count of earlier positions (and the GRU's step count) still includes earlier positions that day; because the close-split filter drops some rows partly by the member's vote, this can carry a same-day outcome into a count by one, far below the intervals.
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
| Day-ahead versions | Own past rate, logistic regression and boosting with history up to the previous sitting day; a GRU (one seed) that gets the previous sitting day's defection rates at the first position of each day instead of the previous outcome |
| Boosting + split, after the vote | Adds how split the member's group was (with the member, so it partly contains the answer) and what their national colleagues in the group did: an upper reference, not a forecast |
| Boosting + colleagues, after the vote | Adds only what the national colleagues in the group did |

Group identity isn't a feature, because groups changed between terms (Identity and Democracy ended; Patriots for Europe and Europe of Sovereign Nations are new in the test term). Group and national party history come from earlier roll calls only, so colleagues' positions on the same roll call never enter.

## Main results

Test term: the 10th (16 July 2024 to October 2026), 3.7 million positions, 8.1% defections ([notebook 01](../notebooks/01_model_comparison.ipynb)). All models know every group's realised line.

| Model | Line accuracy | Defection AUC | Average precision |
|---|---|---|---|
| Group line | 91.9% | 0.500 | 0.081 |
| *One roll call ahead* | | | |
| Own past rate | 91.9% | 0.726 | 0.202 |
| Logistic regression | 92.0% | 0.799 | 0.325 |
| Gradient boosting | 92.1% | 0.802 | 0.332 |
| GRU, three seeds | 92.6% | 0.834 to 0.839 | 0.415 to 0.425 |
| *Day-ahead* | | | |
| Own past rate | 91.8% | 0.718 | 0.191 |
| Logistic regression | 91.8% | 0.734 | 0.228 |
| Gradient boosting | 92.0% | 0.730 | 0.230 |
| GRU, one seed | 91.7% | 0.698 | 0.192 |
| *After the vote* | | | |
| Boosting + group split and colleagues | 96.0% | 0.967 | 0.827 |
| Boosting + colleagues only | 95.6% | 0.942 | 0.764 |

By kind of roll call, one roll call ahead:

| | Group with most groups (6.4% defections) | Group against most groups (11.1%) |
|---|---|---|
| Group line accuracy | 93.6% | 88.9% |
| GRU line accuracy | 94.1% | 90.0% |
| Own past rate AUC | 0.737 | 0.673 |
| Boosting AUC | 0.800 | 0.784 |
| GRU AUC | 0.838 to 0.840 | 0.815 to 0.823 |

- **Bootstrap** (1,000 draws of sitting days and members):
  - one roll call ahead, logistic regression over the own past rate (AUC +0.073) and the GRU over boosting (+0.038, from +0.033 to +0.042) are clear on all positions and on both kinds of roll call. Boosting over logistic regression isn't established;
  - day-ahead, logistic regression and boosting beat the own past rate by +0.015 and +0.011 AUC (lower bounds +0.009 and +0.002), only on roll calls where the group went against most groups. The day-ahead GRU is worse than day-ahead boosting (−0.032).
- **Same-day history carries the gain:** it adds +0.072 AUC to boosting and +0.14 to the GRU, but only +0.008 to the own past rate. Roll calls are a median 14 seconds apart and 89% follow one on the same text, so much of what predicts a defection is a defection earlier in the same sitting, mostly in runs of votes on the same text (same-text persistence isn't separated from persistence within a sitting in general).
- **First positions of the day** (1.7% of positions): no step between models is established.
- **Over time and by group:** AUC drops by about 0.02 from the validation half-years to the first half of the 10th term, then holds. The validation half-years were used for model choices. One roll call ahead, the GRU beats boosting in every group (single seed, no intervals).
- **Close-split filter:** it removes 0.24% of test positions but 2.9% of test defections, so the test defection rate is 8.13% instead of 8.35%.
- **Not yet checked:** other near-unanimous cutoffs (`--min-minority-share` exists; the USA ran 1% and 5%).

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-13 | Use the group line without the member as the target's reference | Entity decision for prediction; the member's own vote mustn't define what they are compared with |
| 2026-11-13 | Give every model the realised lines of all groups | Counterpart of the USA's known party positions; voting lists are published before votes but aren't in the data |
| 2026-11-13 | Split results by whether the member's group took the line of most other groups | Counterpart of party-opposed and bipartisan roll calls; the analysis rules ask for contested votes separately |
| 2026-11-13 | Decide the close-split filter on the whole group, including the member | Keeps or drops a group's members together and matches the USA. It selects on the member's vote in close groups, and the alternative selects worse; the defections removed are counted in notebook 01 |
| 2026-11-13 | Report one-roll-call-ahead and day-ahead models separately | Method review: roll calls are seconds apart, and most of the one-roll-call-ahead models' gain over the own past rate turned out to rest on the same day's earlier positions |
| 2026-11-13 | Bootstrap sitting days and members together, 1,000 draws | Roll calls on the same day are runs of votes on the same texts, so resampling roll calls would understate uncertainty; 1,000 draws as in the USA |
| 2026-11-13 | Allow up to 1,500 trees for the after-the-vote variants | The split variant hit the 600 cap |
| 2026-11-13 | Compute group and national party history over earlier roll calls only | Colleagues on the same roll call vote at the same time; tested in `test_features.py` |
| 2026-11-13 | Train on the 9th term and test on the 10th, with July 2023 to July 2024 for model choices | Project rule on time splits; the 10th term is the latest and has new groups |
| 2026-11-13 | Leave group identity out of the features | Groups changed between terms, so group dummies wouldn't carry over to the test term |

## Running

```
python run_model.py            # reuses saved predictions and weights in outputs/
python run_model.py --refit    # refits every model
python run_model.py --rebuild  # also rebuilds the feature table
```

`--seeds` and `--day-ahead-seeds` set the number of GRU seeds (3 and 1).

Outputs in `outputs/` aren't committed.

## Documentation

- [USA vote prediction](../../../../usa/pipeline/vote_prediction/docs/README.md)
- [Vote explanation](../../vote_explanation/docs/README.md)
