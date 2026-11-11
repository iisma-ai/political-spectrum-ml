# Vote prediction: decisions and documentation

## Question

How well can a member's next vote be predicted, especially when they break from their party, and can a sequence model track their position over time?

Type: predictive, using `run_model.py`, with notebooks for evaluation.

## Setup

- **Target:** whether a Democrat or Republican Yea or Nay goes against their own party's position on that roll call. A vote prediction follows from it: the party position, flipped when the predicted probability is above one half.
- **Party position:** the majority of the member's party on that roll call, leaving the member out. The other party's position is that party's majority.
  - Both are the realised positions from the final votes. They stand in for whip positions announced before the vote, which aren't in the data.
  - So every model, including the party-line baseline, is conditional on knowing how both parties actually voted.
- **Rows kept:**
  - roll calls that aren't near-unanimous (smaller side at least 2.5%). This share comes from the final vote, so it is information from after the vote, like the party positions. When the parties agree, the smaller side is mostly defectors, so those roll calls are kept only if enough members defected, and a member's own vote can decide whether a roll call at the threshold is kept;
  - Democrats and Republicans only. Independents are left out of both the rows and the party positions;
  - a party is dropped for a roll call when its whole margin, including the member, is 0 or 1 vote. Deciding this without the member would keep or drop rows depending on the member's own vote.
- **Member history:** one sequence per person, chamber and party, ordered by date and roll number. A party switch starts a new sequence. History starts in 1981, so members serving in 1991 have a record.
- **Periods:** train on 1991 to 2010 (Congresses 102 to 111), choose the number of trees and GRU epochs on 2011 to 2014 (112 and 113), and test on 2015 to 2024 (114 to 118). Vote types exist from 1991.

## Models

| Model | Inputs |
|---|---|
| Party line | Everyone votes with their party; the reference |
| Own past rate | The member's earlier defection rate on the same kind of vote (party-opposed or not), no fitting |
| Logistic regression | Nine history rates, roll-call features (vote type, own party Yea or Nay, party-opposed), member traits (party and region, tenure, freshman, majority), and history times party-opposed |
| Gradient boosting | The same features, without the interactions |
| GRU | The member's whole vote sequence: each roll call's features plus whether they defected on the previous one. Three seeds |
| Gradient boosting + party split sizes | Adds how split each party was, known only after the vote. The member's own party split counts its defectors, so this partly contains the answer: an upper reference, not a forecast |

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-02 | Split by time | Random splits leak the future |
| 2026-11-02 | Compare with the party-majority baseline | Party-line voting makes raw accuracy look high |
| 2026-11-02 | Treat defection prediction as the main target | It is the hard and informative case |
| 2026-11-02 | Add RNNs after logistic and gradient boosting baselines | A complex model has to beat simple ones to be justified |
| 2026-11-05 | Leave the member out of their own party's position | Otherwise the member's vote helps define what they are compared with |
| 2026-11-05 | Compute every history feature from earlier votes only, and feed the GRU only earlier outcomes | Tested in `test_features.py` and `test_sequence_model.py` |
| 2026-11-05 | Use roll-call vote types, not topics | Topics stop in 2014; vote types cover 1991 on |
| 2026-11-06 | Decide the close-split filter and split sizes on the whole party, including the member | The review found both depended on the member's own vote; fixed and every model refitted |
| 2026-11-06 | Report single GRU seeds against gradient boosting, with a bootstrap over roll calls and members | An average of three seeds against one boosting model isn't like for like, and votes cluster by member as well as roll call |
| 2026-11-06 | Compare the GRU's member states with previous-Congress Nokken-Poole | Same-Congress scores come from the same votes, so agreement with them is partly built in |
| 2026-11-06 | No graph network or bill text yet | The GRU's gain over boosting is modest; see open questions |
| 2026-11-11 | Bootstrap roll calls and members together (two-way), 1,000 draws, on all test votes and separately on party-opposed roll calls and roll calls where the parties agree | The analysis rules ask for contested votes separately; the GRU's AUC gain over boosting on party-opposed roll calls is marginal, which pooled results hid |
| 2026-11-11 | Show seed 0 of the GRU in by-Congress, calibration and subgroup results | Like-for-like with the one boosting model; the 3-seed average is an ensemble |
| 2026-11-11 | Keep the split-size variant's cohesion on the whole party, including the member, and don't read its gain as "the residual is about the roll call" | Leaving the member out makes it differ between a defector and loyal colleagues on the same roll call (a per-row leak). Including them means it counts defectors, so its gain partly comes from containing the answer |
| 2026-11-11 | Treat GRU propensity against previous-Congress Nokken-Poole as description, not validation | Both are built from the same earlier votes |

## Main results

Test years 2015 to 2024, single GRU seeds. Details and limits are in the notebooks.

| Model | AUC | Average precision |
|---|---|---|
| Everyone votes with their party | 0.500 | 0.064 |
| Own past rate on this kind of vote | 0.773 | 0.270 |
| Logistic regression | 0.831 | 0.362 |
| Gradient boosting | 0.850 | 0.379 |
| GRU, single seeds | 0.854 to 0.858 | 0.401 to 0.409 |
| Gradient boosting + party split sizes (after the vote, upper reference) | 0.953 | 0.608 |

Vote accuracy: 93.6% for the party line, 94.1% to 94.2% for single GRU seeds (95.3% and 95.5% to 95.6% on party-opposed roll calls).

## Open questions

- Would tuning boosting close the gap with the GRU?
- Why is the 115th House (2017 to 2018) hard for every model?
- Does the result hold with the near-unanimous cutoff at 1% or 5%?
- What does the GRU's hidden state encode? A linear probe per Congress, scored on test years, would test it.

## Running

```
python run_model.py            # reuses saved predictions and weights in outputs/
python run_model.py --refit    # refits every model
python run_model.py --rebuild  # also rebuilds the feature table
```

The full refit takes about 55 minutes on a 16-thread CPU (`--threads 16`) (run on 2026-11-11 on a machine with 32 GB of RAM; peak memory wasn't measured). Each model's predictions are saved as soon as it finishes, so an interrupted run resumes. Outputs in `outputs/` aren't committed.

## Notebooks

- [01_model_comparison](../notebooks/01_model_comparison.ipynb): test results against the baselines, bootstrap intervals, results by Congress, calibration, and where models help.
- [02_member_states](../notebooks/02_member_states.ipynb): what the GRU's member states hold, compared with Nokken-Poole and actual defection.

## Documentation

- [Analysis options and risks](../../../docs/methodology/analysis_options.md)
