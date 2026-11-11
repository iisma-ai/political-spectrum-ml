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
  - roll calls that aren't near-unanimous (smaller side at least 2.5%);
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
| Gradient boosting + party split sizes | Adds how split each party was, known only after the vote; an upper reference, not a forecast |

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

## Running

```
python run_model.py            # reuses saved predictions and weights in outputs/
python run_model.py --refit    # refits every model
python run_model.py --rebuild  # also rebuilds the feature table
```

The full refit takes about 75 minutes on a CPU. Each model's predictions are saved as soon as it finishes, so an interrupted run resumes. Outputs in `outputs/` aren't committed.

## Notebooks

- [01_model_comparison](../notebooks/01_model_comparison.ipynb): test results against the baselines, bootstrap intervals, results by Congress, calibration, and where models help.
- [02_member_states](../notebooks/02_member_states.ipynb): what the GRU's member states hold, compared with Nokken-Poole and actual defection.

## Documentation

- [Analysis options and risks](../../../docs/methodology/analysis_options.md)
