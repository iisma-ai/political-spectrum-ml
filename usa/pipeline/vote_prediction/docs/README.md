# Vote prediction: decisions and documentation

## Question

How well can a member's next vote be predicted, especially when they break from their party, and can a sequence model track their position over time? Asked in two settings: one roll call ahead, and before the day's first roll call (day-ahead).

Type: predictive, using `run_model.py`, with notebooks for evaluation.

## Setup

- **Target:** whether a Democrat or Republican Yea or Nay goes against their own party's position on that roll call. A vote prediction follows from it: the party position, flipped when the predicted probability is above one half.
- **Party position:** the majority of the member's party on that roll call, leaving the member out. The other party's position is that party's majority.
  - Both are the realised positions from the final votes. They stand in for whip positions announced before the vote, which aren't in the data.
  - So every model, including the party-line baseline, is conditional on knowing how both parties actually voted.
- **Rows kept:**
  - roll calls that aren't near-unanimous (smaller side at least 2.5%). This share comes from the final vote, so it is information from after the vote, like the party positions. When the parties agree, the smaller side is mostly defectors, so those roll calls are kept only if enough members defected, and a member's own vote can decide whether a roll call at the threshold is kept;
  - Democrats and Republicans only. Independents are left out of both the rows and the party positions;
  - a party is dropped for a roll call when its whole margin, including the member, is 0 or 1 vote, so all its members are kept or dropped together. Given the other members' votes, this drops defectors and keeps loyal members where the others split by 1 or 2 votes (corrected 2026-11-13; the earlier wording said the reverse). A filter on the others' margin instead would keep only the minority side of every 1-vote split, all as defectors. Neither filter is clean in close parties; the share of votes affected is small.
- **Member history:** one sequence per person, chamber and party, ordered by date and roll number. A party switch starts a new sequence. History starts in 1981, so members serving in 1991 have a record.
- **Two settings:**
  - **one roll call ahead:** history includes the member's votes on earlier roll calls the same day (lower roll numbers on the same date);
  - **day-ahead:** history stops at the end of the member's previous day with votes, which can be weeks earlier after a recess. Voteview has calendar dates but no times, so a session running past midnight spans two dates, and day-ahead history can then include votes from the same session. The day-ahead GRU's step count still includes earlier kept votes the same day.
- **Periods:** train on 1991 to 2010 (Congresses 102 to 111), choose the number of trees and GRU epochs on 2011 to 2014 (112 and 113), and test on 2015 to 2024 (114 to 118). Vote types exist from 1991.

## Models

| Model | Inputs |
|---|---|
| Party line | Everyone votes with their party; the reference |
| Own past rate | The member's earlier defection rate on the same kind of vote (party-opposed or not), no fitting |
| Logistic regression | Nine history rates, roll-call features (vote type, own party Yea or Nay, party-opposed), member traits (party and region, tenure, freshman, majority), and history times party-opposed |
| Gradient boosting | The same features, without the interactions |
| GRU | The member's whole vote sequence: each roll call's features plus whether they defected on the previous one. Three seeds |
| Day-ahead versions | Own past rate, logistic regression and boosting with history (and the count of earlier votes) up to the previous day; a GRU, three seeds, that gets the previous day's defection rates at the first vote of each day instead of the previous outcome |
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
| 2026-11-06 | Decide the close-split filter and split sizes on the whole party, including the member | The review found both depended on the member's own vote; fixed and every model refitted. Corrected 2026-11-13: the whole-party filter is the one that depends on the member's vote in close parties; kept, see "Rows kept" |
| 2026-11-06 | Report single GRU seeds against gradient boosting, with a bootstrap over roll calls and members | An average of three seeds against one boosting model isn't like for like, and votes cluster by member as well as roll call. Resampling days instead of roll calls ([notebook 04](../notebooks/04_day_ahead.ipynb)) gives nearly the same intervals |
| 2026-11-06 | Compare the GRU's member states with previous-Congress Nokken-Poole | Same-Congress scores come from the same votes, so agreement with them is partly built in |
| 2026-11-06 | No graph network or bill text yet | The GRU's gain over boosting is modest; see open questions. Since 2026-11-14 it is also known to depend on same-day outcomes |
| 2026-11-11 | Bootstrap roll calls and members together (two-way), 1,000 draws, on all test votes and separately on party-opposed roll calls and roll calls where the parties agree | The analysis rules ask for contested votes separately; the GRU's AUC gain over boosting on party-opposed roll calls is marginal, which pooled results hid (with days resampled it isn't established: −0.003 to +0.006) |
| 2026-11-11 | Show seed 0 of the GRU in by-Congress, calibration and subgroup results | Like-for-like with the one boosting model; the 3-seed average is an ensemble |
| 2026-11-11 | Keep the split-size variant's cohesion on the whole party, including the member, and don't read its gain as "the residual is about the roll call" | Leaving the member out makes it differ between a defector and loyal colleagues on the same roll call (a per-row leak). Including them means it counts defectors, so its gain partly comes from containing the answer |
| 2026-11-11 | Treat GRU propensity against previous-Congress Nokken-Poole as description, not validation | Both are built from the same earlier votes |
| 2026-11-11 | Keep the 2.5% near-unanimous cutoff; compare only the order of models, not levels, across cutoffs on roll calls where the parties agree | Refitting at 1% and 5% leaves the ranking ladder, the accuracy gain and the two-behaviours shares unchanged ([notebook 03](../notebooks/03_cutoff_sensitivity.ipynb)). The cutoff uses the final vote, so where the parties agree it selects on defection: the defection rate there goes from 11.2% to 15.7% |
| 2026-11-14 | Add day-ahead versions of every model and report both settings | The European Parliament's one-roll-call-ahead gains turned out to rest mostly on same-day history; the same check was needed here before comparing entities |
| 2026-11-14 | Bootstrap chamber-days and members for the day-ahead comparison | Roll calls on the same day are often runs on the same bill, so days are the cautious unit, as in the European Parliament |
| 2026-11-14 | Fit three day-ahead GRU seeds | Method review: one seed that stopped at epoch 3 was too weak to say the day-ahead GRU is below boosting |

## Main results

Test years 2015 to 2024, single GRU seeds. Details and limits are in the notebooks.

| Model | AUC | Average precision |
|---|---|---|
| Everyone votes with their party | 0.500 | 0.064 |
| Own past rate on this kind of vote | 0.773 | 0.270 |
| Logistic regression | 0.831 | 0.362 |
| Gradient boosting | 0.850 | 0.379 |
| GRU, single seeds | 0.854 to 0.858 | 0.401 to 0.409 |
| *Day-ahead* ([notebook 04](../notebooks/04_day_ahead.ipynb)) | | |
| Own past rate on this kind of vote | 0.770 | 0.267 |
| Logistic regression | 0.818 | 0.330 |
| Gradient boosting | 0.840 | 0.347 |
| GRU, three seeds | 0.828 to 0.834 | 0.323 to 0.338 |
| Gradient boosting + party split sizes (after the vote, upper reference) | 0.953 | 0.608 |

Vote accuracy: 93.6% for the party line, 94.1% to 94.2% for single GRU seeds (95.3% and 95.5% to 95.6% on party-opposed roll calls), and 93.9% for day-ahead boosting.

**Day-ahead** ([notebook 04](../notebooks/04_day_ahead.ipynb), bootstrap of chamber-days and members):
- **Ranking mostly holds.** Day-ahead boosting beats the own past rate on party-opposed roll calls and where the parties agree, by +0.070 AUC over all votes (+0.061 to +0.079), against +0.076 one roll call ahead. Same-day history adds +0.009 AUC to boosting and at most +0.024 to the GRU, mostly in runs of roll calls on the same bill.
- **Flagging defections holds less.** Over the own past rate, average precision gains +0.081 day-ahead against +0.140 for the GRU one roll call ahead. Defections caught at 0.5 fall from 19.1% (GRU) to 9.5% (day-ahead boosting). The accuracy gain over the party line falls from 0.56 to 0.26 points.
- **The day-ahead GRU is below day-ahead boosting** in all three seeds (bootstrapped for seed 0: −0.010, −0.014 to −0.006), so the GRU's edge over boosting depends on same-day outcomes in this design.
- **Where the parties agree, boosting is slightly below logistic regression** one roll call ahead in both the roll-call bootstrap (−0.013 to −0.001, notebook 01) and the day bootstrap (−0.013 to −0.0004); both only just exclude zero. Over all votes the GRU over boosting is +0.004 (+0.001 to +0.008), and on party-opposed roll calls it isn't established (−0.003 to +0.006).

**Other near-unanimous cutoffs** ([notebook 03](../notebooks/03_cutoff_sensitivity.ipynb)): refitting every model (GRU seed 0 only) at 1% and 5% gives the same ladder, with each step over all votes larger than chance. Over all votes, GRU AUC is 0.854 at 1% and 0.857 at 5%. The GRU beats the party line on accuracy by the same margin (94.3% against 93.8% at 1%, 94.1% against 93.5% at 5%). The subset findings hold: the GRU's AUC gain over boosting on party-opposed roll calls is marginal at every cutoff, and boosting doesn't beat logistic regression where the parties agree at any cutoff (at 1% and 5%, that it is worse isn't established; at 2.5% both bootstraps put it just below, see above). Where the parties agree, average precision rises and AUC falls with the cutoff, so levels there aren't comparable across cutoffs. The day-ahead models weren't refitted at other cutoffs. Intervals don't include refit variation (seed, early stopping), and calibration and subgroup results weren't rechecked.

## Open questions

- Would tuning boosting close the gap with the GRU? Day-ahead, boosting is already ahead.
- Would vote times (for example Senate roll-call XML, unverified) change the day-ahead results for sessions running past midnight?
- Would a day-ahead GRU that sees earlier days' outcome sequences, not three rates, beat boosting?
- Why is the 115th House (2017 to 2018) hard for every model?
- What does the GRU's hidden state encode? A linear probe per Congress, scored on test years, would test it.

## Running

```
python run_model.py            # reuses saved predictions and weights in outputs/
python run_model.py --refit    # refits every model
python run_model.py --rebuild  # also rebuilds the feature table
python run_model.py --rebuild --refit --seeds 1 --min-minority-share 0.01  # another near-unanimous cutoff, saved in outputs/cutoff_0.01/
```

`--day-ahead-seeds` sets the number of day-ahead GRU seeds (default 1; the results use 3). Adding the day-ahead models to saved outputs took about 16 minutes plus 14 per extra seed.

The full refit takes about 55 minutes on a 16-thread CPU (`--threads 16`) (run on 2026-11-11 on a machine with 32 GB of RAM; peak memory wasn't measured). Each model's predictions are saved as soon as it finishes, so an interrupted run resumes. Outputs in `outputs/` aren't committed.

## Notebooks

- [01_model_comparison](../notebooks/01_model_comparison.ipynb): test results against the baselines, bootstrap intervals, results by Congress, calibration, and where models help.
- [02_member_states](../notebooks/02_member_states.ipynb): what the GRU's member states hold, compared with Nokken-Poole and actual defection.
- [03_cutoff_sensitivity](../notebooks/03_cutoff_sensitivity.ipynb): every model refitted at near-unanimous cutoffs of 1% and 5%; the ladder, accuracy, bootstrap intervals and the two kinds of defection compared with the standard 2.5%.
- [04_day_ahead](../notebooks/04_day_ahead.ipynb): day-ahead models against one-roll-call-ahead ones, what same-day history adds and where, and the comparison with the European Parliament.

## Documentation

- [Analysis options and risks](../../../docs/methodology/analysis_options.md)
