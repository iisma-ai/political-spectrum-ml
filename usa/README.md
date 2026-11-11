# USA

The most important conclusions of the analysis so far. Decisions are in [docs/README.md](docs/README.md).

Last updated: 2026-11-11

Scope so far: party and member-level analysis from 1857 to September 2026, using Voteview. The 119th Congress is unfinished and excluded from comparisons.

## Conclusions

- **The parties converged most around 1969 in the House and 1967 in the Senate.** Raw votes, agreement networks (including after adjusting for seat shares) and embeddings all agree on these years, at every near-unanimous cutoff tested. Then, 23% of House members and 33% of senators sat in a voting community led by the other party. Score gaps show a broader trough, reaching their lowest earlier, in the 1940s. See [notebook 01](pipeline/polarization_clustering/notebooks/01_party_gap_baseline.ipynb) and [notebook 02](pipeline/polarization_clustering/notebooks/02_agreement_networks.ipynb).
- **Mid-century convergence came from a cross-party middle.** In the House, two-thirds of members crossing into the other party's community were Southern Democrats. In 1969, 78% of Southern Democrats agreed more with Republicans than with non-Southern Democrats. In the Senate, crossing went both ways: Southern Democrats were 46% of crossers and non-Southern Republicans voting with Democrats 34% ([notebook 03](pipeline/polarization_clustering/notebooks/03_embeddings_factions.ipynb)).
- **Southern Democrats formed a distinct group** before and during this period, at times splitting off as their own community. That Southern structure inside the Democratic Party has largely disappeared.
- **Since 1969 the parties have separated, mainly through replacement.** New members differed from those they replaced. This holds for two measures, both chambers and start years from 1965 to 1979. In the House, sitting members contributed about a quarter of the change on Nokken-Poole, mainly Democrats moving left. Party switching was small. Most of the movement was on the Republican side ([notebook 04](pipeline/polarization_clustering/notebooks/04_turnover_conversion.ipynb)).
- **Recent Congresses split cleanly by party.** No large mixed community has existed since 1979, and no member has sat between the parties on DW-NOMINATE since the early 2000s. The DW-NOMINATE gap in 2023 was the widest of any completed Congress: 0.894 in the House and 0.895 in the Senate.
- **"Most polarized ever" depends on the measure.** Only the DW-NOMINATE gap has passed its peak of around 1900. Nokken-Poole, raw votes and seat-adjusted network modularity, averaged over 2015 to 2023, are all still below their pre-1945 extremes.

### Explaining votes

- **The parties grew apart within almost every topic.** The rise wasn't mainly a shift toward divisive topics: the changing topic mix explains 3% or less of the House change and 7% to 13% of the Senate change, in every window tested from 1901 to 2013 ([notebook](pipeline/vote_explanation/notebooks/01_issue_polarization.ipynb)). Agenda selection within topics isn't ruled out.
- **Social Welfare and Government Management are the most divided topics.** Social Welfare was also the least convergent at mid-century. A text classifier suggests division on both stayed high from 2015 to 2023 ([notebook](pipeline/vote_explanation/notebooks/02_topic_classifier.ipynb)).
- **For recent years, vote type matters more than topic.** House rules and procedural votes are almost always party-line (93% to 97% of contested ones), and House suspension votes rarely (14%). The Senate floor has shifted to nominations and cloture, from 8% to 69% of contested roll calls since the early 1990s, which accounts for about half the recent rise in its party-opposed share.
- **Defection was concentrated in two regional groups, and region's role reversed** ([notebook](pipeline/vote_explanation/notebooks/03_defection_model.ipynb)).
  - Within their party, Southern Democrats were less likely than Midwestern Democrats to defect before 1935 (odds ratio 0.58), and more likely after (2.89, then 2.12).
  - Northeastern Republicans went from 0.40 to 1.97 to 2.49 against Midwestern Republicans.
  - Southern Democrats now rarely defect. Northeastern House Republicans still defect more than any other group (12.9% of party-opposed votes in the 2020s).
- **Defection persists within members.** Their previous Congress's defection rate predicts later defection better than region, tenure or topic, and a full model improves on it only a little (AUC 0.707 against 0.697). Majority party members and freshmen defect slightly less. Tenure and age show no meaningful association.

### Predicting votes

Test years 2015 to 2024, models trained on 1991 to 2010. Every model is given how both parties actually voted, so these predict who breaks from a known party line, not how a roll call turns out.

- **The party line is hard to beat on accuracy.** Everyone voting with their party is right on 93.6% of votes (95.3% on party-opposed roll calls). The best single model, a GRU over each member's vote sequence, reaches 94.1% to 94.2% (95.5% to 95.6%) ([notebook 01](pipeline/vote_prediction/notebooks/01_model_comparison.ipynb)).
- **On ranking who defects, each richer model adds a little over all votes together.** Defection AUC and average precision, where no information gives 0.064:
  - the member's own past rate: 0.773 and 0.270;
  - logistic regression: 0.831 and 0.362;
  - gradient boosting: 0.850 and 0.379;
  - GRU, single seeds: 0.854 to 0.858 and 0.401 to 0.409.

  Each step is larger than chance in a bootstrap over roll calls and members. The steps differ by kind of roll call:
  - on party-opposed roll calls, the GRU's AUC gain over boosting is marginal;
  - where the parties agree, boosting doesn't beat logistic regression.
- **These results don't depend on the near-unanimous cutoff.** Refitted at 1% and 5% instead of 2.5%, the models keep the same order with each step larger than chance, the GRU beats the party line on accuracy by the same margin, and the two-behaviours shares below move by at most 1.4 points. Where the parties agree, levels do depend on the cutoff, because it selects on the final vote: the defection rate there goes from 11.2% to 15.7% ([notebook 03](pipeline/vote_prediction/notebooks/03_cutoff_sensitivity.ipynb)). Calibration and the chamber and vote-type results weren't rechecked.
- **The GRU's gain is clearest in the Senate** (average precision 0.46 to 0.49 against 0.41 for boosting), especially on nominations. In the House it comes mainly from amendments, and it is worse than boosting on procedural votes and resolutions. Neither model was tuned beyond when to stop.
- **The GRU stays close to calibrated on later years,** though defection fell from 11.7% in training years to 6.4% (shown for seed 0). Boosting over-predicts modestly at low risk; logistic regression roughly doubles it.
- **"Defection" covers two behaviours** ([notebook 02](pipeline/vote_prediction/notebooks/02_member_states.ipynb)):
  - on party-opposed roll calls, 71% to 83% of defections come from the half of the party nearer the other party;
  - on roll calls where the parties agree, 64% to 82% come from the outer half, members voting against bipartisan majorities;
  - this uses previous-Congress Nokken-Poole, which is partly built from those same kinds of votes.
- **Frequent defection since 2021 is mostly Republican:** 10 of the 12 most frequent House defectors and all 12 in the Senate. Some, like Graham, defect mainly on nominations and cloture rather than across the board.
- The GRU's learned propensity tracks earlier Nokken-Poole positions (within-party rank correlation 0.59 to 0.81), but both come from the same earlier votes, so this is a description, not a validation.

## How to read these

- They describe voting behaviour on votes that reached the floor, not beliefs. Leaders' agenda control affects which votes those are.
- Comparing eras assumes a stable agenda and scale, which hasn't been tested.
- Percentages depend on the near-unanimous vote cutoff. The figures above use 2.5%. The main conclusions were checked at other cutoffs: clustering at several, vote prediction at 1% and 5%.
- Shares in the turnover decomposition carry no uncertainty estimate. The direction is more reliable than the exact numbers.
- "South" means the 11 former Confederate states.
- The data is internally consistent with Voteview. Individual votes are externally confirmed only for a sample of 40 House roll calls from 2003 on.

## Not yet known

- Why the score gap and votes disagree on when the mid-century trough was deepest.
- How much of the pre-1969 narrowing reflects a changing agenda rather than members changing.
- Whether Southern Democrats were a stable third bloc or a spread within the party. The answer depends on Louvain's resolution, and only key Congresses were tested at higher resolutions.
- How much district partisanship and electoral margins explain defection. These aren't in Voteview yet.
- Whether a tuned gradient boosting model would match the GRU, and why the 115th House (2017 to 2018) is hard for every model.

## Conclusions by problem

| Problem | Status | Main result |
|---|---|---|
| [polarization_clustering](pipeline/polarization_clustering/docs/README.md) | Steps 1 and 2 done | Convergence peaked about 1969 through a Southern Democratic and Republican middle; separation since then came mainly from replacement |
| [vote_explanation](pipeline/vote_explanation/docs/README.md) | Step 3 done | Division grew within topics rather than through the topic mix; defection is regional and persists within members |
| [vote_prediction](pipeline/vote_prediction/docs/README.md) | Step 4 done (graph network and bill text deferred) | A GRU over each member's votes ranks defections best (AUC 0.854 to 0.858), modestly ahead of boosting; accuracy barely beats the party line |
