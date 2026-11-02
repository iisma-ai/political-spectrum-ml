# USA

The most important conclusions of the analysis so far. Decisions are in [docs/README.md](docs/README.md).

Last updated: 2026-11-02

Scope so far: party-level measures from 1857 to September 2026, using Voteview. The 119th Congress is unfinished and excluded from comparisons. Clustering hasn't started.

## Conclusions

- **The parties have separated since the 1970s, after a broad mid-century trough.** Separation was also high around 1900. Both NOMINATE scores and raw votes show this general shape. These results come from [the party gap notebook](pipeline/polarization_clustering/notebooks/01_party_gap_baseline.ipynb).
- **The DW-NOMINATE gap in 2023 was the widest of any completed Congress:** 0.894 in the House and 0.895 in the Senate, with delegates excluded.
- **No member has sat between the two parties on DW-NOMINATE since the early 2000s.** The last overlap was in the Congress beginning in 2001 in the House and 2003 in the Senate.
- **"Most polarized ever" depends on the measure.** Only the DW-NOMINATE gap has passed its earlier peak. Nokken-Poole and raw-vote averages for 2015 to 2023 haven't. House party majorities opposed each other on 80.1% of contested votes, against 94.0% in 1903.
- **The closest point between the parties isn't one clear year.** Score gaps show a broad trough, with bootstrap ranges as wide as 1935 to 1977. On votes, House cross-party agreement peaks in 1969 at every cutoff tested. The Senate peaks in 1967, but that moves to 1931 under a stricter cutoff.
- **The DW-NOMINATE gap can't show sitting members converging.** Scores are fixed within a party spell, so it moves only through turnover and party switches.

## How to read these

- They describe voting behaviour on votes that reached the floor, not beliefs. Leaders' agenda control affects which votes those are.
- Comparing 1900 with today assumes a stable agenda and scale, which hasn't been tested.
- Percentages depend on the near-unanimous vote cutoff. The figures above use 2.5%.
- The data is internally consistent with Voteview. Individual votes are externally confirmed only for a sample of 40 House roll calls from 2003 on. Senate votes and earlier House votes aren't.

## Not yet known

- How much of the separation comes from turnover versus sitting members changing. A first look is mixed and isn't a decomposition.
- Whether the DW-NOMINATE gap tracks short-run change. Links between Congress-to-Congress changes are weak or undetectable, and the cause is unresolved.

## Expectation to test

Recent Congresses have one dominant voting axis, so two-cluster results there are likely to reproduce party labels. This is why step 2 focuses on periods, issues, factions and turnover.

## Conclusions by problem

| Problem | Status | Main result |
|---|---|---|
| [polarization_clustering](pipeline/polarization_clustering/docs/README.md) | Baseline done, clustering not started | Mid-century trough, then rising separation; peak claims depend on the measure |
| [vote_explanation](pipeline/vote_explanation/docs/README.md) | Not started | None yet |
| [vote_prediction](pipeline/vote_prediction/docs/README.md) | Not started | None yet |
