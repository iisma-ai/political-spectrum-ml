# USA

The most important conclusions of the analysis so far. Decisions are in [docs/README.md](docs/README.md).

Last updated: 2026-11-03

Scope so far: party and member-level analysis from 1857 to September 2026, using Voteview. The 119th Congress is unfinished and excluded from comparisons.

## Conclusions

- **The parties converged most around 1969 in the House and 1967 in the Senate.** Raw votes, agreement networks (including after adjusting for seat shares) and embeddings all agree on these years, at every near-unanimous cutoff tested. Then, 23% of House members and 33% of senators sat in a voting community led by the other party. Score gaps show a broader trough, reaching their lowest earlier, in the 1940s. See [notebook 01](pipeline/polarization_clustering/notebooks/01_party_gap_baseline.ipynb) and [notebook 02](pipeline/polarization_clustering/notebooks/02_agreement_networks.ipynb).
- **Mid-century convergence came from a cross-party middle.** In the House, two-thirds of members crossing into the other party's community were Southern Democrats. In 1969, 78% of Southern Democrats agreed more with Republicans than with non-Southern Democrats. In the Senate, crossing went both ways: Southern Democrats were 46% of crossers and non-Southern Republicans voting with Democrats 34% ([notebook 03](pipeline/polarization_clustering/notebooks/03_embeddings_factions.ipynb)).
- **Southern Democrats formed a distinct group** before and during this period, at times splitting off as their own community. That Southern structure inside the Democratic Party has largely disappeared.
- **Since 1969 the parties have separated, mainly through replacement.** New members differed from those they replaced. This holds for two measures, both chambers and start years from 1965 to 1979. In the House, sitting members contributed about a quarter of the change on Nokken-Poole, mainly Democrats moving left. Party switching was small. Most of the movement was on the Republican side ([notebook 04](pipeline/polarization_clustering/notebooks/04_turnover_conversion.ipynb)).
- **Recent Congresses split cleanly by party.** No large mixed community has existed since 1979, and no member has sat between the parties on DW-NOMINATE since the early 2000s. The DW-NOMINATE gap in 2023 was the widest of any completed Congress: 0.894 in the House and 0.895 in the Senate.
- **"Most polarized ever" depends on the measure.** Only the DW-NOMINATE gap has passed its peak of around 1900. Nokken-Poole, raw votes and seat-adjusted network modularity, averaged over 2015 to 2023, are all still below their pre-1945 extremes.

## How to read these

- They describe voting behaviour on votes that reached the floor, not beliefs. Leaders' agenda control affects which votes those are.
- Comparing eras assumes a stable agenda and scale, which hasn't been tested.
- Percentages depend on the near-unanimous vote cutoff. The figures above use 2.5%.
- Shares in the turnover decomposition carry no uncertainty estimate. The direction is more reliable than the exact numbers.
- "South" means the 11 former Confederate states.
- The data is internally consistent with Voteview. Individual votes are externally confirmed only for a sample of 40 House roll calls from 2003 on.

## Not yet known

- Why the score gap and votes disagree on when the mid-century trough was deepest.
- How much of the pre-1969 narrowing reflects a changing agenda rather than members changing.
- Whether Southern Democrats were a stable third bloc or a spread within the party. The answer depends on Louvain's resolution, and only key Congresses were tested at higher resolutions.

## Conclusions by problem

| Problem | Status | Main result |
|---|---|---|
| [polarization_clustering](pipeline/polarization_clustering/docs/README.md) | Steps 1 and 2 done | Convergence peaked about 1969 through a Southern Democratic and Republican middle; separation since then came mainly from replacement |
| [vote_explanation](pipeline/vote_explanation/docs/README.md) | Not started | None yet |
| [vote_prediction](pipeline/vote_prediction/docs/README.md) | Not started | None yet |
