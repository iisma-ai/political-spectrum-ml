# Analysis options and risks

## Clustering methods, in order of priority

| Priority | Method | Question | Notes |
|---|---|---|---|
| 1 | Pairwise agreement matrix, cross-party agreement rate | How often do members of different parties vote together? | The readable reference for everything else |
| 2 | Agreement network, modularity, Leiden or Louvain | How many blocs, how strongly divided? | Doesn't assume party structure |
| 3 | PCA or SVD with Gaussian mixtures or HDBSCAN | Are there factions and moderates? | Compare with party using adjusted Rand index |
| 4 | Spectral clustering, including across time | How do communities change? | Handles changing membership |
| 5 | Signed networks | Is there opposition as well as agreement? | May show coalitions |
| 6 | Bipartite clustering of members and votes | Which votes define each bloc? | Links clusters to issues |

## Convergence measures

Calculate per Congress, by chamber and issue:

- gap between party means
- overlap between the parties
- within-party spread
- modularity
- share of votes with party majorities on opposite sides

Compare periods with multi-Congress averages, and test each result at cutoffs of 0%, 2.5% and 10%.

Separate **replacement** (new members differ from those who left) from **conversion** (the same members change). Link people by Bioguide ID, so party switchers count as conversion. [Snyder's work](https://politicalscience.yale.edu/node/15160) discusses within-member drift.

Embeddings fitted separately per Congress don't share a scale. Use time-aware methods, align through members who serve in both Congresses, or use Nokken-Poole scores.

## Feasible tasks

| Type | Task | Notes |
|---|---|---|
| Descriptive | Polarization by chamber and issue, 1789 to now | Includes change-point detection |
| Structure | Factions and moderates over time | Clusters plus member trajectories |
| Explanation | Which issues divide or unite parties | Join Policy Agendas topics |
| Explanation | What is associated with defection | Gradient boosting with SHAP; not causal |
| Prediction | A member's vote on a bill | Yea or Nay |
| Prediction | Defection from party majority | Most informative target |
| Prediction | Polarization or member drift | Few Congress-level data points, so large uncertainty |
| Classification | Party from votes or speech | Accuracy is itself a polarization measure |

## RNN design

About 119 Congress-level time steps per chamber are too few for an RNN to reliably beat a simple baseline. A better design:

1. A GRU or LSTM updates each legislator's state, vote by vote.
2. It combines that state with a bill representation to predict the next vote.
3. The hidden state becomes a time-varying ideology measure, which can feed the clustering.
4. Compare that measure with Nokken-Poole scores as a check.

## Risks

1. **Party-line votes inflate accuracy.** Use the party-majority baseline, and report on contested votes and defections.
2. **Random splits leak the future.** Hold out later Congresses or sessions.
3. **Votes aren't pure ideology.** [Agenda control](https://www.ipr.northwestern.edu/documents/working-papers/2010/IPR-WP-10-11.pdf) shapes what gets a roll call, and [ideology and party discipline are hard to separate](https://repec.cepr.org/repec/cpr/ceprdp/DP14291.pdf).
4. **The agenda changes over time**, so comparisons across decades are [questioned](https://politicalscience.yale.edu/node/15160). Issue-level analysis reduces this.
5. **Vote codes need deliberate handling.** See [the source documentation](../datasets/us_voting_sources.md).
6. **Two-party results can be trivial.** Two clusters may simply reproduce party labels.
