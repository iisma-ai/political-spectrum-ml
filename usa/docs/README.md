# USA: decisions and documentation

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-02 | Use Voteview as the primary roll-call source | House and Senate from the 1st to the 119th Congress, in one consistent format |
| 2026-11-02 | Don't use the ProPublica Congress API or GovTrack bulk data | Both have been shut down |
| 2026-11-02 | Start with agreement measures and networks, not deep learning | They are readable and give a reference for later models |
| 2026-11-02 | Use an RNN at vote level, not Congress level | About 119 Congress-level time steps per chamber are too few |
| 2026-11-02 | Study issues, factions and replacement versus conversion, not only party totals | Two-party clustering of recent Congresses is likely to be close to trivial |
| 2026-11-02 | Use several measures of separation, never DW-NOMINATE alone | Measures disagree on whether the earlier peak has been passed, and DW-NOMINATE can't show sitting members changing |
| 2026-11-02 | Follow people by Bioguide ID | Party switchers get a new ICPSR |
| 2026-11-03 | Use network modularity with nulls and HDBSCAN for structure; don't use BIC mixtures to count groups | Step 2 showed mixtures split parties into subgroups whose number reflects cluster shape |

## How this entity differs

Two dominant parties, two chambers, and a long continuous roll-call record. Separate voting behaviour from beliefs, because leaders influence which bills reach a roll call.

## Roadmap

1. **Foundation (done 2026-11-02):** download Voteview, build the member-by-vote matrices, reproduce the party gap, and check it against DW-NOMINATE.
2. **Clustering (done 2026-11-03):** agreement, modularity and embeddings per Congress, then time-aware communities and the replacement versus conversion split.
3. **Explanation (done 2026-11-04):** add topics and member covariates, then polarization by issue and defection modelling.
4. **Prediction (done 2026-11-11):** logistic baseline, gradient boosting and GRU. A graph network and bill text are deferred, because the GRU's gain over boosting is modest. Day-ahead versions added 2026-11-14 to compare with the European Parliament ([notebook 04](../pipeline/vote_prediction/notebooks/04_day_ahead.ipynb)).
5. **Other entities (started 2026-11-12):** European Parliament first, using the same pipeline; see [its roadmap](../../european_parliament/docs/README.md#roadmap).

## Documentation

- [US voting data sources](datasets/us_voting_sources.md)
- [Data sources for other entities](datasets/other_entities.md)
- [Similar projects](literature/similar_projects.md)
- [Analysis options and risks](methodology/analysis_options.md)

Problems:

- [Data loading](../pipeline/data_loading/docs/README.md)
- [Polarization clustering](../pipeline/polarization_clustering/docs/README.md)
- [Vote explanation](../pipeline/vote_explanation/docs/README.md)
- [Vote prediction](../pipeline/vote_prediction/docs/README.md)
