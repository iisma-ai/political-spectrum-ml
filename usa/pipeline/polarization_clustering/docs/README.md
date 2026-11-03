# Polarization clustering: decisions and documentation

## Question

How has the voting distance between and within US parties changed, and when, where and on which issues have they converged?

Type: analytical, using notebooks.

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-02 | Start with agreement rates and modularity before embeddings | They give a readable reference |
| 2026-11-02 | Use UMAP and t-SNE for pictures only | Their distances don't reliably measure convergence |
| 2026-11-02 | Don't use the DW-NOMINATE gap to study sitting members | Scores are fixed within a party spell |
| 2026-11-02 | Link people across Congresses by Bioguide ID, not ICPSR | Voteview gives party switchers a new ICPSR, which would count conversion as turnover |
| 2026-11-02 | Compare periods with multi-Congress averages | Single Congresses swing with the mix of votes |
| 2026-11-02 | Report every result at cutoffs of 0%, 2.5% and 10% | The Senate's peak agreement year moves from 1967 to 1931 at 10% |
| 2026-11-02 | Don't compare PC1 scores or separation between eras | Components fitted separately aren't on a common scale |
| 2026-11-03 | Build networks as complete weighted agreement graphs, with no edge for pairs sharing under 20 votes | Follows Waugh et al. (2009) and avoids thresholds |
| 2026-11-03 | Report party modularity adjusted for seat shares alongside raw modularity | Raw modularity is capped lower when one party holds most seats; the Senate's 1937 low came from this |
| 2026-11-03 | Judge communities by party alignment and resolution tests, not by their count | Louvain on a complete positive graph tends to produce two halves |
| 2026-11-03 | Test within-party structure against 10 shuffled-vote nulls, and call it structure, not factions | The null removes all member consistency, so beating it doesn't prove discrete groups |
| 2026-11-03 | Don't count blocs or factions with Gaussian mixtures chosen by BIC | They split parties into pure subgroups whose number reflects cluster shape; BIC picks 2 on true two-Gaussian data |
| 2026-11-03 | Measure arrivals and departures from the party's earlier mean in the turnover decomposition | Otherwise switching and replacement depend on where the scale's zero is |
| 2026-11-03 | Treat party switching as its own part, separate from conversion and replacement | It is a distinct mechanism, and Bioguide linking identifies it |
| 2026-11-03 | Test turnover results at start years from 1965 to 1979 | Conversion on the defection rate is sensitive to the start year |

Reviewed by `method-reviewer` on 2026-11-02 (step 1) and 2026-11-03 (step 2). Confirmed findings were fixed before conclusions were recorded. The fixes for step 2 were not reviewed a second time.

## Notebooks

- [01_party_gap_baseline](../notebooks/01_party_gap_baseline.ipynb): rebuilds the party gap, compares it with raw-vote measures, and tests when the parties were closest. Step 1.
- [02_agreement_networks](../notebooks/02_agreement_networks.ipynb): modularity, Louvain communities, cross-party blocs, resolution and cutoff tests, community tracking, and within-party structure. Step 2.
- [03_embeddings_factions](../notebooks/03_embeddings_factions.ipynb): multidimensional scaling maps, Gaussian mixtures and HDBSCAN, with checks on why the mixture adds components. Step 2.
- [04_turnover_conversion](../notebooks/04_turnover_conversion.ipynb): exact split of change into conversion, switching and replacement, for the Nokken-Poole gap and the defection rate. Step 2.

Shared code is in `clustering.py`, tested by `test_clustering.py`.

## Open questions

- A full scan of Louvain resolutions across all Congresses, to tell whether Southern Democrats were a stable third bloc.
- Uncertainty estimates for the turnover shares, for example by bootstrapping roll calls.
- Why score gaps and vote measures disagree on when the mid-century trough was deepest.
- Topic-level polarization using Voteview issue codes or Policy Agendas topics.

## Documentation

- [Analysis options and risks](../../../docs/methodology/analysis_options.md)
