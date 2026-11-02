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

## Notebooks

- [01_party_gap_baseline](../notebooks/01_party_gap_baseline.ipynb): rebuilds the party gap, compares it with raw-vote measures, and tests when the parties were closest. Step 1 of the roadmap.

## Documentation

- [Analysis options and risks](../../../docs/methodology/analysis_options.md)
