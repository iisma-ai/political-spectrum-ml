# Polarization clustering: decisions and documentation

## Question

How united are the European Parliament's political groups, which groups vote together, and how is the chamber divided? How did this change between the 9th term (2019 to 2024) and the 10th term so far?

Same problem name as the [USA](../../../../usa/pipeline/polarization_clustering/docs/README.md), so results can be compared.

Type: analytical, using notebooks.

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-13 | Measure group cohesion with the agreement index over For, Against and Abstention, after Hix, Noury and Roland (2005) ([paper](https://doi.org/10.1017/S0007123405000128), confirmed; formula unverified against the paywalled text) | The standard European Parliament measure; abstention is often a group's line. A For/Against version is reported as a sensitivity check |
| 2026-11-13 | Count a group on a roll call only when at least 5 of its members take an active position (`MIN_GROUP_VOTES`) | Avoids lines and indexes from one or two members |
| 2026-11-13 | Define a group's line as its most common of the three positions, missing on a tie | Matches the group-line decision in [`docs/README.md`](../../../docs/README.md) |
| 2026-11-13 | Take the winning side from positions (more For than Against), not from the export's `result` | `result` is filled for only 8,097 of 25,686 roll calls; positions match it on 8,082 |
| 2026-11-13 | Compare national delegations with random same-size draws from their group | Small sets agree more by chance; the draw separates national splits from size |
| 2026-11-13 | Study splits between "S&D–Renew" (both taking the same line) and the "ECR pair" (ECR with ID in the 9th term, or PfE in the 10th), counting only For against Against splits | The closest pairs on each side of the EPP. The names are the groups', not claims about ideology. An abstaining side can't win, so abstention splits would count as losses by construction |
| 2026-11-13 | Test the arithmetic of wins directly: on splits where the EPP joins the ECR pair, compare the For or Against votes cast by EPP, ECR, ID or PfE and ESN with all votes on the other side | Group sizes count members, but outcomes depend on votes cast, and turnout changed between terms |
| 2026-11-13 | Give each member the group they sat in for most of their roll calls in the term | Members change group within terms (79 and 21 among members in the matrices; 100 and 21 member-terms over all group spells); networks need one label |
| 2026-11-13 | Compute group modularity over members of a group, without non-attached members, and also report mean agreement within and between groups | Non-attached members aren't a group. Size-adjusted modularity depends on the number of groups, so it isn't compared with the USA's two parties; within and between agreement is |
| 2026-11-13 | Use the same agreement networks, modularity, Louvain and classical scaling as the USA, fitted per term, and don't compare map coordinates or eigenvalue shares between terms | Same measure names and definitions across entities; separately fitted maps aren't on a common scale |
| 2026-11-13 | Don't interpret the second dimension of the maps | The maps bend into an arch: a quadratic in dimension 1 explains 49% and 34% of dimension 2, and the shape can't separate an artefact from a centre-against-ends pattern |
| 2026-11-13 | Read Louvain results for which bloc the EPP joins, not for the number of blocs | The count changes between resolutions 0.9 and 1.1; the EPP's placement doesn't |
| 2026-11-13 | Report every result at cutoffs of 0%, 2.5% and 10%, and with abstention counted against | Same cutoffs as the USA, plus the abstention rule required by the entity decisions |

## Notebooks

- [01_group_cohesion](../notebooks/01_group_cohesion.ipynb): agreement index by group, term, half-year and kind of roll call; abstention as a group line; national delegations inside groups.
- [02_coalitions](../notebooks/02_coalitions.ipynb): line agreement between groups, who is on the winning side, winning coalitions, and the EPP's position when the centre-left and the right split.
- [03_networks_embeddings](../notebooks/03_networks_embeddings.ipynb): member agreement networks, group modularity against a shuffled null, Louvain communities and resolutions, maps per term with a horseshoe check, and where non-attached members sit.

Shared code is in `clustering.py`, tested by `test_clustering.py`.

Reviewed by `method-reviewer` on 2026-11-13. Its findings were fixed before the conclusions were recorded: the USA modularity comparison was dropped, the group-size explanation was replaced by a direct count of votes cast, abstention splits were separated, and wording on the horseshoe, the group order, Louvain blocs and national delegations was corrected. The fixes weren't reviewed a second time.

## Open questions

- Whether the EPP's change between terms comes from re-elected members voting differently or from new members (the USA turnover decomposition, adapted to groups).
- How much of each group's disagreement lies between national delegations, with a proper within/between decomposition.
- Cohesion and coalitions by subject, using the Legislative Observatory subjects and responsible committees in `subjects`.
- Group line against national party line: which one members follow when they differ (roadmap step 3).
- An outside measure of party positions, such as an expert survey, to say whether the main ordering is left and right.
- Whether which roll calls get requested changed between terms, since all results describe requested roll calls only.

## Documentation

- [Entity decisions](../../../docs/README.md)
- [Data loading](../../data_loading/docs/README.md)
