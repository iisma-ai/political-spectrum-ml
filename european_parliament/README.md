# European Parliament

The most important conclusions of the analysis so far. Decisions are in [docs/README.md](docs/README.md).

Last updated: 2026-11-13

Scope so far: plenary roll calls from July 2019 to October 2026 (the 9th term and the 10th term so far), from HowTheyVote.eu (Open Database License). The 10th term is unfinished.

## Conclusions

- **Political groups vote cohesively, and those to the right of the EPP less so.** On contested roll calls, the mean agreement index is 0.85 to 0.97 for the five groups from Greens/EFA to the EPP in both terms. It is about 0.8 for ECR, 0.65 for Identity and Democracy (9th term) and 0.77 for Patriots for Europe (10th). In ECR, ID and PfE, national delegations agree far more than the group as a whole (0.13 to 0.32 above a random set of the same size, against 0.02 to 0.06 in the centre groups), and these groups abstain as a group on 7% to 12% of contested roll calls ([notebook 01](pipeline/polarization_clustering/notebooks/01_group_cohesion.ipynb)).
- **One ordering organises the chamber in both terms:** Greens/EFA and The Left, then S&D, Renew, the EPP and ECR, then ID or PfE and ESN. Agreement between groups falls with distance along it: neighbours agree on 50% to 89% of contested roll calls, the two ends on 13% to 25%. Members at the centre are on the winning side most often. The ordering describes who votes with whom; it isn't a measured left-right scale ([notebook 03](pipeline/polarization_clustering/notebooks/03_networks_embeddings.ipynb)).
- **In the 10th term the EPP voted more with S&D and Renew and less with ECR.** Its agreement with S&D rose from 70.5% to 78.6% and with Renew from 78.7% to 86.7%. With ECR it fell from 59.8% to 50.3%. It was on the winning side of 94% of contested roll calls, against 82%. In the member network it is never grouped with ECR, PfE and ESN in the 10th term, at any resolution tested, while in the 9th term it is grouped with ECR and ID at resolution 1 or below ([notebook 02](pipeline/polarization_clustering/notebooks/02_coalitions.ipynb), [notebook 03](pipeline/polarization_clustering/notebooks/03_networks_embeddings.ipynb)).
- **Yet a majority to the right of S&D and Renew wins more often.** When S&D–Renew and the ECR pair (ECR with ID, then PfE) vote For against Against, the EPP joins the ECR pair less often in the 10th term (19%, against 35%). But when it does, that side wins 73% of the time, against 18%.
  - Overall the ECR pair's line won 14.0% of these splits, against 6.3%.
  - On 66% of the 10th-term roll calls where the EPP joined the ECR pair, the EPP and the groups to its right cast more votes than the whole other side, against 3.7% in the 9th term.
  - A majority without the EPP won 17.6% of contested roll calls in the 9th term and 6.4% in the 10th ([notebook 02](pipeline/polarization_clustering/notebooks/02_coalitions.ipynb)).

## How to read these

- Only roll calls are recorded, and they are taken on request, so they are a selected sample of plenary votes. Show-of-hands votes are missing. All results describe voting behaviour on requested roll calls, not beliefs.
- The two terms vote on different agendas and with different members (the 2024 election, members changing group, the UK leaving in 2020). A difference between terms isn't the same members changing their votes. Separating member replacement from change is an open question.
- Results hold at near-unanimous cutoffs of 0%, 2.5% and 10% and with abstention counted against.
- The size-adjusted modularity used in the USA can't be compared across chambers with different numbers of groups, so the European Parliament isn't ranked against the USA on it.
- The data matches the European Parliament's Open Data Portal: no roll call missing on any sitting day, and no disagreement on individual positions in a random sample of 200 roll calls ([data loading](pipeline/data_loading/docs/README.md)).

## Conclusions by problem

| Problem | Status | Main result |
|---|---|---|
| [data_loading](pipeline/data_loading/docs/README.md) | Foundation done | Internally consistent, complete against the Open Data Portal, and matching it on a 200 roll-call sample |
| [polarization_clustering](pipeline/polarization_clustering/docs/README.md) | Step 2 done | Cohesive groups along one ordering; the EPP closer to S&D and Renew in the 10th term, while the EPP with the groups to its right now holds enough votes to win when they combine |
