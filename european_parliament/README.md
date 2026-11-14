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

### Explaining votes

- **When a member's national party colleagues and their political group take different lines, members vote with their national colleagues.** Where all of a member's other colleagues from the same national party and group (at least two) vote together against the group's line, the member votes with them on 97.0% (9th term) and 96.1% (10th) of positions, and with the group on 2.3% and 2.9%. When the two lines agree, members vote against united colleagues on under 1%.
  - Random placebo colleagues from other parties in the group are followed only 23% of the time.
  - Measuring against the line of the group's other national parties gives 96.7% to 97.6%.
  - ID and PfE make up about 40% of these conflicts. By group the share ranges from 88% (ESN) to 99%, and counted once per national party and roll call it is 82% to 85% ([notebook 02](pipeline/vote_explanation/notebooks/02_group_or_national_party.ipynb)).
- **Members vote against their group line on 8.8% and 8.2% of positions on contested roll calls, a third of them abstentions.** The rate is lowest in Greens/EFA (2.2% and 4.2%) and S&D (about 5%), and highest in ECR (14.1% and 12.6%), ID (23.3%) and PfE (15.5%).
  - Measured against the line of the group's other national parties, ID's rate is 37.5% and PfE's 20.7%, because a few large delegations largely set those groups' lines.
  - About half of all defections are with united national colleagues. In ID and PfE it is three quarters ([notebook 01](pipeline/vote_explanation/notebooks/01_defection_patterns.ipynb), [notebook 02](pipeline/vote_explanation/notebooks/02_group_or_national_party.ipynb)).
- **Members of smaller national parties defect more.** Where a member's party holds fewer than 10 seats in the group, the odds of defecting are about twice those in larger parties, also after accounting for what the colleagues did. Against the line without the member's own party, the ratio is 1.4 to 1.5, so part of the gap comes from large delegations setting the line.
  - Final votes carry slightly higher odds than amendments; age shows none.
  - Subjects matter less than groups: Women's Rights roll calls see the most defection, Budgetary Control little ([notebook 03](pipeline/vote_explanation/notebooks/03_subjects_and_model.ipynb)).
- **Members from some countries defect more than their group colleagues from other countries in both terms:** Sweden, Malta, Czechia, Denmark and Hungary, by 1.4 to 2.1 times. This mostly reflects their national parties ([notebook 01](pipeline/vote_explanation/notebooks/01_defection_patterns.ipynb)).
- **Explanatory models add little to a member's own record.** Trained on the 9th term and tested on the 10th, a member's own previous half-year rate gives AUC 0.702, and gradient boosting adds about 0.008, which holds when members or roll calls are resampled. Knowing how the member's national colleagues voted on the same roll call gives 0.922, which describes collective defection rather than forecasting it ([notebook 03](pipeline/vote_explanation/notebooks/03_subjects_and_model.ipynb)). The prediction step below tests this with forecasting models; its definitions differ, so the numbers aren't merged.

### Predicting votes

All prediction models know every group's realised line on the roll call (standing in for the voting lists, which aren't in the data), and predict which members break from their group's line without them. Trained on January 2020 to June 2023, tested on the 10th term (8.1% defections), with 95% intervals from resampling sitting days and members ([notebook 01](pipeline/vote_prediction/notebooks/01_model_comparison.ipynb)).

- **Before a sitting day, fitted models add little to a member's own record.** With history only up to the previous sitting day, the member's own past defection rate gives AUC 0.718 and average precision 0.191. Logistic regression and gradient boosting give 0.734 and 0.730 (+0.015 and +0.011), and that gain is established only on roll calls where the member's group went against most other groups. No day-ahead model meaningfully beats the group line on accuracy (91.9%). A day-ahead GRU did worse than the own rate (0.698, one seed), a limit of its design rather than a finding about members.
- **During a sitting, the previous roll calls predict the next.** With history up to the previous roll call, the own past rate gives 0.726, logistic regression 0.799 and boosting 0.802, which can't be told apart from logistic regression. A GRU over each member's sequence of positions gives 0.834 to 0.839 (three seeds), with average precision 0.415 to 0.425 against 0.332 for boosting, clearly on both kinds of roll call.
  - Most of the gain over the own record comes from the same day's earlier positions: +0.072 AUC for boosting and +0.14 for the GRU. Roll calls on a sitting day are a median 14 seconds apart, and 89% follow one on the same text.
  - Line accuracy rises only from 91.9% (group line) to 92.6%, because defections are rare.
- **What national colleagues did on the same roll call** gives AUC 0.942 on its own. It's strongly associated with defection, as in the explanation step, but isn't known before the vote.
- **Limits:** realised group lines, not voting lists; one test term with new groups; intervals exclude refit variation; a filter on near-tied groups removes 2.9% of test defections; other near-unanimous cutoffs not yet checked for prediction.

## How to read these

- Only roll calls are recorded, and they are taken on request, so they are a selected sample of plenary votes. Show-of-hands votes are missing. All results describe voting behaviour on requested roll calls, not beliefs.
- The two terms vote on different agendas and with different members (the 2024 election, members changing group, the UK leaving in 2020). A difference between terms isn't the same members changing their votes. Separating member replacement from change is an open question.
- Clustering and explanation results hold at near-unanimous cutoffs of 0%, 2.5% and 10% and with abstention counted against. Prediction hasn't been checked at other cutoffs yet.
- National party lines are revealed from votes, not recorded instructions. Voting with national colleagues may reflect instructions, shared views or both.
- Defection here is measured against the group's most common position with the member included. The leave-one-out version used as the prediction baseline gives 9.0% and 8.4%.
- The size-adjusted modularity used in the USA can't be compared across chambers with different numbers of groups, so the European Parliament isn't ranked against the USA on it.
- The data matches the European Parliament's Open Data Portal: no roll call missing on any sitting day, and no disagreement on individual positions in a random sample of 200 roll calls ([data loading](pipeline/data_loading/docs/README.md)).

## Conclusions by problem

| Problem | Status | Main result |
|---|---|---|
| [data_loading](pipeline/data_loading/docs/README.md) | Foundation done | Internally consistent, complete against the Open Data Portal, and matching it on a 200 roll-call sample |
| [polarization_clustering](pipeline/polarization_clustering/docs/README.md) | Step 2 done | Cohesive groups along one ordering; the EPP closer to S&D and Renew in the 10th term, while the EPP with the groups to its right now holds enough votes to win when they combine |
| [vote_explanation](pipeline/vote_explanation/docs/README.md) | Step 3 done | Members vote with united national colleagues over the group on about 96% of conflicts; defection is highest on the right and about half collective; explanatory models add little to a member's own record |
| [vote_prediction](pipeline/vote_prediction/docs/README.md) | Step 4 done | Before a sitting, models add little to a member's own rate (AUC 0.73 against 0.72); during it, a GRU reaches 0.84, mostly from the same day's earlier roll calls |
