# Vote explanation: decisions and documentation

## Question

Which issues, member characteristics and circumstances are associated with how members vote, and with when they break from their party?

Type: analytical, using notebooks.

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-02 | Report results as associations | The observational data doesn't support causal claims by default |
| 2026-11-04 | Use Voteview's Clausen categories for topics, 1857 to 2014 | They cover every roll call until 2014 and none after; one scheme for the whole period |
| 2026-11-04 | Split change into within-topic and topic-mix parts with an exact shift-share, counting a topic present in one period only as mix | Tests the agenda explanation without letting new or vanished topics leak into "within" |
| 2026-11-04 | Extend topics past 2014 with a text classifier, reporting only the three large topics and labelling results as predicted | The classifier is accurate on large topics and poor on small ones |
| 2026-11-04 | Model defection only on party-opposed roll calls, one row per member vote | Defection is only defined when the party majorities differ |
| 2026-11-04 | Exclude the roll call's own margin from all defection models | It is computed from the same votes, so it partly contains the outcome |
| 2026-11-04 | Use only previous-Congress features for a member's history and position | Keeps the member's own votes out of the explanation; Nokken-Poole still carries a little information from later Congresses through shared roll-call parameters |
| 2026-11-04 | Count independents with the nearer party by DW-NOMINATE and leave tied chambers missing | Otherwise caucusing independents are dropped and control is wrong, as in the 2021 Senate |
| 2026-11-04 | Sample up to 15,000 votes per Congress and chamber | Keeps models fast; rates from the sample weight Congresses and chambers more equally than all votes |
| 2026-11-04 | Train on 1901 to 1993 and test on 1995 to 2013, with party-line, constant-rate and own-history baselines | Follows the project's time-split and baseline rules |
| 2026-11-04 | Cluster logistic regression errors by member and roll call | Topic and vote type are shared by everyone voting on a roll call |
| 2026-11-04 | Compare regions within each party as well as against one reference group | A single reference mixes regional and party differences |

Reviewed by `method-reviewer` on 2026-11-04. Confirmed findings were fixed before conclusions were recorded. The fixes were not reviewed a second time.

## Notebooks

- [01_issue_polarization](../notebooks/01_issue_polarization.ipynb): party division by topic (1857 to 2014) and vote type (1991 to 2023), with exact splits into within-topic and mix changes.
- [02_topic_classifier](../notebooks/02_topic_classifier.ipynb): TF-IDF and logistic regression topic classifier, tested on 2013 to 2014 and applied from 2015.
- [03_defection_model](../notebooks/03_defection_model.ipynb): who defects, odds ratios by era and within party, time-split prediction against baselines, and vote-type effects.

Shared code is in `explanation.py`, tested by `test_explanation.py`.

## Open questions

- District partisanship and electoral margins aren't in Voteview. Adding them, for example district presidential vote, is the most important missing explanation.
- Splitting the fall in defection by group into replacement and conversion.
- Agenda selection within topics, which six broad categories can't detect.
- Forward-moving validation of the topic classifier over several Congresses, and whether Senate Foreign and Defense Policy really became less divided after 2014.

## Documentation

- [Analysis options and risks](../../../docs/methodology/analysis_options.md)
- [Topic and vote type sources and mapping](../../data_loading/docs/README.md)
