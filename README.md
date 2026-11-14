# Politics

A comparative study of how votes can be explained and how the political spectrum changes over time, using machine learning on parliamentary voting data.

## Countries and organisations

| Entity | Status | Conclusions |
|---|---|---|
| USA | Data foundation, clustering, explanation and prediction done | [usa/README.md](usa/README.md) |
| European Parliament | Data foundation, clustering, explanation and prediction done (2019 to 2026) | [european_parliament/README.md](european_parliament/README.md) |

## Comparing entities

Problems use the same names and, where possible, the same measures in every entity. Some measures can't be compared directly:

- **Size-adjusted modularity** depends on the number of groups as well as on how separated they are, so a multi-group chamber can't be ranked against a two-party one on it. The European Parliament also reports agreement within and between groups, which doesn't have this problem.
- **Vote prediction** is reported in two settings in both entities: one roll call ahead (history includes earlier roll calls the same day) and day-ahead. In the European Parliament, where roll calls run seconds apart on the same texts, most of the models' gain over a member's own rate needs the same day's earlier votes: day-ahead it falls from +0.076 to +0.011 AUC for boosting. In the USA most of it holds day-ahead (+0.076 to +0.070), though flagging defections (average precision, accuracy) holds less. In both, the same-day gain is concentrated in runs of votes on the same text or bill (in the European Parliament not separated from persistence within a sitting in general), and in this GRU design the GRU's edge over boosting needs it. Compare the day-ahead results between entities. The targets also differ (a three-position group plurality against a two-party majority), so AUC levels aren't compared, only gaps.

## How the repo is organised

Each country or political organisation has its own top-level folder:

- `README.md`: the most important conclusions so far
- `docs/`: the most important decisions, with detail grouped by topic
- `pipeline/data_loading/`: data loading
- `pipeline/<problem>/`: one folder per problem, with a script for predictive work or notebooks for analysis, plus its own `docs/`

Project rules for Claude Code are in [CLAUDE.md](CLAUDE.md), with skills and agents in `.claude/`.

## Data

Raw data is not stored in this repo. Each entity documents its sources in `docs/datasets/`.
