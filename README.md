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
- **Vote prediction** in both entities is one roll call ahead, so history includes earlier roll calls on the same day. In the European Parliament, where roll calls run seconds apart on the same texts, most of the models' gain over a member's own rate comes from that; day-ahead models add little. The USA hasn't been checked day-ahead, so its GRU and boosting gains may rest on the same thing. The targets also differ (a three-position group plurality against a two-party majority), so AUC levels aren't compared.

## How the repo is organised

Each country or political organisation has its own top-level folder:

- `README.md`: the most important conclusions so far
- `docs/`: the most important decisions, with detail grouped by topic
- `pipeline/data_loading/`: data loading
- `pipeline/<problem>/`: one folder per problem, with a script for predictive work or notebooks for analysis, plus its own `docs/`

Project rules for Claude Code are in [CLAUDE.md](CLAUDE.md), with skills and agents in `.claude/`.

## Data

Raw data is not stored in this repo. Each entity documents its sources in `docs/datasets/`.
