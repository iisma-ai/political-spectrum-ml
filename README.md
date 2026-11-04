# Politics

A comparative study of how votes can be explained and how the political spectrum changes over time, using machine learning on parliamentary voting data.

## Countries and organisations

| Entity | Status | Conclusions |
|---|---|---|
| USA | Data foundation, clustering and explanation done, prediction next | [usa/README.md](usa/README.md) |

## How the repo is organised

Each country or political organisation has its own top-level folder:

- `README.md`: the most important conclusions so far
- `docs/`: the most important decisions, with detail grouped by topic
- `pipeline/data_loading/`: data loading
- `pipeline/<problem>/`: one folder per problem, with a script for predictive work or notebooks for analysis, plus its own `docs/`

Project rules for Claude Code are in [CLAUDE.md](CLAUDE.md), with skills and agents in `.claude/`.

## Data

Raw data is not stored in this repo. Each entity documents its sources in `docs/datasets/`.
