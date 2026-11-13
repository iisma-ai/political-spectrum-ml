# European Parliament

The most important conclusions of the analysis so far. Decisions are in [docs/README.md](docs/README.md).

Last updated: 2026-11-12

Scope so far: plenary roll calls from July 2019 to October 2026 (the 9th term and the 10th term so far), from HowTheyVote.eu (Open Database License). The 10th term is unfinished.

## Conclusions

No results yet. The data foundation is in place: 25,686 open roll calls and 18.2 million member positions ([data loading](pipeline/data_loading/docs/README.md)).

## How to read these

- Only roll calls are recorded, and they are taken on request, so they are a selected sample of plenary votes. Show-of-hands votes are missing.
- The data matches the European Parliament's Open Data Portal: no roll call missing on any sitting day, and no disagreement on individual positions in a random sample of 200 roll calls. Both come from the Parliament's own records.
- Abstention is an active choice here and changes which roll calls count as contested, so results using the near-unanimous cutoff need a sensitivity check with abstention counted against.

## Conclusions by problem

| Problem | Status | Main result |
|---|---|---|
| [data_loading](pipeline/data_loading/docs/README.md) | Foundation done | Internally consistent, complete against the Open Data Portal, and matching it on a 200 roll-call sample |
