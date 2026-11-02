# Vote prediction: decisions and documentation

## Question

How well can a member's next vote be predicted, especially when they break from their party, and can a sequence model track their position over time?

Type: predictive, using `run_model.py`, with notebooks for evaluation.

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-02 | Split by time | Random splits leak the future |
| 2026-11-02 | Compare with the party-majority baseline | Party-line voting makes raw accuracy look high |
| 2026-11-02 | Treat defection prediction as the main target | It is the hard and informative case |
| 2026-11-02 | Add RNNs after logistic and gradient boosting baselines | A complex model has to beat simple ones to be justified |

## Documentation

- [Analysis options and risks](../../../docs/methodology/analysis_options.md)
