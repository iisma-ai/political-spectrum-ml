"""Tests that the GRU sees only earlier outcomes."""

import numpy as np
import pandas as pd
import torch

from sequence_model import DefectionGRU, SequenceBatches, predict


def rows(defected):
    n = len(defected)
    return pd.DataFrame({
        "person": ["a"] * (n - 2) + ["b"] * 2, "chamber": "House", "party": "D", "congress": 110,
        "vote_type": "passage", "region": "South", "own_yea": True, "party_opposed": True, "other_tied": False,
        "majority": 1.0, "freshman": False, "tenure": 2, "defected": defected,
    })


def test_previous_outcome_is_shifted_within_each_sequence():
    data = SequenceBatches(rows([True, False, True, False, True, True]), np.ones(6, dtype=bool))
    for batch in data.batches:
        _, previous, y, _, index = data.padded(batch)
        for row in range(len(batch)):
            valid = index[row] >= 0
            outcomes = y[row][torch.from_numpy(valid)].numpy()
            prev = previous[row, :, 0][torch.from_numpy(valid)].numpy()
            assert prev[0] == 0 and previous[row, 0, 1] == 1
            assert np.array_equal(prev[1:], outcomes[:-1])


def test_predictions_do_not_depend_on_later_outcomes():
    torch.manual_seed(0)
    first = rows([True, False, True, False, True, True])
    second = rows([True, False, False, True, True, True])  # differs from the third vote of member a on
    model = DefectionGRU(SequenceBatches(first, np.ones(6, dtype=bool)).inputs.shape[1], hidden=8)
    p_first, _ = predict(model, SequenceBatches(first, np.ones(6, dtype=bool)))
    p_second, _ = predict(model, SequenceBatches(second, np.ones(6, dtype=bool)))
    assert np.allclose(p_first[:3], p_second[:3])
    assert not np.allclose(p_first[3], p_second[3])
    assert np.allclose(p_first[4:], p_second[4:])


def test_day_ahead_prediction_ignores_the_same_day():
    torch.manual_seed(0)
    dates = pd.to_datetime(["2007-01-02", "2007-01-02", "2007-01-03", "2007-01-03", "2007-01-02", "2007-01-03"])
    first = rows([True, False, True, False, True, True]).assign(date=dates)
    # Member a's second vote on day one changes: no day-one prediction may move, but day two's may.
    second = rows([True, True, True, False, True, True]).assign(date=dates)
    # Member a's first vote on day two changes: no prediction may move, since day two's outcomes never enter.
    third = rows([True, False, False, False, True, True]).assign(date=dates)
    batches = lambda frame: SequenceBatches(frame, np.ones(6, dtype=bool), day_ahead=True)  # noqa: E731
    model = DefectionGRU(batches(first).inputs.shape[1], hidden=8, n_previous=batches(first).previous.shape[1])
    p_first, _ = predict(model, batches(first))
    p_second, _ = predict(model, batches(second))
    p_third, _ = predict(model, batches(third))
    assert np.allclose(p_first[:2], p_second[:2]) and not np.allclose(p_first[2:4], p_second[2:4])
    assert np.allclose(p_first, p_third)
