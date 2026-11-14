import numpy as np
import pandas as pd
import torch

from sequence_model import DefectionGRU, SequenceBatches, predict


def toy_rows(n_members=3, length=6):
    rows = []
    for member in range(n_members):
        for t in range(length):
            rows.append({"member_id": member, "vote_part": "amendment", "own_line": 0, "country_code": "DEU",
                         "others_same_share": 0.5, "with_most_groups": True, "group_seat_share": 0.2,
                         "log_party_seats": 1.0, "no_party": False, "new_group": 0.0, "age": 50.0, "term": 9,
                         "defected": (t % 2 == 1)})
    return pd.DataFrame(rows)


def test_previous_outcome_is_shifted_within_each_sequence():
    rows = toy_rows()
    data = SequenceBatches(rows, np.ones(len(rows), bool), ["DEU"])
    x, previous, y, weight, index = data.padded(data.batches[0])
    for row in range(previous.shape[0]):
        assert previous[row, 0, 0] == 0 and previous[row, 0, 1] == 1  # nothing before the first step
        assert torch.equal(previous[row, 1:, 0], y[row, :-1])  # step t sees the outcome at t - 1 only


def test_prediction_at_a_step_ignores_later_outcomes():
    rows = toy_rows(n_members=1, length=8)
    torch.manual_seed(0)
    model = DefectionGRU(SequenceBatches(rows, np.ones(len(rows), bool), ["DEU"]).inputs.shape[1])
    before = predict(model, SequenceBatches(rows, np.ones(len(rows), bool), ["DEU"]))
    changed = rows.copy()
    changed.loc[5:, "defected"] = ~changed.loc[5:, "defected"]
    after = predict(model, SequenceBatches(changed, np.ones(len(rows), bool), ["DEU"]))
    assert np.allclose(before[:6], after[:6])  # step 5's outcome first enters at step 6
    assert not np.allclose(before[6:], after[6:])


def test_day_ahead_prediction_ignores_the_same_day():
    rows = toy_rows(n_members=1, length=9)
    rows["date"] = pd.to_datetime(["2020-01-01"] * 3 + ["2020-01-02"] * 3 + ["2020-01-03"] * 3)
    torch.manual_seed(0)
    data = SequenceBatches(rows, np.ones(len(rows), bool), ["DEU"], day_ahead=True)
    model = DefectionGRU(data.inputs.shape[1], n_previous=data.previous.shape[1])
    before = predict(model, data)
    changed = rows.copy()
    changed.loc[3:5, "defected"] = ~changed.loc[3:5, "defected"]  # day 2's outcomes
    after = predict(model, SequenceBatches(changed, np.ones(len(rows), bool), ["DEU"], day_ahead=True))
    assert np.allclose(before[:6], after[:6])  # day 2 itself never sees them
    assert not np.allclose(before[6:], after[6:])  # day 3 does
