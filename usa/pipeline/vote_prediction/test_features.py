"""Unit tests for the vote prediction features, using small hand-written tables."""

import numpy as np
import pandas as pd
import pytest

from features import build_rows, party_positions, prior_rate


def test_own_party_position_leaves_the_member_out():
    votes = pd.DataFrame({
        "congress": 110, "chamber": "House", "rollnumber": 1, "icpsr": [1, 2, 3, 4, 5],
        "party": ["D", "D", "D", "R", "R"], "vote": [1, -1, -1, 1, 1],
    })
    frame = party_positions(votes).set_index("icpsr")
    # Without member 1, the other Democrats voted Nay; without member 2, Democrats are tied.
    assert frame.loc[1, "own_position"] == -1
    assert frame.loc[2, "own_position"] == 0
    assert frame.loc[1, "other_position"] == 1 and frame.loc[4, "other_position"] == -1
    assert frame.loc[4, "own_position"] == 1


def test_prior_rate_uses_only_earlier_rows():
    frame = pd.DataFrame({"key": ["a"] * 5 + ["b"] * 2, "value": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                          "flag": [True, False, True, True, True, True, True]})
    rates = prior_rate(frame, ["key"], "value")
    assert np.isnan(rates[0]) and rates[1] == 1.0 and rates[2] == pytest.approx(0.5) and rates[4] == pytest.approx(0.75)
    assert np.isnan(rates[5]) and rates[6] == 1.0
    masked = prior_rate(frame, ["key"], "value", frame["flag"])
    # Row 1 isn't flagged, so rows 2 and later skip it; row 1 itself still gets the earlier flagged rate.
    assert masked[1] == 1.0 and masked[2] == 1.0 and masked[3] == 1.0 and masked[4] == 1.0
    windowed = prior_rate(frame, ["key"], "value", window=2)
    assert windowed[3] == pytest.approx(0.5) and windowed[4] == 1.0


def test_build_rows_history_is_causal_and_drops_lopsided_votes():
    members = pd.DataFrame({
        "congress": 110, "chamber": "House", "icpsr": [1, 2, 3, 4, 5, 6], "bioguide_id": list("abcdef"),
        "bioname": list("ABCDEF"), "party": ["D"] * 3 + ["R"] * 3, "state_abbrev": ["TX", "NY", "CA", "OH", "ME", "TX"],
        "born": 1950, "nominate_dim1": [-0.3] * 3 + [0.4] * 3, "is_delegate": False,
    })
    rollcalls = pd.DataFrame({
        "congress": 110, "chamber": "House", "rollnumber": [1, 2, 3, 4],
        "date": pd.to_datetime(["2007-01-02", "2007-01-02", "2007-01-03", "2007-01-04"]), "is_lopsided": [False, False, True, False],
    })
    topics = rollcalls[["congress", "chamber", "rollnumber"]].assign(vote_type=["passage", "procedural", "passage", "passage"])
    # Member 1 defects on roll calls 1 and 4; roll call 3 is near-unanimous and dropped.
    vote = {1: [1, -1, -1, -1, -1, -1], 2: [1, 1, 1, -1, -1, -1], 3: [1] * 6, 4: [-1, 1, 1, -1, -1, -1]}
    votes = pd.DataFrame([{"congress": 110, "chamber": "House", "rollnumber": r, "icpsr": i + 1, "vote": v}
                          for r, cast in vote.items() for i, v in enumerate(cast)])
    rows = build_rows(votes, members, rollcalls, topics)
    assert set(rows["rollnumber"]) == {1, 2, 4}
    member = rows[rows["icpsr"] == 1].set_index("rollnumber")
    assert member["defected"].tolist() == [True, False, True]
    assert np.isnan(member.loc[1, "prior_rate_all"]) and member.loc[2, "prior_rate_all"] == 1.0
    assert member.loc[4, "prior_rate_all"] == pytest.approx(0.5)
    assert member.loc[4, "last_defected"] == 0.0
    assert member.loc[4, "prior_rate_same_type"] == 1.0
    # Roll call 1: without member 1 both parties voted Nay. Roll call 2: Democrats Yea, Republicans Nay.
    assert not bool(member.loc[1, "party_opposed"]) and bool(member.loc[2, "party_opposed"])
