"""Unit tests for the vote prediction features, using small hand-written tables."""

import numpy as np
import pandas as pd
import pytest

from features import build_rows, day_ahead_rate, party_positions, prior_rate


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


def test_cohesion_is_the_same_for_everyone_in_a_party():
    votes = pd.DataFrame({
        "congress": 110, "chamber": "House", "rollnumber": 1, "icpsr": range(1, 8),
        "party": ["D"] * 4 + ["R"] * 3, "vote": [1, 1, 1, -1, -1, -1, 1],
    })
    frame = party_positions(votes)
    assert frame.groupby("party")["own_cohesion"].nunique().eq(1).all()
    assert frame.loc[frame["party"] == "D", "own_cohesion"].iloc[0] == pytest.approx(0.5)


def test_build_rows_history_is_causal_and_drops_lopsided_votes_and_close_party_splits():
    members = pd.DataFrame({
        "congress": 110, "chamber": "House", "icpsr": range(1, 9), "bioguide_id": list("abcdefgh"),
        "bioname": list("ABCDEFGH"), "party": ["D"] * 5 + ["R"] * 3, "state_abbrev": ["TX", "NY", "CA", "IL", "MA", "OH", "ME", "TX"],
        "born": 1950, "nominate_dim1": [-0.3] * 5 + [0.4] * 3, "is_delegate": False,
    })
    rollcalls = pd.DataFrame({
        "congress": 110, "chamber": "House", "rollnumber": [1, 2, 3, 4, 5],
        "date": pd.to_datetime(["2007-01-02", "2007-01-02", "2007-01-03", "2007-01-04", "2007-01-05"]),
        "is_lopsided": [False, False, True, False, False],
    })
    topics = rollcalls[["congress", "chamber", "rollnumber"]].assign(vote_type=["passage", "procedural", "passage", "passage", "passage"])
    # Member 1 defects on roll calls 1 and 4; roll call 3 is near-unanimous and dropped. On roll call 5 the
    # Democrats split 3 to 2, so their votes are dropped whichever way member 1 voted; Republicans are kept.
    vote = {1: [1, -1, -1, -1, -1, -1, -1, -1], 2: [1, 1, 1, 1, 1, -1, -1, -1], 3: [1] * 8,
            4: [-1, 1, 1, 1, 1, -1, -1, -1], 5: [1, 1, -1, -1, 1, -1, -1, -1]}
    votes = pd.DataFrame([{"congress": 110, "chamber": "House", "rollnumber": r, "icpsr": i + 1, "vote": v}
                          for r, cast in vote.items() for i, v in enumerate(cast)])
    rows = build_rows(votes, members, rollcalls, topics)
    assert set(rows["rollnumber"]) == {1, 2, 4, 5}
    assert set(rows.loc[rows["rollnumber"] == 5, "party"]) == {"R"}
    member = rows[rows["icpsr"] == 1].set_index("rollnumber")
    assert member["defected"].tolist() == [True, False, True]
    assert np.isnan(member.loc[1, "prior_rate_all"]) and member.loc[2, "prior_rate_all"] == 1.0
    assert member.loc[4, "prior_rate_all"] == pytest.approx(0.5)
    assert member.loc[4, "last_defected"] == 0.0
    assert member.loc[4, "prior_rate_same_type"] == 1.0
    # Roll call 1: without member 1 both parties voted Nay. Roll call 2: Democrats Yea, Republicans Nay.
    assert not bool(member.loc[1, "party_opposed"]) and bool(member.loc[2, "party_opposed"])


def test_build_rows_min_minority_share_overrides_the_standard_flag():
    members = pd.DataFrame({
        "congress": 110, "chamber": "House", "icpsr": range(1, 7), "bioguide_id": list("abcdef"),
        "bioname": list("ABCDEF"), "party": ["D"] * 3 + ["R"] * 3, "state_abbrev": ["TX", "NY", "CA", "OH", "ME", "TX"],
        "born": 1950, "nominate_dim1": [-0.3] * 3 + [0.4] * 3, "is_delegate": False,
    })
    # Roll call 1 has a minority share of 0.03: kept by the standard flag and a 1% cutoff, dropped at 5%.
    rollcalls = pd.DataFrame({
        "congress": 110, "chamber": "House", "rollnumber": [1, 2], "date": pd.to_datetime(["2007-01-02", "2007-01-03"]),
        "is_lopsided": [False, False], "minority_share": [0.03, 0.5],
    })
    topics = rollcalls[["congress", "chamber", "rollnumber"]].assign(vote_type="passage")
    votes = pd.DataFrame([{"congress": 110, "chamber": "House", "rollnumber": r, "icpsr": i + 1, "vote": v}
                          for r, cast in {1: [1, 1, 1, -1, -1, -1], 2: [1, 1, 1, -1, -1, -1]}.items() for i, v in enumerate(cast)])
    assert set(build_rows(votes, members, rollcalls, topics)["rollnumber"]) == {1, 2}
    assert set(build_rows(votes, members, rollcalls, topics, min_minority_share=0.01)["rollnumber"]) == {1, 2}
    assert set(build_rows(votes, members, rollcalls, topics, min_minority_share=0.05)["rollnumber"]) == {2}


def test_day_ahead_rate_ignores_the_same_day():
    frame = pd.DataFrame({"key": ["a"] * 5 + ["b"] * 2, "value": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                          "date": pd.to_datetime(["2007-01-02", "2007-01-02", "2007-01-03", "2007-01-03", "2007-01-05",
                                                  "2007-01-02", "2007-01-03"])})
    rates = day_ahead_rate(frame, ["key"], "value")
    # Day one has no earlier day; day two sees day one only (0.5); day three sees both days (0.75).
    assert np.isnan(rates[0]) and np.isnan(rates[1]) and rates[2] == 0.5 and rates[3] == 0.5 and rates[4] == 0.75
    assert np.isnan(rates[5]) and rates[6] == 1.0
    last = day_ahead_rate(frame, ["key"], "value", window=1)
    assert last[2] == 0.0 and last[4] == 1.0
    changed = frame.assign(value=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])  # a day-two outcome changes
    assert np.allclose(day_ahead_rate(changed, ["key"], "value")[:4], rates[:4], equal_nan=True)


def test_build_rows_day_ahead_features_stop_at_the_previous_day():
    members = pd.DataFrame({
        "congress": 110, "chamber": "House", "icpsr": range(1, 7), "bioguide_id": list("abcdef"),
        "bioname": list("ABCDEF"), "party": ["D"] * 3 + ["R"] * 3, "state_abbrev": ["TX", "NY", "CA", "OH", "ME", "TX"],
        "born": 1950, "nominate_dim1": [-0.3] * 3 + [0.4] * 3, "is_delegate": False,
    })
    rollcalls = pd.DataFrame({
        "congress": 110, "chamber": "House", "rollnumber": [1, 2, 3],
        "date": pd.to_datetime(["2007-01-02", "2007-01-02", "2007-01-03"]), "is_lopsided": False,
    })
    topics = rollcalls[["congress", "chamber", "rollnumber"]].assign(vote_type="passage")
    # Member 1 defects on roll call 1. Five Democrats, so one defector leaves a margin of 3 and the close-split filter
    # keeps them.
    members = pd.concat([members, members.iloc[:2].assign(icpsr=[7, 8], bioguide_id=["g", "h"], bioname=["G", "H"])])
    members.loc[members["icpsr"].isin([7, 8]), "party"] = "D"
    vote = {1: [-1, 1, 1, -1, -1, -1, 1, 1], 2: [1, 1, 1, -1, -1, -1, 1, 1], 3: [1, 1, 1, -1, -1, -1, 1, 1]}
    votes = pd.DataFrame([{"congress": 110, "chamber": "House", "rollnumber": r, "icpsr": i + 1, "vote": v}
                          for r, cast in vote.items() for i, v in enumerate(cast)])
    member = build_rows(votes, members, rollcalls, topics).query("icpsr == 1").set_index("rollnumber")
    assert member["defected"].tolist() == [True, False, False]
    assert member.loc[2, "prior_rate_all"] == 1.0 and np.isnan(member.loc[2, "prior_rate_all_day"])
    assert member.loc[3, "prior_rate_all_day"] == 0.5 and member.loc[3, "last_defected_day"] == 0.0
    assert member["prior_votes_day"].tolist() == [0, 0, 2]


@pytest.mark.parametrize("keys, masked, window", [(["key"], False, None), (["key"], True, None), (["key"], True, 3),
                                                  (["key"], False, 1), (["key", "congress"], True, None)])
def test_day_ahead_rate_matches_prior_rate_at_each_days_first_row(keys, masked, window):
    # At a member's first row of a day, "earlier rows" and "rows on earlier days" are the same set, so the two must
    # agree there.
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({"key": rng.integers(0, 4, n), "congress": rng.integers(0, 2, n),
                          "date": pd.Timestamp("2007-01-01") + pd.to_timedelta(rng.integers(0, 30, n), unit="D"),
                          "value": rng.integers(0, 2, n).astype(float), "flag": rng.random(n) < 0.6})
    frame = frame.sort_values(["key", "congress", "date"] if "congress" in keys else ["key", "date"], kind="stable").reset_index(drop=True)
    mask = frame["flag"] if masked else None
    day = day_ahead_rate(frame, keys, "value", mask, window)
    standard = prior_rate(frame, keys, "value", mask, window)
    first = ~frame.duplicated(keys + ["date"])
    assert np.allclose(day[first], standard[first], equal_nan=True)
    # Changing every outcome on the last day must leave every feature unchanged, including that day's later rows.
    last_day = frame["date"] == frame["date"].max()
    changed = frame.assign(value=np.where(last_day, 1 - frame["value"], frame["value"]))
    assert np.allclose(day_ahead_rate(changed, keys, "value", mask, window), day, equal_nan=True)
