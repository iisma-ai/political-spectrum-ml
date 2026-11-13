import numpy as np
import pandas as pd
import pytest

from features import add_history, build_rows, earlier_rollcall_rate, group_positions, prior_rate


def votes_frame(rows):
    return pd.DataFrame(rows, columns=["vote_id", "member_id", "position", "group_code", "country_code", "national_party_id"])


def group(vote_id, positions, code="EPP", party=1.0, start=0):
    return [(vote_id, start + i, p, code, "DEU", party) for i, p in enumerate(positions)]


def test_own_line_leaves_the_member_out_and_margin_includes_them():
    out = group_positions(votes_frame(group(1, ["FOR"] * 6 + ["AGAINST"])), set()).set_index("member_id")
    assert out.loc[6, "own_line"] == 0  # the Against member sees six For
    assert out.loc[0, "own_line"] == 0
    assert out["own_full_margin"].eq(5).all()  # 6 For against 1 Against, the same for everyone


def test_others_same_share_counts_other_groups_only():
    rows = (group(1, ["FOR"] * 6, "EPP") + group(1, ["FOR"] * 6, "SD", start=10)
            + group(1, ["AGAINST"] * 6, "ECR", start=20) + group(1, ["AGAINST"] * 6, "NI", start=30))
    out = group_positions(votes_frame(rows), set()).set_index("member_id")
    assert out.loc[0, "others_same_share"] == pytest.approx(0.5)  # S&D same, ECR different; NI not counted
    assert out.loc[20, "others_same_share"] == pytest.approx(0.0)


def test_colleagues_against_the_line():
    rows = group(1, ["FOR"] * 6, party=1.0) + group(1, ["AGAINST"] * 3, party=2.0, start=10)
    out = group_positions(votes_frame(rows), set()).set_index("member_id")
    assert out.loc[10, "colleagues"] == "united against the line"
    assert out.loc[0, "colleagues"] == "united with the line"


def test_seats_include_did_not_vote():
    out = group_positions(votes_frame(group(1, ["FOR"] * 6 + ["DID_NOT_VOTE"] * 2)), set())
    assert len(out) == 6 and out["group_seats"].eq(8).all() and out["party_seats"].eq(8).all()


def test_build_rows_drops_close_splits_and_near_unanimous():
    votes = votes_frame(group(1, ["FOR"] * 4 + ["AGAINST"] * 3) + group(2, ["FOR"] * 7) + group(3, ["FOR"] * 6 + ["AGAINST"]))
    rollcalls = pd.DataFrame({
        "vote_id": [1, 2, 3], "is_secret": False, "minority_share": [0.4, 0.0, 0.2],
        "timestamp": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]), "term": 9, "vote_part": "amendment"})
    rollcalls["date"] = rollcalls["timestamp"].dt.normalize()
    members = pd.DataFrame({"member_id": range(7), "date_of_birth": pd.to_datetime("1970-01-01")})
    rows = build_rows(votes, rollcalls, members, pd.Series(dtype=str), set())
    assert set(rows["vote_id"]) == {3}  # 1: margin 1; 2: near-unanimous
    assert rows["defected"].sum() == 1


def test_member_history_uses_earlier_rows_only():
    frame = pd.DataFrame({"member_id": [1] * 4, "timestamp": pd.date_range("2020", periods=4), "vote_id": [1, 2, 3, 4],
                          "defected": [True, False, True, False], "with_most_groups": True, "vote_part": "amendment",
                          "group_code": "EPP", "party_group": "EPP|1"})
    out = add_history(frame)
    assert np.isnan(out.loc[0, "prior_rate_all"])
    assert out["prior_rate_all"].tolist()[1:] == pytest.approx([1.0, 0.5, 2 / 3])
    assert out["last_defected"].tolist()[1:] == [1.0, 0.0, 1.0]


def test_group_history_excludes_colleagues_on_the_same_roll_call():
    frame = pd.DataFrame({"member_id": [1, 2, 1, 2], "vote_id": [1, 1, 2, 2],
                          "timestamp": pd.to_datetime(["2020-01-01"] * 2 + ["2020-01-02"] * 2),
                          "group_code": "EPP", "value": [1.0, 0.0, 1.0, 1.0]})
    rate = earlier_rollcall_rate(frame, ["group_code"], "value")
    assert rate.iloc[:2].isna().all()  # nothing earlier, and the colleague's vote on roll call 1 isn't used
    assert rate.iloc[2:].tolist() == [0.5, 0.5]


def test_prior_rate_window():
    frame = pd.DataFrame({"k": [1] * 5, "v": [1.0, 1.0, 0.0, 0.0, 0.0]})
    assert prior_rate(frame, ["k"], "v", window=2).tolist()[1:] == pytest.approx([1.0, 1.0, 0.5, 0.0])
