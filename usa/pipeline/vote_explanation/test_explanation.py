"""Unit tests for the vote explanation helpers, using small hand-written tables."""

import pandas as pd
import pytest

from explanation import chamber_control, defection_rows, lagged_position, member_covariates, shift_share


def members() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "congress": [100, 100, 100, 101, 101, 101],
            "chamber": ["House"] * 6,
            "icpsr": [1, 2, 3, 1, 2, 4],
            "bioguide_id": ["a", "b", "c", "a", "b", None],
            "party": ["D", "D", "R", "D", "R", "R"],
            "state_abbrev": ["TX", "NY", "OH", "TX", "NY", "KY"],
            "born": [1940, 1950, 1960, 1940, 1950, 1970],
            "nokken_poole_dim1": [-0.1, -0.5, 0.4, -0.2, 0.3, 0.5],
            "nominate_dim1": [-0.1, -0.5, 0.4, -0.2, 0.3, 0.5],
        }
    )


def test_member_covariates():
    frame = member_covariates(members()).set_index(["congress", "icpsr"])
    assert frame.loc[(100, 1), "region"] == "South" and frame.loc[(101, 4), "region"] == "Border"
    assert frame.loc[(100, 1), "tenure"] == 0 and frame.loc[(101, 1), "tenure"] == 1
    assert frame.loc[(101, 4), "freshman"] and not frame.loc[(101, 2), "freshman"]
    assert frame.loc[(100, 1), "age"] == 1987 - 1940
    assert frame.loc[(100, 1), "majority"] and not frame.loc[(100, 3), "majority"]
    assert frame.loc[(100, 3), "party_seat_share"] == pytest.approx(1 / 3)


def test_lagged_position_uses_previous_congress_and_same_party():
    covariates = member_covariates(members())
    lagged = lagged_position(covariates)
    frame = covariates.assign(lagged=lagged).set_index(["congress", "icpsr"])
    # Democrats in 100 had median -0.3; member 1 at -0.1 was 0.2 toward Republicans.
    assert frame.loc[(101, 1), "lagged"] == pytest.approx(0.2)
    # Member 2 switched party, so there is no same-party previous Congress.
    assert pd.isna(frame.loc[(101, 2), "lagged"])
    assert pd.isna(frame.loc[(100, 1), "lagged"])


def test_defection_rows_flag_votes_with_the_other_majority():
    votes = pd.DataFrame(
        {
            "congress": 100, "chamber": "House", "rollnumber": [1] * 6 + [2] * 6,
            "icpsr": [1, 2, 3, 4, 5, 6] * 2,
            "party": ["D", "D", "D", "R", "R", "R"] * 2,
            "vote": [1, 1, -1, -1, -1, -1] + [1, 1, 1, 1, 1, 0],
        }
    )
    rows = defection_rows(votes)
    assert set(rows["rollnumber"]) == {1}
    assert rows.set_index("icpsr")["defected"].to_dict() == {1: False, 2: False, 3: True, 4: False, 5: False, 6: False}


def test_shift_share_adds_up():
    before = pd.DataFrame({"rate": [0.2, 0.8], "weight": [0.7, 0.3]}, index=["a", "b"])
    after = pd.DataFrame({"rate": [0.4, 0.9], "weight": [0.4, 0.6]}, index=["a", "b"])
    parts = shift_share(before, after)
    assert parts["total"] == pytest.approx((0.4 * 0.4 + 0.6 * 0.9) - (0.7 * 0.2 + 0.3 * 0.8))
    assert parts["within_topics"] + parts["topic_mix"] == pytest.approx(parts["total"])


def test_shift_share_counts_a_new_topic_as_mix():
    before = pd.DataFrame({"rate": [0.5], "weight": [1.0]}, index=["a"])
    after = pd.DataFrame({"rate": [0.5, 1.0], "weight": [0.5, 0.5]}, index=["a", "b"])
    parts = shift_share(before, after)
    assert parts["total"] == pytest.approx(0.25)
    assert parts["within_topics"] == pytest.approx(0)
    assert parts["topic_mix"] == pytest.approx(0.25)


def senate(parties, dims):
    return pd.DataFrame({
        "congress": 117, "chamber": "Senate", "icpsr": range(len(parties)), "bioguide_id": [f"p{i}" for i in range(len(parties))],
        "party": parties, "state_abbrev": "VT", "born": 1950, "nominate_dim1": dims, "nokken_poole_dim1": dims,
    })


def test_independents_count_with_the_nearer_party_and_ties_are_marked():
    tied = senate(["D"] * 48 + ["Other"] * 2 + ["R"] * 50, [-0.4] * 48 + [-0.5, -0.2] + [0.5] * 50)
    control = chamber_control(tied).iloc[0]
    assert (control["seats_d"], control["seats_r"], control["control"]) == (50, 50, "tied")
    frame = member_covariates(tied)
    assert frame["majority"].isna().all()
    assert frame.loc[frame["party"] == "D", "party_seat_share"].iloc[0] == pytest.approx(0.5)


def test_independents_can_give_control():
    chamber = senate(["D"] * 49 + ["Other"] * 2 + ["R"] * 49, [-0.4] * 49 + [-0.5, -0.2] + [0.5] * 49)
    frame = member_covariates(chamber)
    assert chamber_control(chamber).iloc[0]["control"] == "D"
    assert frame.loc[frame["party"] == "D", "majority"].all()
    assert not frame.loc[frame["party"] == "R", "majority"].any()
    assert frame.loc[frame["party"] == "Other", "majority"].isna().all()
