"""Unit tests for the data cleaning rules, using small hand-written tables."""

import pandas as pd
import pytest

import voteview
from build_dataset import CAST_TO_VOTE, clean_members, clean_rollcalls, clean_votes
from validate import session_year
from voteview import pivot_votes


def raw_members() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "congress": [100, 100, 100, 100],
            "chamber": ["House", "House", "House", "President"],
            "icpsr": [1, 2, 3, 99],
            "bioname": ["A", "B", "C", "P"],
            "bioguide_id": ["a", "b", "c", "p"],
            "party_code": [100, 200, 100, 200],
            "state_abbrev": ["NY", "TX", "DC", "USA"],
            "district_code": [1, 2, 0, 0],
            "born": [1950, 1950, 1950, 1950],
            "nominate_dim1": [-0.3, 0.4, -0.4, None],
            "nominate_dim2": [0.0, 0.0, 0.0, None],
            "nokken_poole_dim1": [-0.3, 0.4, -0.4, None],
            "nokken_poole_dim2": [0.0, 0.0, 0.0, None],
        }
    )


def raw_votes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "congress": [100] * 7,
            "chamber": ["House"] * 7,
            "rollnumber": [1, 1, 1, 1, 2, 2, 3],
            "icpsr": [1, 2, 3, 99, 1, 2, 2],
            "cast_code": [3, 5, 1, 1, 9, 7, 0],
        }
    )


def test_members_drop_presidents_and_flag_delegates():
    members = clean_members(raw_members())
    assert 99 not in set(members["icpsr"])
    assert members.set_index("icpsr")["is_delegate"].to_dict() == {1: False, 2: False, 3: True}


def test_votes_drop_presidents_and_non_members():
    votes = clean_votes(raw_votes(), clean_members(raw_members()))
    assert 99 not in set(votes["icpsr"])
    assert 0 not in set(votes["cast_code"])


@pytest.mark.parametrize(
    "code, vote", [(1, 1), (2, 1), (3, 1), (4, -1), (5, -1), (6, -1), (7, 0), (8, 0), (9, 0)]
)
def test_every_cast_code_maps_correctly(code, vote):
    assert CAST_TO_VOTE[code] == vote


def rollcall_with(yea: int, nay: int) -> pd.Series:
    icpsr = list(range(1, yea + nay + 1))
    members = pd.DataFrame(
        {
            "congress": 100, "chamber": "House", "icpsr": icpsr, "is_delegate": False,
        }
    )
    votes = pd.DataFrame(
        {
            "congress": 100, "chamber": "House", "rollnumber": 1, "icpsr": icpsr,
            "cast_code": [1] * yea + [6] * nay, "vote": [1] * yea + [-1] * nay,
        }
    )
    raw = pd.DataFrame({"congress": [100], "chamber": ["House"], "rollnumber": [1], "date": ["1987-01-01"]})
    return clean_rollcalls(raw, votes, members).iloc[0]


def test_lopsided_boundary():
    assert rollcall_with(98, 2)["is_lopsided"]
    assert not rollcall_with(97, 3)["is_lopsided"]
    assert not rollcall_with(975, 25)["is_lopsided"]


def test_rollcalls_exclude_delegates_and_flag_empty_as_lopsided():
    members = clean_members(raw_members())
    votes = clean_votes(raw_votes(), members)
    raw = pd.DataFrame(
        {"congress": [100, 100], "chamber": ["House", "House"], "rollnumber": [1, 2], "date": ["1987-01-01"] * 2}
    )
    rollcalls = clean_rollcalls(raw, votes, members).set_index("rollnumber")
    assert rollcalls.loc[1, "yea"] == 1 and rollcalls.loc[1, "nay"] == 1
    assert not rollcalls.loc[1, "is_lopsided"]
    assert rollcalls.loc[2, "is_lopsided"]


def test_absent_is_missing_not_nay():
    votes = clean_votes(raw_votes(), clean_members(raw_members()))
    matrix = pivot_votes(votes)
    assert matrix.loc[1, 1] == 1
    assert pd.isna(matrix.loc[1].get(2))


def test_vote_matrix_applies_filters(monkeypatch):
    members = clean_members(raw_members())
    votes = clean_votes(raw_votes(), members)
    rollcalls = pd.DataFrame({"congress": [100, 100], "chamber": ["House", "House"], "rollnumber": [1, 2], "is_lopsided": [False, True]})
    monkeypatch.setattr(voteview, "load_members", lambda: members)
    monkeypatch.setattr(voteview, "load_rollcalls", lambda: rollcalls)
    monkeypatch.setattr(voteview, "load_votes", lambda congress, chamber: votes)
    matrix = voteview.vote_matrix(100, "House", min_votes=1)
    assert list(matrix.columns) == [1]
    assert set(matrix.index) == {1, 2}


def test_session_year():
    assert session_year(118, 1) == 2023
    assert session_year(112, 2) == 2012
