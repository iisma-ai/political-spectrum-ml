import numpy as np
import pandas as pd
import pytest

from clustering import (
    agreement_index, agreement_matrix, classical_mds, group_counts, group_line, line_agreement, modal_group, orient,
    winning_side,
)


def votes_frame(rows):
    return pd.DataFrame(rows, columns=["vote_id", "member_id", "group_code", "position", "country_code"])


def test_group_counts_ignore_did_not_vote():
    votes = votes_frame([
        (1, 1, "EPP", "FOR", "DE"), (1, 2, "EPP", "AGAINST", "DE"), (1, 3, "EPP", "DID_NOT_VOTE", "FR"),
        (1, 4, "SD", "ABSTENTION", "FR"),
    ])
    counts = group_counts(votes)
    assert counts.loc[(1, "EPP")].tolist() == [1, 1, 0]
    assert counts.loc[(1, "SD")].tolist() == [0, 0, 1]


def test_agreement_index_bounds_and_abstention():
    counts = pd.DataFrame({"for": [10, 4, 6, 0], "against": [0, 4, 0, 0], "abstention": [0, 4, 6, 0]})
    index = agreement_index(counts)
    assert index[0] == pytest.approx(1.0)
    assert index[1] == pytest.approx(0.0)  # an even three-way split
    assert index[2] == pytest.approx(0.25)  # (6 - 3) / 12
    assert np.isnan(index[3])
    assert agreement_index(counts, abstention=False)[2] == pytest.approx(1.0)


def test_group_line_ties_and_minimum():
    counts = pd.DataFrame({"for": [6, 5, 2, 1], "against": [1, 5, 0, 1], "abstention": [0, 0, 1, 7]})
    assert group_line(counts).tolist()[0] == "FOR"
    assert pd.isna(group_line(counts)[1])  # tie
    assert pd.isna(group_line(counts)[2])  # fewer than 5 active positions
    assert group_line(counts)[3] == "ABSTENTION"


def test_line_agreement_counts_only_shared_lines():
    lines = pd.Series(
        ["FOR", "FOR", "AGAINST", "FOR", np.nan, "ABSTENTION"],
        index=pd.MultiIndex.from_tuples([(1, "A"), (1, "B"), (2, "A"), (2, "B"), (3, "A"), (3, "B")]),
    )
    table = line_agreement(lines)
    assert table.loc["A", "B"] == pytest.approx(0.5)


def test_winning_side():
    rollcalls = pd.DataFrame({"n_for": [10, 3, 5], "n_against": [2, 8, 5]})
    assert winning_side(rollcalls).tolist()[:2] == ["FOR", "AGAINST"]
    assert pd.isna(winning_side(rollcalls)[2])


def test_modal_group_counts_switches():
    votes = votes_frame([
        (1, 1, "ID", "FOR", "IT"), (2, 1, "ID", "FOR", "IT"), (3, 1, "NI", "FOR", "IT"), (1, 2, "EPP", "FOR", "DE"),
    ])
    modal = modal_group(votes)
    assert modal.loc[1, "group"] == "ID" and modal.loc[1, "groups_held"] == 2
    assert modal.loc[2, "groups_held"] == 1


def test_mds_separates_two_blocs_and_orient_flips():
    matrix = pd.DataFrame(np.vstack([np.ones((4, 30)), -np.ones((4, 30))]), index=list("abcdefgh"))
    agreement = agreement_matrix(matrix)
    coordinates, shares = classical_mds(agreement)
    assert shares[0] > 0.99
    groups = pd.Series(["L"] * 4 + ["R"] * 4, index=list("abcdefgh"))
    oriented = orient(coordinates, groups, low="R")
    assert oriented.loc[groups == "R", "dim1"].mean() < oriented.loc[groups == "L", "dim1"].mean()
