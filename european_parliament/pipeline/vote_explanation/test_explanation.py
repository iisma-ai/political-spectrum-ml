import numpy as np
import pandas as pd
import pytest

from explanation import defection_rows, follows, independent_party_ids, lagged_rate, shift_share


def votes_frame(rows):
    return pd.DataFrame(rows, columns=["vote_id", "member_id", "position", "group_code", "national_party_id"])


def group_of(vote_id, positions, group="EPP", party=None):
    return [(vote_id, 100 * vote_id + i, p, group, party) for i, p in enumerate(positions)]


def test_group_line_includes_the_member():
    rows = defection_rows(votes_frame(group_of(1, ["FOR"] * 6 + ["AGAINST"])))
    assert rows["defected_group"].tolist() == [False] * 6 + [True]


def test_line_needs_five_votes_and_no_tie():
    even = defection_rows(votes_frame(group_of(1, ["FOR"] * 3 + ["AGAINST"] * 3)))
    assert even["defected_group"].isna().all()  # an even split has no line, so nobody defects
    small = defection_rows(votes_frame(group_of(2, ["FOR"] * 3 + ["AGAINST"])))
    assert small["defected_group"].isna().all()  # only four active members
    three_way = defection_rows(votes_frame(group_of(3, ["FOR"] * 3 + ["AGAINST"] * 3 + ["ABSTENTION"])))
    assert three_way["defected_group"].isna().all()


def test_national_party_line_needs_united_colleagues():
    split = defection_rows(votes_frame([(1, i, p, "EPP", 7.0) for i, p in enumerate(["FOR", "FOR", "AGAINST", "AGAINST"])]
                                       + [(1, 10 + i, "FOR", "EPP", 8.0) for i in range(5)]))
    assert split.loc[split["national_party_id"] == 7.0, "defected_party"].isna().all()
    united = defection_rows(votes_frame([(2, i, p, "EPP", 7.0) for i, p in enumerate(["FOR", "FOR", "AGAINST"])]))
    assert united["defected_party"].tolist()[2] == True  # noqa: E712
    assert united["defected_party"].isna().tolist()[:2] == [True, True]  # each For member's others disagree


def test_did_not_vote_and_non_attached():
    rows = defection_rows(votes_frame(group_of(1, ["FOR"] * 6 + ["DID_NOT_VOTE"]) + group_of(2, ["FOR"] * 6, group="NI")))
    assert len(rows) == 12
    assert rows.loc[rows["group_code"] == "NI", "defected_group"].isna().all()


def test_abstention_counts_as_defection_but_not_in_for_against():
    rows = defection_rows(votes_frame(group_of(1, ["FOR"] * 6 + ["ABSTENTION"])))
    last = rows.iloc[-1]
    assert last["defected_group"] == True  # noqa: E712
    assert pd.isna(last["defected_group_for_against"])


def test_conflict_and_follows():
    epp = [(1, i, "FOR", "EPP", 10.0) for i in range(6)]
    national = [(1, 10 + i, "AGAINST", "EPP", 20.0) for i in range(3)]
    rows = defection_rows(votes_frame(epp + national))
    member = rows[rows["member_id"] == 10].iloc[0]
    assert member["conflict"] == True and member["defected_group"] == True  # noqa: E712
    assert follows(rows[rows["member_id"] == 10]).iloc[0] == "national party"
    assert rows.loc[rows["national_party_id"] == 10.0, "conflict"].eq(False).all()


def test_missing_national_party_has_no_line():
    rows = defection_rows(votes_frame(group_of(1, ["FOR"] * 7, party=np.nan)))
    assert rows["defected_party"].isna().all()


def test_lagged_rate_uses_earlier_periods_only():
    rows = pd.DataFrame({"member_id": [1, 1, 1, 1], "half": [1, 1, 2, 3], "defected_group": [1.0, 0.0, 1.0, 0.0]})
    lagged = lagged_rate(rows, "half").set_index("half")
    assert np.isnan(lagged.loc[1, "lagged_rate"])
    assert lagged.loc[2, "lagged_rate"] == pytest.approx(0.5)
    assert lagged.loc[3, "lagged_rate"] == pytest.approx(1.0)


def test_shift_share_adds_up():
    before = pd.DataFrame({"rate": [0.1, 0.3], "weight": [0.5, 0.5]}, index=["a", "b"])
    after = pd.DataFrame({"rate": [0.2, 0.3], "weight": [0.2, 0.8]}, index=["a", "b"])
    parts = shift_share(before, after)
    assert parts["within_topics"] + parts["topic_mix"] == pytest.approx(parts["total"])


def test_independents_have_no_party_line():
    votes = votes_frame([(1, i, "FOR", "EPP", 9.0) for i in range(6)] + [(1, 10, "AGAINST", "EPP", 9.0)])
    assert defection_rows(votes)["defected_party"].iloc[-1] == 1  # the Against member faces six united colleagues
    assert defection_rows(votes, no_party_ids={9.0})["defected_party"].isna().all()
    parties = pd.DataFrame({"national_party_id": [1, 2, 3], "label": ["Independent", "Volt", "Indépendant"]})
    assert independent_party_ids(parties) == {1, 3}
