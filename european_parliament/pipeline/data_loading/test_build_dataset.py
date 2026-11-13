import pandas as pd

from build_dataset import clean_rollcalls, clean_votes, term_of, vote_part
from howtheyvote import build_matrix
from validate import compare


def member_votes(rows):
    raw = pd.DataFrame(rows, columns=["vote_id", "member_id", "position", "group_code"])
    raw["country_code"] = "FRA"
    raw["national_party_id"] = 1
    return clean_votes(raw)


def raw_rollcalls(rows):
    raw = pd.DataFrame(rows, columns=["id", "timestamp", "count_for", "count_against", "count_abstention"])
    for column in ["display_title", "reference", "description", "amendment_number", "amendment_subject",
                   "procedure_reference", "procedure_title", "procedure_type", "procedure_stage", "majority_type",
                   "result", "texts_adopted_reference"]:
        raw[column] = None
    raw["is_main"] = False
    raw["count_did_not_vote"] = 0
    return raw


def test_positions_map_to_votes_and_old_left_group_label_is_merged():
    votes = member_votes([
        (1, 10, "FOR", "EPP"), (1, 11, "AGAINST", "GUE_NGL_1995_0"),
        (1, 12, "ABSTENTION", "SD"), (1, 13, "DID_NOT_VOTE", "SD"),
    ])
    assert votes["vote"].tolist() == [1, -1, 0, 0]
    assert votes["position"].astype(str).tolist() == ["FOR", "AGAINST", "ABSTENTION", "DID_NOT_VOTE"]
    assert "GUE_NGL_1995_0" not in set(votes["group_code"].astype(str))


def test_term_starts_on_the_first_day_of_the_term():
    dates = pd.Series(pd.to_datetime(["2019-07-15 10:00", "2024-07-15 23:59", "2024-07-16 09:00"]))
    assert term_of(dates).tolist() == [9, 9, 10]


def test_rollcalls_minority_share_ignores_abstentions_and_flags_secret_ballots():
    votes = member_votes(
        [(1, m, "FOR", "EPP") for m in range(97)] + [(1, 200, "AGAINST", "SD")]
        + [(1, 300 + m, "ABSTENTION", "SD") for m in range(50)]
        + [(2, m, "FOR", "EPP") for m in range(60)] + [(2, 100 + m, "AGAINST", "SD") for m in range(40)]
        + [(3, m, "DID_NOT_VOTE", "EPP") for m in range(5)]
    )
    rollcalls = clean_rollcalls(raw_rollcalls([
        (2, "2024-07-17 12:00:00", 60, 40, 0),
        (1, "2019-07-15 17:09:37.025000", 97, 1, 50),
        (3, "2019-09-17 12:05:31", 0, 0, 0),
    ]), votes).set_index("vote_id")
    # 1 of 98 For or Against votes is under 2.5%, whatever the abstentions.
    assert rollcalls.loc[1, "minority_share"] == 1 / 98
    assert rollcalls.loc[1, "is_lopsided"] and not rollcalls.loc[2, "is_lopsided"]
    assert rollcalls.loc[3, "is_secret"] and not rollcalls.loc[1, "is_secret"]
    assert rollcalls.loc[1, "n_abstention"] == 50
    # Roll numbers restart in each term and follow time order.
    assert rollcalls.loc[[1, 3, 2], "rollnumber"].tolist() == [1, 2, 1]
    assert rollcalls.loc[[1, 2], "term"].tolist() == [9, 10]


def test_vote_part_puts_main_votes_and_procedural_requests_first():
    rollcalls = pd.DataFrame({
        "description": ["Am 7", "§ 12/2", "Renvoi (article 59, paragraphe 4 du règlement)", "Après le § 2 - Am 10",
                        "Proposition de résolution (ensemble du texte)", None, "Vote unique"],
        "display_title": ["X", "X", "X", "X", "X", "Mardi - demande du groupe GUE/NGL", "X"],
        "is_main": [False, False, False, False, True, False, False],
    })
    assert vote_part(rollcalls).tolist() == [
        "amendment", "text part", "procedural", "amendment", "final", "procedural", "other"]


def test_matrix_drops_secret_and_lopsided_roll_calls_and_treats_abstention_as_missing():
    votes = member_votes(
        [(1, m, "FOR", "EPP") for m in range(10)] + [(1, 10, "ABSTENTION", "SD"), (1, 11, "AGAINST", "SD")]
        + [(2, m, "FOR", "EPP") for m in range(12)]
    )
    rollcalls = pd.DataFrame({"vote_id": [1, 2, 3], "is_secret": [False, False, True],
                              "minority_share": [1 / 11, 0.0, None]})
    matrix = build_matrix(votes, rollcalls, min_votes=1)
    assert matrix.columns.tolist() == [1]
    assert matrix.loc[11, 1] == -1 and 10 not in matrix.index


def test_official_comparison_counts_disagreements_missing_and_extra_members():
    ours = pd.DataFrame({"member_id": [1, 2, 3, 4], "position": ["FOR", "AGAINST", "ABSTENTION", "DID_NOT_VOTE"]})
    assert compare(ours, {1: "FOR", 2: "AGAINST", 3: "ABSTENTION"}) == {
        "official": 3, "disagree": 0, "missing": 0, "extra": 0}
    assert compare(ours, {1: "AGAINST", 2: "AGAINST", 5: "FOR"}) == {
        "official": 3, "disagree": 1, "missing": 1, "extra": 1}


def test_vote_part_reads_english_agenda_titles_and_title_amendments_only_without_description():
    rollcalls = pd.DataFrame({
        "description": [None, None, "Décision d'engager des négociations interinstitutionnelles", None],
        "display_title": ["Tuesday’s agenda – Request by the PfE Group", "14 02 01 10 - Am 865/1", "X",
                          "Ongoing hearings under Article 7(1) TEU"],
        "is_main": [False, False, True, False],
    })
    assert vote_part(rollcalls).tolist() == ["procedural", "amendment", "procedural", "other"]
