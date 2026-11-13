"""Clean the raw HowTheyVote.eu export into the parquet tables every problem uses.

Data source: HowTheyVote.eu (Open Database License), https://howtheyvote.eu.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 1 for For, -1 for Against, 0 for Abstention and Did not vote. `position` keeps Abstention and Did not vote apart:
# filtering on `vote != 0` drops abstentions, which are active choices, so use `position` for anything but
# For/Against matrices.
POSITION_TO_VOTE = {"FOR": 1, "AGAINST": -1, "ABSTENTION": 0, "DID_NOT_VOTE": 0}
# Same rule as the USA: drop roll calls whose smaller side, of For and Against, is under 2.5%.
LOPSIDED_SHARE = 0.025
# First day of each parliamentary term, as in group_memberships.csv.
TERM_STARTS = {9: "2019-07-02", 10: "2024-07-16"}
# The Left group's earlier name, used in the export only for three UK members who left at Brexit.
GROUP_ALIASES = {"GUE_NGL_1995_0": "GUE_NGL"}

# What a roll call votes on, from the French `description` (and the title, for agenda votes), checked in order.
# The export's amendment_number is only filled from 2024, so it can't be compared across terms.
VOTE_PARTS = [
    ("procedural", r"renvoi|demande de|demande du groupe|engager des négociations|ordre du jour|agenda|request by"),
    ("amendment", r"\bam\b|amendement"),
    ("text part", r"§|considérant|visa|annexe|article|paragraphe|point"),
]

VOTE_COLUMNS =["vote_id", "member_id", "position", "country_code", "group_code", "national_party_id"]


def read(name: str, **options) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / f"{name}.csv.gz", low_memory=False, **options)


def term_of(dates: pd.Series) -> pd.Series:
    starts = sorted((pd.Timestamp(day), term) for term, day in TERM_STARTS.items())
    term = pd.Series(pd.NA, index=dates.index, dtype="Int8")
    for start, number in starts:
        term[dates >= start] = number
    return term


def vote_part(rollcalls: pd.DataFrame) -> pd.Series:
    """procedural first, then final for the main vote on a text, then the first matching part, else other.

    Procedural comes before final because the export marks votes on opening negotiations as main votes in the
    10th term only.
    """
    description = rollcalls["description"].fillna("").str.lower()
    # Agenda votes and some budget amendments have no description and are named in the title. A title applies to
    # every vote under it and can name an article, so it is read only when the description is empty, and only for
    # procedural votes and amendments.
    title = rollcalls["display_title"].fillna("").str.lower().where(description.eq(""), "")
    part = pd.Series("other", index=rollcalls.index)
    for label, pattern in reversed(VOTE_PARTS):
        matches = description.str.contains(pattern, regex=True)
        if label in ("procedural", "amendment"):
            matches |= title.str.contains(pattern, regex=True)
        part[matches] = label
    main = rollcalls["is_main"].fillna(False).astype(bool)
    part[main & part.ne("procedural")] = "final"
    return part


def clean_groups(raw: pd.DataFrame) -> pd.DataFrame:
    groups = raw[~raw["code"].isin(GROUP_ALIASES)].copy()
    return groups[["code", "short_label", "label", "official_label"]].reset_index(drop=True)


def clean_group_memberships(raw: pd.DataFrame) -> pd.DataFrame:
    spells = raw.copy()
    spells["group_code"] = spells["group_code"].replace(GROUP_ALIASES)
    spells["start_date"] = pd.to_datetime(spells["start_date"])
    spells["end_date"] = pd.to_datetime(spells["end_date"])  # empty while still a member
    return spells.sort_values(["member_id", "start_date"]).reset_index(drop=True)


def clean_members(raw: pd.DataFrame) -> pd.DataFrame:
    # Contact details and social media links aren't needed for any analysis, so they aren't kept.
    members = raw.rename(columns={"id": "member_id"})
    members["date_of_birth"] = pd.to_datetime(members["date_of_birth"])
    return members[["member_id", "first_name", "last_name", "country_code", "date_of_birth"]]


def clean_votes(raw: pd.DataFrame) -> pd.DataFrame:
    votes = raw[VOTE_COLUMNS].copy()
    votes["group_code"] = votes["group_code"].replace(GROUP_ALIASES)
    votes["vote"] = votes["position"].map(POSITION_TO_VOTE).astype("int8")
    votes["position"] = votes["position"].astype("category")
    for column in ["country_code", "group_code"]:
        votes[column] = votes[column].astype("category")
    votes["national_party_id"] = votes["national_party_id"].astype("Int32")
    return votes


def clean_rollcalls(raw: pd.DataFrame, votes: pd.DataFrame) -> pd.DataFrame:
    rollcalls = raw.rename(columns={"id": "vote_id"}).copy()
    rollcalls["timestamp"] = pd.to_datetime(rollcalls["timestamp"], format="ISO8601")
    rollcalls["date"] = rollcalls["timestamp"].dt.normalize()
    rollcalls["term"] = term_of(rollcalls["timestamp"])
    rollcalls = rollcalls.sort_values(["timestamp", "vote_id"]).reset_index(drop=True)
    rollcalls["rollnumber"] = (rollcalls.groupby("term").cumcount() + 1).astype("int32")
    rollcalls["vote_part"] = vote_part(rollcalls)

    counts = (
        votes.assign(n_for=votes["position"].eq("FOR"), n_against=votes["position"].eq("AGAINST"),
                     n_abstention=votes["position"].eq("ABSTENTION"))
        .groupby("vote_id")[["n_for", "n_against", "n_abstention"]]
        .sum()
    )
    rollcalls = rollcalls.join(counts, on="vote_id")
    rollcalls[["n_for", "n_against", "n_abstention"]] = rollcalls[["n_for", "n_against", "n_abstention"]].fillna(0).astype("int32")
    # Secret ballots are in the export with zero published counts and every member as Did not vote.
    published = rollcalls[["count_for", "count_against", "count_abstention"]]
    positions = rollcalls[["n_for", "n_against", "n_abstention"]].sum(axis=1)
    rollcalls["is_secret"] = published.notna().all(axis=1) & published.sum(axis=1).eq(0) & positions.eq(0)
    total = (rollcalls["n_for"] + rollcalls["n_against"]).where(lambda s: s > 0)
    rollcalls["minority_share"] = rollcalls[["n_for", "n_against"]].min(axis=1) / total
    rollcalls["is_lopsided"] = ~(rollcalls["minority_share"] >= LOPSIDED_SHARE)
    # Sensitivity measure: abstention counted with Against, as the side not supporting the text. Not used by default.
    with_abstention = (total + rollcalls["n_abstention"]).where(lambda s: s > 0)
    rollcalls["minority_share_with_abstention"] = (
        pd.concat([rollcalls["n_for"], rollcalls["n_against"] + rollcalls["n_abstention"]], axis=1).min(axis=1)
        / with_abstention)
    columns = [
        "vote_id", "term", "rollnumber", "timestamp", "date", "display_title", "reference", "description",
        "is_main", "vote_part", "amendment_number", "amendment_subject", "procedure_reference", "procedure_title",
        "procedure_type", "procedure_stage", "majority_type", "result", "texts_adopted_reference",
        "count_for", "count_against", "count_abstention", "count_did_not_vote",
        "n_for", "n_against", "n_abstention", "is_secret", "minority_share", "is_lopsided",
        "minority_share_with_abstention",
    ]
    return rollcalls[columns]


def clean_subjects() -> pd.DataFrame:
    """Subjects of each roll call, long format: OEIL subject codes, EuroVoc concepts and responsible committees."""
    oeil = read("oeil_subject_votes", dtype={"oeil_subject_code": str}).merge(
        read("oeil_subjects", dtype={"code": str}), left_on="oeil_subject_code", right_on="code", how="left")
    eurovoc = read("eurovoc_concept_votes", dtype={"eurovoc_concept_id": str}).merge(
        read("eurovoc_concepts", dtype={"id": str}), left_on="eurovoc_concept_id", right_on="id", how="left")
    committees = read("responsible_committee_votes").merge(read("committees"), left_on="committee_code",
                                                           right_on="code", how="left")
    return pd.concat([
        pd.DataFrame({"vote_id": oeil["vote_id"], "scheme": "oeil_subject", "code": oeil["oeil_subject_code"],
                      "label": oeil["label"]}),
        pd.DataFrame({"vote_id": eurovoc["vote_id"], "scheme": "eurovoc", "code": eurovoc["eurovoc_concept_id"],
                      "label": eurovoc["label"]}),
        pd.DataFrame({"vote_id": committees["vote_id"], "scheme": "committee", "code": committees["committee_code"],
                      "label": committees["label"]}),
    ], ignore_index=True)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    votes = clean_votes(read("member_votes"))
    rollcalls = clean_rollcalls(read("votes"), votes)
    # Secret ballots carry no positions, so their rows aren't votes.
    votes = votes[~votes["vote_id"].isin(rollcalls.loc[rollcalls["is_secret"], "vote_id"])]
    votes = votes.merge(rollcalls[["vote_id", "term"]], on="vote_id", how="left").reset_index(drop=True)

    tables = {
        "members": clean_members(read("members")),
        "groups": clean_groups(read("groups")),
        "group_memberships": clean_group_memberships(read("group_memberships")),
        "national_parties": read("national_parties").rename(columns={"id": "national_party_id"}),
        "national_party_memberships": read("national_party_memberships"),
        "countries": read("countries"),
        "rollcalls": rollcalls,
        "votes": votes,
        "subjects": clean_subjects(),
    }
    for name, table in tables.items():
        table.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    print("  ".join(f"{name} {len(table):,}" for name, table in tables.items()))


if __name__ == "__main__":
    main()
