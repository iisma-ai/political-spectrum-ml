"""Build the European Parliament vote prediction table: one row per group member's active position, with only prior
information.

Setting: when a member votes, every political group's line on the roll call is known, as from the groups' voting
lists, and so is everything that happened on earlier roll calls. The group lines used are the realised ones from the
final votes, the counterpart of the USA's party positions. Nothing else about the roll call's outcome is used, except
in the variant labelled as using the split of the member's group and national colleagues.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1] / "data_loading"))
sys.path.append(str(Path(__file__).parents[1] / "vote_explanation"))
from explanation import (  # noqa: E402
    CODES, MIN_GROUP_VOTES, MIN_PARTY_VOTES, NON_ATTACHED, POSITIONS, independent_party_ids, plurality_line, totals_by,
    unanimous_others_line,
)
from howtheyvote import DEFAULT_MINORITY_SHARE, load, load_rollcalls, load_votes  # noqa: E402

SEQUENCE_KEYS = ["member_id"]
ORDER = ["timestamp", "vote_id"]
VOTE_PARTS = ["amendment", "text part", "final", "procedural", "other"]
LINES = POSITIONS  # FOR, AGAINST, ABSTENTION, coded 0 to 2


def group_positions(votes: pd.DataFrame, no_party_ids: set) -> pd.DataFrame:
    """Add each member's group line without their own vote, and how the other groups lined up.

    Columns added:
    - own_line: the group's most common position leaving the member out (0 For, 1 Against, 2 Abstention, -1 none);
    - own_full_margin: votes between the group's two most common positions with the member, the same for everyone in
      the group on the roll call, so a filter on it doesn't depend on how this member voted;
    - others_same_share: the share of the other groups with a line whose line is the member's own_line;
    - group_seats, group_seat_share, party_seats: members listed (any position, including Did not vote);
    - after the vote only, for the labelled variant: own_cohesion (the group's largest position share with the member)
      and colleagues (what the member's national colleagues in the group did, against own_line).
    """
    listed = votes.copy()
    listed["group_code"] = listed["group_code"].astype(str)
    group_seats = listed.groupby(["vote_id", "group_code"]).size().rename("group_seats")
    all_seats = listed.groupby("vote_id").size().rename("all_seats")
    party = listed["national_party_id"].where(~listed["national_party_id"].isin(no_party_ids))
    listed["party_key"] = party.fillna(-listed["member_id"].astype(float))  # members without a party stand alone
    party_seats = listed.groupby(["vote_id", "group_code", "party_key"]).size().rename("party_seats")

    active = listed[listed["position"].isin(POSITIONS)].copy()
    code = active["position"].astype(str).map(CODES).to_numpy()
    onehot = np.eye(3, dtype="int32")[code]
    active["position_code"] = code.astype("int8")
    totals = totals_by(["vote_id", "group_code"], active, onehot)
    active["own_line"] = plurality_line(totals - onehot, MIN_GROUP_VOTES)
    ordered = np.sort(totals, axis=1)
    active["own_full_margin"] = (ordered[:, 2] - ordered[:, 1]).astype("int32")
    active["own_cohesion"] = (ordered[:, 2] / np.maximum(totals.sum(axis=1), 1)).astype("float32")
    full_line = plurality_line(totals, MIN_GROUP_VOTES)

    # How many groups (non-attached members excluded) took each line on each roll call.
    lines = pd.DataFrame({"vote_id": active["vote_id"], "group_code": active["group_code"], "line": full_line})
    lines = lines[(lines["group_code"] != NON_ATTACHED)].drop_duplicates(["vote_id", "group_code"])
    lines = lines[lines["line"] >= 0]
    per_line = lines.groupby(["vote_id", "line"]).size().unstack(fill_value=0).reindex(columns=[0, 1, 2], fill_value=0)
    per_line["n"] = per_line.sum(axis=1)
    joined = per_line.reindex(active["vote_id"].to_numpy()).fillna(0).to_numpy()
    own = active["own_line"].to_numpy()
    own_counted = (full_line >= 0) & (active["group_code"] != NON_ATTACHED).to_numpy()
    same = np.where(own >= 0, joined[np.arange(len(own)), np.clip(own, 0, 2)], 0) - (own_counted & (full_line == own))
    others = joined[:, 3] - own_counted
    active["others_same_share"] = np.where(others > 0, same / np.maximum(others, 1), np.nan).astype("float32")
    active["other_groups"] = others.astype("int16")

    # National colleagues in the same group: known only after the vote, for the labelled variant.
    keyed = active.assign(national_party_id=active["party_key"])
    party_totals = totals_by(["vote_id", "group_code", "national_party_id"], keyed, onehot)
    colleague_line = unanimous_others_line(party_totals, onehot, MIN_PARTY_VOTES)
    colleagues = party_totals.sum(axis=1) - 1
    active["colleagues"] = np.select(
        [colleagues < MIN_PARTY_VOTES, colleague_line < 0, colleague_line == own],
        ["too few colleagues", "colleagues divided", "united with the line"], "united against the line")

    active = active.join(group_seats, on=["vote_id", "group_code"]).join(all_seats, on="vote_id")
    active = active.join(party_seats, on=["vote_id", "group_code", "party_key"])
    active["group_seat_share"] = (active["group_seats"] / active["all_seats"]).astype("float32")
    active["no_party"] = party.reindex(active.index).isna()
    return active.drop(columns="all_seats")


def prior_rate(frame: pd.DataFrame, keys: list, value: str, mask=None, window: int | None = None) -> pd.Series:
    """For each row, the mean of value over earlier rows of the same group where mask holds.

    Same as the USA's. frame must be sorted in time order within each group; only rows strictly before the current
    one count. Use it only for keys with one row per roll call (a member); for groups of members, see
    earlier_rollcall_rate.
    """
    mask = pd.Series(True, index=frame.index) if mask is None else mask
    masked = frame.loc[mask, keys + [value]]
    grouped = masked.groupby(keys, sort=False)[value]
    total = grouped.cumsum()
    count = grouped.cumcount() + 1
    if window is not None:
        total = total - total.groupby([masked[k] for k in keys], sort=False).shift(window).fillna(0)
        count = count.clip(upper=window)
    inclusive = pd.Series(np.nan, index=frame.index)
    inclusive.loc[masked.index] = (total / count).to_numpy()
    group_ids = frame.groupby(keys, sort=False).ngroup()
    return inclusive.groupby(group_ids).shift(1).groupby(group_ids).ffill()


def earlier_rollcall_rate(frame: pd.DataFrame, keys: list, value: str, window: int | None = None) -> pd.Series:
    """For each row, the mean of value over the rows of the same group on earlier roll calls only.

    Colleagues on the same roll call vote at the same time, so their positions aren't earlier information.
    """
    per_call = frame.groupby(keys + ["vote_id"], sort=False).agg(
        total=(value, "sum"), count=(value, "size"), timestamp=("timestamp", "first")).reset_index()
    per_call = per_call.sort_values(keys + ["timestamp", "vote_id"], kind="stable")
    grouped = per_call.groupby(keys, sort=False)
    if window is None:
        total, count = grouped["total"].cumsum(), grouped["count"].cumsum()
    else:
        total = grouped["total"].transform(lambda s: s.rolling(window, min_periods=1).sum())
        count = grouped["count"].transform(lambda s: s.rolling(window, min_periods=1).sum())
    per_call["rate"] = (total / count).groupby([per_call[k] for k in keys], sort=False).shift(1)
    merged = frame[keys + ["vote_id"]].merge(per_call[keys + ["vote_id", "rate"]], on=keys + ["vote_id"], how="left")
    return pd.Series(merged["rate"].to_numpy(), index=frame.index, dtype="float32")


def add_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Add history features from earlier roll calls: the member's own, their national party's and their group's."""
    frame = frame.sort_values(SEQUENCE_KEYS + ORDER, kind="stable").reset_index(drop=True)
    frame["defected_f"] = frame["defected"].astype("float32")
    with_most = frame["with_most_groups"]
    frame["prior_rate_all"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f")
    frame["prior_rate_with_most"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f", with_most)
    frame["prior_rate_against_most"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f", ~with_most)
    frame["recent_rate_all_100"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f", window=100)
    frame["recent_rate_against_most_50"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f", ~with_most, window=50)
    frame["prior_rate_same_part"] = prior_rate(frame, SEQUENCE_KEYS + ["vote_part"], "defected_f")
    frame["last_defected"] = frame.groupby(SEQUENCE_KEYS, sort=False)["defected_f"].shift(1)
    frame["prior_votes"] = frame.groupby(SEQUENCE_KEYS, sort=False).cumcount().astype("int32")
    previous_group = frame.groupby(SEQUENCE_KEYS, sort=False)["group_code"].shift(1)
    frame["new_group"] = (previous_group.notna() & (previous_group != frame["group_code"])).astype("float32")
    frame["party_recent_rate_200"] = earlier_rollcall_rate(frame.assign(party=frame["party_group"]), ["party"],
                                                           "defected_f", window=200)
    frame["group_recent_rate_200"] = earlier_rollcall_rate(frame, ["group_code"], "defected_f", window=200)
    history = ["prior_rate_all", "prior_rate_with_most", "prior_rate_against_most", "recent_rate_all_100",
               "recent_rate_against_most_50", "prior_rate_same_part", "last_defected", "party_recent_rate_200",
               "group_recent_rate_200"]
    for column in history:
        frame[column] = frame[column].astype("float32")
    return frame.drop(columns="defected_f")


def build_rows(votes: pd.DataFrame, rollcalls: pd.DataFrame, members: pd.DataFrame, committees: pd.Series,
               no_party_ids: set, min_minority_share: float = DEFAULT_MINORITY_SHARE) -> pd.DataFrame:
    """Build the prediction table from member votes, roll calls, members and responsible committees.

    Keeps group members' active positions (non-attached members excluded) on roll calls that aren't near-unanimous,
    where the group has a line without the member. A group is dropped for a roll call when its two most common
    positions, with the member, are 0 or 1 vote apart: deciding this without the member would keep or drop rows
    depending on how the member voted.
    """
    rollcalls = rollcalls[~rollcalls["is_secret"]]
    keep = rollcalls.loc[rollcalls["minority_share"] >= min_minority_share, "vote_id"]
    frame = group_positions(votes[votes["vote_id"].isin(keep)], no_party_ids)
    frame = frame[(frame["group_code"] != NON_ATTACHED)]
    close = frame["own_full_margin"] <= 1
    print(f"Dropped {close.sum():,} of {len(frame):,} positions ({close.mean():.3%}) where the member's group split "
          f"by 0 or 1 votes")
    frame = frame[~close & (frame["own_line"] >= 0)].drop(columns="own_full_margin")
    frame = frame.merge(rollcalls[["vote_id", "timestamp", "date", "term", "vote_part"]], on="vote_id")
    frame["committee"] = committees.reindex(frame["vote_id"].to_numpy()).fillna("NONE").to_numpy()
    birth = members.set_index("member_id")["date_of_birth"].dt.year
    frame["age"] = (frame["date"].dt.year - birth.reindex(frame["member_id"]).to_numpy()).astype("float32")
    frame["defected"] = frame["position_code"].to_numpy() != frame["own_line"].to_numpy()
    frame["with_most_groups"] = frame["others_same_share"].fillna(0) >= 0.5
    # National party within the group; members without a party (or independents) each stand alone.
    frame["party_group"] = frame["group_code"] + "|" + frame["party_key"].astype(str)
    frame["log_party_seats"] = np.log(frame["party_seats"]).astype("float32")
    return add_history(frame)


def load_rows(min_minority_share: float = DEFAULT_MINORITY_SHARE) -> pd.DataFrame:
    """Build the prediction table for both terms from the processed HowTheyVote tables."""
    rollcalls = load_rollcalls()
    members = load("members")
    subjects = load("subjects").query("scheme == 'committee'")
    joint = subjects.groupby("vote_id")["code"].nunique() > 1
    first = subjects.drop_duplicates("vote_id").set_index("vote_id")["code"]
    committees = first.where(~joint.reindex(first.index), "JOINT")
    no_party_ids = independent_party_ids(load("national_parties"))
    votes = pd.concat([load_votes(term) for term in [9, 10]], ignore_index=True)
    votes = votes[["vote_id", "member_id", "position", "group_code", "country_code", "national_party_id"]]
    return build_rows(votes, rollcalls, members, committees, no_party_ids, min_minority_share)
