"""Build the vote prediction table: one row per Democrat or Republican Yea or Nay, with only prior information.

Setting: when a member votes, their own party's position and the other party's position are known, as from whip
notices, and so is everything the member did on earlier roll calls. Nothing else about the roll call's outcome is
used, except in the variant that is labelled as using the size of each party's split.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parents[1] / "data_loading"))
sys.path.append(str(Path(__file__).parents[1] / "vote_explanation"))
from explanation import ROLLCALL_KEYS, member_covariates  # noqa: E402
from voteview import load_members, load_rollcalls, load_topics, load_votes  # noqa: E402

SEQUENCE_KEYS = ["person", "chamber", "party"]
ORDER = ["date", "rollnumber"]
# History starts in 1981 so members serving in 1991 have a record; vote types exist from 1991 (102nd Congress).
HISTORY_START, MODEL_START, INCOMPLETE_CONGRESS = 97, 102, 119
VOTE_TYPES = ["amendment", "procedural", "suspension", "passage", "resolution", "cloture", "nomination",
              "conference report", "veto override", "other", "unknown"]
REGIONS = ["South", "Border", "Northeast", "Midwest", "West", "Other"]
MIN_PREVIOUS_CONGRESS_VOTES = 20


def party_positions(votes: pd.DataFrame) -> pd.DataFrame:
    """Add each member's own-party position without their own vote, and the other party's position.

    Positions are +1 (Yea), -1 (Nay) or 0 (tied or no voters). Leaving the member out stops their own vote from
    deciding the position they are compared with.
    """
    counts = votes.assign(yea=votes["vote"].eq(1), nay=votes["vote"].eq(-1)).groupby(ROLLCALL_KEYS + ["party"])[["yea", "nay"]].sum()
    counts = counts.unstack("party", fill_value=0).reindex(columns=pd.MultiIndex.from_product([["yea", "nay"], ["D", "R"]]), fill_value=0)
    counts.columns = [f"{kind}_{party}" for kind, party in counts.columns]
    frame = votes[ROLLCALL_KEYS].join(counts, on=ROLLCALL_KEYS)
    is_d = (votes["party"] == "D").to_numpy()
    own_yea = np.where(is_d, frame["yea_D"], frame["yea_R"]) - votes["vote"].eq(1).to_numpy()
    own_nay = np.where(is_d, frame["nay_D"], frame["nay_R"]) - votes["vote"].eq(-1).to_numpy()
    other_yea = np.where(is_d, frame["yea_R"], frame["yea_D"])
    other_nay = np.where(is_d, frame["nay_R"], frame["nay_D"])
    out = votes.copy()
    out["own_position"] = np.sign(own_yea - own_nay).astype("int8")
    out["other_position"] = np.sign(other_yea - other_nay).astype("int8")
    # The whole party's margin, with the member: the same for everyone in the party on a roll call, so filters and
    # features built from it don't depend on how this member voted.
    full_yea = own_yea + votes["vote"].eq(1).to_numpy()
    full_nay = own_nay + votes["vote"].eq(-1).to_numpy()
    out["own_full_margin"] = (full_yea - full_nay).astype("int32")
    # Only for the labelled variant: how split each party was, known only after the vote.
    out["own_cohesion"] = (np.abs(full_yea - full_nay) / np.maximum(full_yea + full_nay, 1)).astype("float32")
    out["other_cohesion"] = (np.abs(other_yea - other_nay) / np.maximum(other_yea + other_nay, 1)).astype("float32")
    return out


def prior_rate(frame: pd.DataFrame, keys: list, value: str, mask=None, window: int | None = None) -> pd.Series:
    """For each row, the mean of value over earlier rows of the same group where mask holds.

    frame must already be sorted in time order within each group. Only rows strictly before the current one
    count, so the current vote never enters its own feature. window limits it to the last window masked rows.
    Missing when there is no earlier masked row.
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


def add_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Add history features computed from each member's earlier votes in the same chamber and party."""
    frame = frame.sort_values(SEQUENCE_KEYS + ORDER, kind="stable").reset_index(drop=True)
    frame["defected_f"] = frame["defected"].astype("float32")
    opposed = frame["party_opposed"]
    frame["prior_rate_all"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f")
    frame["prior_rate_opposed"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f", opposed)
    frame["prior_rate_bipartisan"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f", ~opposed)
    frame["recent_rate_opposed_100"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f", opposed, window=100)
    frame["recent_rate_all_50"] = prior_rate(frame, SEQUENCE_KEYS, "defected_f", window=50)
    frame["congress_rate_opposed"] = prior_rate(frame, SEQUENCE_KEYS + ["congress"], "defected_f", opposed)
    frame["prior_rate_same_type"] = prior_rate(frame, SEQUENCE_KEYS + ["vote_type"], "defected_f")
    frame["last_defected"] = frame.groupby(SEQUENCE_KEYS, sort=False)["defected_f"].shift(1)
    frame["prior_votes"] = frame.groupby(SEQUENCE_KEYS, sort=False).cumcount().astype("int32")
    previous = (frame[opposed].groupby(SEQUENCE_KEYS + ["congress"])["defected_f"].agg(["mean", "count"]).reset_index())
    previous = previous[previous["count"] >= MIN_PREVIOUS_CONGRESS_VOTES]
    previous = previous.assign(congress=previous["congress"] + 1).rename(columns={"mean": "previous_congress_rate_opposed"})
    frame = frame.merge(previous[SEQUENCE_KEYS + ["congress", "previous_congress_rate_opposed"]], on=SEQUENCE_KEYS + ["congress"], how="left")
    for column in ["prior_rate_all", "prior_rate_opposed", "prior_rate_bipartisan", "recent_rate_opposed_100", "recent_rate_all_50",
                   "congress_rate_opposed", "prior_rate_same_type", "last_defected", "previous_congress_rate_opposed"]:
        frame[column] = frame[column].astype("float32")
    return frame.drop(columns="defected_f")


def build_rows(votes: pd.DataFrame, members: pd.DataFrame, rollcalls: pd.DataFrame, topics: pd.DataFrame) -> pd.DataFrame:
    """Build the prediction table from votes, members, roll calls and topics already limited to the wanted Congresses.

    Keeps Democrat and Republican Yea and Nay votes on roll calls that aren't near-unanimous. A party whose whole
    margin, including the member, is 0 or 1 vote is dropped for that roll call: deciding this without the member would
    keep or drop rows depending on how the member voted.
    """
    covariates = member_covariates(members)
    covariates = covariates[~covariates["is_delegate"] & covariates["party"].isin(["D", "R"])]
    keep = rollcalls.loc[~rollcalls["is_lopsided"], ROLLCALL_KEYS + ["date"]]
    frame = votes[votes["vote"] != 0].merge(keep, on=ROLLCALL_KEYS)
    member_columns = ["congress", "chamber", "icpsr", "person", "party", "region", "tenure", "freshman", "majority", "bioname", "state_abbrev"]
    frame = frame.merge(covariates[member_columns].drop_duplicates(["congress", "chamber", "icpsr"]), on=["congress", "chamber", "icpsr"])
    frame = party_positions(frame)
    close = frame["own_full_margin"].abs() <= 1
    print(f"Dropped {close.sum():,} of {len(frame):,} votes ({close.mean():.3%}) where the member's party split by 0 or 1 votes")
    frame = frame[~close].drop(columns="own_full_margin")
    frame = frame.merge(topics[ROLLCALL_KEYS + ["vote_type"]], on=ROLLCALL_KEYS, how="left")
    frame["vote_type"] = frame["vote_type"].fillna("unknown")
    frame["defected"] = frame["vote"].to_numpy() != frame["own_position"].to_numpy()
    frame["party_opposed"] = (frame["own_position"] == -frame["other_position"]).to_numpy()
    frame["other_tied"] = frame["other_position"] == 0
    frame["own_yea"] = frame["own_position"] == 1
    frame["majority"] = frame["majority"].astype("float32")  # missing in tied chambers
    return add_history(frame)


def load_rows(start: int = HISTORY_START, end: int = INCOMPLETE_CONGRESS) -> pd.DataFrame:
    """Build the prediction table for Congresses start to end - 1 from the processed Voteview tables."""
    members = load_members()
    rollcalls = load_rollcalls()
    topics = load_topics()
    votes = load_votes()
    votes = votes[(votes["congress"] >= start) & (votes["congress"] < end)][["congress", "chamber", "rollnumber", "icpsr", "vote"]]
    in_range = lambda table: table[(table["congress"] >= start) & (table["congress"] < end)]  # noqa: E731
    # Tenure needs every earlier Congress, so members are limited only after the covariates are built.
    rows = build_rows(votes, members, in_range(rollcalls), in_range(topics))
    return rows[rows["congress"] >= start]
