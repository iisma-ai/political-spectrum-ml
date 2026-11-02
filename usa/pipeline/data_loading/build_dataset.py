"""Clean the raw Voteview CSVs into the parquet tables every problem uses."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# 1 for Yea positions, -1 for Nay positions, 0 for present or not voting.
CAST_TO_VOTE = {1: 1, 2: 1, 3: 1, 4: -1, 5: -1, 6: -1, 7: 0, 8: 0, 9: 0}
PARTIES = {100: "D", 200: "R"}
TERRITORIES = {"AS", "DC", "GU", "MP", "PR", "VI"}
LOPSIDED_SHARE = 0.025

VOTE_DTYPES = {"congress": "int16", "rollnumber": "int32", "icpsr": "int32", "cast_code": "int8"}


def clean_members(raw: pd.DataFrame) -> pd.DataFrame:
    members = raw[raw["chamber"] != "President"].copy()
    members["party"] = members["party_code"].map(PARTIES).fillna("Other")
    members["is_delegate"] = (members["chamber"] == "House") & members["state_abbrev"].isin(TERRITORIES)
    columns = [
        "congress", "chamber", "icpsr", "bioname", "bioguide_id", "party_code", "party", "state_abbrev",
        "district_code", "born", "is_delegate", "nominate_dim1", "nominate_dim2",
        "nokken_poole_dim1", "nokken_poole_dim2",
    ]
    return members[columns].reset_index(drop=True)


def clean_votes(raw: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    # Presidents' positions are stored under House and Senate rows, so keep chamber members only.
    keys = members[["congress", "chamber", "icpsr"]]
    votes = raw.merge(keys, on=["congress", "chamber", "icpsr"], how="inner")
    votes = votes[votes["cast_code"] != 0].copy()
    votes["vote"] = votes["cast_code"].map(CAST_TO_VOTE).astype("int8")
    return votes.reset_index(drop=True)


def clean_rollcalls(raw: pd.DataFrame, votes: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    rollcall_keys = ["congress", "chamber", "rollnumber"]
    delegates = members.loc[members["is_delegate"], ["congress", "chamber", "icpsr"]]
    voters = votes.merge(delegates, on=["congress", "chamber", "icpsr"], how="left", indicator=True)
    voters = voters[voters["_merge"] == "left_only"]
    positions = (
        voters.assign(yea=voters["vote"].eq(1), nay=voters["vote"].eq(-1))
        .groupby(rollcall_keys)[["yea", "nay"]]
        .sum()
        .reset_index()
    )
    rollcalls = raw.merge(positions, on=rollcall_keys, how="left")
    rollcalls[["yea", "nay"]] = rollcalls[["yea", "nay"]].fillna(0).astype("int32")
    total = (rollcalls["yea"] + rollcalls["nay"]).where(lambda s: s > 0)
    rollcalls["minority_share"] = rollcalls[["yea", "nay"]].min(axis=1) / total
    rollcalls["is_lopsided"] = ~(rollcalls["minority_share"] >= LOPSIDED_SHARE)
    rollcalls["date"] = pd.to_datetime(rollcalls["date"])
    return rollcalls


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_members = pd.read_csv(RAW_DIR / "HSall_members.csv", low_memory=False)
    raw_rollcalls = pd.read_csv(RAW_DIR / "HSall_rollcalls.csv", low_memory=False)
    raw_votes = pd.read_csv(
        RAW_DIR / "HSall_votes.csv",
        usecols=["congress", "chamber", "rollnumber", "icpsr", "cast_code"],
        dtype=VOTE_DTYPES,
    )

    members = clean_members(raw_members)
    votes = clean_votes(raw_votes, members)
    rollcalls = clean_rollcalls(raw_rollcalls, votes, members)

    members.to_parquet(PROCESSED_DIR / "members.parquet", index=False)
    rollcalls.to_parquet(PROCESSED_DIR / "rollcalls.parquet", index=False)
    votes.to_parquet(PROCESSED_DIR / "votes.parquet", index=False)
    print(f"members {len(members):,}  rollcalls {len(rollcalls):,}  votes {len(votes):,}")


if __name__ == "__main__":
    main()
