"""Load the processed Voteview tables and build member-by-vote matrices."""

from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


def load_members() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "members.parquet")


def load_rollcalls() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "rollcalls.parquet")


def load_votes(congress: int | None = None, chamber: str | None = None) -> pd.DataFrame:
    filters = []
    if congress is not None:
        filters.append(("congress", "==", congress))
    if chamber is not None:
        filters.append(("chamber", "==", chamber))
    return pd.read_parquet(PROCESSED_DIR / "votes.parquet", filters=filters or None)


def pivot_votes(votes: pd.DataFrame) -> pd.DataFrame:
    """Return members by rollnumber with 1 for Yea, -1 for Nay and NaN for anything else."""
    positions = votes[votes["vote"] != 0]
    return positions.pivot(index="icpsr", columns="rollnumber", values="vote").astype("float32")


def vote_matrix(
    congress: int,
    chamber: str,
    include_lopsided: bool = False,
    include_delegates: bool = False,
    min_votes: int = 20,
) -> pd.DataFrame:
    """Build the member-by-vote matrix for one Congress and chamber."""
    votes = load_votes(congress, chamber)
    rollcalls = load_rollcalls()
    selected = rollcalls[(rollcalls["congress"] == congress) & (rollcalls["chamber"] == chamber)]
    if not include_lopsided:
        selected = selected[~selected["is_lopsided"]]
    votes = votes[votes["rollnumber"].isin(selected["rollnumber"])]
    if not include_delegates:
        members = load_members()
        delegates = members[
            (members["congress"] == congress) & (members["chamber"] == chamber) & members["is_delegate"]
        ]
        votes = votes[~votes["icpsr"].isin(delegates["icpsr"])]
    matrix = pivot_votes(votes)
    return matrix[matrix.notna().sum(axis=1) >= min_votes]
