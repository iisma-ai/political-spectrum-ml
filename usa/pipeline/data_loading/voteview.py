"""Load the processed Voteview tables and build member-by-vote matrices."""

from pathlib import Path

import pandas as pd

from build_dataset import LOPSIDED_SHARE

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
# One threshold for both: rollcalls.is_lopsided and the default matrix filter.
DEFAULT_MINORITY_SHARE = LOPSIDED_SHARE


def load_members() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "members.parquet")


def load_rollcalls() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "rollcalls.parquet")


def load_topics() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "topics.parquet")


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


def build_matrix(
    votes: pd.DataFrame,
    rollcalls: pd.DataFrame,
    members: pd.DataFrame,
    min_minority_share: float = DEFAULT_MINORITY_SHARE,
    include_delegates: bool = False,
    min_votes: int = 20,
) -> pd.DataFrame:
    """Build the member-by-vote matrix from tables already limited to one Congress and chamber.

    Roll calls whose smaller side is below min_minority_share are dropped; 0 keeps every roll call.
    """
    selected = rollcalls
    if min_minority_share > 0:
        selected = selected[selected["minority_share"] >= min_minority_share]
    votes = votes[votes["rollnumber"].isin(selected["rollnumber"])]
    if not include_delegates:
        votes = votes[~votes["icpsr"].isin(members.loc[members["is_delegate"], "icpsr"])]
    matrix = pivot_votes(votes)
    return matrix[matrix.notna().sum(axis=1) >= min_votes]


def vote_matrix(congress: int, chamber: str, **options) -> pd.DataFrame:
    """Build the member-by-vote matrix for one Congress and chamber. Options go to build_matrix."""
    rollcalls = load_rollcalls()
    members = load_members()
    return build_matrix(
        load_votes(congress, chamber),
        rollcalls[(rollcalls["congress"] == congress) & (rollcalls["chamber"] == chamber)],
        members[(members["congress"] == congress) & (members["chamber"] == chamber)],
        **options,
    )
