"""Load the processed European Parliament tables and build member-by-vote matrices."""

from pathlib import Path

import pandas as pd

from build_dataset import LOPSIDED_SHARE

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
# One threshold for both: rollcalls.is_lopsided and the default matrix filter.
DEFAULT_MINORITY_SHARE = LOPSIDED_SHARE


def load(name: str) -> pd.DataFrame:
    """Any processed table: members, groups, group_memberships, national_parties, national_party_memberships,
    countries, rollcalls, votes or subjects."""
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")


def load_members() -> pd.DataFrame:
    return load("members")


def load_rollcalls() -> pd.DataFrame:
    return load("rollcalls")


def load_group_memberships() -> pd.DataFrame:
    return load("group_memberships")


def load_subjects() -> pd.DataFrame:
    return load("subjects")


def load_votes(term: int | None = None) -> pd.DataFrame:
    filters = [("term", "==", term)] if term is not None else None
    return pd.read_parquet(PROCESSED_DIR / "votes.parquet", filters=filters)


def pivot_votes(votes: pd.DataFrame) -> pd.DataFrame:
    """Return members by vote_id with 1 for For, -1 for Against and NaN for Abstention or Did not vote."""
    positions = votes[votes["vote"] != 0]
    return positions.pivot(index="member_id", columns="vote_id", values="vote").astype("float32")


def build_matrix(
    votes: pd.DataFrame,
    rollcalls: pd.DataFrame,
    min_minority_share: float = DEFAULT_MINORITY_SHARE,
    min_votes: int = 20,
) -> pd.DataFrame:
    """Build the member-by-vote matrix from tables already limited to one term.

    Secret ballots are always dropped. Roll calls whose smaller side is below min_minority_share are dropped;
    0 keeps every other roll call.
    """
    selected = rollcalls[~rollcalls["is_secret"]]
    if min_minority_share > 0:
        selected = selected[selected["minority_share"] >= min_minority_share]
    matrix = pivot_votes(votes[votes["vote_id"].isin(selected["vote_id"])])
    return matrix[matrix.notna().sum(axis=1) >= min_votes]


def vote_matrix(term: int, **options) -> pd.DataFrame:
    """Build the member-by-vote matrix for one term. Options go to build_matrix."""
    rollcalls = load_rollcalls()
    return build_matrix(load_votes(term), rollcalls[rollcalls["term"] == term], **options)
