"""Shared code for the European Parliament vote explanation notebooks.

- A group's line is its most common position over For, Against and Abstention, with the member included, as the
  USA defection rows include the member in the party majority. Leaving the member out would make every member of an
  evenly split group a defector. Non-attached members have no group line.
- A national party's line is the position all the member's other active national party colleagues took, when there
  are at least two of them. National parties are small, so a plurality would depend heavily on the member's own vote
  (with the member in) or turn even splits into mutual defection (with the member out).
"""

import numpy as np
import pandas as pd

POSITIONS = ["FOR", "AGAINST", "ABSTENTION"]
CODES = {position: code for code, position in enumerate(POSITIONS)}
NON_ATTACHED = "NI"
# Smallest number of active members needed for a group line, as in polarization_clustering.
MIN_GROUP_VOTES = 5
# Smallest number of other active national party colleagues needed for a national party line.
MIN_PARTY_VOTES = 2
# The export files members without a party under a per-country "Independent" entry. These aren't parties, so their
# members get no national party line.
INDEPENDENT_LABELS = {"Independent", "Independiente", "Indépendant", "Indépendent", "Bezpartyjna"}


def independent_party_ids(national_parties: pd.DataFrame) -> set:
    """IDs of the export's national party entries that stand for independents, not parties."""
    return set(national_parties.loc[national_parties["label"].str.strip().isin(INDEPENDENT_LABELS), "national_party_id"])


def totals_by(keys: list, frame: pd.DataFrame, onehot: np.ndarray) -> np.ndarray:
    """For each row, the For, Against and Abstention counts over all rows sharing `keys`."""
    return pd.DataFrame(onehot, index=frame.index).groupby([frame[key] for key in keys], observed=True,
                                                           dropna=False).transform("sum").to_numpy().copy()


def plurality_line(totals: np.ndarray, minimum: int) -> np.ndarray:
    """The most common position, coded 0 to 2; -1 with fewer than `minimum` positions or a tie at the top."""
    largest = totals.max(axis=1)
    ties = (totals == largest[:, None]).sum(axis=1) > 1
    return np.where(ties | (totals.sum(axis=1) < minimum), -1, totals.argmax(axis=1)).astype("int8")


def unanimous_others_line(totals: np.ndarray, onehot: np.ndarray, minimum: int) -> np.ndarray:
    """The position every other row took, coded 0 to 2; -1 with fewer than `minimum` others or any disagreement."""
    others = totals - onehot
    count = others.sum(axis=1)
    united = others.max(axis=1) == count
    return np.where(united & (count >= minimum), others.argmax(axis=1), -1).astype("int8")


def defection_rows(votes: pd.DataFrame, no_party_ids: set = frozenset()) -> pd.DataFrame:
    """One row per active position (For, Against or Abstention), with the member's group line and national party line.

    Columns added:
    - position_code, group_line, party_line: 0 For, 1 Against, 2 Abstention, -1 no line;
    - defected_group: an active position different from the group line, where the group has one;
    - defected_party: the same against the national party line;
    - conflict: both lines exist and differ;
    - defected_group_for_against: position and group line are both For or Against and differ (secondary measure);
    - party_others: the member's other active national party colleagues in the same group;
    - colleagues: "too few colleagues", "colleagues divided", "united with the group" or "united against the group";
    - outside_for, outside_against, outside_abstention: the group's positions without the member's national party
      (without the member alone for members without one), and group_line_outside, defected_group_outside and
      conflict_outside measured against that line. A check for national delegations that make up much of their
      group's line;
    - defected_group_leave_one_out: against the group line without the member, the baseline used for prediction.
    Non-attached members get no group line. Members without a national party, or filed under one of
    `no_party_ids` (independents), get no party line. National party colleagues are counted within the same group.
    """
    active = votes[votes["position"].isin(POSITIONS)].copy()
    active["group_code"] = active["group_code"].astype(str)
    code = active["position"].astype(str).map(CODES).to_numpy()
    onehot = np.eye(3, dtype="int32")[code]
    active["position_code"] = code.astype("int8")
    non_attached = active["group_code"].eq(NON_ATTACHED).to_numpy()
    group_totals = totals_by(["vote_id", "group_code"], active, onehot)
    active["group_line"] = np.where(non_attached, -1, plurality_line(group_totals, MIN_GROUP_VOTES)).astype("int8")

    party = active["national_party_id"].where(~active["national_party_id"].isin(no_party_ids))
    keyed = active.assign(national_party_id=party.fillna(-1))
    party_totals = totals_by(["vote_id", "group_code", "national_party_id"], keyed, onehot)
    no_party = party.isna().to_numpy()
    party_totals[no_party] = onehot[no_party]  # members without a party stand alone
    party_line = unanimous_others_line(party_totals, onehot, MIN_PARTY_VOTES)
    active["party_line"] = np.where(no_party, -1, party_line).astype("int8")
    active["party_others"] = (party_totals.sum(axis=1) - 1).astype("int16")

    has_group, has_party = active["group_line"] >= 0, active["party_line"] >= 0
    # Floats: 1 yes, 0 no, NaN where the line is missing, so means are rates.
    active["defected_group"] = (active["position_code"] != active["group_line"]).astype("float32").where(has_group)
    active["defected_party"] = (active["position_code"] != active["party_line"]).astype("float32").where(has_party)
    active["conflict"] = (active["group_line"] != active["party_line"]).astype("float32").where(has_group & has_party)
    for_against = has_group & (active["position_code"] < 2) & (active["group_line"] < 2)
    active["defected_group_for_against"] = (
        (active["position_code"] != active["group_line"]).astype("float32").where(for_against))
    active["colleagues"] = pd.Series(np.select(
        [active["party_others"] < MIN_PARTY_VOTES, ~has_party, active["party_line"] == active["group_line"]],
        ["too few colleagues", "colleagues divided", "united with the group"], "united against the group"),
        index=active.index).where(has_group)

    outside = group_totals - party_totals
    for column, name in enumerate(["outside_for", "outside_against", "outside_abstention"]):
        active[name] = outside[:, column].astype("int16")
    active["group_line_outside"] = np.where(non_attached, -1, plurality_line(outside, MIN_GROUP_VOTES)).astype("int8")
    has_outside = active["group_line_outside"] >= 0
    active["defected_group_outside"] = (
        (active["position_code"] != active["group_line_outside"]).astype("float32").where(has_outside))
    active["conflict_outside"] = (
        (active["group_line_outside"] != active["party_line"]).astype("float32").where(has_outside & has_party))
    leave_one_out = np.where(non_attached, -1, plurality_line(group_totals - onehot, MIN_GROUP_VOTES))
    defected = pd.Series((active["position_code"].to_numpy() != leave_one_out).astype("float32"), index=active.index)
    active["defected_group_leave_one_out"] = defected.where(leave_one_out >= 0)
    return active


def follows(rows: pd.DataFrame, group_line: str = "group_line") -> pd.Series:
    """On conflicts, whether the member took the group line, the national party line, or something else."""
    return pd.Series(np.select(
        [rows["position_code"] == rows["party_line"], rows["position_code"] == rows[group_line]],
        ["national party", "group"], "neither"), index=rows.index)


def placebo_colleagues(rows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Replace each member's national party colleagues with as many random members of the same group from other
    national parties, and rebuild the national party line and conflict from them.

    rows need group_line, party_others and the outside_* counts. Returns placebo_line (-1 when the random colleagues
    aren't unanimous or are too few) and placebo_conflict. If following the national party were only a matter of how
    the line is defined, members would follow placebo colleagues about as often as real ones.
    """
    pool = rows[["outside_for", "outside_against", "outside_abstention"]].to_numpy().astype("int64")
    sizes = np.minimum(rows["party_others"].to_numpy().astype("int64"), pool.sum(axis=1))
    draws = np.array([rng.multivariate_hypergeometric(colors, size) for colors, size in zip(pool, sizes)])
    count = draws.sum(axis=1)
    united = (draws.max(axis=1) == count) & (count >= MIN_PARTY_VOTES)
    placebo_line = np.where(united, draws.argmax(axis=1), -1)
    group_line = rows["group_line"].to_numpy()
    conflict = np.where((placebo_line >= 0) & (group_line >= 0), (placebo_line != group_line).astype(float), np.nan)
    return pd.DataFrame({"placebo_line": placebo_line, "placebo_conflict": conflict}, index=rows.index)


def lagged_rate(rows: pd.DataFrame, period: str, column: str = "defected_group") -> pd.DataFrame:
    """Each member's rate of `column` in the last earlier period in which they had rows, and how many rows it rests on.

    Keyed by member and period; missing for a member's first period. Only earlier periods are used, so the member's
    own votes in the period being explained don't enter.
    """
    rates = rows.dropna(subset=[column]).groupby(["member_id", period])[column].agg(["mean", "count"]).reset_index()
    rates = rates.sort_values(["member_id", period])
    rates["lagged_rate"] = rates.groupby("member_id")["mean"].shift(1)
    rates["lagged_count"] = rates.groupby("member_id")["count"].shift(1)
    return rates[["member_id", period, "lagged_rate", "lagged_count"]]


def shift_share(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Split the change in an overall rate into change within topics and change in the topic mix.

    Same as usa/pipeline/vote_explanation/explanation.py. before and after are indexed by topic with columns "rate"
    and "weight" (weights sum to 1). A topic present in only one period counts entirely as topic mix.
    """
    joined = before.join(after, suffixes=("_before", "_after"), how="outer")
    joined["rate_before"] = joined["rate_before"].fillna(joined["rate_after"])
    joined["rate_after"] = joined["rate_after"].fillna(joined["rate_before"])
    joined = joined.fillna(0)
    within = ((joined["weight_before"] + joined["weight_after"]) / 2 * (joined["rate_after"] - joined["rate_before"])).sum()
    mix = ((joined["rate_before"] + joined["rate_after"]) / 2 * (joined["weight_after"] - joined["weight_before"])).sum()
    total = (joined["weight_after"] * joined["rate_after"]).sum() - (joined["weight_before"] * joined["rate_before"]).sum()
    return {"total": total, "within_topics": within, "topic_mix": mix}


def two_way(frame: pd.DataFrame) -> dict:
    """Cluster standard errors by member and by roll call, for statsmodels."""
    return {"groups": np.column_stack([pd.factorize(frame["member_id"])[0], pd.factorize(frame["vote_id"])[0]])}
