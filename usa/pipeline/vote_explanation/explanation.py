"""Shared code for the vote explanation notebooks."""

import numpy as np
import pandas as pd

ROLLCALL_KEYS = ["congress", "chamber", "rollnumber"]

# South is the 11 former Confederate states, as in polarization_clustering. Border states are kept apart
# so they can be tested separately; the other regions follow the Census Bureau.
REGIONS = {
    "South": {"AL", "AR", "FL", "GA", "LA", "MS", "NC", "SC", "TN", "TX", "VA"},
    "Border": {"DE", "KY", "MD", "MO", "OK", "WV"},
    "Northeast": {"CT", "ME", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"},
    "Midwest": {"IL", "IN", "IA", "KS", "MI", "MN", "NE", "ND", "OH", "SD", "WI"},
    "West": {"AK", "AZ", "CA", "CO", "HI", "ID", "MT", "NV", "NM", "OR", "UT", "WA", "WY"},
}
STATE_REGION = {state: region for region, states in REGIONS.items() for state in states}


def first_year(congress):
    return 1789 + 2 * (congress - 1)


def chamber_control(members: pd.DataFrame) -> pd.DataFrame:
    """Seats and control for each Congress and chamber.

    Members outside the two parties are counted with the party whose mean DW-NOMINATE dimension 1 is closer,
    as a stand-in for caucusing (Sanders and King with Democrats, for example). A tie is marked as tied, since
    the Vice President decided control. Mid-Congress replacements count as extra seats, and a change of control
    during a Congress (as in the Senate in 2001) isn't captured.
    """
    frame = members.drop_duplicates(["congress", "chamber", "icpsr"]).copy()
    means = frame[frame["party"].isin(["D", "R"])].groupby(["congress", "chamber", "party"])["nominate_dim1"].mean().unstack()
    others = frame[~frame["party"].isin(["D", "R"])].join(means, on=["congress", "chamber"])
    nearer = np.where((others["nominate_dim1"] - others["D"]).abs() <= (others["nominate_dim1"] - others["R"]).abs(), "D", "R")
    frame.loc[others.index, "caucus"] = np.where(others["nominate_dim1"].isna(), None, nearer)
    frame["caucus"] = frame["caucus"].where(~frame["party"].isin(["D", "R"]), frame["party"])
    seats = frame.dropna(subset=["caucus"]).groupby(["congress", "chamber", "caucus"]).size().unstack(fill_value=0)
    seats = seats.reindex(columns=["D", "R"], fill_value=0)
    control = pd.DataFrame({
        "seats_d": seats["D"], "seats_r": seats["R"],
        "control": np.select([seats["D"] > seats["R"], seats["R"] > seats["D"]], ["D", "R"], "tied"),
    })
    return control.reset_index()


def member_covariates(members: pd.DataFrame) -> pd.DataFrame:
    """One row per member record with region, tenure, age, chamber control and party seat share.

    Tenure counts earlier Congresses the person served in either chamber, linked by Bioguide ID. majority is True
    for the controlling party, False for the other, and missing for a tied chamber.
    """
    frame = members.copy()
    frame["person"] = frame["bioguide_id"].fillna("icpsr:" + frame["icpsr"].astype(str))
    frame["region"] = frame["state_abbrev"].map(STATE_REGION).fillna("Other")
    served = frame[["person", "congress"]].drop_duplicates().sort_values(["person", "congress"])
    served["tenure"] = served.groupby("person").cumcount()
    frame = frame.merge(served, on=["person", "congress"])
    frame["freshman"] = frame["tenure"] == 0
    frame["age"] = first_year(frame["congress"]) - frame["born"]
    control = chamber_control(members)
    frame = frame.merge(control, on=["congress", "chamber"], how="left")
    own_seats = np.where(frame["party"] == "D", frame["seats_d"], np.where(frame["party"] == "R", frame["seats_r"], np.nan))
    frame["party_seat_share"] = own_seats / (frame["seats_d"] + frame["seats_r"])
    frame["majority"] = (frame["party"] == frame["control"]).astype("boolean")
    frame.loc[(frame["control"] == "tied") | ~frame["party"].isin(["D", "R"]), "majority"] = pd.NA
    return frame.drop(columns=["seats_d", "seats_r"])


def party_shares(votes: pd.DataFrame) -> pd.DataFrame:
    """Each party's Yea share and voter count on each roll call, Democrats and Republicans only."""
    positions = votes[(votes["vote"] != 0) & votes["party"].isin(["D", "R"])]
    grouped = positions.assign(yea=positions["vote"].eq(1)).groupby(ROLLCALL_KEYS + ["party"])["yea"].agg(["mean", "count"]).unstack()
    return pd.DataFrame({
        "D": grouped[("mean", "D")], "R": grouped[("mean", "R")],
        "n_D": grouped[("count", "D")], "n_R": grouped[("count", "R")],
    }).dropna()


def defection_rows(votes: pd.DataFrame) -> pd.DataFrame:
    """One row per Democrat or Republican Yea or Nay on a party-opposed roll call, flagging votes with the other party."""
    shares = party_shares(votes)
    majority = np.sign(shares[["D", "R"]] - 0.5)
    opposed = majority[(majority["D"] * majority["R"]) == -1].rename(columns={"D": "majority_d", "R": "majority_r"})
    positions = votes[(votes["vote"] != 0) & votes["party"].isin(["D", "R"])]
    rows = positions.merge(opposed.reset_index(), on=ROLLCALL_KEYS)
    other = np.where(rows["party"] == "D", rows["majority_r"], rows["majority_d"])
    rows["defected"] = rows["vote"].to_numpy() == other
    return rows.drop(columns=["majority_d", "majority_r"])


def shift_share(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Split the change in an overall rate into change within topics and change in the topic mix.

    before and after are indexed by topic with columns "rate" and "weight" (weights sum to 1). Using the average
    of the two periods' weights and rates makes the two parts add up exactly to the change. A topic present in
    only one period takes its rate from that period, so its whole contribution counts as topic mix.
    """
    joined = before.join(after, lsuffix="_before", rsuffix="_after", how="outer")
    joined["rate_before"] = joined["rate_before"].fillna(joined["rate_after"])
    joined["rate_after"] = joined["rate_after"].fillna(joined["rate_before"])
    joined = joined.fillna(0)
    within = ((joined["weight_before"] + joined["weight_after"]) / 2 * (joined["rate_after"] - joined["rate_before"])).sum()
    mix = ((joined["rate_before"] + joined["rate_after"]) / 2 * (joined["weight_after"] - joined["weight_before"])).sum()
    total = (joined["weight_after"] * joined["rate_after"]).sum() - (joined["weight_before"] * joined["rate_before"]).sum()
    return {"total": total, "within_topics": within, "topic_mix": mix}


def lagged_position(covariates: pd.DataFrame, column: str = "nokken_poole_dim1") -> pd.Series:
    """Each member's position toward the other party in their previous Congress, in the same chamber and party.

    Positive values mean closer to the other party than their party's median was then. Missing for members without
    a previous Congress. Using the previous Congress keeps the member's own score out of the votes being explained,
    but Nokken-Poole reuses DW-NOMINATE roll-call parameters fitted on all Congresses, so a little information from
    later Congresses remains.
    """
    frame = covariates[covariates["party"].isin(["D", "R"])].dropna(subset=[column])
    median = frame.groupby(["congress", "chamber", "party"])[column].transform("median")
    direction = np.where(frame["party"] == "D", 1.0, -1.0)
    toward = pd.DataFrame({
        "person": frame["person"], "chamber": frame["chamber"], "party": frame["party"],
        "congress": frame["congress"] + 1, "lagged_toward_other": (frame[column] - median) * direction,
    }).drop_duplicates(["person", "chamber", "party", "congress"])
    merged = covariates.merge(toward, on=["person", "chamber", "party", "congress"], how="left")
    return pd.Series(merged["lagged_toward_other"].to_numpy(), index=covariates.index)
