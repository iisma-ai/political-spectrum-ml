"""Validate the processed tables against Voteview's own totals and official House records."""

import argparse
import sys
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

from build_dataset import RAW_DIR
from voteview import load_members, load_rollcalls, load_votes

ROLLCALL_KEYS = ["congress", "chamber", "rollnumber"]
OFFICIAL_YEA = {"Yea", "Aye"}
OFFICIAL_NAY = {"Nay", "No"}
CLERK_URL = "https://clerk.house.gov/evs/{year}/roll{number:03d}.xml"
MIN_MATCHED_MEMBERS = 300


def report(name: str, passed: bool, detail: str, failures: list[str]) -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    if not passed:
        failures.append(name)


def check_tables(members, rollcalls, votes, failures) -> None:
    duplicates = votes.duplicated(ROLLCALL_KEYS + ["icpsr"]).sum()
    report("unique votes", duplicates == 0, f"{duplicates} duplicates", failures)

    raw = pd.read_csv(RAW_DIR / "HSall_votes.csv", usecols=["congress", "chamber", "icpsr"], dtype={"congress": "int16", "icpsr": "int32"})
    dropped = raw.merge(members[["congress", "chamber", "icpsr"]].drop_duplicates(), how="left", indicator=True)
    dropped = dropped[dropped["_merge"] == "left_only"]
    presidents = pd.read_csv(RAW_DIR / "HSall_members.csv", usecols=["chamber", "icpsr"])
    presidents = set(presidents.loc[presidents["chamber"] == "President", "icpsr"])
    non_presidents = (~dropped["icpsr"].isin(presidents)).sum()
    report("only presidents' rows were dropped", non_presidents == 0, f"{len(dropped):,} dropped, {non_presidents} non-presidents", failures)

    missing = len(votes) - len(votes.merge(rollcalls[ROLLCALL_KEYS], on=ROLLCALL_KEYS))
    report("every vote has a roll call", missing == 0, f"{missing} orphans", failures)

    codes = sorted(votes["cast_code"].unique().tolist())
    report("cast codes", set(codes) <= set(range(1, 10)), str(codes), failures)

    latest = rollcalls["date"].max().date()
    report("date range", rollcalls["date"].min().year == 1789, f"{rollcalls['date'].min().date()} to {latest}", failures)


def voting_positions(tagged: pd.DataFrame) -> pd.DataFrame:
    voting = tagged[~tagged["delegate"]]
    return pd.DataFrame(
        {
            "position_yea": voting["cast_code"].isin([1, 2, 3]).groupby([voting[k] for k in ROLLCALL_KEYS]).sum(),
            "position_nay": voting["cast_code"].isin([4, 5, 6]).groupby([voting[k] for k in ROLLCALL_KEYS]).sum(),
        }
    )


def check_counts(members, rollcalls, votes, failures) -> None:
    """Voteview's yea_count and nay_count hold codes 1 and 6, with delegates counted on only some votes."""
    delegates = members.loc[members["is_delegate"], ["congress", "chamber", "icpsr"]].assign(delegate=True)
    tagged = votes.merge(delegates, on=["congress", "chamber", "icpsr"], how="left")
    tagged["delegate"] = tagged["delegate"].eq(True)
    tagged["yea"] = tagged["cast_code"].eq(1)
    tagged["nay"] = tagged["cast_code"].eq(6)
    everyone = tagged.groupby(ROLLCALL_KEYS)[["yea", "nay"]].sum().add_prefix("all_")
    voting = tagged[~tagged["delegate"]].groupby(ROLLCALL_KEYS)[["yea", "nay"]].sum().add_prefix("voting_")
    joined = rollcalls.set_index(ROLLCALL_KEYS).join(everyone).join(voting).fillna(0)

    with_delegates = joined["all_yea"].eq(joined["yea_count"]) & joined["all_nay"].eq(joined["nay_count"])
    without_delegates = joined["voting_yea"].eq(joined["yea_count"]) & joined["voting_nay"].eq(joined["nay_count"])
    matched = with_delegates | without_delegates
    report("roll-call totals", matched.all(), f"{matched.sum():,} of {len(joined):,} match", failures)

    positions = voting_positions(tagged)
    built = rollcalls.set_index(ROLLCALL_KEYS)[["yea", "nay"]].join(positions).fillna(0)
    consistent = built["yea"].eq(built["position_yea"]) & built["nay"].eq(built["position_nay"])
    report("derived yea and nay", consistent.all(), f"{consistent.sum():,} of {len(built):,} match cast codes", failures)


def check_party_means(members, failures) -> None:
    """Confirm keys and filters by rebuilding Voteview's means, delegates included, from its own scores."""
    published = pd.read_csv(RAW_DIR / "HSall_parties.csv")
    published = published[published["chamber"] != "President"]
    without_mean = published["nominate_dim1_mean"].isna().sum()
    published = published.dropna(subset=["nominate_dim1_mean"])
    computed = (
        members.groupby(["congress", "chamber", "party_code"])
        .agg(mean_dim1=("nominate_dim1", "mean"), n=("icpsr", "count"))
        .reset_index()
    )
    joined = published.merge(computed, on=["congress", "chamber", "party_code"], how="left")
    difference = (joined["mean_dim1"] - joined["nominate_dim1_mean"]).abs()
    largest = difference.max()
    unmatched = joined["mean_dim1"].isna()
    report(
        "party means match Voteview",
        largest < 0.001 and not unmatched.any(),
        f"{len(joined):,} compared, largest difference {largest:.5f}, {unmatched.sum()} unmatched, "
        f"{without_mean} with no published mean",
        failures,
    )


def fetch_house_vote(year: int, number: int) -> ET.Element:
    request = urllib.request.Request(
        CLERK_URL.format(year=year, number=number), headers={"User-Agent": "Mozilla/5.0 (research data validation)"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return ET.fromstring(response.read())


def session_year(congress: int, session: int) -> int:
    return 1789 + 2 * (congress - 1) + session - 1


def official_positions(root: ET.Element) -> dict[str, int]:
    official = {}
    for item in root.iter("recorded-vote"):
        cast = item.find("vote").text
        official[item.find("legislator").get("name-id")] = 1 if cast in OFFICIAL_YEA else -1 if cast in OFFICIAL_NAY else 0
    return official


def compare_rollcall(ours: pd.DataFrame, official: dict[str, int]) -> dict[str, int]:
    # Only codes 1 and 6 are official votes; pairs and announcements are not voting in Clerk files.
    ours = ours.assign(official_vote=ours["vote"].where(ours["cast_code"].isin([1, 6]), 0))
    matched = ours[ours["bioguide_id"].isin(official)]
    expected = matched["bioguide_id"].map(official)
    missing = [member for member, vote in official.items() if vote != 0 and member not in set(ours["bioguide_id"])]
    return {
        "compared": len(matched),
        "yea_nay": int((expected != 0).sum()),
        "disagreements": int((matched["official_vote"] != expected).sum()),
        "missing": len(missing),
    }


def check_official_house(members, rollcalls, sample: int, failures) -> None:
    """Compare individual votes with the House Clerk's records for a sample of roll calls."""
    available = rollcalls[(rollcalls["chamber"] == "House") & rollcalls["clerk_rollnumber"].notna()]
    available = available[available["date"].dt.year >= 2003]
    chosen = available.sample(sample, random_state=0).sort_values("date")
    ids = members[members["chamber"] == "House"][["congress", "icpsr", "bioguide_id"]]

    totals = {"compared": 0, "yea_nay": 0, "disagreements": 0, "missing": 0}
    skipped = 0
    smallest = None
    control = None
    for row in chosen.itertuples():
        year = session_year(row.congress, int(row.session))
        number = int(row.clerk_rollnumber)
        try:
            official = official_positions(fetch_house_vote(year, number))
        except Exception as error:
            print(f"      skipped {row.congress}-{row.rollnumber}: {error!r}")
            skipped += 1
            continue
        ours = load_votes(row.congress, "House")
        ours = ours[ours["rollnumber"] == row.rollnumber].merge(ids[ids["congress"] == row.congress], on=["congress", "icpsr"])
        result = compare_rollcall(ours, official)
        for key in totals:
            totals[key] += result[key]
        smallest = result["compared"] if smallest is None else min(smallest, result["compared"])
        if control is None and row.minority_share >= 0.25:
            control = (ours, year, number)

    report(
        "votes match House Clerk",
        totals["disagreements"] == 0 and totals["missing"] == 0 and skipped == 0 and (smallest or 0) >= MIN_MATCHED_MEMBERS,
        f"{len(chosen)} roll calls, {totals['compared']:,} member records ({totals['yea_nay']:,} Yea or Nay), "
        f"{totals['disagreements']} disagreements, {totals['missing']} missing, {skipped} skipped, "
        f"fewest matched on one roll call {smallest}",
        failures,
    )

    # The comparison must fail when given the wrong roll call, or it proves nothing.
    ours, year, number = control
    for offset in (1, -1, 2):
        wrong = compare_rollcall(ours, official_positions(fetch_house_vote(year, number + offset)))
        if wrong["disagreements"] > 0:
            break
    report("negative control detects a wrong roll call", wrong["disagreements"] > 0, f"{wrong['disagreements']} disagreements", failures)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-sample", type=int, default=0, help="House roll calls to check against the Clerk")
    args = parser.parse_args()

    members, rollcalls, votes = load_members(), load_rollcalls(), load_votes()
    failures: list[str] = []
    check_tables(members, rollcalls, votes, failures)
    check_counts(members, rollcalls, votes, failures)
    check_party_means(members, failures)
    if args.official_sample:
        check_official_house(members, rollcalls, args.official_sample, failures)

    print(f"\n{len(failures)} failed" + (f": {', '.join(failures)}" if failures else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
