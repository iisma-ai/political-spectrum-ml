"""Check the processed European Parliament tables for internal consistency and against official records.

The official check compares individual positions on a random sample of roll calls with the European Parliament's
Open Data Portal (https://data.europarl.europa.eu), which lists the members voting For, Against and Abstention
on each roll call. HowTheyVote's vote IDs are the portal's `notation_votingId`, and its member IDs are the
portal's person IDs. It also checks, for every sitting day fetched, that no official roll call is missing from ours.

`--describe` prints the descriptive numbers quoted in the docs.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import LOPSIDED_SHARE
from howtheyvote import load, load_votes

API = "https://data.europarl.europa.eu/api/v2/meetings/MTG-PL-{date}/decisions"
OFFICIAL_DIR = Path(__file__).parent / "data" / "official"
OFFICIAL_LISTS = {"had_voter_favor": "FOR", "had_voter_against": "AGAINST", "had_voter_abstention": "ABSTENTION"}
PAGE = 500


def report(name: str, passed: bool, detail: str, failures: list[str]) -> None:
    print(f"{'PASS' if passed else 'FAIL'}  {name}: {detail}")
    if not passed:
        failures.append(name)


def check_tables(rollcalls, votes, failures) -> None:
    duplicates = votes.duplicated(["vote_id", "member_id"]).sum()
    report("unique member votes", duplicates == 0, f"{duplicates} duplicates", failures)
    orphans = (~votes["vote_id"].isin(rollcalls["vote_id"])).sum()
    report("votes joined to roll calls", orphans == 0, f"{orphans} without a roll call", failures)
    no_term = rollcalls["term"].isna().sum()
    report("every roll call in a term", no_term == 0, f"{no_term} without a term", failures)
    open_ballots = rollcalls[~rollcalls["is_secret"]]
    empty = (~open_ballots["vote_id"].isin(votes["vote_id"])).sum()
    report("every open roll call has member votes", empty == 0, f"{empty} without", failures)


def check_counts(rollcalls, failures) -> None:
    open_ballots = rollcalls[~rollcalls["is_secret"]]
    match = (
        (open_ballots["n_for"] == open_ballots["count_for"])
        & (open_ballots["n_against"] == open_ballots["count_against"])
        & (open_ballots["n_abstention"] == open_ballots["count_abstention"])
    )
    report("member positions add up to published counts", bool(match.all()),
           f"{match.sum():,} of {len(match):,} roll calls match", failures)
    # A roll call with published counts but no positions would be a loss of member rows, not a secret ballot.
    no_positions = rollcalls[["n_for", "n_against", "n_abstention"]].sum(axis=1).eq(0)
    unexplained = (no_positions & ~rollcalls["is_secret"]).sum()
    missing_counts = rollcalls[["count_for", "count_against", "count_abstention"]].isna().any(axis=1).sum()
    report("secret ballots are the only roll calls without positions", unexplained == 0 and missing_counts == 0,
           f"{rollcalls['is_secret'].sum()} secret ballots; {unexplained} other roll calls without positions; "
           f"{missing_counts} with missing published counts", failures)


def member_days(votes, rollcalls) -> pd.DataFrame:
    """One row per member, sitting day and recorded group."""
    days = votes[["vote_id", "member_id", "group_code"]].merge(rollcalls[["vote_id", "date"]], on="vote_id")
    days["group_code"] = days["group_code"].astype(str)
    return days.drop_duplicates(["member_id", "date", "group_code"])[["member_id", "date", "group_code"]]


def spell_on(days: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    """The group spell covering each member-day; the latest-starting spell if two cover it."""
    joined = days.merge(spells[["member_id", "group_code", "start_date", "end_date"]], on="member_id",
                        suffixes=("", "_spell"))
    covering = joined[(joined["date"] >= joined["start_date"])
                      & (joined["end_date"].isna() | (joined["date"] <= joined["end_date"]))]
    return covering.sort_values("start_date").drop_duplicates(["member_id", "date", "group_code"], keep="last")


def check_groups(votes, rollcalls, spells, failures) -> None:
    """The group recorded on each vote should be the member's group spell on that day."""
    days = member_days(votes, rollcalls)
    covered = spell_on(days, spells)
    uncovered = len(days) - len(covered)
    agree = (covered["group_code"] == covered["group_code_spell"]).sum()
    report("group on each vote matches the membership spell", uncovered == 0 and agree == len(covered),
           f"{agree:,} of {len(days):,} member-days agree; {uncovered} without a covering spell", failures)


def check_membership(votes, rollcalls, spells, failures) -> None:
    """Members holding a seat should be listed on every open roll call of the day.

    A member listed on some of a day's roll calls but not others means lost rows, so that fails. A member left out
    of a whole day is reported: it happened for new members around the start of their mandate.
    """
    open_ballots = rollcalls.loc[~rollcalls["is_secret"], ["vote_id", "date"]]
    sitting = pd.DataFrame({"date": open_ballots["date"].unique()})
    holding = sitting.merge(spells[["member_id", "start_date", "end_date"]], how="cross")
    holding = holding[(holding["date"] >= holding["start_date"])
                      & (holding["end_date"].isna() | (holding["date"] <= holding["end_date"]))]
    holding = holding[["member_id", "date"]].drop_duplicates()
    rows = votes[["vote_id", "member_id"]].merge(open_ballots, on="vote_id")
    listed = rows.groupby(["member_id", "date"]).size().rename("listed").reset_index()
    per_day = open_ballots.groupby("date").size().rename("rollcalls")
    days = holding.merge(listed, on=["member_id", "date"], how="left").join(per_day, on="date")
    days["listed"] = days["listed"].fillna(0)
    partial = days[(days["listed"] > 0) & (days["listed"] < days["rollcalls"])]
    whole = days[days["listed"] == 0]
    members = votes.groupby("vote_id").size()
    report("members holding a seat are listed on all or none of a day's roll calls", partial.empty,
           f"{len(partial)} member-days listed on only some roll calls; {len(whole)} member-days left out entirely "
           f"({sorted(zip(whole['member_id'], whole['date'].dt.strftime('%Y-%m-%d')))}); "
           f"members per roll call {members.min()} to {members.max()}", failures)


def fetch_day(date: str) -> list[dict]:
    """All decisions of one plenary day from the Open Data Portal, cached under data/official/."""
    OFFICIAL_DIR.mkdir(parents=True, exist_ok=True)
    cache = OFFICIAL_DIR / f"decisions_{date}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    # On 2026-11-12 the portal ignored `limit` and returned the whole day on every page, so stop when a page adds
    # nothing new rather than when it is short.
    records: dict[str, dict] = {}
    offset = 0
    while True:
        url = f"{API.format(date=date)}?format=application%2Fld%2Bjson&offset={offset}&limit={PAGE}"
        request = urllib.request.Request(url, headers={"Accept": "application/ld+json",
                                                       "User-Agent": "political-spectrum-ml research script"})
        with urllib.request.urlopen(request, timeout=120) as response:
            page = json.load(response).get("data", [])
        new = {record["id"]: record for record in page if record.get("id") not in records}
        records.update(new)
        if len(page) < PAGE or not new:
            break
        offset += PAGE
        time.sleep(1)
    records = list(records.values())
    cache.write_text(json.dumps(records), encoding="utf-8")
    time.sleep(1)
    return records


def official_positions(record: dict) -> dict[int, str]:
    return {int(person.rsplit("/", 1)[-1]): position
            for key, position in OFFICIAL_LISTS.items() for person in record.get(key, [])}


def compare(ours: pd.DataFrame, official: dict[int, str]) -> dict[str, int]:
    positions = dict(zip(ours["member_id"], ours["position"].astype(str)))
    active = {member for member, position in positions.items() if position != "DID_NOT_VOTE"}
    return {
        "official": len(official),
        "disagree": sum(positions.get(member) != position for member, position in official.items() if member in positions),
        "missing": sum(member not in positions for member in official),
        "extra": len(active - set(official)),
    }


def check_official(rollcalls, votes, sample: int, failures) -> None:
    chosen = rollcalls[~rollcalls["is_secret"]].sample(sample, random_state=0)
    sitting_days = sorted(rollcalls["date"].unique())
    totals = {"official": 0, "disagree": 0, "missing": 0, "extra": 0}
    found = {}
    for vote_id, date in zip(chosen["vote_id"], chosen["date"]):
        # Votes after midnight, as in the 2020 remote sessions, are filed under the sitting day that began before it.
        previous = sitting_days[max(sitting_days.index(date) - 1, 0)]
        for day in (date, previous):
            records = {r.get("notation_votingId"): r for r in fetch_day(pd.Timestamp(day).strftime("%Y-%m-%d"))}
            if str(vote_id) in records:
                found[vote_id] = records[str(vote_id)]
                break
        if vote_id not in found:
            continue
        result = compare(votes[votes["vote_id"] == vote_id], official_positions(found[vote_id]))
        for key in totals:
            totals[key] += result[key]
    clean = len(found) == sample and totals["disagree"] == totals["missing"] == totals["extra"] == 0
    # Rule of three: with no errors in n roll calls, the 95% upper bound on the share with an error is about 3/n.
    report("individual positions against the Open Data Portal", clean,
           f"{sample} random roll calls, {len(found)} found; {totals['official']:,} official positions, "
           f"{totals['disagree']} disagreements, {totals['missing']} missing from ours, "
           f"{totals['extra']} in ours but not official; 95% upper bound on roll calls with an error "
           f"{3 / sample:.1%}", failures)

    # Negative control: another roll call's positions must disagree, or the check can't fail.
    first, second = list(found)[:2]
    wrong = compare(votes[votes["vote_id"] == second], official_positions(found[first]))
    report("negative control (wrong roll call)", wrong["disagree"] + wrong["missing"] + wrong["extra"] > 0,
           f"{wrong['disagree']} disagreements, {wrong['missing']} missing, {wrong['extra']} extra", failures)


def check_complete(rollcalls, failures) -> None:
    """Every official roll call with voter lists, on every cached sitting day, should be in ours."""
    ours = set(rollcalls["vote_id"].astype(str))
    official, missing, days = 0, [], 0
    for path in sorted(OFFICIAL_DIR.glob("decisions_*.json")):
        days += 1
        for record in json.loads(path.read_text(encoding="utf-8")):
            if not any(record.get(key) for key in OFFICIAL_LISTS):
                continue
            official += 1
            if record.get("notation_votingId") not in ours:
                missing.append((path.stem.removeprefix("decisions_"), record.get("notation_votingId")))
    report("no official roll call missing from ours (cached days)", not missing,
           f"{days} sitting days, {official:,} official roll calls with voter lists, {len(missing)} missing"
           + (f": {missing[:10]}" if missing else ""), failures)


def describe(rollcalls, votes, spells) -> None:
    """Numbers quoted in the docs."""
    open_ballots = rollcalls[~rollcalls["is_secret"]]
    print("\nRoll calls by term and vote part (share):")
    print(pd.crosstab(open_ballots["term"], open_ballots["vote_part"], normalize="index").round(3).to_string())
    print("\nPositions by term (share of member rows):")
    print(pd.crosstab(votes["term"], votes["position"], normalize="index").round(3).to_string())

    lopsided = open_ballots["is_lopsided"]
    lopsided_with = ~(open_ballots["minority_share_with_abstention"] >= LOPSIDED_SHARE)
    print(f"\nNear-unanimous (smaller side under {LOPSIDED_SHARE:.1%}) by term, For against Against, and with abstention "
          "counted against:")
    print(pd.DataFrame({"For/Against": lopsided, "with abstention": lopsided_with,
                        "lopsided only without abstention": lopsided & ~lopsided_with,
                        "term": open_ballots["term"]}).groupby("term").mean().round(3).to_string())

    switches = spells.groupby(["member_id", "term"])["group_code"].nunique()
    print(f"\nMember-terms with more than one group (after merging GUE_NGL_1995_0): {(switches > 1).sum()} "
          f"({(switches > 1).groupby('term').sum().to_dict()} by term)")
    print("Groups per term:", spells.groupby("term")["group_code"].unique().map(sorted).to_dict())
    last9 = spells[spells["term"] == 9].sort_values("start_date").groupby("member_id")["group_code"].last()
    first10 = spells[spells["term"] == 10].sort_values("start_date").groupby("member_id")["group_code"].first()
    both = pd.concat([last9.rename("9th term, last"), first10.rename("10th term, first")], axis=1).dropna()
    print(f"\nRe-elected members ({len(both)}), last group in the 9th term against first group in the 10th:")
    print(pd.crosstab(both["9th term, last"], both["10th term, first"]).to_string())

    subjects = load("subjects")
    print("\nRoll calls with each subject scheme:", subjects.groupby("scheme")["vote_id"].nunique().to_dict())

    rows = []
    for path in OFFICIAL_DIR.glob("decisions_*.json"):
        for record in json.loads(path.read_text(encoding="utf-8")):
            if record.get("notation_votingId") and record.get("activity_start_date"):
                rows.append((int(record["notation_votingId"]), record["activity_start_date"]))
    if rows:
        official = pd.DataFrame(rows, columns=["vote_id", "official"]).drop_duplicates("vote_id")
        official = official.merge(rollcalls[["vote_id", "timestamp"]], on="vote_id")
        offset = official["official"].str[19:]
        hours = ((official["timestamp"] - pd.to_datetime(official["official"].str[:19])).dt.total_seconds() / 3600)
        print(f"\nOur timestamp minus official local time, hours, by official UTC offset ({len(official):,} roll calls):")
        print(pd.crosstab(offset, hours.round(2)).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-sample", type=int, default=0,
                        help="number of random roll calls to check against the Open Data Portal (0 skips it)")
    parser.add_argument("--describe", action="store_true", help="print the descriptive numbers quoted in the docs")
    args = parser.parse_args()

    rollcalls = load("rollcalls")
    votes = load_votes()
    spells = load("group_memberships")
    failures: list[str] = []
    check_tables(rollcalls, votes, failures)
    check_counts(rollcalls, failures)
    check_groups(votes, rollcalls, spells, failures)
    check_membership(votes, rollcalls, spells, failures)
    if args.official_sample:
        check_official(rollcalls, votes, args.official_sample, failures)
    if OFFICIAL_DIR.exists():
        check_complete(rollcalls, failures)
    print("all checks passed" if not failures else f"failed: {', '.join(failures)}")
    if args.describe:
        describe(rollcalls, votes, spells)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
