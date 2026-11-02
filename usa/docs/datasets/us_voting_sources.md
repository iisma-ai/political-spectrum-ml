# US voting data sources

Researched on 2026-11-02. This list can't be proven complete; it covers what the search found.

Status: **confirmed** means the source's own page was read or found. **Unverified** means the link or detail still needs checking.

## Core roll-call data

| Source | Coverage | Access | Use | Status |
|---|---|---|---|---|
| [Voteview](https://voteview.com/data) | House and Senate, 1st to 119th Congress | CSV and JSON, updated live; full database `/static/db/current.zip` (about 500 MB, weekly) | Primary source | Confirmed |
| [Congress.gov API, House votes](https://blogs.loc.gov/law/2025/08/congress-gov-enhanced-access-to-house-roll-call-votes-is-now-available/) | House only, 2023 onward, beta | API | Official recent House data | Confirmed |
| [House Clerk vote XML](https://clerk.house.gov/evs/2023/roll100.xml) | Individual House votes, listing members by Bioguide ID | XML, `clerk.house.gov/evs/{year}/roll{number}.xml` | Validating member votes | Confirmed, used in `validate.py` |
| [Senate XML](https://www.senate.gov/general/XML.htm) | Vote lists and individual votes | XML | Official Senate data | Direct requests and web fetches returned 403 on 2026-11-02; range unverified |
| [House roll-call XML dictionary](https://xml.house.gov/rollcall/rollcall-vote.HTML) | Schema reference | Docs | Parsing official files | Confirmed |
| [unitedstates/congress](https://github.com/unitedstates/congress) | Bills, amendments, votes, bill text | Python scrapers, JSON and XML | Bill text and metadata | Recent activity unverified |

## Voteview files

All four files were downloaded on 2026-11-02, with data through 2026-09-30. `download.py` records the source dates in `data/raw/manifest.json`.

- `https://voteview.com/static/data/out/votes/HSall_votes.csv` (702 MB, about 26.4 million rows)
  - Columns: `congress, chamber, rollnumber, icpsr, cast_code, prob`
- `https://voteview.com/static/data/out/rollcalls/HSall_rollcalls.csv` (113,568 roll calls)
  - Columns: `congress, chamber, rollnumber, date, session, clerk_rollnumber, majority_requirement, yea_count, nay_count, nominate_mid_1, nominate_mid_2, nominate_spread_1, nominate_spread_2, nominate_log_likelihood, bill_number, vote_result, vote_desc, vote_question, dtl_desc`
  - `issue_codes`, `crs_policy_area`, `clausen_codes` and `peltzman_codes` are [JSON only](https://voteview.com/articles/data_help_rollcalls).
- `https://voteview.com/static/data/out/members/HSall_members.csv`, which also holds DW-NOMINATE and Nokken-Poole scores
- `https://voteview.com/static/data/out/parties/HSall_parties.csv`, with published party means used for validation

[Vote codes](https://voteview.com/articles/data_help_votes):

| Code | Meaning | Project mapping |
|---|---|---|
| 0 | Not a member when the vote was taken | Excluded |
| 1, 2, 3 | Yea, paired Yea, announced Yea | Yea |
| 4, 5, 6 | Announced Nay, paired Nay, Nay | Nay |
| 7, 8 | Present | Not a Yea or Nay |
| 9 | Not voting | Not a Yea or Nay |

## Quirks found in the Voteview data

- **Presidents appear inside vote rows.** The votes file has 79,586 rows under `House` or `Senate` that belong to presidents' IDs. Filtering `chamber = President` doesn't remove them, so the build keeps only IDs listed as members of that chamber.
- **Duplicates.** 152 duplicate rows, all in the 115th Congress, disappear with that filter.
- **`yea_count` and `nay_count` hold only codes 1 and 6.** They leave out paired and announced positions.
- **Delegates are counted inconsistently.** In the House, totals include territorial delegates on some roll calls but not others, all from the 111th Congress on.
- **DW-NOMINATE scores are fixed for each member.** No member's `nominate_dim1` changes between Congresses, while Nokken-Poole scores do vary.
- **Code 0 didn't appear** in the current votes file, though the documentation lists it.

## Sources not to use

- [ProPublica Congress API](https://projects.propublica.org/api-docs/congress-api/sunlight/): no longer available.
- [GovTrack bulk data and API](https://congressionaldata.org/ending-govtracks-bulk-data-and-api/): ended.
- [Legacy Voteview pages](https://legacy.voteview.com/dw-nominate.htm): outdated formats.

## Data for explaining votes

| Source | Adds | Status |
|---|---|---|
| [Policy Agendas roll-call votes](https://www.maxwell.syr.edu/research/campbell-public-affairs-institute/united-states-policy-agendas-project/datasets-and-codebooks/roll-call-votes) | Topic code for every roll call | Page found, files unverified |
| [Congressional Bills Project](https://www.maxwell.syr.edu/research/campbell-public-affairs-institute/united-states-policy-agendas-project/datasets-and-codebooks/congressional-bills) | Bill sponsors, topics, progress | Page found, files unverified |
| Congressional Record parsed speeches, Gentzkow, Shapiro, Taddy (43rd to 114th Congress) | Speech text and phrase counts | Unverified: citation gives `data.stanford.edu/congress_text` |
| [ICPSR 33501](https://www.icpsr.umich.edu/web/ICPSR/studies/33501/staff) | Congressional Record text, 104th to 110th | Page found |
| [Nokken-Poole scores](https://legacy.voteview.com/Nokken-Poole.htm) | Member positions that vary by Congress | Confirmed, in `HSall_members.csv` |
| DIME (donor-based ideology, CFscores) | A measure of ideology that doesn't use votes | Unverified: search failed; possibly `data.stanford.edu/dime` |
