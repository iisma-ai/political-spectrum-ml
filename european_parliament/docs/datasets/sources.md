# European Parliament data sources

Checked 2026-11-12.

| Source | Covers | Use | Status |
|---|---|---|---|
| [HowTheyVote.eu export](https://github.com/HowTheyVote/data) | Plenary roll calls from the 9th term (July 2019) on, with each member's position, political group and national party on the day of the vote; subjects and responsible committees. Updated weekly | Primary source | Confirmed ([about](https://howtheyvote.eu/about), [data and license](https://howtheyvote.eu/data)) |
| [European Parliament Open Data Portal](https://data.europarl.europa.eu) | Official plenary decisions, with the members voting For, Against and Abstention on each roll call | External check of individual positions on a sample, and of completeness on every sitting day (`validate.py`) | Confirmed: vote IDs and member IDs match HowTheyVote (checked 2026-11-12) |
| [europarl.europa.eu roll-call XML](https://www.europarl.europa.eu/doceo/document/) | Roll-call annexes to the plenary minutes | Not used | Refuses automated requests (HTTP 403, 2026-11-12) |
| [VoteWatch data](https://cadmus.eui.eu/handle/1814/74918) | European Parliament 2004 to 2022 | Possible extension to earlier terms | Page found 2026-11-02; data not downloaded or inspected (unverified) |

## HowTheyVote.eu

- **License:** Open Database License, with individual contents under the Database Contents License; MEP photos and vote summaries excluded ([source](https://howtheyvote.eu/data)). Attribution required: "Data source: HowTheyVote.eu (Open Database License)". A shared derived database must stay under the ODbL. This repo doesn't store the data.
- **Which votes:** "Currently, only roll-call votes can be found on HowTheyVote.eu. Other types of votes, such as secret votes or simple votes by show of hands cannot be found" ([source](https://howtheyvote.eu/about)). Two secret ballots from September 2019 are in the export anyway, with no member positions.
- **Positions:** For, Against, Abstention, or Did not vote. The first three are choices the member made; Did not vote covers absence, with or without excuse ([source](https://howtheyvote.eu/data)). Vote corrections made after the vote aren't collected ([source](https://howtheyvote.eu/about)).
- **Groups:** `member_votes.group_code` is the member's group on the day of the vote ([source](https://howtheyvote.eu/data)).
- **Official sources used by HowTheyVote:** roll-call results in the plenary minutes, the Legislative Observatory and press releases ([source](https://howtheyvote.eu/about)).
- **Format warning:** "The dataset is experimental and the format of the tables may change in the future" ([source](https://github.com/HowTheyVote/data/releases/latest)). `download.py` records the release tag in `data/raw/manifest.json`.
