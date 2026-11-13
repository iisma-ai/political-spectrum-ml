# Data loading: decisions and documentation

## Question

How do we load, clean and validate European Parliament voting data into a form every problem can use?

## How to run

From `european_parliament/pipeline/data_loading/`:

```
python download.py                  # latest weekly HowTheyVote.eu release; --tag 2026-11-07 for a fixed one
python build_dataset.py
python validate.py --official-sample 200 --describe
python -m pytest test_build_dataset.py
```

`--describe` prints every descriptive number quoted in these docs and in `european_parliament/docs/README.md`.

Problems load data through `howtheyvote.py`: `load(name)` for any table, `load_members`, `load_rollcalls`, `load_votes(term)`, `load_group_memberships`, `load_subjects` and `vote_matrix(term)`.

Data source: [HowTheyVote.eu](https://howtheyvote.eu) (Open Database License). Release used for the results below: 2026-11-07, with a sha256 for each file in `data/raw/manifest.json`.

## Tables

All in `data/processed/`, rebuilt from `data/raw/` in about 20 seconds.

| Table | One row per | Notes |
|---|---|---|
| `rollcalls` | roll call (25,688) | `term`, `rollnumber` (restarting each term, in time order), `timestamp`, `date`, titles, procedure, `vote_part`, published counts, counts from member positions (`n_for`, `n_against`, `n_abstention`), `is_secret`, `minority_share`, `is_lopsided`, `minority_share_with_abstention` |
| `votes` | member and roll call (18,218,348) | `position` (FOR, AGAINST, ABSTENTION, DID_NOT_VOTE), `vote` (1, -1, 0), and the member's `group_code`, `country_code` and `national_party_id` on the day of the vote |
| `members` | member (1,280) | Name, country, date of birth. Contact details and social media links in the export aren't kept |
| `group_memberships` | spell in a political group (1,899) | `start_date`, `end_date` (empty while current), `term` |
| `groups`, `national_parties`, `national_party_memberships`, `countries` | as named | |
| `subjects` | roll call and subject (230,310) | Long format: `scheme` is `oeil_subject` (Legislative Observatory, 24,563 roll calls), `eurovoc` (15,000) or `committee` (responsible committee, 17,660) |

### Vote positions

| Position | `vote` | In For/Against matrices | Share of member rows, 9th term | 10th term |
|---|---|---|---|---|
| FOR | 1 | 1 | 52.8% | 40.3% |
| AGAINST | -1 | -1 | 34.8% | 41.9% |
| ABSTENTION | 0 | missing | 5.0% | 5.0% |
| DID_NOT_VOTE | 0 | missing | 7.4% | 12.8% |

Abstention is an active choice, unlike Did not vote, so `position` keeps them apart. **`vote == 0` merges the two:** use `position` for anything but For/Against matrices. Matrices use For and Against only, like the USA's Yea and Nay.

Did not vote nearly doubles in the 10th term. Whether that is behaviour or a change in recording hasn't been checked, so compare absence between terms with care.

### Near-unanimous roll calls

`minority_share` is the smaller of For and Against over For plus Against, the same rule as the USA, and `is_lopsided` marks those under 2.5%. Abstentions count on neither side. In the European Parliament abstention is more common and is often a group's stance, so `minority_share_with_abstention` counts it with Against, as not supporting the text:

| Term | Near-unanimous, For against Against | With abstention counted against | Near-unanimous only without abstention |
|---|---|---|---|
| 9th | 6.9% | 2.2% | 4.8% |
| 10th | 2.9% | 0.7% | 2.2% |

So the choice changes which roll calls count as contested, mostly ones where a group abstained. The standard stays For against Against, for comparison with the USA. Analyses should report the other as a sensitivity check.

### Vote parts

`vote_part` says what a roll call votes on. It is read from the French `description`, and from the title only when the description is empty (agenda requests and some budget amendments), checked in this order:

| Part | Rule | Share, 9th term | Share, 10th term |
|---|---|---|---|
| procedural | referral back (renvoi), requests (demande de, demande du groupe, request by), opening negotiations, agenda (ordre du jour) | 1.3% | 1.5% |
| final | the export's `is_main` (the vote on a whole text) | 9.4% | 10.9% |
| amendment | "am" or "amendement" | 56.3% | 62.5% |
| text part | a paragraph, recital, citation, annex or article voted separately or in parts (§, considérant, visa, annexe, article, paragraphe, point) | 30.4% | 23.7% |
| other | anything else, such as single votes, decisions and budget lines without a description | 2.8% | 1.4% |

- **Why procedural comes first:** the export marks votes on opening negotiations as main votes in the 10th term only.
- **Why not `amendment_number`:** the export fills it only from 2024, so it can't be used across terms.
- **Hand check:** 120 sampled roll calls had 10 wrong (English agenda titles, amendments named in the title, negotiation votes as final), and those rules were fixed. A new sample of 100, 10 per part and term, then had 1 wrong: a text-part vote named only in its title, labelled other.

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-12 | Download HowTheyVote.eu release assets by tag, and record the tag and each file's sha256 in `data/raw/manifest.json` | Weekly releases; assets can be replaced and the export calls its format experimental |
| 2026-11-12 | Map positions to 1, -1, 0, and keep the position itself | Abstention is an active choice; Did not vote is absence |
| 2026-11-12 | Take the member's group, country and national party from each vote row, not from their current membership | Members change group during a term (121 member-terms); the export records the group on the day |
| 2026-11-12 | Merge the group code `GUE_NGL_1995_0` into `GUE_NGL` | It is the Left group's earlier name (Confederal Group of the European United Left – Nordic Green Left), used only for 3 UK members who left at Brexit |
| 2026-11-12 | Keep the 2 secret ballots in `rollcalls` with `is_secret`, and drop their member rows | Their rows record every member as Did not vote, which isn't a position |
| 2026-11-12 | `minority_share` on For and Against, `is_lopsided` below 2.5%; also store `minority_share_with_abstention` | Same rule as the USA; abstention changes which roll calls are contested here, so it must be tested |
| 2026-11-12 | Assign terms by date: 9th from 2019-07-02, 10th from 2024-07-16 | The first days of the terms in `group_memberships` |
| 2026-11-12 | Derive `vote_part` from the description text, procedural before final | Consistent across both terms, unlike `amendment_number` and `is_main` |
| 2026-11-12 | Check against the Open Data Portal's plenary decisions: positions on a sample, and completeness on every sitting day | Its vote and member IDs match HowTheyVote's. Votes after midnight in the 2020 remote sessions are filed under the day the session began |

## Validation results

Run on 2026-11-12, release 2026-11-07:

| Check | Result |
|---|---|
| Unique member votes; every vote joined to a roll call; every roll call in a term | Passed |
| Member positions against published For, Against and Abstention counts | 25,686 of 25,686 open roll calls match |
| Roll calls without positions | Only the 2 secret ballots; no roll call has missing published counts |
| Group on each vote against the member's group spell on that day | 254,290 of 254,290 member-days agree; none without a spell |
| Members holding a seat listed on every roll call of the day | No member is missing from only some of a day's roll calls. 3 member-days are missing entirely: two new members around the start of their mandate (2024-09-18; 2026-06-15 and 16). Neither appears in the official voter lists on those days either |
| Individual positions against the Open Data Portal | 200 random roll calls (149 from the 9th term, 51 from the 10th): 128,554 official positions, 0 disagreements, 0 missing, 0 extra. With no errors in 200, the 95% upper bound on the share of roll calls with an error is about 1.5% |
| Negative control | Another roll call's positions produce 572 differences, so the official check can fail |
| Completeness against the Open Data Portal | All 357 sitting days: all 25,680 official roll calls with voter lists are in ours |

**What this covers:** the count, group and membership checks compare the export with itself. The position and completeness checks are external.

- **The 6 extra roll calls:** ours has 25,686 open roll calls against the portal's 25,680. Five have no official record (four on 2021-02-09, one on 2025-03-12), and one has a record without voter lists (2024-04-23). They are kept.
- **What the external checks show:** HowTheyVote reads the roll-call results in the plenary minutes, and the portal publishes the same official records. So the checks confirm the export matches the Parliament's records, not that those records are free of errors.

**Not checked:** votes corrected after the vote (HowTheyVote doesn't collect corrections), roll-call metadata (`is_main`, `result`, `majority_type`, procedure fields), subjects, and national party memberships.

**Timestamps** are in a fixed UTC+1. Checked on all 25,680 roll calls in the portal, they equal official Brussels time in winter and are one hour behind in summer. A vote just after midnight in summer can get the previous day's date. Neither matters for splitting by term or year.

Reviewed by `method-reviewer` on 2026-11-12. Its findings were fixed or recorded here before the foundation was marked done.

## Documentation

- [Data sources](../../../docs/datasets/sources.md)
