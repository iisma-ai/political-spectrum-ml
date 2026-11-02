# Data loading: decisions and documentation

## Question

How do we load, clean and validate US voting data into a form every problem can use?

## How to run

From `usa/pipeline/data_loading/`:

```
python download.py
python build_dataset.py
python validate.py --official-sample 40
python -m pytest test_build_dataset.py
```

Problems load data through `voteview.py`: `load_members`, `load_rollcalls`, `load_votes` and `vote_matrix`.

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-02 | Start from the Voteview votes, rollcalls, members and parties CSVs | One consistent source across all Congresses |
| 2026-11-02 | Map cast codes 1 to 3 to Yea, 4 to 6 to Nay, and treat 7 to 9 as neither | Absence is not a Nay |
| 2026-11-02 | Keep only IDs listed as chamber members | Presidents' positions are stored under House and Senate rows |
| 2026-11-02 | Flag territorial delegates and exclude them from matrices by default | They aren't full voting members, and Voteview counts them inconsistently |
| 2026-11-02 | Exclude roll calls where the smaller side is under 2.5% by default | They carry little information about divisions; the share includes paired and announced positions, and the threshold is a choice that must be tested |
| 2026-11-02 | Keep Voteview's ICPSR as the vote key, and use Bioguide ID to follow people | A party switch creates a new ICPSR; 41 of 12,399 people have more than one |
| 2026-11-02 | Use `party` values D, R and Other from `party_code` | Independents, such as those caucusing with a party, stay in Other |
| 2026-11-02 | Keep raw and processed data in `data/`, outside git | Files are large and can be rebuilt |

## Validation results

Run on 2026-11-02:

| Check | Result |
|---|---|
| Unique votes, and all votes joined to roll calls | Passed |
| Rows dropped by the build | 79,586, all presidents |
| Roll-call totals against Voteview's counts | 113,568 of 113,568 match, with delegates either included or excluded |
| Derived `yea` and `nay` against cast codes | 113,568 of 113,568 match |
| Party means against Voteview's published values | All 727 with a published mean match, delegates included |
| Individual votes against House Clerk records | 40 random roll calls from 2003 on: 17,310 member records, 16,684 of them Yea or Nay, 0 disagreements, 0 missing |
| Negative control | A wrong roll call produces disagreements, so the Clerk check can fail |

**What this covers:** most checks compare Voteview with itself, so they show internal consistency. The only external check is the House sample. With 40 clean roll calls, some roll calls could still contain errors. It also uses a single fixed random seed.

**Not externally confirmed:** individual Senate votes, because `senate.gov` refused automated requests, and all votes before 2003.

Reviewed by `method-reviewer` on 2026-11-02. Its findings were fixed before conclusions were recorded.

## Documentation

- [US voting data sources](../../../docs/datasets/us_voting_sources.md)
