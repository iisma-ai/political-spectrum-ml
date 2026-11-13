# European Parliament: decisions and documentation

## Roadmap

Same steps as the [USA](../../usa/docs/README.md#roadmap), with the same problem names so results can be compared.

1. **Foundation (done 2026-11-12):** download HowTheyVote.eu, build the member-by-vote tables, and check them against the official Open Data Portal.
2. **Clustering (done 2026-11-13):** cohesion by political group, coalitions between groups, member agreement networks and maps per term ([decisions](../pipeline/polarization_clustering/docs/README.md)).
3. **Explanation (next):** defection from the political group and from the national party, by subject, country and group.
4. **Prediction:** the same ladder as the USA (group line, own past rate, logistic regression, gradient boosting, GRU).

## Most important decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-12 | Use HowTheyVote.eu as the primary source | Every plenary roll call since July 2019 with each member's position, group and national party on the day; open license; weekly updates |
| 2026-11-12 | Check individual positions against the European Parliament's Open Data Portal | Official, machine-readable, and uses the same vote and member IDs; the europarl.europa.eu roll-call XML refuses automated requests |
| 2026-11-12 | Treat the political group as the "party" for party-line baselines, and keep national party alongside | Groups organise plenary voting, but members are elected on national lists, so both lines matter |
| 2026-11-12 | Define the group line over three options (For, Against, Abstention): the group's most common position, leaving the member out, with defection meaning a different active position. Report the For/Against version as secondary | A group's line can be to abstain, which a For/Against line would miss |
| 2026-11-12 | Leave non-attached members out of group-line baselines and report them separately | They have no group whip, so a "non-attached majority" isn't a line |
| 2026-11-12 | Keep the term as the main period, and start roll numbers again in each term | Terms are the counterpart of Congresses; group composition changes between them |
| 2026-11-12 | Use the same near-unanimous rule as the USA (smaller of For and Against under 2.5%), and test abstention counted against as a sensitivity check | Comparable across entities. But here the choice matters: in the 9th term 6.9% of roll calls are near-unanimous on For against Against and 2.2% with abstention counted against |

## How this entity differs

Numbers marked "from the data" are printed by `pipeline/data_loading/validate.py --describe`.

- **Party system:** multi-party. Seven political groups in the 9th term and eight in the 10th, plus non-attached members, each made of national parties (from the data). Members are elected in national elections. 121 member-terms show a change of group within a term, 100 in the 9th and 21 in the 10th so far (from the data). That no group holds a majority alone is expected from group sizes but hasn't been checked against seat totals here.
- **Groups change between terms:** Identity and Democracy (9th term) no longer exists in the 10th, which has Patriots for Europe and Europe of Sovereign Nations instead. Of 23 re-elected Identity and Democracy members, 21 started the 10th term in Patriots for Europe, 1 in Europe of Sovereign Nations and 1 in ECR; Patriots for Europe also drew 4 from ECR, 3 from Renew and 5 non-attached (from the data: 341 re-elected members). Cross-term comparisons of groups need an explicit mapping.
- **Which votes are recorded:** only roll calls, which are taken on request, so they are a selected sample of plenary votes. Show-of-hands and secret votes aren't in the data ([source, confirmed](https://howtheyvote.eu/about)). Most roll calls are on amendments and parts of texts, not final votes (about 10% are final votes).
- **Three positions:** For, Against and Abstention, all active choices, plus Did not vote ([source, confirmed](https://howtheyvote.eu/data)). Abstention is 5% of member positions in both terms (from the data), so it may carry meaning that For/Against matrices drop.
- **Short record:** two terms, about seven years, against more than 160 years for the USA. Changes over time can be studied only within this window.
- **Size:** 751 seats until the UK left on 31 January 2020, 705 after, 720 in the 10th term (unverified against an official source). Members listed per roll call range from 696 to 751, never above these numbers (from the data).

## Documentation

- [Data sources](datasets/sources.md)
- [Data loading](../pipeline/data_loading/docs/README.md)
- [Polarization clustering](../pipeline/polarization_clustering/docs/README.md)
