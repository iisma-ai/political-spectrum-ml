---
name: record-findings
description: Update an entity's conclusions and decision docs after an analysis or model run. Use when a result is finished, when a decision is made, or when a result changes an earlier conclusion.
---

# Record findings

1. Identify what changed: a new result, a changed conclusion, a decision, or a method limit.
2. For a result that changes a conclusion, get a check from the `method-reviewer` agent first.
3. Update `<entity>/README.md`:
   - Keep only the most important conclusions, and rewrite old ones instead of appending a log.
   - State each conclusion with its key number, its period, and a link to the notebook or script.
   - Include negative results and limits that affect how it should be read.
   - Update "Last updated" and the problem table.
4. Add lasting decisions to the problem's `docs/README.md`, or to `<entity>/docs/README.md` if they affect several problems. Include the date and the reason.
5. Put longer explanations in a `docs/` topic folder and link to them.
6. If a change affects cross-entity comparison, note it in the root `README.md`.

Only record what the notebooks or scripts actually show. Label expectations and untested ideas as such.
