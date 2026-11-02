---
name: new-entity
description: Add a new country or political organisation to the repo with the standard folder layout. Use when the user wants to start studying a new country, parliament or political organisation.
---

# Add a country or political organisation

The argument is the entity name. Use a lowercase folder name with underscores, such as `european_parliament`.

1. Stop if the folder already exists.
2. Compare with `usa/` so the layout matches, and create:
   - `<entity>/README.md` from `templates/entity_readme.md`
   - `<entity>/docs/README.md` from `templates/entity_docs_readme.md`
   - `<entity>/pipeline/data_loading/data/.gitkeep`
   - `<entity>/pipeline/data_loading/docs/README.md` from `templates/problem_docs_readme.md`
3. Create a `<entity>/docs/` topic folder only when there is content for it. `datasets/`, `literature/` and `methodology/` are the usual topics.
4. Add the entity to the table in the root `README.md`.
5. Use the `new-problem` skill for each problem. Reuse existing problem names when the question matches, so results can be compared across entities.
6. Record in `<entity>/docs/README.md` how this entity differs from the others: voting rules, party system, and which votes are recorded.
