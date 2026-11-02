---
name: new-problem
description: Add a new problem or type of analysis to an entity's pipeline with the standard folder layout. Use when the user wants to start a new analysis, clustering study, explanation or prediction task.
---

# Add a problem to a pipeline

The arguments are the entity and problem name. Use a lowercase folder name with underscores. If the same question exists in another entity, reuse its name.

1. Stop if `<entity>/pipeline/<problem>/` already exists.
2. Decide the type:
   - **Predictive:** create `run_model.py`, which runs the model and generates predictions, plus `notebooks/` for evaluation.
   - **Analytical:** create `notebooks/`, for plots and written commentary.
3. Create `docs/README.md` from `../new-entity/templates/problem_docs_readme.md`, and state the question clearly.
4. Add `.gitkeep` to any empty folder. Add a `docs/` topic folder only when there is content for it.
5. Add the problem to the table in `<entity>/README.md`.
6. Link the problem's `docs/README.md` from `<entity>/docs/README.md`.
