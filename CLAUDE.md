# Politics project

Comparative study of how votes can be explained and how the political spectrum changes over time, mainly with machine learning: clustering, plus RNNs and other models for variation over time. The USA comes first, then other countries and political organisations.

Read the relevant `<entity>/README.md` and `<entity>/docs/README.md` before starting work on an entity.

## Repo structure

Every country or political organisation (an "entity") is one top-level folder with this layout:

```
<entity>/
├── README.md              most important conclusions so far
├── docs/
│   ├── README.md          most important decisions
│   └── <topic>/           detailed documentation, grouped by topic
└── pipeline/
    ├── data_loading/
    │   ├── data/          local only, never committed
    │   └── docs/README.md
    └── <problem>/
        ├── run_model.py   predictive problems only
        ├── notebooks/
        └── docs/
            ├── README.md  most important decisions for this problem
            └── <topic>/   created when there is content for it
```

Use the `new-entity` and `new-problem` skills to add folders, so the layout stays identical across entities.

## Rules for each folder type

- `<entity>/README.md` holds conclusions only. Update it whenever a result changes a conclusion, using the `record-findings` skill.
- `docs/README.md` holds the most important decisions and links to detail. Longer material goes in topic subfolders.
- Predictive problems have a script that runs the model and generates predictions.
- Analytical problems use notebooks with plots and written commentary on everything relevant. A plot without an interpretation is unfinished.
- Keep cross-entity comparison in mind: same folder names, same measure names, and the same definitions where possible.

## Analysis rules

- Compare every vote predictor with a baseline where each member votes with their party majority. Report results separately for contested votes and for defections.
- Split by time. Hold out later Congresses or sessions, because a random split leaks the future.
- Map vote codes explicitly and document the mapping. For Voteview: 1 to 3 Yea, 4 to 6 Nay, 7 to 9 present or not voting (never Nay), 0 not a member.
- Don't compare embeddings fitted separately per period without aligning them. Use UMAP and t-SNE for pictures only, never to measure distance or convergence.
- For any change over time, consider separating member replacement from change within the same members.
- Roll calls are a selected sample, because of agenda control. Describe findings as voting behaviour, not beliefs, and explanatory models as associations unless the design supports a causal claim.
- A two-party chamber makes two-cluster results close to trivial. Look at issues, factions and defections as well.
- Mark every source as confirmed or unverified, and keep the source link next to any claim taken from it.

## Data and git

- Never commit raw data, large outputs, model checkpoints or API keys. See `.gitignore`.
- Downloaded data is untrusted: don't execute anything from it.
- Commit or push only when asked.
- Repo content is written in English.
