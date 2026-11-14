"""Train the US vote prediction models on earlier Congresses and generate predictions for later ones.

Each model predicts the probability that a member votes against their own party's position on a roll call. A vote
prediction follows from it: the party position, flipped when that probability is above one half. Training uses the
102nd to 111th Congresses (1991 to 2010), model choices use the 112th and 113th (2011 to 2014), and the 114th to
118th (2015 to 2024) are the test set, used once. Outputs go to outputs/, which is not committed.

Two settings: one roll call ahead (history includes earlier roll calls on the same day) and day-ahead (models ending
in _day: history only up to the member's previous day with votes, as if all of a day's predictions were made before
its first roll call).
"""

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from features import SEQUENCE_KEYS, add_day_history, load_rows
from sequence_model import DefectionGRU, SequenceBatches, fit_gru, predict

OUTPUTS = Path(__file__).parent / "outputs"
TRAIN, VALID, TEST = (102, 111), (112, 113), (114, 118)
HISTORY = ["prior_rate_all", "prior_rate_opposed", "prior_rate_bipartisan", "recent_rate_opposed_100", "recent_rate_all_50",
           "congress_rate_opposed", "prior_rate_same_type", "last_defected", "previous_congress_rate_opposed"]
# previous_congress_rate_opposed already stops at the end of the previous Congress.
HISTORY_DAY = [f"{c}_day" for c in HISTORY[:-1]] + ["previous_congress_rate_opposed"]
ROLLCALL = ["own_yea", "party_opposed", "other_tied"]
MEMBER = ["tenure", "freshman", "majority"]
CATEGORIES = ["vote_type", "party_region", "chamber"]
# Known only after the vote: how split each party was. Used only in the variant labelled as such.
SPLIT = ["own_cohesion", "other_cohesion"]
KEEP = ["congress", "chamber", "rollnumber", "date", "icpsr", "person", "bioname", "party", "region", "state_abbrev", "vote_type",
        "party_opposed", "own_position", "other_position", "vote", "defected", "freshman", "tenure", "prior_votes"]


def period(rows, bounds):
    return rows["congress"].between(*bounds).to_numpy()


def load_or_build(rebuild: bool, min_minority_share: float | None = None) -> pd.DataFrame:
    cache = OUTPUTS / "rows.parquet"
    if cache.exists() and not rebuild:
        rows = pd.read_parquet(cache)
    else:
        rows = load_rows(min_minority_share=min_minority_share)
        rows["party_region"] = rows["party"] + " " + rows["region"]
        rows["log_prior_votes"] = np.log1p(rows["prior_votes"]).astype("float32")
        rows["log_prior_votes_day"] = np.log1p(rows["prior_votes_day"]).astype("float32")
        rows.to_parquet(cache)
    if "log_prior_votes_day" not in rows:
        # A cache from before the day-ahead features: add them without rebuilding. Same sort as add_history.
        rows = rows.sort_values(SEQUENCE_KEYS + ["date", "rollnumber"], kind="stable").reset_index(drop=True)
        rows = add_day_history(rows)
        rows["log_prior_votes_day"] = np.log1p(rows["prior_votes_day"]).astype("float32")
        rows.to_parquet(cache)
    # Categories save memory; the sequence keys stay plain so grouping on them doesn't build every combination.
    for column in ["bioname", "state_abbrev", "region", "vote_type", "party_region"]:
        rows[column] = rows[column].astype("category")
    return rows


def own_past_rate(rows, train, suffix=""):
    """Baseline without fitting: the member's earlier defection rate on the same kind of vote (party-opposed or not)."""
    rate = np.where(rows["party_opposed"], rows[f"prior_rate_opposed{suffix}"], rows[f"prior_rate_bipartisan{suffix}"])
    fallback = np.where(rows["party_opposed"], rows.loc[train & rows["party_opposed"], "defected"].mean(),
                        rows.loc[train & ~rows["party_opposed"], "defected"].mean())
    return np.where(np.isnan(rate), fallback, rate)


def logistic_design(rows, train, history=HISTORY, count="log_prior_votes"):
    """Numeric history filled with training means plus missing flags, history times party-opposed, and one-hot categories."""
    parts = {}
    for column in history + [count, "tenure"]:
        values = rows[column].astype("float32")
        parts[column] = values.fillna(values[train].mean())
        if values.isna().any():
            parts[f"{column}_missing"] = values.isna().astype("float32")
    opposed = rows["party_opposed"].astype("float32")
    for column in history:
        parts[f"{column}_x_opposed"] = parts[column] * opposed
    for column in ROLLCALL + ["freshman"]:
        parts[column] = rows[column].astype("float32")
    parts["majority"] = rows["majority"].fillna(0.5).astype("float32")
    design = pd.DataFrame(parts)
    dummies = pd.get_dummies(rows[CATEGORIES], dtype="float32")
    opposed_dummies = dummies.mul(opposed, axis=0).add_suffix("_x_opposed")
    return pd.concat([design, dummies, opposed_dummies], axis=1).to_numpy("float32")


def boosting_design(rows, extra=(), history=HISTORY, count="log_prior_votes"):
    frame = rows[history + [count] + ROLLCALL + MEMBER + list(extra)].astype("float32")
    for column in CATEGORIES:
        frame[column] = pd.Categorical(rows[column]).codes
    return frame


def fit_boosting(rows, train, valid, name, extra=(), history=HISTORY, count="log_prior_votes", seed=0, log=print):
    """Gradient boosting; the number of trees is chosen by validation log loss."""
    design = boosting_design(rows, extra, history, count)
    categorical = [design.columns.get_loc(c) for c in CATEGORIES]
    model = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
                                           l2_regularization=1.0, categorical_features=categorical, early_stopping=False,
                                           random_state=seed)
    model.fit(design[train], rows.loc[train, "defected"])
    losses = [log_loss(rows.loc[valid, "defected"], p[:, 1]) for p in model.staged_predict_proba(design[valid])]
    best = int(np.argmin(losses)) + 1
    log(f"{name}: best number of trees {best} of {len(losses)}, validation log loss {losses[best - 1]:.4f}")
    for stage, probability in enumerate(model.staged_predict_proba(design), start=1):
        if stage == best:
            return probability[:, 1]


def summarise(rows, probability):
    """Headline metrics: vote accuracy overall and on party-opposed roll calls, and defection ranking and recall."""
    y = rows["defected"].to_numpy()
    opposed = rows["party_opposed"].to_numpy()
    flipped = probability > 0.5
    correct = flipped == y
    return {
        "vote_accuracy": correct.mean(), "vote_accuracy_party_opposed": correct[opposed].mean(),
        "defection_auc": roc_auc_score(y, probability) if np.ptp(probability) > 0 else 0.5,
        "defection_average_precision": average_precision_score(y, probability),
        "defection_log_loss": log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6)),
        "defections_caught": (flipped & y).sum() / y.sum(), "flag_precision": (flipped & y).sum() / max(flipped.sum(), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the feature table instead of using the cached one")
    parser.add_argument("--seeds", type=int, default=3, help="number of GRU training seeds")
    parser.add_argument("--day-ahead-seeds", type=int, default=1, help="number of day-ahead GRU training seeds")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--refit", action="store_true", help="refit every model instead of reusing saved predictions and weights")
    parser.add_argument("--min-minority-share", type=float, default=None,
                        help="near-unanimous cutoff for a sensitivity check (default: the standard 2.5%% flag); outputs go to outputs/cutoff_<value>/")
    args = parser.parse_args()
    global OUTPUTS
    if args.min_minority_share is not None:
        OUTPUTS = OUTPUTS / f"cutoff_{args.min_minority_share:g}"
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    def log(message):
        print(message, flush=True)
        with open(OUTPUTS / "run_log.txt", "a", encoding="utf-8") as file:
            file.write(message + "\n")

    started = time.time()
    rows = load_or_build(args.rebuild, args.min_minority_share)
    rows = rows.sort_values(SEQUENCE_KEYS + ["date", "rollnumber"], kind="stable").reset_index(drop=True)
    train, valid, test = period(rows, TRAIN), period(rows, VALID), period(rows, TEST)
    log(f"{len(rows):,} rows; train {train.sum():,}, validation {valid.sum():,}, test {test.sum():,} ({time.time() - started:.0f}s)")

    # Each model's probabilities are saved as soon as it finishes, so an interrupted run resumes where it stopped.
    def saved(name):
        return (OUTPUTS / f"p_{name}.npy").exists() and not args.refit

    def save(name, probability):
        np.save(OUTPUTS / f"p_{name}.npy", np.asarray(probability, dtype="float32"))

    save("party_line", np.zeros(len(rows)))
    save("training_rate", np.full(len(rows), rows.loc[train, "defected"].mean()))
    save("own_past_rate", own_past_rate(rows, train))
    save("own_past_rate_day", own_past_rate(rows, train, suffix="_day"))

    # Rows before 1991 only supply history; the feature-based models use 1991 on.
    modelled = rows["congress"].to_numpy() >= TRAIN[0]
    standard, day = dict(history=HISTORY, count="log_prior_votes"), dict(history=HISTORY_DAY, count="log_prior_votes_day")
    feature_models = {"logistic": ("logistic", (), standard), "boosting": ("boosting", (), standard),
                      "boosting_with_split": ("boosting", SPLIT, standard),
                      "logistic_day": ("logistic", (), day), "boosting_day": ("boosting", (), day)}
    if not all(saved(name) for name in feature_models):
        # Only the columns the feature-based models use, so the copy stays small.
        needed = list(dict.fromkeys(HISTORY + HISTORY_DAY + ["log_prior_votes", "log_prior_votes_day"] + ROLLCALL + MEMBER
                                    + CATEGORIES + SPLIT + ["defected"]))
        subset = rows.loc[modelled, needed].reset_index(drop=True)
        sub_train, sub_valid = train[modelled], valid[modelled]
        for name, (kind, extra, setting) in feature_models.items():
            if saved(name):
                continue
            probability = np.full(len(rows), np.nan)
            if kind == "logistic":
                design = logistic_design(subset, sub_train, **setting)
                scaler = StandardScaler().fit(design[sub_train])
                logistic = LogisticRegression(max_iter=500, C=1.0).fit(scaler.transform(design[sub_train]), subset.loc[sub_train, "defected"])
                probability[modelled] = logistic.predict_proba(scaler.transform(design))[:, 1]
                del design
            else:
                probability[modelled] = fit_boosting(subset, sub_train, sub_valid, name, extra=extra, log=log, **setting)
            save(name, probability)
            log(f"{name} done ({time.time() - started:.0f}s)")
        del subset
        gc.collect()

    # Member states are averaged over each member's party-opposed votes in each Congress, for seed 0 only.
    opposed = rows["party_opposed"].to_numpy()
    member_keys = rows.loc[opposed, ["person", "chamber", "party", "congress"]]
    state_groups = np.full(len(rows), -1, dtype="int64")
    state_groups[opposed] = member_keys.groupby(["person", "chamber", "party", "congress"], sort=False).ngroup().to_numpy()
    grus = [(f"gru_seed{s}", s, False) for s in range(args.seeds)] + [(f"gru_day_seed{s}", s, True) for s in range(args.day_ahead_seeds)]
    for name, seed, day_ahead in grus:
        weights = OUTPUTS / f"{name}.pt"
        needs_states = name == "gru_seed0" and (args.refit or not (OUTPUTS / "gru_member_states.parquet").exists())
        if saved(name) and not needs_states:
            continue
        if weights.exists() and not args.refit:
            data = SequenceBatches(rows, train, day_ahead=day_ahead)
            model = DefectionGRU(data.inputs.shape[1], n_previous=data.previous.shape[1])
            model.load_state_dict(torch.load(weights))
            log(f"{name}: loaded saved weights")
        else:
            model, data = fit_gru(rows, train, valid, seed=seed, day_ahead=day_ahead, log=log)
            torch.save(model.state_dict(), weights)
        probability, sums = predict(model, data, state_groups=state_groups if needs_states else None)
        save(name, probability)
        if needs_states:
            write_member_states(rows, opposed, state_groups, sums, probability)
        del model, data
        gc.collect()
        log(f"{name} done ({time.time() - started:.0f}s)")

    names = (["party_line", "training_rate", "own_past_rate", "logistic", "boosting", "boosting_with_split", "own_past_rate_day",
              "logistic_day", "boosting_day"] + [name for name, _, _ in grus])
    selected = valid | test
    output = rows.loc[selected, KEEP].copy()
    output["split"] = np.where(test[selected], "test", "validation")
    for name in names:
        output[f"p_{name}"] = np.load(OUTPUTS / f"p_{name}.npy")[selected]
    output.to_parquet(OUTPUTS / "predictions.parquet")

    in_test = output["split"] == "test"
    metrics = pd.DataFrame({name: summarise(output[in_test], output.loc[in_test, f"p_{name}"].to_numpy()) for name in names}).T
    metrics.to_csv(OUTPUTS / "test_metrics.csv")
    log(metrics.round(4).to_string())
    log(f"done in {time.time() - started:.0f}s")


def write_member_states(rows, opposed, state_groups, sums, probability):
    """Average GRU state, predicted and actual defection per member and Congress, on party-opposed votes.

    Covers every Congress, including training years, so it is for describing members, not for evaluation.
    """
    part = rows.loc[opposed, ["person", "chamber", "party", "congress", "icpsr", "bioname", "region", "state_abbrev", "defected"]]
    part = part.assign(group=state_groups[opposed], p_gru=probability[opposed])
    member_congress = part.groupby("group").agg(
        person=("person", "first"), chamber=("chamber", "first"), party=("party", "first"), congress=("congress", "first"),
        icpsr=("icpsr", "first"), bioname=("bioname", "first"), region=("region", "first"), state_abbrev=("state_abbrev", "first"),
        votes=("defected", "size"), defection_rate=("defected", "mean"), mean_p_gru=("p_gru", "mean"))
    means = sums[member_congress.index.to_numpy()] / member_congress["votes"].to_numpy()[:, None]
    for i in range(means.shape[1]):
        member_congress[f"h{i}"] = means[:, i].astype("float32")
    member_congress.reset_index(drop=True).to_parquet(OUTPUTS / "gru_member_states.parquet")


if __name__ == "__main__":
    main()
