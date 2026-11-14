"""Train the European Parliament vote prediction models on earlier roll calls and generate predictions for later ones.

Each model predicts the probability that a group member's position differs from their group's line on a roll call
(For, Against or Abstention, the group's most common position without the member). Training uses January 2020 to
June 2023, model choices use July 2023 to the end of the 9th term, and the 10th term (from 16 July 2024) is the test
set, used once. Roll calls from 2019 only supply history. Outputs go to outputs/, which is not committed.

Standard models predict one roll call ahead: history includes the member's positions seconds earlier on the same
sitting day. Day-ahead models (suffix _day) use history up to the end of the previous sitting day only.
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
TRAIN = (pd.Timestamp("2020-01-01"), pd.Timestamp("2023-07-01"))
VALID = (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-07-16"))
TEST = (pd.Timestamp("2024-07-16"), pd.Timestamp("2100-01-01"))
HISTORY = ["prior_rate_all", "prior_rate_with_most", "prior_rate_against_most", "recent_rate_all_100",
           "recent_rate_against_most_50", "prior_rate_same_part", "last_defected", "party_recent_rate_200",
           "group_recent_rate_200"]
HISTORY_DAY = [f"{c}_day" for c in HISTORY]
ROLLCALL = ["others_same_share", "with_most_groups", "group_seat_share"]
MEMBER = ["log_party_seats", "no_party", "new_group", "age"]
CATEGORIES = ["vote_part", "own_line", "committee", "country_code"]
# Known only after the vote: how split the member's group was (with the member) and what their national colleagues in
# the group did. Used only in the variant labelled as such.
SPLIT_NUMERIC = ["own_cohesion"]
SPLIT_CATEGORIES = ["colleagues"]
KEEP = ["vote_id", "member_id", "timestamp", "date", "term", "group_code", "country_code", "national_party_id",
        "vote_part", "committee", "own_line", "position_code", "with_most_groups", "others_same_share", "defected",
        "prior_votes", "no_party", "colleagues", "own_cohesion"]


def period(rows, bounds):
    return ((rows["date"] >= bounds[0]) & (rows["date"] < bounds[1])).to_numpy()


def load_or_build(rebuild: bool, min_minority_share: float | None) -> pd.DataFrame:
    cache = OUTPUTS / "rows.parquet"
    if cache.exists() and not rebuild:
        rows = pd.read_parquet(cache)
        if not set(HISTORY_DAY) <= set(rows.columns):  # cache built before the day-ahead features
            rows = add_day_history(rows)
            rows.to_parquet(cache)
        return rows
    options = {} if min_minority_share is None else {"min_minority_share": min_minority_share}
    rows = load_rows(**options)
    rows["log_prior_votes"] = np.log1p(rows["prior_votes"]).astype("float32")
    for column in ["group_code", "country_code", "vote_part", "committee", "colleagues", "party_group"]:
        rows[column] = rows[column].astype(str)
    rows.to_parquet(cache)
    return rows


def own_past_rate(rows, train, suffix=""):
    """Baseline without fitting: the member's earlier defection rate on the same kind of roll call (own group with
    most other groups or not). suffix "_day" uses earlier sitting days only."""
    with_most = rows["with_most_groups"].to_numpy()
    rate = np.where(with_most, rows[f"prior_rate_with_most{suffix}"], rows[f"prior_rate_against_most{suffix}"])
    fallback = np.where(with_most, rows.loc[train & with_most, "defected"].mean(),
                        rows.loc[train & ~with_most, "defected"].mean())
    return np.where(np.isnan(rate), fallback, rate)


def logistic_design(rows, train, history=HISTORY):
    """Numeric history filled with training means plus missing flags, history times against-most, and one-hot categories."""
    parts = {}
    for column in history + ["log_prior_votes", "age"]:
        values = rows[column].astype("float32")
        parts[column] = values.fillna(values[train].mean())
        if values.isna().any():
            parts[f"{column}_missing"] = values.isna().astype("float32")
    against = (~rows["with_most_groups"]).astype("float32")
    for column in history:
        parts[f"{column}_x_against_most"] = parts[column] * against
    for column in ROLLCALL + ["log_party_seats", "no_party", "new_group"]:
        parts[column] = rows[column].astype("float32").fillna(0.5 if column == "others_same_share" else 0)
    dummies = pd.get_dummies(rows[CATEGORIES].astype(str), dtype="float32")
    return pd.concat([pd.DataFrame(parts), dummies], axis=1).to_numpy("float32")


def boosting_design(rows, history=HISTORY, extra_numeric=(), extra_categories=()):
    frame = rows[history + ["log_prior_votes"] + ROLLCALL + MEMBER + list(extra_numeric)].astype("float32")
    for column in CATEGORIES + list(extra_categories):
        frame[column] = pd.Categorical(rows[column].astype(str)).codes
    return frame


def fit_boosting(rows, train, valid, name, history=HISTORY, extra_numeric=(), extra_categories=(), max_iter=600,
                 seed=0, log=print):
    """Gradient boosting; the number of trees is chosen by validation log loss."""
    design = boosting_design(rows, history, extra_numeric, extra_categories)
    categorical = [design.columns.get_loc(c) for c in CATEGORIES + list(extra_categories)]
    model = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
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
    """Headline metrics: line accuracy overall and where the group went against most groups, and defection ranking.

    Line accuracy is the share of positions correctly classified as with or against the group line: the group-line
    baseline predicts "with the line" everywhere.
    """
    y = rows["defected"].to_numpy()
    against = ~rows["with_most_groups"].to_numpy()
    flagged = probability > 0.5
    correct = flagged == y
    return {
        "line_accuracy": correct.mean(), "line_accuracy_against_most": correct[against].mean(),
        "defection_auc": roc_auc_score(y, probability) if np.ptp(probability) > 0 else 0.5,
        "defection_average_precision": average_precision_score(y, probability),
        "defection_log_loss": log_loss(y, np.clip(probability, 1e-6, 1 - 1e-6)),
        "defections_caught": (flagged & y).sum() / y.sum(), "flag_precision": (flagged & y).sum() / max(flagged.sum(), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="rebuild the feature table instead of using the cached one")
    parser.add_argument("--seeds", type=int, default=3, help="number of GRU training seeds")
    parser.add_argument("--day-ahead-seeds", type=int, default=1, help="number of day-ahead GRU training seeds")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--refit", action="store_true", help="refit every model instead of reusing saved predictions and weights")
    parser.add_argument("--min-minority-share", type=float, default=None,
                        help="near-unanimous cutoff for a sensitivity check (default 2.5%%); outputs go to outputs/cutoff_<value>/")
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
    rows = rows.sort_values(SEQUENCE_KEYS + ["timestamp", "vote_id"], kind="stable").reset_index(drop=True)
    train, valid, test = period(rows, TRAIN), period(rows, VALID), period(rows, TEST)
    log(f"{len(rows):,} rows; train {train.sum():,}, validation {valid.sum():,}, test {test.sum():,} "
        f"({time.time() - started:.0f}s)")
    countries = sorted(rows["country_code"].unique())

    def saved(name):
        return (OUTPUTS / f"p_{name}.npy").exists() and not args.refit

    def save(name, probability):
        np.save(OUTPUTS / f"p_{name}.npy", np.asarray(probability, dtype="float32"))

    save("group_line", np.zeros(len(rows)))
    save("training_rate", np.full(len(rows), rows.loc[train, "defected"].mean()))
    save("own_past_rate", own_past_rate(rows, train))
    save("own_past_rate_day", own_past_rate(rows, train, "_day"))

    # Rows from 2019 only supply history; the feature-based models use 2020 on.
    modelled = (rows["date"] >= TRAIN[0]).to_numpy()
    # After-the-vote variants get more trees: the split variant hit a 600 cap.
    feature_models = {
        "logistic": dict(history=HISTORY), "logistic_day": dict(history=HISTORY_DAY),
        "boosting": dict(history=HISTORY), "boosting_day": dict(history=HISTORY_DAY),
        "boosting_with_split": dict(extra_numeric=SPLIT_NUMERIC, extra_categories=SPLIT_CATEGORIES, max_iter=1500),
        "boosting_with_colleagues": dict(extra_categories=SPLIT_CATEGORIES, max_iter=1500),
    }
    if not all(saved(name) for name in feature_models):
        needed = list(dict.fromkeys(HISTORY + HISTORY_DAY + ["log_prior_votes"] + ROLLCALL + MEMBER + CATEGORIES
                                    + SPLIT_NUMERIC + SPLIT_CATEGORIES + ["defected"]))
        subset = rows.loc[modelled, needed].reset_index(drop=True)
        sub_train, sub_valid = train[modelled], valid[modelled]
        for name, options in feature_models.items():
            if saved(name):
                continue
            probability = np.full(len(rows), np.nan)
            if name.startswith("logistic"):
                design = logistic_design(subset, sub_train, options["history"])
                scaler = StandardScaler().fit(design[sub_train])
                logistic = LogisticRegression(max_iter=500, C=1.0).fit(scaler.transform(design[sub_train]),
                                                                       subset.loc[sub_train, "defected"])
                probability[modelled] = logistic.predict_proba(scaler.transform(design))[:, 1]
                del design
            else:
                probability[modelled] = fit_boosting(subset, sub_train, sub_valid, name, log=log, **options)
            save(name, probability)
            log(f"{name} done ({time.time() - started:.0f}s)")
        del subset
        gc.collect()

    grus = ([(f"gru_seed{s}", s, False) for s in range(args.seeds)]
            + [(f"gru_day_seed{s}", s, True) for s in range(args.day_ahead_seeds)])
    for name, seed, day_ahead in grus:
        weights = OUTPUTS / f"{name}.pt"
        if saved(name):
            continue
        if weights.exists() and not args.refit:
            data = SequenceBatches(rows, train, countries, day_ahead=day_ahead)
            model = DefectionGRU(data.inputs.shape[1], n_previous=data.previous.shape[1])
            model.load_state_dict(torch.load(weights))
            log(f"{name}: loaded saved weights")
        else:
            model, data = fit_gru(rows, train, valid, countries, seed=seed, day_ahead=day_ahead, log=log)
            torch.save(model.state_dict(), weights)
        save(name, predict(model, data))
        del model, data
        gc.collect()
        log(f"{name} done ({time.time() - started:.0f}s)")

    names = (["group_line", "training_rate", "own_past_rate", "own_past_rate_day"] + list(feature_models)
             + [name for name, _, _ in grus])
    selected = valid | test
    output = rows.loc[selected, KEEP].copy()
    output["split"] = np.where(test[selected], "test", "validation")
    for name in names:
        output[f"p_{name}"] = np.load(OUTPUTS / f"p_{name}.npy")[selected]
    output.to_parquet(OUTPUTS / "predictions.parquet")

    in_test = output["split"] == "test"
    metrics = pd.DataFrame({name: summarise(output[in_test], output.loc[in_test, f"p_{name}"].to_numpy())
                            for name in names}).T
    metrics.to_csv(OUTPUTS / "test_metrics.csv")
    log(metrics.round(4).to_string())
    log(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
