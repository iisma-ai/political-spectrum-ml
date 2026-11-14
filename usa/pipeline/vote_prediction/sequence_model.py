"""GRU that reads each member's votes in order and predicts whether they break from their party on the next one.

Each member, chamber and party spell is one sequence, ordered by date and roll number. At step t the input holds
the roll call's features, the member's fixed features, and whether the member defected at step t - 1. The output is
the probability of defecting at step t. So the hidden state is a running summary of the member's past votes, like
the hand-built history features given to the other models, but learned.

The day-ahead version gets no outcome from the same day: instead of the previous outcome, it gets the previous day's
defection rates at the member's first vote of each day.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from features import REGIONS, SEQUENCE_KEYS, VOTE_TYPES


def step_inputs(rows: pd.DataFrame) -> np.ndarray:
    """Per-vote inputs, without the previous outcome. rows must be sorted by sequence and time."""
    vote_type = pd.Categorical(rows["vote_type"], categories=VOTE_TYPES)
    region = pd.Categorical(rows["region"], categories=REGIONS)
    group = rows.groupby(SEQUENCE_KEYS, sort=False).ngroup().to_numpy()
    new_sequence = np.r_[True, group[1:] != group[:-1]]
    congress = rows["congress"].to_numpy()
    new_congress = new_sequence | np.r_[True, congress[1:] != congress[:-1]]
    columns = [
        np.eye(len(VOTE_TYPES), dtype="float32")[vote_type.codes],
        np.eye(len(REGIONS), dtype="float32")[region.codes],
        np.column_stack([
            rows["party"].eq("R"), rows["chamber"].eq("Senate"), rows["own_yea"], rows["party_opposed"], rows["other_tied"],
            rows["majority"].fillna(0.5), rows["freshman"], np.log1p(rows["tenure"]) / 3, new_congress,
        ]).astype("float32"),
    ]
    return np.concatenate(columns, axis=1)


def previous_inputs(rows: pd.DataFrame, target: np.ndarray, day_ahead: bool) -> np.ndarray:
    """Outcome inputs for each step. rows must be sorted by sequence and time.

    Standard: the outcome at the previous step and a first-step flag. Day-ahead: the member's defection rates on their
    previous day with votes (all, party-opposed, bipartisan; the latter two fall back to the first when empty) at the
    first vote of each day, a first-of-day flag and a first-step flag.
    """
    group = rows.groupby(SEQUENCE_KEYS, sort=False).ngroup().to_numpy()
    new_sequence = np.r_[True, group[1:] != group[:-1]]
    if not day_ahead:
        previous = np.zeros((len(rows), 2), dtype="float32")
        previous[1:, 0] = target[:-1]
        previous[new_sequence, 0] = 0
        previous[new_sequence, 1] = 1
        return previous
    date = rows["date"].to_numpy()
    new_day = new_sequence | np.r_[True, date[1:] != date[:-1]]
    day = np.cumsum(new_day) - 1
    opposed = rows["party_opposed"].to_numpy().astype(bool)
    days = pd.DataFrame({"day": day, "y": target, "opposed": np.where(opposed, target, np.nan),
                         "bipartisan": np.where(opposed, np.nan, target)}).groupby("day").mean()
    days["opposed"] = days["opposed"].fillna(days["y"])
    days["bipartisan"] = days["bipartisan"].fillna(days["y"])
    rates = days[["y", "opposed", "bipartisan"]].to_numpy("float32")
    previous = np.zeros((len(rows), 5), dtype="float32")
    first_rows = np.flatnonzero(new_day)
    has_earlier_day = ~new_sequence[first_rows]
    previous[first_rows[has_earlier_day], :3] = rates[day[first_rows[has_earlier_day]] - 1]
    previous[new_day, 3] = 1
    previous[new_sequence, 4] = 1
    return previous


class DefectionGRU(nn.Module):
    def __init__(self, n_inputs: int, hidden: int = 64, n_previous: int = 2):
        super().__init__()
        # Extra inputs for earlier outcomes and flags: see previous_inputs.
        self.gru = nn.GRU(n_inputs + n_previous, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_inputs, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, previous, h=None):
        out, h = self.gru(torch.cat([x, previous], dim=-1), h)
        # The head also sees the current roll call directly, so the state only has to carry the member's history.
        return self.head(torch.cat([out, x], dim=-1)).squeeze(-1), out, h


class SequenceBatches:
    """Padded batches of whole sequences, grouped by length to limit padding."""

    def __init__(self, rows: pd.DataFrame, train_mask: np.ndarray, batch_size: int = 64, day_ahead: bool = False):
        self.inputs = step_inputs(rows)
        self.target = rows["defected"].to_numpy().astype("float32")
        self.previous = previous_inputs(rows, self.target, day_ahead)
        self.train_mask = train_mask.astype("float32")
        group = rows.groupby(SEQUENCE_KEYS, sort=False).ngroup().to_numpy()
        starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
        self.spans = list(zip(starts, np.r_[starts[1:], len(rows)]))
        order = np.argsort([end - start for start, end in self.spans])
        self.batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]

    def padded(self, batch):
        spans = [self.spans[i] for i in batch]
        length = max(end - start for start, end in spans)
        x = np.zeros((len(spans), length, self.inputs.shape[1]), dtype="float32")
        previous = np.zeros((len(spans), length, self.previous.shape[1]), dtype="float32")
        y = np.zeros((len(spans), length), dtype="float32")
        weight = np.zeros((len(spans), length), dtype="float32")
        index = np.full((len(spans), length), -1, dtype="int64")
        for row, (start, end) in enumerate(spans):
            n = end - start
            x[row, :n] = self.inputs[start:end]
            y[row, :n] = self.target[start:end]
            previous[row, :n] = self.previous[start:end]
            weight[row, :n] = self.train_mask[start:end]
            index[row, :n] = np.arange(start, end)
        return torch.from_numpy(x), torch.from_numpy(previous), torch.from_numpy(y), torch.from_numpy(weight), index


def train_epoch(model, data: SequenceBatches, optimiser, rng, chunk: int = 256) -> float:
    """One pass with truncated backpropagation: gradients flow within chunks, and the state carries across them."""
    model.train()
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    total, weight_sum = 0.0, 0.0
    for b in rng.permutation(len(data.batches)):
        x, previous, y, weight, _ = data.padded(data.batches[b])
        if weight.sum() == 0:
            continue
        h = None
        for start in range(0, x.shape[1], chunk):
            part = slice(start, start + chunk)
            w = weight[:, part]
            logits, _, h = model(x[:, part], previous[:, part], h)
            h = h.detach()
            if w.sum() == 0:
                continue
            loss = (loss_fn(logits, y[:, part]) * w).sum() / w.sum()
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            total += loss.item() * w.sum().item()
            weight_sum += w.sum().item()
    return total / max(weight_sum, 1)


@torch.no_grad()
def predict(model, data: SequenceBatches, chunk: int = 2048, state_groups: np.ndarray | None = None):
    """Probabilities for every row, and optionally the summed hidden state per group.

    state_groups gives each row a group number, or -1 to leave it out. Summing per group, instead of keeping the
    state at every row, keeps memory small.
    """
    model.eval()
    probability = np.zeros(len(data.target), dtype="float32")
    sums = None
    if state_groups is not None:
        sums = np.zeros((state_groups.max() + 1, model.gru.hidden_size), dtype="float64")
    for batch in data.batches:
        x, previous, _, _, index = data.padded(batch)
        h = None
        for start in range(0, x.shape[1], chunk):
            part = slice(start, start + chunk)
            logits, out, h = model(x[:, part], previous[:, part], h)
            valid = index[:, part] >= 0
            rows = index[:, part][valid]
            probability[rows] = torch.sigmoid(logits).numpy()[valid]
            if sums is not None:
                groups = state_groups[rows]
                keep = groups >= 0
                np.add.at(sums, groups[keep], out.numpy()[valid][keep])
    return probability, sums


def fit_gru(rows: pd.DataFrame, train_mask, valid_mask, seed: int = 0, hidden: int = 64, max_epochs: int = 12,
            patience: int = 2, day_ahead: bool = False, log=print):
    """Train on train_mask rows and stop when log loss on valid_mask rows stops improving. Returns the best model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = SequenceBatches(rows, train_mask, day_ahead=day_ahead)
    model = DefectionGRU(data.inputs.shape[1], hidden, n_previous=data.previous.shape[1])
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    target = data.target[valid_mask]
    best, best_state, stale = np.inf, None, 0
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, data, optimiser, rng)
        probability, _ = predict(model, data)
        p = np.clip(probability[valid_mask], 1e-6, 1 - 1e-6)
        valid_loss = float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))
        log(f"{'day-ahead ' if day_ahead else ''}seed {seed} epoch {epoch + 1}: train log loss {train_loss:.4f}, validation log loss {valid_loss:.4f}")
        if valid_loss < best - 1e-5:
            best, best_state, stale = valid_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, data
