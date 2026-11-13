"""GRU that reads each member's positions in order and predicts whether they break from their group's line next.

Adapted from usa/pipeline/vote_prediction/sequence_model.py. Each member is one sequence across both terms, ordered by
time, with a flag when they change group or term. At step t the input holds the roll call's features, the member's
features, and whether the member defected at step t - 1. The output is the probability of defecting at step t.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from features import LINES, SEQUENCE_KEYS, VOTE_PARTS


def step_inputs(rows: pd.DataFrame, countries: list) -> np.ndarray:
    """Per-position inputs, without the previous outcome. rows must be sorted by sequence and time."""
    part = pd.Categorical(rows["vote_part"], categories=VOTE_PARTS)
    country = pd.Categorical(rows["country_code"].astype(str), categories=countries)
    group = rows.groupby(SEQUENCE_KEYS, sort=False).ngroup().to_numpy()
    new_sequence = np.r_[True, group[1:] != group[:-1]]
    term = rows["term"].to_numpy()
    new_term = new_sequence | np.r_[True, term[1:] != term[:-1]]
    columns = [
        np.eye(len(VOTE_PARTS), dtype="float32")[part.codes],
        np.eye(len(LINES), dtype="float32")[rows["own_line"].to_numpy()],
        np.eye(len(countries), dtype="float32")[country.codes],
        np.column_stack([
            rows["others_same_share"].fillna(0.5), rows["with_most_groups"], rows["group_seat_share"],
            rows["log_party_seats"] / 3, rows["no_party"], rows["new_group"], rows["age"].fillna(55) / 100, new_term,
        ]).astype("float32"),
    ]
    return np.concatenate(columns, axis=1)


class DefectionGRU(nn.Module):
    def __init__(self, n_inputs: int, hidden: int = 64):
        super().__init__()
        # Two extra inputs: the previous outcome and a flag for the first step of a sequence.
        self.gru = nn.GRU(n_inputs + 2, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_inputs, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, previous, h=None):
        out, h = self.gru(torch.cat([x, previous], dim=-1), h)
        # The head also sees the current roll call directly, so the state only has to carry the member's history.
        return self.head(torch.cat([out, x], dim=-1)).squeeze(-1), out, h


class SequenceBatches:
    """Padded batches of whole sequences, grouped by length to limit padding."""

    def __init__(self, rows: pd.DataFrame, train_mask: np.ndarray, countries: list, batch_size: int = 32):
        self.inputs = step_inputs(rows, countries)
        self.target = rows["defected"].to_numpy().astype("float32")
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
        previous = np.zeros((len(spans), length, 2), dtype="float32")
        y = np.zeros((len(spans), length), dtype="float32")
        weight = np.zeros((len(spans), length), dtype="float32")
        index = np.full((len(spans), length), -1, dtype="int64")
        for row, (start, end) in enumerate(spans):
            n = end - start
            x[row, :n] = self.inputs[start:end]
            y[row, :n] = self.target[start:end]
            previous[row, 1:n, 0] = self.target[start:end - 1]
            previous[row, 0, 1] = 1.0
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
def predict(model, data: SequenceBatches, chunk: int = 2048) -> np.ndarray:
    """Probabilities for every row."""
    model.eval()
    probability = np.zeros(len(data.target), dtype="float32")
    for batch in data.batches:
        x, previous, _, _, index = data.padded(batch)
        h = None
        for start in range(0, x.shape[1], chunk):
            part = slice(start, start + chunk)
            logits, _, h = model(x[:, part], previous[:, part], h)
            valid = index[:, part] >= 0
            probability[index[:, part][valid]] = torch.sigmoid(logits).numpy()[valid]
    return probability


def fit_gru(rows: pd.DataFrame, train_mask, valid_mask, countries: list, seed: int = 0, hidden: int = 64,
            max_epochs: int = 12, patience: int = 2, log=print):
    """Train on train_mask rows and stop when log loss on valid_mask rows stops improving. Returns the best model."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = SequenceBatches(rows, train_mask, countries)
    model = DefectionGRU(data.inputs.shape[1], hidden)
    optimiser = torch.optim.Adam(model.parameters(), lr=2e-3)
    target = data.target[valid_mask]
    best, best_state, stale = np.inf, None, 0
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, data, optimiser, rng)
        probability = predict(model, data)
        p = np.clip(probability[valid_mask], 1e-6, 1 - 1e-6)
        valid_loss = float(-np.mean(target * np.log(p) + (1 - target) * np.log(1 - p)))
        log(f"seed {seed} epoch {epoch + 1}: train log loss {train_loss:.4f}, validation log loss {valid_loss:.4f}")
        if valid_loss < best - 1e-5:
            best, best_state, stale = valid_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, data
