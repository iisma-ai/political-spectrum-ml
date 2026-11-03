"""Measures shared by the polarization clustering notebooks."""

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

SOUTH = {"AL", "AR", "FL", "GA", "LA", "MS", "NC", "SC", "TN", "TX", "VA"}
LARGE_COMMUNITY = 0.05
FACTION_SIZE = 0.10
BLOC_PURITY = 0.80


def agreement_matrix(matrix: pd.DataFrame, min_shared: int = 20) -> pd.DataFrame:
    """Share of the Yea or Nay votes two members both cast on which they voted the same way."""
    values = np.nan_to_num(matrix.to_numpy(dtype="float64"))
    present = (values != 0).astype("float64")
    shared = present @ present.T
    same = (values @ values.T + shared) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        agreement = same / shared
    agreement[shared < min_shared] = np.nan
    np.fill_diagonal(agreement, np.nan)
    return pd.DataFrame(agreement, index=matrix.index, columns=matrix.index)


def agreement_graph(agreement: pd.DataFrame) -> nx.Graph:
    """Complete weighted graph with agreement as edge weight, as in Waugh et al. (2009)."""
    values = agreement.to_numpy()
    rows, columns = np.triu_indices_from(values, k=1)
    weights = values[rows, columns]
    keep = ~np.isnan(weights) & (weights > 0)
    labels = agreement.index.to_numpy()
    graph = nx.Graph()
    graph.add_nodes_from(labels)
    graph.add_weighted_edges_from(zip(labels[rows[keep]], labels[columns[keep]], weights[keep]))
    return graph


def labels_to_communities(labels: pd.Series) -> list[set]:
    return [set(index) for index in labels.groupby(labels).groups.values()]


def communities_to_labels(communities) -> pd.Series:
    ordered = sorted(communities, key=len, reverse=True)
    return pd.Series({node: number for number, community in enumerate(ordered) for node in community}).sort_index()


def partition_modularity(graph: nx.Graph, labels: pd.Series) -> float:
    return modularity(graph, labels_to_communities(labels), weight="weight")


def normalized_modularity(graph: nx.Graph, labels: pd.Series) -> float:
    """Modularity divided by its maximum for these group sizes (Newman's assortativity coefficient).

    Raw modularity is capped at 1 - sum of squared group weight shares, so a chamber dominated by one party
    has lower modularity even when the parties are perfectly separated. This removes that effect.
    """
    strength = pd.Series(dict(graph.degree(weight="weight")))
    shares = strength.groupby(labels.reindex(strength.index)).sum() / strength.sum()
    ceiling = 1 - (shares**2).sum()
    return partition_modularity(graph, labels) / ceiling if ceiling > 0 else np.nan


def best_partition(graph: nx.Graph, seeds=range(5), resolution: float = 1.0):
    """Run Louvain with several seeds. Return the best modularity, its labels, and the mean pairwise ARI between runs."""
    runs = []
    for seed in seeds:
        communities = louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
        runs.append((modularity(graph, communities, weight="weight", resolution=resolution), communities_to_labels(communities)))
    best_modularity, best_labels = max(runs, key=lambda run: run[0])
    pairs = [adjusted_rand_score(a[1], b[1].reindex(a[1].index)) for a, b in combinations(runs, 2)]
    return best_modularity, best_labels, float(np.mean(pairs)) if pairs else np.nan


def cross_party_share(labels: pd.Series, parties: pd.Series) -> float:
    """Share of Democrats and Republicans whose community is mostly made up of the other party."""
    frame = pd.DataFrame({"community": labels, "party": parties.reindex(labels.index)})
    frame = frame[frame["party"].isin(["D", "R"])]
    majority = frame.groupby("community")["party"].agg(lambda s: s.value_counts().idxmax())
    return float((frame["party"] != frame["community"].map(majority)).mean())


def describe_communities(labels: pd.Series, parties: pd.Series, south: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"community": labels, "party": parties.reindex(labels.index), "south": south.reindex(labels.index)})
    table = frame.groupby("community").agg(
        size=("party", "size"),
        share_d=("party", lambda s: s.eq("D").mean()),
        share_r=("party", lambda s: s.eq("R").mean()),
        share_south=("south", "mean"),
    )
    table["share_of_chamber"] = table["size"] / len(frame)
    table["kind"] = np.select(
        [table["share_d"] >= BLOC_PURITY, table["share_r"] >= BLOC_PURITY], ["D bloc", "R bloc"], "mixed"
    )
    return table.sort_values("size", ascending=False)


def permuted_matrix(matrix: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle each roll call's entries among members, keeping roll-call margins but removing member consistency."""
    values = matrix.to_numpy().copy()
    for column in range(values.shape[1]):
        values[:, column] = rng.permutation(values[:, column])
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)


def faction_summary(matrix: pd.DataFrame, south: pd.Series, seeds, null_seed: int, null_draws: int = 10) -> dict:
    """Community structure within one party, against shuffled-vote nulls that keep roll-call margins.

    The null removes all member consistency, so beating it shows structure, not necessarily discrete groups.
    """
    graph = agreement_graph(agreement_matrix(matrix))
    observed, labels, stability = best_partition(graph, seeds)
    nulls = []
    for draw in range(null_draws):
        rng = np.random.default_rng(null_seed + draw)
        null_graph = agreement_graph(agreement_matrix(permuted_matrix(matrix, rng)))
        nulls.append(best_partition(null_graph, [null_seed + draw])[0])
    sizes = labels.value_counts(normalize=True)
    factions = sizes[sizes >= FACTION_SIZE].index
    south_shares = south.reindex(labels.index).groupby(labels).mean().reindex(factions)
    return {
        "modularity": observed,
        "null_modularity": float(np.mean(nulls)),
        "null_max": float(np.max(nulls)),
        "excess_modularity": observed - float(np.mean(nulls)),
        "stability": stability,
        "factions": len(factions),
        "south_gap": float(south_shares.max() - south_shares.min()) if len(factions) >= 2 else np.nan,
    }


def network_summary(matrix, parties, south, seeds=range(5), factions=True, null_seed=0, resolution=1.0):
    """Network measures for one Congress and chamber. Returns the measures and the best community labels."""
    parties = parties.reindex(matrix.index).fillna("Other")
    south = south.reindex(matrix.index).fillna(False)
    agreement = agreement_matrix(matrix)
    graph = agreement_graph(agreement)
    best, labels, stability = best_partition(graph, seeds, resolution)
    two_party = parties.isin(["D", "R"])
    sizes = labels.value_counts(normalize=True)
    pairs = len(agreement) * (len(agreement) - 1)
    result = {
        "members": len(matrix),
        "missing_pairs": float(agreement.isna().to_numpy().sum() - len(agreement)) / pairs if pairs else np.nan,
        "party_modularity": partition_modularity(graph, parties),
        "party_assortativity": normalized_modularity(graph, parties),
        "best_modularity": best,
        "seed_stability": stability,
        "large_communities": int((sizes >= LARGE_COMMUNITY).sum()),
        "party_ari": adjusted_rand_score(parties[two_party], labels[two_party]),
        "cross_party_share": cross_party_share(labels, parties),
    }
    if factions:
        for party in ["D", "R"]:
            party_matrix = matrix[parties == party]
            party_matrix = party_matrix.loc[:, party_matrix.notna().sum() >= 5]
            for key, value in faction_summary(party_matrix, south, list(seeds)[:3], null_seed).items():
                result[f"{party}_{key}"] = value
    return result, labels


def classical_mds(agreement: pd.DataFrame, dimensions: int = 2) -> tuple[pd.DataFrame, float]:
    """Embed members from 1 - agreement without imputing votes. Also return the share of positive eigenvalue mass kept."""
    distance = 1 - agreement.to_numpy(dtype="float64")
    np.fill_diagonal(distance, 0)
    distance = np.where(np.isnan(distance), np.nanmean(distance), distance)
    size = len(distance)
    centring = np.eye(size) - np.ones((size, size)) / size
    inner = -0.5 * centring @ (distance**2) @ centring
    values, vectors = np.linalg.eigh(inner)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    coordinates = vectors[:, :dimensions] * np.sqrt(np.clip(values[:dimensions], 0, None))
    kept = values[:dimensions].clip(0).sum() / values.clip(0).sum()
    return pd.DataFrame(coordinates, index=agreement.index, columns=[f"dim{i + 1}" for i in range(dimensions)]), float(kept)


def choose_mixture(points: np.ndarray, max_components: int, seed: int = 0):
    """Fit Gaussian mixtures with 1 to max_components components and keep the lowest BIC."""
    fits = []
    for components in range(1, max_components + 1):
        model = GaussianMixture(n_components=components, covariance_type="full", n_init=5, random_state=seed).fit(points)
        fits.append((model.bic(points), components, model.predict(points)))
    bic, components, labels = min(fits, key=lambda fit: fit[0])
    return components, labels


def match_communities(before: pd.Series, after: pd.Series) -> pd.DataFrame:
    """Jaccard overlap between communities in consecutive Congresses, over people present in both."""
    common = before.index.intersection(after.index)
    rows = []
    for old, old_people in before[common].groupby(before[common]).groups.items():
        for new, new_people in after[common].groupby(after[common]).groups.items():
            union = len(set(old_people) | set(new_people))
            rows.append({"before": old, "after": new, "jaccard": len(set(old_people) & set(new_people)) / union})
    return pd.DataFrame(rows)


def decompose_change(before: pd.Series, after: pd.Series, others_before, others_after) -> dict:
    """Split the change in one party's mean score into conversion, switching and replacement.

    before and after map person to score for the party in two consecutive Congresses. others_before and
    others_after are the people in the other party, used to identify switchers. Arrivals and departures are
    measured from the earlier mean, so the parts add up exactly and don't depend on where the scale's zero is.
    """
    reference = before.mean()
    stayers = before.index.intersection(after.index)
    arrived = after.index.difference(before.index)
    departed = before.index.difference(after.index)
    switch_in = arrived.intersection(others_before)
    switch_out = departed.intersection(others_after)
    entrants = arrived.difference(switch_in)
    leavers = departed.difference(switch_out)
    size = len(after)
    return {
        "total": after.mean() - reference,
        "conversion": (after[stayers] - before[stayers]).sum() / size,
        "switching": ((after[switch_in] - reference).sum() - (before[switch_out] - reference).sum()) / size,
        "replacement": ((after[entrants] - reference).sum() - (before[leavers] - reference).sum()) / size,
        "stayers": len(stayers),
        "switchers": len(switch_in) + len(switch_out),
        "switched_in": list(switch_in),
        "switched_out": list(switch_out),
    }


def defection_rates(votes: pd.DataFrame) -> pd.DataFrame:
    """Share of party-opposed roll calls on which each member voted with the other party's majority."""
    keys = ["congress", "chamber", "rollnumber"]
    positions = votes[(votes["vote"] != 0) & votes["party"].isin(["D", "R"])]
    shares = positions.assign(yea=positions["vote"].eq(1)).groupby(keys + ["party"])["yea"].mean().unstack()
    majority = np.sign(shares - 0.5)
    opposed = majority[(majority["D"] * majority["R"]) == -1].rename(columns={"D": "majority_d", "R": "majority_r"})
    merged = positions.merge(opposed.reset_index(), on=keys)
    other = np.where(merged["party"] == "D", merged["majority_r"], merged["majority_d"])
    merged["defected"] = merged["vote"].to_numpy() == other
    return merged.groupby(["congress", "chamber", "icpsr", "party"])["defected"].agg(["mean", "count"]).reset_index()
