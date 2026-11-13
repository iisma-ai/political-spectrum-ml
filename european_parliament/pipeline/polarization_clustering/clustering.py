"""Measures shared by the European Parliament polarization clustering notebooks.

The network and embedding functions use the same definitions as usa/pipeline/polarization_clustering/clustering.py,
so results can be compared. The group measures are specific to a multi-party chamber with three active positions.
"""

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.community import louvain_communities, modularity
from sklearn.metrics import adjusted_rand_score

POSITIONS = ["FOR", "AGAINST", "ABSTENTION"]
NON_ATTACHED = "NI"
# Smallest number of active positions a group needs on a roll call for its line or cohesion to count.
MIN_GROUP_VOTES = 5


# Group measures ---------------------------------------------------------------------------------------------------

def group_counts(votes: pd.DataFrame) -> pd.DataFrame:
    """For, Against and Abstention counts for each roll call and group, from member rows (Did not vote dropped)."""
    active = votes[votes["position"].isin(POSITIONS)]
    counts = (
        active.groupby(["vote_id", "group_code"], observed=True)["position"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=POSITIONS, fill_value=0)
    )
    counts.columns = [column.lower() for column in counts.columns]
    counts.columns.name = None
    return counts.astype("int32")


def agreement_index(counts: pd.DataFrame, abstention: bool = True) -> pd.Series:
    """Agreement index of each group on each roll call, after Hix, Noury and Roland (2005).

    (largest position - half of the other positions) / all positions, from 0 (an even three-way split) to 1 (all
    the same). With abstention=False, abstentions are dropped and only For and Against count.
    """
    columns = ["for", "against", "abstention"] if abstention else ["for", "against"]
    values = counts[columns]
    total = values.sum(axis=1)
    largest = values.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        index = (largest - 0.5 * (total - largest)) / total
    return index.where(total > 0)


def group_line(counts: pd.DataFrame, min_votes: int = MIN_GROUP_VOTES) -> pd.Series:
    """Each group's most common position (FOR, AGAINST or ABSTENTION) on each roll call.

    Missing when the group cast fewer than min_votes active positions or when two positions tie.
    """
    values = counts[["for", "against", "abstention"]]
    largest = values.max(axis=1)
    ties = values.eq(largest, axis=0).sum(axis=1) > 1
    line = values.idxmax(axis=1).str.upper()
    return line.where(~ties & (values.sum(axis=1) >= min_votes))


def line_agreement(lines: pd.Series) -> pd.DataFrame:
    """Share of roll calls on which two groups took the same line, over roll calls where both had one."""
    wide = lines.unstack()
    groups = list(wide.columns)
    table = pd.DataFrame(np.nan, index=groups, columns=groups)
    for first, second in combinations(groups, 2):
        both = wide[[first, second]].dropna()
        if len(both):
            table.loc[first, second] = table.loc[second, first] = (both[first] == both[second]).mean()
    for group in groups:
        table.loc[group, group] = 1.0
    return table


def winning_side(rollcalls: pd.DataFrame) -> pd.Series:
    """FOR when For outnumbers Against, AGAINST when Against outnumbers For, missing on a tie.

    A majority of votes cast, with abstentions counting for neither side. The few roll calls needing a majority of
    component members can differ; the notebooks check this against the export's recorded result.
    """
    side = pd.Series(pd.NA, index=rollcalls.index, dtype="object")
    side[rollcalls["n_for"] > rollcalls["n_against"]] = "FOR"
    side[rollcalls["n_for"] < rollcalls["n_against"]] = "AGAINST"
    return side


def modal_group(votes: pd.DataFrame) -> pd.DataFrame:
    """Each member's most common group over the given rows, the number of groups they sat in, and their country."""
    counts = votes.groupby(["member_id", "group_code"], observed=True).size().rename("rows").reset_index()
    counts = counts.sort_values(["member_id", "rows"], ascending=[True, False])
    modal = counts.drop_duplicates("member_id").set_index("member_id")["group_code"].astype(str)
    groups = counts.groupby("member_id").size()
    country = votes.drop_duplicates("member_id").set_index("member_id")["country_code"].astype(str)
    return pd.DataFrame({"group": modal, "groups_held": groups, "country": country.reindex(modal.index)})


# Member agreement networks, as in the USA ----------------------------------------------------------------------------

def agreement_matrix(matrix: pd.DataFrame, min_shared: int = 20) -> pd.DataFrame:
    """Share of the For or Against votes two members both cast on which they voted the same way."""
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
    """Modularity divided by its maximum for these group sizes (Newman's assortativity coefficient)."""
    strength = pd.Series(dict(graph.degree(weight="weight")))
    shares = strength.groupby(labels.reindex(strength.index)).sum() / strength.sum()
    ceiling = 1 - (shares**2).sum()
    return partition_modularity(graph, labels) / ceiling if ceiling > 0 else np.nan


def best_partition(graph: nx.Graph, seeds=range(5), resolution: float = 1.0):
    """Run Louvain with several seeds. Return the best modularity, its labels, and the mean pairwise ARI between runs."""
    runs = []
    for seed in seeds:
        communities = louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
        runs.append((modularity(graph, communities, weight="weight", resolution=resolution),
                     communities_to_labels(communities)))
    best_modularity, best_labels = max(runs, key=lambda run: run[0])
    pairs = [adjusted_rand_score(a[1], b[1].reindex(a[1].index)) for a, b in combinations(runs, 2)]
    return best_modularity, best_labels, float(np.mean(pairs)) if pairs else np.nan


def classical_mds(agreement: pd.DataFrame, dimensions: int = 2) -> tuple[pd.DataFrame, np.ndarray]:
    """Embed members from 1 - agreement without imputing votes.

    Also return each dimension's share of the positive eigenvalue mass, for the first ten dimensions.
    """
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
    shares = values[:10].clip(0) / values.clip(0).sum()
    columns = [f"dim{i + 1}" for i in range(dimensions)]
    return pd.DataFrame(coordinates, index=agreement.index, columns=columns), shares


def orient(coordinates: pd.DataFrame, groups: pd.Series, low: str, column: str = "dim1") -> pd.DataFrame:
    """Flip a dimension's sign so that group `low` has the lower mean. Signs of eigenvectors are arbitrary."""
    oriented = coordinates.copy()
    if oriented.loc[groups.reindex(oriented.index).eq(low), column].mean() > 0:
        oriented[column] = -oriented[column]
    return oriented


def permuted_matrix(matrix: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle each roll call's entries among members, keeping roll-call margins but removing member consistency."""
    values = matrix.to_numpy().copy()
    for column in range(values.shape[1]):
        values[:, column] = rng.permutation(values[:, column])
    return pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
