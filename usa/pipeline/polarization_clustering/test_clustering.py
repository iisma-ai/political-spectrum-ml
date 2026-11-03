"""Unit tests for the clustering measures, using small hand-written tables."""

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from clustering import (
    normalized_modularity,
    partition_modularity,
    agreement_graph,
    agreement_matrix,
    choose_mixture,
    classical_mds,
    cross_party_share,
    decompose_change,
    defection_rates,
    match_communities,
    network_summary,
)


def test_agreement_counts_only_shared_votes():
    matrix = pd.DataFrame([[1, 1, -1, np.nan], [1, -1, -1, 1], [np.nan, np.nan, 1, 1]], index=["a", "b", "c"])
    agreement = agreement_matrix(matrix, min_shared=1)
    assert agreement.loc["a", "b"] == pytest.approx(2 / 3)
    assert agreement.loc["b", "c"] == pytest.approx(1 / 2)
    assert agreement.loc["a", "c"] == pytest.approx(0)
    assert np.isnan(agreement.loc["a", "a"])
    assert np.isnan(agreement_matrix(matrix, min_shared=2).loc["a", "c"])


def test_graph_skips_missing_and_zero_agreement():
    agreement = pd.DataFrame([[np.nan, 0.5, 0.0], [0.5, np.nan, np.nan], [0.0, np.nan, np.nan]], index=list("abc"), columns=list("abc"))
    graph = agreement_graph(agreement)
    assert set(graph.nodes) == {"a", "b", "c"}
    assert list(graph.edges(data="weight")) == [("a", "b", 0.5)]


def test_cross_party_share():
    labels = pd.Series({"d1": 0, "d2": 0, "r1": 0, "r2": 1, "r3": 1})
    parties = pd.Series({"d1": "D", "d2": "D", "r1": "R", "r2": "R", "r3": "R"})
    assert cross_party_share(labels, parties) == pytest.approx(1 / 5)


def test_network_summary_separates_two_blocs():
    rng = np.random.default_rng(0)
    dem = np.where(rng.random((12, 60)) < 0.9, 1, -1)
    rep = -np.where(rng.random((12, 60)) < 0.9, 1, -1)
    matrix = pd.DataFrame(np.vstack([dem, rep]), index=range(24)).astype(float)
    parties = pd.Series(["D"] * 12 + ["R"] * 12, index=range(24))
    south = pd.Series(False, index=range(24))
    result, labels = network_summary(matrix, parties, south, seeds=range(3), factions=False)
    assert result["party_ari"] == pytest.approx(1)
    assert result["cross_party_share"] == 0
    assert result["party_modularity"] > 0


BEFORE = pd.Series({"stay": 0.2, "leaves": 0.6, "switches_out": 0.0})
AFTER = pd.Series({"stay": 0.4, "enters": 0.8, "switches_in": 0.5, "other_entrant": 0.1})


def decompose(shift=0.0):
    return decompose_change(BEFORE + shift, AFTER + shift, others_before=pd.Index(["switches_in"]), others_after=pd.Index(["switches_out"]))


def test_decomposition_adds_up_and_finds_switchers():
    parts = decompose()
    reference = BEFORE.mean()
    assert parts["total"] == pytest.approx(AFTER.mean() - BEFORE.mean())
    assert parts["conversion"] + parts["switching"] + parts["replacement"] == pytest.approx(parts["total"])
    assert parts["conversion"] == pytest.approx(0.2 / 4)
    assert parts["switching"] == pytest.approx(((0.5 - reference) - (0.0 - reference)) / 4)
    assert parts["replacement"] == pytest.approx(((0.8 - reference) + (0.1 - reference) - (0.6 - reference)) / 4)
    assert parts["switchers"] == 2


def test_decomposition_does_not_depend_on_the_scale_zero():
    plain, shifted = decompose(), decompose(shift=1.0)
    for part in ["total", "conversion", "switching", "replacement"]:
        assert shifted[part] == pytest.approx(plain[part])


def test_switcher_at_the_party_mean_contributes_nothing():
    before = pd.Series({"a": 0.2, "b": 0.6})
    after = pd.Series({"a": 0.2, "b": 0.6, "switcher": 0.4})
    parts = decompose_change(before, after, others_before=pd.Index(["switcher"]), others_after=pd.Index([]))
    assert parts["switching"] == pytest.approx(0)
    assert parts["replacement"] == pytest.approx(0)


def test_normalized_modularity_is_one_for_separate_groups():
    graph = nx.Graph()
    graph.add_weighted_edges_from([("a", "b", 1.0), ("c", "d", 1.0), ("d", "e", 1.0), ("c", "e", 1.0)])
    labels = pd.Series({"a": "D", "b": "D", "c": "R", "d": "R", "e": "R"})
    assert normalized_modularity(graph, labels) == pytest.approx(1)
    assert partition_modularity(graph, labels) < 0.5


def test_classical_mds_recovers_distances():
    points = np.array([[0.0, 0.0], [0.3, 0.0], [0.0, 0.4], [0.3, 0.4]])
    distance = np.sqrt(((points[:, None] - points[None]) ** 2).sum(axis=2))
    agreement = pd.DataFrame(1 - distance, index=list("abcd"), columns=list("abcd"))
    coordinates, kept = classical_mds(agreement)
    recovered = np.sqrt(((coordinates.to_numpy()[:, None] - coordinates.to_numpy()[None]) ** 2).sum(axis=2))
    assert np.allclose(recovered, distance)
    assert kept == pytest.approx(1)


def test_choose_mixture_finds_two_groups():
    rng = np.random.default_rng(0)
    points = np.vstack([rng.normal(-3, 0.3, (40, 2)), rng.normal(3, 0.3, (40, 2))])
    components, labels = choose_mixture(points, 4)
    assert components == 2
    assert len(set(labels[:40])) == 1 and labels[0] != labels[-1]


def test_match_communities_uses_people_in_both():
    before = pd.Series({"a": 0, "b": 0, "c": 1, "gone": 1})
    after = pd.Series({"a": 5, "b": 5, "c": 5, "new": 6})
    matches = match_communities(before, after).set_index(["before", "after"])["jaccard"]
    assert matches[(0, 5)] == pytest.approx(2 / 3)
    assert matches[(1, 5)] == pytest.approx(1 / 3)


def test_defection_rates_count_votes_with_other_majority():
    votes = pd.DataFrame(
        {
            "congress": 100, "chamber": "House", "rollnumber": [1] * 6 + [2] * 6,
            "icpsr": [1, 2, 3, 4, 5, 6] * 2,
            "party": ["D", "D", "D", "R", "R", "R"] * 2,
            "vote": [1, 1, -1, -1, -1, -1] + [1, 1, 1, 1, 1, -1],
        }
    )
    rates = defection_rates(votes).set_index("icpsr")
    assert rates.loc[3, "mean"] == 1 and rates.loc[3, "count"] == 1
    assert rates.loc[1, "mean"] == 0
