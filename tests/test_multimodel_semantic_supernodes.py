import numpy as np
import pytest

from aspd.multimodel.discovery import semantic_supernodes as sn


def dist(anchors, masses):
    return sn.Distribution(np.array(anchors), np.array(masses))


def test_exact_transport_and_sinkhorn_report_euclidean_cost_and_feasible_plans():
    z = np.array([[0.0], [2.0], [5.0]])
    left, right = dist([0, 1], [1, 3]), dist([1, 2], [1, 1])
    exact, plan = sn.compare(left, right, z)
    assert exact == pytest.approx(2)
    np.testing.assert_allclose(plan.sum(0), right.mass)
    np.testing.assert_allclose(plan.sum(1), left.mass)
    reverse, _ = sn.compare(right, left, z)
    smooth, smoothed = sn.compare(left, right, z, epsilon=0.5)
    assert reverse == pytest.approx(exact)
    assert smooth >= exact - 1e-7
    np.testing.assert_allclose(smoothed.sum(1), left.mass, atol=1e-7)
    np.testing.assert_allclose(smoothed.sum(0), right.mass, atol=1e-7)


def test_uniform_mixture_preserves_common_anchor_mass():
    result = sn.mixture([dist([0, 1], [1, 1]), dist([1, 3], [1, 3])])
    assert result.anchors.tolist() == [0, 1, 3]
    np.testing.assert_allclose(result.mass, [0.25, 0.375, 0.375])


def test_average_linkage_preserves_singletons_and_eq5_cost():
    matrix = np.array([[0.0, 0.3, 1.0], [0.3, 0.0, 1.1], [1.0, 1.1, 0.0]])
    _, groups = sn.tree_and_partition(matrix, threshold=0.5)
    assert groups == [[0, 1], [2]]
    assert sn.cost_summary(groups, matrix) == {"objective": 0.3, "pairs": 1, "mean": 0.3}
    assert sn.tree_and_partition(matrix, count=3)[1] == [[0], [1], [2]]
    matrix[0, 2] = np.nan
    with pytest.raises(ValueError):
        sn.tree_and_partition(matrix, count=2)


def test_taylor_signed_cancellation_and_partial_denominator():
    source = ["gemma.L09.q/2", "gemma.L09.k/1", "gemma.L10.mlp_up/3"]
    a, b = "gemma.L09.mlp_down/3", "gemma.L10.mlp_down/2"
    observed = {
        (source[0], a): 2,
        (source[1], a): -2,
        (source[0], b): 1,
        (source[1], b): 0,
        (source[2], b): -2,
    }
    complete = sn.aggregate_taylor(source, [a, b], observed, {a: 1, b: 0.25})
    assert complete["score"] == pytest.approx(0.125)
    assert complete["observedPairs"] == 5
    assert complete["targets"][0]["signed"] == 0
    del observed[(source[1], a)]
    partial = sn.aggregate_taylor(source, [a, b], observed, {a: 1, b: 0.25})
    assert partial["score"] is None and partial["status"] == "truncated"
    assert partial["observedTargetMean"] > complete["score"]
    assert sn.aggregate_taylor([a], [source[0]], {}, {})["status"] == "undefined"
    assert sn.aggregate_taylor(source, [a], {}, {})["status"] == "unavailable"


def test_hidden_state_concatenation_preserves_layer_order_and_anchor_rows():
    layers = [np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])]
    np.testing.assert_array_equal(sn.concatenate_hidden_layers(layers), [[1, 3], [2, 4]])
    np.testing.assert_array_equal(sn.concatenate_hidden_layers(layers, [1]), layers[1])


def test_same_document_transport_control_is_not_a_measured_zero():
    cost = np.array([[0.0, 1.0], [1.0, 0.0]])
    excluded = np.eye(2, dtype=bool)
    assert sn.cost_without_pairs(cost, [0.5, 0.5], [0.5, 0.5], excluded) == pytest.approx(1.0)
    assert sn.cost_without_pairs(cost, [0.9, 0.1], [0.9, 0.1], excluded) is None


def test_native_reference_joins_exact_anchors_and_checks_seal(tmp_path):
    import json

    from aspd.multimodel.discovery.supernode_reference import load_native_reference
    from aspd.multimodel.posthoc.artifacts import sha256_file

    path = tmp_path / "gemma.npz"
    np.savez(path, anchors=["second", "first"], layers=[0, 1], hidden=[[[2], [1]], [[4], [3]]])
    marker = tmp_path / "COMPLETE.json"
    marker.write_text(
        json.dumps(
            {
                "measurementContract": "contract",
                "models": {
                    "gemma": {
                        "location": "decoder-block-output",
                        "sha256": sha256_file(path),
                        "allLayers": [0, 1],
                    }
                },
            }
        )
    )
    vectors, metadata = load_native_reference(
        tmp_path, "gemma", ["first", "second"], "contract", "all-hidden"
    )
    np.testing.assert_array_equal(vectors, [[1, 3], [2, 4]])
    assert metadata["dimensions"] == 2
    with pytest.raises(ValueError, match="contracts differ"):
        load_native_reference(tmp_path, "gemma", ["first"], "wrong", "all-hidden")
