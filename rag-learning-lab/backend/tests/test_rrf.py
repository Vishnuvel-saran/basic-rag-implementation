from __future__ import annotations

import pytest

from app.retrieval.rrf import fuse_results


def test_rrf_scoring_formula():
    """Test that RRF calculates scores correctly using the formula: 1/(k+rank)."""
    # Create simple result sets with known scores
    result_sets = {
        "method1": [
            {"chunk_id": "a", "text": "doc a", "retrieval_score": 0.9},
            {"chunk_id": "b", "text": "doc b", "retrieval_score": 0.8},
        ],
        "method2": [
            {"chunk_id": "a", "text": "doc a", "retrieval_score": 0.7},
            {"chunk_id": "c", "text": "doc c", "retrieval_score": 0.6},
        ],
    }

    fused = fuse_results(result_sets, k_constant=60)

    # Find doc 'a' in results - it appears in both sets at rank 1
    doc_a = next((r for r in fused if r["chunk_id"] == "a"), None)
    assert doc_a is not None

    # RRF for doc_a: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.032787
    expected_score = 1 / (60 + 1) + 1 / (60 + 1)
    assert abs(doc_a["rrf_score"] - expected_score) < 0.0001

    # Doc b should be second (rank 1 in method1, not in method2)
    doc_b = next((r for r in fused if r["chunk_id"] == "b"), None)
    assert doc_b is not None
    expected_b_score = 1 / (60 + 2)
    assert abs(doc_b["rrf_score"] - expected_b_score) < 0.0001


def test_rrf_accumulates_scores_from_multiple_sets():
    """Test that documents appearing in multiple result sets accumulate RRF scores."""
    result_sets = {
        "set1": [
            {"chunk_id": "doc1", "text": "content1"},
            {"chunk_id": "doc2", "text": "content2"},
        ],
        "set2": [
            {"chunk_id": "doc1", "text": "content1"},
            {"chunk_id": "doc3", "text": "content3"},
        ],
    }

    fused = fuse_results(result_sets, k_constant=60)

    doc1 = next((r for r in fused if r["chunk_id"] == "doc1"), None)
    assert doc1 is not None

    # Doc1 appears at rank 1 in both sets
    # Score = 1/(60+1) + 1/(60+1) = 2/61
    expected = 2 / 61
    assert abs(doc1["rrf_score"] - expected) < 0.0001


def test_rrf_handles_document_only_in_one_set():
    """Test RRF correctly scores documents that appear in only one result set."""
    result_sets = {
        "set1": [
            {"chunk_id": "doc1", "text": "content1"},
            {"chunk_id": "doc2", "text": "content2"},
        ],
        "set2": [
            {"chunk_id": "doc3", "text": "content3"},
        ],
    }

    fused = fuse_results(result_sets, k_constant=60)

    doc2 = next((r for r in fused if r["chunk_id"] == "doc2"), None)
    doc3 = next((r for r in fused if r["chunk_id"] == "doc3"), None)

    assert doc2 is not None
    assert doc3 is not None

    # Doc2 at rank 2 in set1, not in set2: score = 1/(60+2)
    expected_doc2 = 1 / (60 + 2)
    assert abs(doc2["rrf_score"] - expected_doc2) < 0.0001

    # Doc3 at rank 1 in set2, not in set1: score = 1/(60+1)
    expected_doc3 = 1 / (60 + 1)
    assert abs(doc3["rrf_score"] - expected_doc3) < 0.0001


def test_rrf_respects_top_k():
    """Test that RRF returns only top_k results."""
    result_sets = {
        "set1": [
            {"chunk_id": f"doc{i}", "text": f"content{i}"} for i in range(10)
        ],
        "set2": [
            {"chunk_id": f"doc{i}", "text": f"content{i}"} for i in range(10, 20)
        ],
    }

    fused_all = fuse_results(result_sets, k_constant=60, top_k=None)
    fused_top5 = fuse_results(result_sets, k_constant=60, top_k=5)

    assert len(fused_all) == 20
    assert len(fused_top5) == 5


def test_rrf_preserves_metadata():
    """Test that RRF preserves all metadata from original results."""
    result_sets = {
        "set1": [
            {
                "chunk_id": "doc1",
                "text": "content",
                "metadata": {"page": 1, "source": "file.pdf"},
                "retrieval_method": "method1",
            }
        ],
    }

    fused = fuse_results(result_sets, k_constant=60)

    assert fused[0]["chunk_id"] == "doc1"
    assert fused[0]["text"] == "content"
    assert fused[0]["metadata"]["page"] == 1
    assert fused[0]["metadata"]["source"] == "file.pdf"


def test_rrf_includes_rank_metadata():
    """Test that RRF includes rank and score metadata for each method."""
    result_sets = {
        "set1": [
            {"chunk_id": "doc1", "text": "content", "retrieval_score": 0.95},
        ],
        "set2": [
            {"chunk_id": "doc1", "text": "content", "retrieval_score": 0.80},
        ],
    }

    fused = fuse_results(result_sets, k_constant=60)

    result = fused[0]
    assert "set1_rank" in result
    assert "set1_score" in result
    assert "set2_rank" in result
    assert "set2_score" in result
    assert result["set1_rank"] == 1
    assert result["set1_score"] == 0.95
    assert result["set2_rank"] == 1
    assert result["set2_score"] == 0.80


def test_rrf_with_different_k_constants():
    """Test that different k constants produce different scores."""
    result_sets = {
        "set1": [
            {"chunk_id": "doc1", "text": "content"},
        ],
    }

    fused_k30 = fuse_results(result_sets, k_constant=30)
    fused_k60 = fuse_results(result_sets, k_constant=60)
    fused_k100 = fuse_results(result_sets, k_constant=100)

    score_k30 = fused_k30[0]["rrf_score"]
    score_k60 = fused_k60[0]["rrf_score"]
    score_k100 = fused_k100[0]["rrf_score"]

    # Larger k should give smaller scores
    assert score_k30 > score_k60 > score_k100


def test_rrf_handles_empty_result_sets():
    """Test that RRF handles empty result sets gracefully."""
    result_sets = {
        "set1": [],
        "set2": [],
    }

    fused = fuse_results(result_sets, k_constant=60)
    assert fused == []


def test_rrf_handles_partial_empty_result_sets():
    """Test RRF when one set is empty."""
    result_sets = {
        "set1": [
            {"chunk_id": "doc1", "text": "content1"},
        ],
        "set2": [],
    }

    fused = fuse_results(result_sets, k_constant=60)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "doc1"


def test_rrf_sorts_by_score_descending():
    """Test that RRF results are sorted by score in descending order."""
    result_sets = {
        "set1": [
            {"chunk_id": "doc1", "text": "content1"},
            {"chunk_id": "doc2", "text": "content2"},
            {"chunk_id": "doc3", "text": "content3"},
        ],
    }

    fused = fuse_results(result_sets, k_constant=60)

    # Scores should be in descending order
    scores = [r["rrf_score"] for r in fused]
    assert scores == sorted(scores, reverse=True)

    # First result should have rank 1 (highest score)
    assert fused[0]["set1_rank"] == 1


def test_rrf_with_no_chunk_ids_handles_gracefully():
    """Test that RRF skips results without chunk_id."""
    result_sets = {
        "set1": [
            {"chunk_id": "doc1", "text": "content1"},
            {"text": "content_no_id"},  # Missing chunk_id
        ],
    }

    fused = fuse_results(result_sets, k_constant=60)
    assert len(fused) == 1
    assert fused[0]["chunk_id"] == "doc1"
