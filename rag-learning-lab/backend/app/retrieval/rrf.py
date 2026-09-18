from __future__ import annotations

from typing import Any


def fuse_results(
    result_sets: dict[str, list[dict[str, Any]]],
    k_constant: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Fuse multiple retrieval result sets using Reciprocal Rank Fusion (RRF).

    RRF combines rankings from different retrieval methods by assigning each
    document a score based on its rank in each result set, then returning the
    top-k documents by fused score.

    Formula: RRF_score(doc) = Σ 1 / (k + rank)
    where rank is the 1-indexed position in each result set.

    Args:
        result_sets: Dict mapping method name (e.g., "bm25", "semantic") to
                     list of result dicts. Each result must have a "chunk_id".
        k_constant: The RRF k constant (default 60, standard for RRF)
        top_k: Number of top results to return. If None, returns all unique docs.

    Returns:
        List of fused result dicts sorted by RRF score (descending).
        Each result includes:
        - All fields from the original result dicts
        - rrf_score: float (the fused RRF score)
        - <method>_rank: int | None (the rank in that method's results, or None)
        - <method>_score: float | None (the score in that method's results, or None)
    """
    if not result_sets:
        return []

    # Collect all unique documents and track their scores/ranks across methods
    doc_scores: dict[str, dict[str, Any]] = {}
    doc_metadata: dict[str, dict[str, Any]] = {}

    for method, results in result_sets.items():
        for rank_idx, result in enumerate(results):
            chunk_id = result.get("chunk_id")
            if not chunk_id:
                continue

            # Calculate RRF score for this document in this method
            rank = rank_idx + 1  # 1-indexed rank
            rrf_component = 1.0 / (k_constant + rank)

            # Initialize document if not seen before
            if chunk_id not in doc_scores:
                doc_scores[chunk_id] = {"rrf_score": 0.0}
                doc_metadata[chunk_id] = result.copy()

            # Accumulate RRF score
            doc_scores[chunk_id]["rrf_score"] += rrf_component

            # Store method-specific rank and score
            retrieval_method = result.get("retrieval_method", method)
            doc_scores[chunk_id][f"{retrieval_method}_rank"] = rank
            doc_scores[chunk_id][f"{retrieval_method}_score"] = result.get("retrieval_score")

    # Sort by RRF score (descending)
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda item: item[1]["rrf_score"],
        reverse=True,
    )

    # Build final results
    final_results = []
    for chunk_id, scores in sorted_docs:
        # Start with the original result metadata
        result = doc_metadata[chunk_id].copy()

        # Add RRF-specific fields
        result["rrf_score"] = round(scores["rrf_score"], 6)

        # Add rank/score from each method (for debugging and transparency)
        for method in result_sets.keys():
            result[f"{method}_rank"] = scores.get(f"{method}_rank")
            result[f"{method}_score"] = scores.get(f"{method}_score")

        final_results.append(result)

    # Return top-k or all if top_k not specified
    if top_k is not None:
        return final_results[:top_k]
    return final_results
