"""Regression test for the page-misattribution bug in commit 707a76f.

The old chunk_document_pages() re-found each chunk's page by searching for
its exact word sequence in the flattened document. That search returns the
*first* match at or after a cursor, so when the same phrase (running
headers, footers, disclaimers, repeated titles) appears on more than one
page, a chunk can silently get attributed to the wrong page with no error.

These cases fix that by having each chunker report the exact word-index
span it covers while it builds the chunk, so page attribution is a direct
index lookup instead of a re-search.
"""

from app.ingestion.chunker import chunk_document_pages


def _pages(*texts: str) -> list[dict]:
    return [
        {
            "page_number": index + 1,
            "document_id": "doc-1",
            "filename": "doc.pdf",
            "text": text,
        }
        for index, text in enumerate(texts)
    ]


def test_sentence_chunks_are_correctly_paged_despite_repeated_header():
    pages = _pages(
        "Confidential Internal Report. Section Overview. The revenue grew by 12 percent this quarter.",
        "Confidential Internal Report. Section Details. The revenue grew by 12 percent this quarter due to APAC sales.",
        "Confidential Internal Report. Section Risks. Currency fluctuations pose a risk for next year outlook.",
    )

    chunks = chunk_document_pages(pages, chunk_size=12, chunk_overlap=0, strategy="sentence")

    by_text = {chunk["chunk_text"]: chunk["page_ids"] for chunk in chunks}
    assert by_text["Section Details."] == [2]
    assert by_text["Confidential Internal Report. Section Risks."] == [3]
    assert (
        by_text["The revenue grew by 12 percent this quarter due to APAC sales."]
        == [2]
    )


def test_recursive_overlap_stays_on_correct_page_with_repeated_sentence():
    pages = _pages(
        "Standard boilerplate notice text here. Alpha unique content for page one only.",
        "Standard boilerplate notice text here. Beta unique content for page two only extended further words.",
        "Standard boilerplate notice text here. Gamma unique content for page three only.",
    )

    chunks = chunk_document_pages(
        pages, chunk_size=6, chunk_overlap=3, strategy="recursive"
    )

    page_two_chunks = [c for c in chunks if "Beta" in c["chunk_text"]]
    assert page_two_chunks
    for chunk in page_two_chunks:
        assert 2 in chunk["page_ids"]

    page_three_chunks = [c for c in chunks if "Gamma" in c["chunk_text"]]
    assert page_three_chunks
    for chunk in page_three_chunks:
        assert 3 in chunk["page_ids"]
