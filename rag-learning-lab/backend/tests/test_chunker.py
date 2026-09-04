from app.ingestion.chunker import split_text_into_chunks


def test_split_text_into_chunks_uses_overlap():
    text = " ".join(f"word{i}" for i in range(30))

    chunks = split_text_into_chunks(text, chunk_size=10, chunk_overlap=2)

    assert len(chunks) > 1
    assert chunks[0].startswith("word0")
    assert chunks[1].startswith("word8")
    assert chunks[2].startswith("word16")


def test_split_text_into_chunks_rejects_invalid_overlap():
    try:
        split_text_into_chunks("hello world", chunk_size=5, chunk_overlap=10)
        assert False, "Expected ValueError for invalid overlap"
    except ValueError:
        pass
