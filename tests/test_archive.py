from pathlib import Path

import pytest

from tesla_lab.archive import (ArchiveIndex, NebiusEmbedder, build_index, chunk_text, clean_ocr, hits_as_sources,
                               load_index, merge_evidence, save_index, search)
from tesla_lab.schema import Source

ROOT = Path(__file__).resolve().parents[1]


def test_clean_ocr_fixes_patent_scan_noise():
    assert clean_ocr("tranSmiSSion of the COnductOrs paSS") == "transmission of the Conductors pass"


def test_chunking_overlaps_and_covers():
    text = " ".join(f"word{i}." for i in range(600))
    chunks = chunk_text(text, size=500, overlap=100)
    assert len(chunks) > 3 and all(len(c) <= 520 for c in chunks)
    assert "word599." in chunks[-1]


def _make_pdf(path: Path):
    from fpdf import FPDF

    pdf = FPDF()
    for body in ["Page one is about bicycles and gears. " * 20,
                 "Page two describes a resonant coil tuned to the frequency of the receiver. " * 15]:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, body)
    pdf.output(str(path))


def test_ingest_and_cite_page(tmp_path):
    (tmp_path / "pdfs").mkdir()
    _make_pdf(tmp_path / "pdfs" / "demo.pdf")
    (tmp_path / "sources.json").write_text('{"demo.pdf": {"id": "D1", "title": "Demo doc", "url": "https://x.org/d"}}')
    idx = build_index(tmp_path, log=lambda *a: None)
    save_index(idx, tmp_path)
    hits = search("resonant coil tuned receiver", archive_dir=tmp_path)
    assert hits and hits[0].doc_id == "D1" and hits[0].page == 2 and hits[0].title == "Demo doc"
    src = hits_as_sources(hits)
    assert src[0].id == "A1" and src[0].url.endswith("#page=2")


def test_shipped_sample_archive_finds_patent_page():
    idx = load_index(ROOT / "archive")
    assert idx is not None and len(idx.meta["docs"]) >= 5
    hits = search("earth stationary waves terrestrial globe resonance nodal regions", archive_dir=ROOT / "archive")
    assert hits[0].doc_id == "US787412A" and hits[0].page == 5
    hits = search("fluid adhesion viscosity spaced discs turbine", archive_dir=ROOT / "archive")
    assert hits[0].doc_id == "US1061206A"
    assert not search("perpetual motion magnets free energy", archive_dir=ROOT / "archive", min_score=0.44)


def test_merge_evidence_interleaves():
    a = [Source(id="A1", url="a1", title="a", score=0.9), Source(id="A2", url="a2", title="a", score=0.5)]
    w = [Source(id="S1", url="s1", title="s", score=0.8), Source(id="S2", url="s2", title="s", score=0.1)]
    ids = [s.id for s in merge_evidence(a, w)]
    assert set(ids[:2]) == {"A1", "S1"} and len(ids) == 4


def test_nebius_embedder_with_fake_client():
    class E:
        def __init__(self, n):
            self.data = [type("D", (), {"embedding": [1.0, 0.0, 0.0]})() for _ in range(n)]

    class C:
        class embeddings:
            calls = []

            @staticmethod
            def create(model, input, encoding_format):
                C.embeddings.calls.append(model)
                return E(len(input))

    emb = NebiusEmbedder(model="BAAI/bge-en-icl", client=C())
    v = emb.embed(["a", "b"])
    assert v.shape == (2, 3) and C.embeddings.calls == ["BAAI/bge-en-icl"]


def test_hybrid_search_with_vectors():
    import numpy as np

    from tesla_lab.archive import Chunk

    chunks = [Chunk(id=0, doc_id="X", title="t", page=1, text="alpha beta"), Chunk(id=1, doc_id="Y", title="t", page=3, text="gamma delta")]
    idx = ArchiveIndex(chunks=chunks, vectors=np.array([[1, 0], [0, 1]], dtype="float32"))

    class Emb:
        name, model = "fake", "fake"

        def embed(self, texts, is_query=False):
            return np.array([[0.0, 1.0]], dtype="float32")

    hits = idx.search("nothing lexical", embedder=Emb())
    assert hits[0].doc_id == "Y" and hits[0].page == 3


def test_sentence_transformers_backend_if_installed():
    pytest.importorskip("sentence_transformers")
    from tesla_lab.archive import STEmbedder

    v = STEmbedder().embed(["wireless power", "turbine"])
    assert v.shape[0] == 2
