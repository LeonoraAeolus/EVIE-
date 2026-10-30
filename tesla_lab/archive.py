"""Tesla Archive: local retrieval over primary documents, cited by document + page.

Where the documents come from (all public):
  * Tesla's patents      Google Patents PDFs (scripts/ingest_archive.py --patent US645576A ...)
  * arXiv papers         arXiv API (scripts/ingest_archive.py --arxiv "query")
  * CIA reading room     https://www.cia.gov/readingroom  -> download PDFs, drop into archive/pdfs/
  * FBI Vault, Tesla     https://vault.fbi.gov/nikola-tesla blocks scripts (HTTP 403 to non-browsers),
                         so download the parts in a browser and drop them into archive/pdfs/
Every PDF in archive/pdfs/ is split into pages, cleaned, chunked, embedded and indexed into
archive/index/. Search returns ArchiveHit(title, page, snippet, score) so the UI can say
"found in US 645,576, page 3".

Embedding backends (ARCHIVE_EMBEDDER):
  lexical   BM25 over the chunks; pure Python, no download. Default, used by the sample index.
  st        sentence-transformers (local, e.g. all-MiniLM-L6-v2), hybrid with BM25.
            pip install sentence-transformers
  nebius    Nebius Token Factory /v1/embeddings (OpenAI-compatible). The docs' example model is
            BAAI/bge-en-icl: https://docs.tokenfactory.nebius.com/api-reference/examples/create-embeddings
            (not yet tested with a key), hybrid with BM25.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .schema import ArchiveHit, Source

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", ROOT / "archive"))
STOP = set("""a an and are as at be been by for from has have in into is it its of on or that the
their this to was were which with will can may not but also such these those than then there
so if all any each other some our we i my he his she her they them you your one two said""".split())


# ----------------------------------------------------------------------------- text
def clean_ocr(text: str) -> str:
    """Fix the commonest OCR damage in 1890-1915 patent scans (stray capital S, line numbers)."""
    t = text.replace("\u00ad", "")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)  # hyphenated line breaks
    t = re.sub(r"(?<=[a-z])S(?=[a-z]|\b)", "s", t)  # tranS -> trans, paSS -> pass
    t = re.sub(r"(?<=[a-z])S", "s", t)
    t = re.sub(r"(?<=[a-z])([A-Z])(?=[a-z]|\b)", lambda m: m.group(1).lower(), t)  # flоW, SerVe
    t = re.sub(r"\b([A-Z])([A-Z])(?=[a-z]{2,})", lambda m: m.group(1) + m.group(2).lower(), t)  # COnduct
    t = re.sub(r"\b(?:\d{1,3}\?|\?\d|[0-9]{2,3})\s*$", "", t, flags=re.M)  # margin line numbers
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip()


def tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2]


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):  # end on a sentence or word boundary
            k = max(text.rfind(". ", i, j), text.rfind(" ", i + size // 2, j))
            j = k + 1 if k > i else j
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return [c for c in out if len(c) > 40]


def pdf_pages(path: str | Path, ocr: bool = True) -> list[tuple[int, str]]:
    """[(page_number, text)] for a PDF. Pages with no text layer are OCR'd if pytesseract and
    pdf2image are installed (optional; scanned FOIA/FBI files usually need it)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        if len(txt.strip()) < 40 and ocr:
            txt = _ocr_page(path, i) or txt
        pages.append((i, clean_ocr(txt)))
    return pages


def _ocr_page(path: str | Path, page: int) -> str:
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return ""
    try:
        img = convert_from_path(str(path), dpi=300, first_page=page, last_page=page)[0]
        return pytesseract.image_to_string(img)
    except Exception:
        return ""


# ----------------------------------------------------------------------------- embedders
class Embedder:
    name = "base"
    model = ""

    def embed(self, texts: list[str], is_query: bool = False):  # -> numpy array (n, d), L2-normalised
        raise NotImplementedError


class STEmbedder(Embedder):
    name = "st"

    def __init__(self, model: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = model or os.getenv("ARCHIVE_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self._m = SentenceTransformer(self.model)

    def embed(self, texts, is_query=False):
        return self._m.encode(list(texts), normalize_embeddings=True, convert_to_numpy=True, batch_size=32)


class NebiusEmbedder(Embedder):
    name = "nebius"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("NEBIUS_EMBED_MODEL", "BAAI/bge-en-icl")
        if client is None:
            from openai import OpenAI

            from .config import get_settings

            s = get_settings("live")
            if not s.nebius_api_key:
                raise RuntimeError("NEBIUS_API_KEY is needed for ARCHIVE_EMBEDDER=nebius")
            client = OpenAI(base_url=s.nebius_base_url, api_key=s.nebius_api_key)
        self.client = client

    def embed(self, texts, is_query=False):
        import numpy as np

        vecs = []
        texts = list(texts)
        for i in range(0, len(texts), 32):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + 32], encoding_format="float")
            vecs += [d.embedding for d in resp.data]
        a = np.asarray(vecs, dtype="float32")
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)


_EMBEDDERS: dict[str, Embedder] = {}


def get_embedder(name: str | None = None) -> Embedder | None:
    name = (name or os.getenv("ARCHIVE_EMBEDDER", "lexical")).strip().lower()
    if name in ("", "lexical", "bm25"):
        return None
    if name in _EMBEDDERS:
        return _EMBEDDERS[name]
    if name in ("st", "sentence-transformers", "local"):
        _EMBEDDERS[name] = STEmbedder()
        return _EMBEDDERS[name]
    if name == "nebius":
        _EMBEDDERS[name] = NebiusEmbedder()
        return _EMBEDDERS[name]
    raise ValueError(f"ARCHIVE_EMBEDDER must be lexical, st or nebius, not {name!r}")


# ----------------------------------------------------------------------------- index
@dataclass
class Chunk:
    id: int
    doc_id: str
    title: str
    page: int
    text: str
    source_url: str = ""
    file: str = ""
    kind: str = "other"


@dataclass
class ArchiveIndex:
    chunks: list[Chunk]
    meta: dict = field(default_factory=dict)
    vectors: object = None  # numpy array or None
    _df: Counter = field(default_factory=Counter)
    _tfs: list = field(default_factory=list)
    _avgdl: float = 1.0

    def __post_init__(self):
        self._tfs = [Counter(tokenize(c.title + " " + c.text)) for c in self.chunks]
        for tf in self._tfs:
            self._df.update(tf.keys())
        self._avgdl = sum(sum(tf.values()) for tf in self._tfs) / max(1, len(self._tfs))

    # BM25 (Robertson/Sparck Jones), k1=1.5, b=0.75
    def bm25(self, query: str) -> list[float]:
        q = tokenize(query)
        n = len(self.chunks)
        scores = []
        for tf in self._tfs:
            dl = sum(tf.values()) or 1
            s = 0.0
            for w in q:
                if w in tf:
                    idf = math.log(1 + (n - self._df[w] + 0.5) / (self._df[w] + 0.5))
                    s += idf * tf[w] * 2.5 / (tf[w] + 1.5 * (1 - 0.75 + 0.75 * dl / self._avgdl))
            scores.append(s)
        return scores

    def search(self, query: str, k: int = 5, embedder: Embedder | None = None, min_score: float = 0.0,
               per_doc: int = 2) -> list[ArchiveHit]:
        """Ranked hits. score is an absolute 0..1 match strength (so a threshold means the same for
        every query): keyword strength = BM25 per query term / 2.5, semantic = cosine similarity."""
        if not self.chunks:
            return []
        lex = self.bm25(query)
        nq = max(1, len(tokenize(query)))
        strength = [min(1.0, x / nq / 2.5) for x in lex]
        order_lex = [i for i in sorted(range(len(lex)), key=lambda i: -lex[i]) if lex[i] > 0]
        if self.vectors is not None and embedder is not None:
            import numpy as np

            qv = embedder.embed([query], is_query=True)[0]
            dense = np.asarray(self.vectors) @ qv
            order_dense = [int(i) for i in np.argsort(-dense)]
            # Reciprocal-rank fusion of keyword and semantic rankings
            rrf: dict[int, float] = {}
            for order in (order_lex, order_dense):
                for r, i in enumerate(order[:50]):
                    rrf[i] = rrf.get(i, 0) + 1 / (60 + r)
            ranked = sorted(rrf, key=lambda i: -rrf[i])
            score = {i: max(strength[i], float(dense[i])) for i in ranked}
        else:
            ranked = order_lex
            score = {i: strength[i] for i in ranked}
        hits, per, pages = [], Counter(), set()
        for i in ranked:
            c = self.chunks[i]
            if per[c.doc_id] >= per_doc or score[i] < min_score or (c.doc_id, c.page) in pages:
                continue
            per[c.doc_id] += 1
            pages.add((c.doc_id, c.page))
            hits.append(ArchiveHit(doc_id=c.doc_id, title=c.title, page=c.page, snippet=_snippet(c.text, query),
                                   score=round(float(score[i]), 3), source_url=c.source_url, file=c.file, kind=c.kind))
            if len(hits) >= k:
                break
        return hits


def _snippet(text: str, query: str, width: int = 320) -> str:
    q = set(tokenize(query))
    words = text.split()
    if len(" ".join(words)) <= width:
        return text
    best, best_i = -1, 0
    for i in range(0, max(1, len(words) - 10), 5):
        s = sum(1 for w in words[i:i + 45] if re.sub(r"\W", "", w.lower()) in q)
        if s > best:
            best, best_i = s, i
    out = " ".join(words[best_i:best_i + 50])
    return ("..." if best_i else "") + out[:width] + "..."


# ----------------------------------------------------------------------------- build / load
def load_catalog(archive_dir: Path = ARCHIVE_DIR) -> dict:
    p = archive_dir / "sources.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def build_index(archive_dir: Path = ARCHIVE_DIR, embedder: Embedder | None = None, ocr: bool = True,
                log=print) -> ArchiveIndex:
    archive_dir = Path(archive_dir)
    catalog = load_catalog(archive_dir)
    pdfs = sorted((archive_dir / "pdfs").rglob("*.pdf"))
    chunks: list[Chunk] = []
    docs = []
    for pdf in pdfs:
        rel = str(pdf.relative_to(archive_dir))
        info = catalog.get(pdf.name, {})
        doc_id = info.get("id", pdf.stem)
        title = info.get("title", pdf.stem.replace("_", " "))
        pages = pdf_pages(pdf, ocr=ocr)
        n0 = len(chunks)
        for page_no, text in pages:
            for piece in chunk_text(text):
                chunks.append(Chunk(id=len(chunks), doc_id=doc_id, title=title, page=page_no, text=piece,
                                    source_url=info.get("url", ""), file=rel, kind=info.get("kind", "other")))
        empty = sum(1 for _, t in pages if len(t) < 40)
        docs.append({"id": doc_id, "title": title, "file": rel, "pages": len(pages), "chunks": len(chunks) - n0,
                     "pages_without_text": empty})
        log(f"  {rel}: {len(pages)} pages, {len(chunks) - n0} chunks" + (f", {empty} pages had no text (scanned? install OCR extras)" if empty else ""))
    vectors = None
    if embedder is not None and chunks:
        log(f"  embedding {len(chunks)} chunks with {embedder.name}:{embedder.model}")
        vectors = embedder.embed([c.title + "\n" + c.text for c in chunks])
    meta = {"built": time.strftime("%Y-%m-%d %H:%M"), "embedder": embedder.name if embedder else "lexical",
            "model": getattr(embedder, "model", "") if embedder else "bm25", "docs": docs, "n_chunks": len(chunks)}
    return ArchiveIndex(chunks=chunks, meta=meta, vectors=vectors)


def save_index(index: ArchiveIndex, archive_dir: Path = ARCHIVE_DIR) -> Path:
    out = Path(archive_dir) / "index"
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(index.meta, indent=2), encoding="utf-8")
    with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in index.chunks:
            f.write(json.dumps(c.__dict__) + "\n")
    vec = out / "vectors.npy"
    if index.vectors is not None:
        import numpy as np

        np.save(vec, np.asarray(index.vectors, dtype="float32"))
    elif vec.exists():
        vec.unlink()
    return out


_CACHE: dict[str, ArchiveIndex] = {}


def load_index(archive_dir: Path = ARCHIVE_DIR) -> ArchiveIndex | None:
    d = Path(archive_dir) / "index"
    key = str(d.resolve())
    if key in _CACHE:
        return _CACHE[key]
    if not (d / "chunks.jsonl").exists():
        return None
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
    chunks = [Chunk(**json.loads(line)) for line in (d / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    vectors = None
    if (d / "vectors.npy").exists():
        import numpy as np

        vectors = np.load(d / "vectors.npy")
    idx = ArchiveIndex(chunks=chunks, meta=meta, vectors=vectors)
    _CACHE[key] = idx
    return idx


def search(query: str, k: int = 5, archive_dir: Path = ARCHIVE_DIR, min_score: float = 0.0) -> list[ArchiveHit]:
    idx = load_index(archive_dir)
    if idx is None:
        return []
    emb = None
    if idx.vectors is not None:
        try:
            emb = get_embedder(idx.meta.get("embedder"))
        except Exception:
            emb = None  # fall back to keyword search if the embedder isn't installed
    return idx.search(query, k=k, embedder=emb, min_score=min_score)


def hits_as_sources(hits: Iterable[ArchiveHit], start: int = 1) -> list[Source]:
    """Archive hits as citable sources (A1, A2, ...) so Nemotron can cite them next to Tavily's S-ids."""
    out = []
    for n, h in enumerate(hits, start=start):
        url = (h.source_url or h.file) + (f"#page={h.page}" if h.source_url or h.file else "")
        out.append(Source(id=f"A{n}", url=url, title=f"[Tesla Archive] {h.title}, page {h.page}", content=h.snippet,
                          score=h.score))
    return out


def merge_evidence(archive_sources: list[Source], web_sources: list[Source], k: int = 60) -> list[Source]:
    """Interleave archive and web evidence by reciprocal rank so neither drowns the other."""
    scored: dict[str, tuple[float, Source]] = {}
    for lst in (archive_sources, web_sources):
        ranked = sorted(lst, key=lambda s: -(s.score or 0))
        for r, s in enumerate(ranked):
            prev = scored.get(s.url, (0.0, s))[0]
            scored[s.url] = (prev + 1 / (k + r), s)
    return [s for _, s in sorted(scored.values(), key=lambda t: -t[0])]


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m tesla_lab.archive", description="Search the Tesla Archive")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args(argv)
    idx = load_index()
    if idx is None:
        print("No index yet. Run: python scripts/ingest_archive.py")
        return 1
    print(f"index: {idx.meta.get('n_chunks')} chunks, embedder={idx.meta.get('embedder')}")
    for h in search(a.query, k=a.k):
        print(f"\n[{h.score:.2f}] {h.title}, page {h.page}\n  {h.snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
