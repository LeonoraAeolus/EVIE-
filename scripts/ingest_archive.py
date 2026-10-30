"""Build the Tesla Archive index.

  python scripts/ingest_archive.py                       # index every PDF in archive/pdfs/
  python scripts/ingest_archive.py --patent US645576A US1119732A   # download from Google Patents first
  python scripts/ingest_archive.py --arxiv "wireless power transfer resonant" --max 5
  python scripts/ingest_archive.py --embedder st         # semantic + keyword (pip install sentence-transformers)
  python scripts/ingest_archive.py --embedder nebius     # Nebius Token Factory embeddings (needs NEBIUS_API_KEY)

Manual sources (they block or discourage scripts, so download in a browser and drop the PDFs in):
  archive/pdfs/fbi_vault/   https://vault.fbi.gov/nikola-tesla   (returns HTTP 403 to scripts)
  archive/pdfs/cia/         https://www.cia.gov/readingroom      (search "Tesla"; PDFs per document)
Add a line per file to archive/sources.json ({"file.pdf": {"id", "title", "url", "kind"}}) for nice citations.
Scanned PDFs with no text layer need OCR: pip install pytesseract pdf2image (+ the tesseract and poppler binaries).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tesla_lab.archive import ARCHIVE_DIR, build_index, get_embedder, save_index  # noqa: E402
from tesla_lab.gaps import arxiv_get, arxiv_query_string, parse_arxiv_atom  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0 (TeslaLab archive ingest; research use)"}


def fetch_patent(number: str, catalog: dict, dest: Path) -> None:
    """Google Patents page -> its PDF link on patentimages.storage.googleapis.com (verified 2026-10-13)."""
    number = number.upper().replace(",", "").replace(" ", "")
    if not re.search(r"[A-Z]\d*$", number):
        number += "A"
    page = requests.get(f"https://patents.google.com/patent/{number}/en", headers=UA, timeout=60)
    page.raise_for_status()
    m = re.search(r"https://patentimages\.storage\.googleapis\.com/[^\"]+\.pdf", page.text)
    if not m:
        raise RuntimeError(f"no PDF link found for {number}")
    title = re.search(r"<title>\s*([^<]+?)\s*-\s*Google Patents", page.text)
    date = re.search(r'itemprop="publicationDate" datetime="(\d{4})', page.text)
    pdf = requests.get(m.group(0), headers=UA, timeout=120)
    pdf.raise_for_status()
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{number}.pdf").write_bytes(pdf.content)
    raw = title.group(1) if title else number
    name = raw.split(" - ", 1)[-1].strip()
    num = re.sub(r"^US(\d+)[A-Z]\d*$", lambda x: f"US {int(x.group(1)):,}", number)
    catalog[f"{number}.pdf"] = {"id": number, "title": f"{num} - {name}" + (f" ({date.group(1)})" if date else ""),
                                "url": f"https://patents.google.com/patent/{number}/en", "kind": "patent"}
    print(f"  downloaded {number}: {name}")


def fetch_arxiv(query: str, n: int, catalog: dict, dest: Path) -> None:
    r = arxiv_get({"search_query": arxiv_query_string(query), "max_results": n, "sortBy": "relevance"}, timeout=60)
    dest.mkdir(parents=True, exist_ok=True)
    for lead in parse_arxiv_atom(r.text, query):
        aid = lead.url.rstrip("/").split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{aid}"
        time.sleep(3.1)  # arXiv asks for ~1 request / 3 s
        pdf = requests.get(pdf_url, headers=UA, timeout=120)
        if pdf.status_code != 200 or not pdf.content.startswith(b"%PDF"):
            print(f"  ! skipped {aid}")
            continue
        fn = f"arxiv_{aid.replace('/', '_')}.pdf"
        (dest / fn).write_bytes(pdf.content)
        catalog[fn] = {"id": f"arXiv:{aid}", "title": f"arXiv {aid} - {lead.title} ({(lead.published or '')[:4]})",
                       "url": lead.url, "kind": "arxiv"}
        print(f"  downloaded arXiv {aid}: {lead.title[:70]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patent", nargs="*", default=[], help="patent numbers, e.g. US645576A")
    ap.add_argument("--arxiv", help="arXiv search query to download papers for")
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--embedder", default=None, help="lexical (default) | st | nebius")
    ap.add_argument("--archive-dir", default=str(ARCHIVE_DIR))
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
    adir = Path(a.archive_dir)
    cat_path = adir / "sources.json"
    catalog = json.loads(cat_path.read_text()) if cat_path.exists() else {}
    for p in a.patent:
        try:
            fetch_patent(p, catalog, adir / "pdfs" / "patents")
        except Exception as e:
            print(f"  ! {p}: {e}")
    if a.arxiv:
        fetch_arxiv(a.arxiv, a.max, catalog, adir / "pdfs" / "arxiv")
    cat_path.write_text(json.dumps(catalog, indent=2))
    emb = get_embedder(a.embedder) if a.embedder else None
    print(f"Indexing {adir / 'pdfs'} (embedder: {emb.name + ':' + emb.model if emb else 'lexical BM25'})")
    idx = build_index(adir, embedder=emb, ocr=not a.no_ocr)
    out = save_index(idx, adir)
    print(f"Done: {idx.meta['n_chunks']} chunks from {len(idx.meta['docs'])} documents -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
