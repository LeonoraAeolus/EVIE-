import io

from pypdf import PdfReader

from tesla_lab.design import build_spec
from tesla_lab.schema import ProductConcept
from tesla_lab.spec_sheet import to_markdown, to_pdf

CONCEPT = ProductConcept(based_on_claim="C1", name="Tesla Shelf", pitch="Power without cables.", who_its_for="x",
                         how_it_works="y", honest_limits="Short range only.", image_prompt="z")


def test_markdown_spec_sheet():
    spec = build_spec("wireless_power", {}, title="Tesla Shelf")
    md = to_markdown(spec, CONCEPT, "Two tuned coils")
    assert md.startswith("# Tesla Shelf")
    assert "ICNIRP" in md and "Bill of materials (estimates)" in md and spec.headline[0] in md


def test_pdf_spec_sheet_has_text():
    for design in ("wireless_power", "disc_pump", "h2_backup"):
        spec = build_spec(design, {}, title="Demo product")
        pdf = to_pdf(spec, CONCEPT, "claim")
        assert pdf[:4] == b"%PDF"
        text = " ".join(p.extract_text() for p in PdfReader(io.BytesIO(pdf)).pages)
        assert "Demo product" in text and "Key numbers" in text
