"""Spec sheet exports: Markdown and PDF (fpdf2), from a ProductSpec + concept."""
from __future__ import annotations

from .schema import ProductConcept, ProductSpec

_ASCII = {"µ": "u", "Ω": "ohm", "≈": "~", "≤": "<=", "≥": ">=", "λ": "lambda", "√": "sqrt", "·": "*", "×": "x",
          "—": "-", "–": "-", "’": "'", "‘": "'", "“": '"', "”": '"', "₂": "2", "°": " deg", "ψ": "psi", "ω": "w",
          "ν": "nu", "ρ": "rho", "η": "eta", "²": "^2", "…": "...", "→": "->", "∝": "~"}


def _a(s) -> str:
    s = str(s)
    for k, v in _ASCII.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "replace").decode("latin-1")


def _val(item) -> str:
    v = item.value
    if isinstance(v, float):
        v = f"{v:,.4g}" if abs(v) < 1e4 else f"{v:,.0f}"
    return f"{v} {item.unit}".strip()


BASIS_LABEL = {"requirement": "req", "computed": "calc", "assumed": "assumed", "standard": "std", "estimate": "est."}


def to_markdown(spec: ProductSpec, concept: ProductConcept | None = None, claim: str = "") -> str:
    L = [f"# {spec.title} - spec sheet", ""]
    if concept:
        L += [f"_{concept.pitch}_", ""]
    if claim:
        L += [f"Derived from the claim that WORKS: {claim}", ""]
    L += ["## Key numbers", ""] + [f"- {h}" for h in spec.headline] + [""]
    group = None
    for it in spec.specs:
        if it.group != group:
            group = it.group
            L += ["", f"## {group}", "", "| Spec | Value | Basis | Note |", "|---|---|---|---|"]
        L.append(f"| {it.name} | {_val(it)} | {BASIS_LABEL.get(it.basis, it.basis)} | {it.note} |")
    if spec.safety:
        L += ["", "## Exposure / safety screen", "", "| Check | Value | Limit | Result | Source |", "|---|---|---|---|---|"]
        for s in spec.safety:
            res = "PASS" if s.passes else "FAIL"
            L.append(f"| {s.name} | {s.value:g} {s.unit} | {s.limit:g} {s.unit} | {res}{' (approx.)' if s.approximate else ''} | [{s.source}]({s.url}) |")
        L += [""] + [f"- {s.note}" for s in spec.safety[:1]]
    if spec.bom:
        L += ["", "## Bill of materials (estimates)", "", "| Part | Qty | Unit USD | Total USD | Note |", "|---|---|---|---|---|"]
        for b in spec.bom:
            L.append(f"| {b.part} | {b.qty:g} | {b.unit_cost_usd:,.2f} | {b.total:,.2f} | {b.note} |")
        L += [f"| **Total (estimate)** | | | **{spec.bom_total_usd:,.0f}** | |", "", f"_{spec.cost_note}_"]
    if concept:
        L += ["", "## Honest limits", "", concept.honest_limits]
    if spec.math_check and spec.math_check.status != "skipped":
        L += ["", "## Math check (OpenMath-Nemotron)", "", spec.math_check.headline]
        if spec.math_check.source == "canned":
            L += ["", "_Canned derivation (mock mode), not model output._"]
    L += ["", "## Caveats", ""] + [f"- {c}" for c in spec.caveats]
    L += ["", "## Render prompt (generated from the numbers above)", "", f"> {spec.render_prompt}"]
    L += ["", "## References", ""] + [f"- [{r.title}]({r.url})" if r.url else f"- {r.title}" for r in spec.references]
    L += ["", "Basis: req = requirement you set; calc = computed by EVIE's design calculator; assumed = an "
          "engineering assumption to check on a prototype; est. = rough cost estimate.", ""]
    return "\n".join(L)


def to_pdf(spec: ProductSpec, concept: ProductConcept | None = None, claim: str = "", image_path: str | None = None) -> bytes:
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    W = pdf.w - 30
    pdf.set_fill_color(17, 24, 39)
    pdf.rect(0, 0, pdf.w, 32, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_xy(15, 8)
    pdf.cell(W, 10, _a(f"{spec.title}  |  spec sheet"))
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(15, 19)
    pdf.cell(W, 6, _a("EVIE Product Design Engine - numbers computed in Python, not guessed"))
    pdf.set_text_color(20, 20, 20)
    pdf.set_y(38)
    if concept:
        pdf.set_font("Helvetica", "I", 11)
        pdf.multi_cell(W, 6, _a(concept.pitch), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    if claim:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(W, 5, _a(f"Derived from the claim that WORKS: {claim}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(20, 20, 20)
    if image_path:
        try:
            pdf.ln(2)
            pdf.image(image_path, w=W * 0.62, x=15 + W * 0.19)
        except Exception:
            pass
    pdf.ln(2)
    _h(pdf, "Key numbers")
    pdf.set_font("Helvetica", "", 10)
    for h in spec.headline:
        pdf.multi_cell(W, 5.5, _a(f"- {h}"), new_x="LMARGIN", new_y="NEXT")
    if spec.series and spec.series.get("x"):
        _chart(pdf, spec, W)
    group = None
    for it in spec.specs:
        if it.group != group:
            group = it.group
            _h(pdf, group)
        _row(pdf, W, [it.name, _val(it), BASIS_LABEL.get(it.basis, it.basis)], [0.55, 0.30, 0.15])
        if it.note:
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(110, 110, 110)
            pdf.multi_cell(W, 4, _a("   " + it.note), new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(20, 20, 20)
    if spec.safety:
        _h(pdf, "Exposure / safety screen (approximate)")
        for s in spec.safety:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(22, 163, 74) if s.passes else pdf.set_text_color(220, 38, 38)
            pdf.set_x(15)
            pdf.cell(18, 5, "PASS" if s.passes else "FAIL")
            pdf.set_text_color(20, 20, 20)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(W - 18, 5, _a(f"{s.name}: {s.value:g} vs limit {s.limit:g} {s.unit}. {s.source}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 7.5)
        pdf.multi_cell(W, 4, _a(spec.safety[0].note), new_x="LMARGIN", new_y="NEXT")
    if spec.bom:
        _h(pdf, "Bill of materials (estimates)")
        for b in spec.bom:
            _row(pdf, W, [b.part, f"{b.qty:g} x ${b.unit_cost_usd:,.2f}", f"${b.total:,.0f}"], [0.62, 0.23, 0.15])
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(W, 6, _a(f"Total (estimate): ${spec.bom_total_usd:,.0f}"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(W, 4, _a(spec.cost_note), new_x="LMARGIN", new_y="NEXT")
    if concept:
        _h(pdf, "Honest limits")
        pdf.set_font("Helvetica", "", 9.5)
        pdf.multi_cell(W, 5, _a(concept.honest_limits), new_x="LMARGIN", new_y="NEXT")
    if spec.math_check and spec.math_check.status != "skipped":
        _h(pdf, "Math check (OpenMath-Nemotron)")
        pdf.set_font("Helvetica", "", 9.5)
        tag = " (canned derivation in mock mode, not model output)" if spec.math_check.source == "canned" else ""
        pdf.multi_cell(W, 5, _a(spec.math_check.headline + tag), new_x="LMARGIN", new_y="NEXT")
    _h(pdf, "Caveats")
    pdf.set_font("Helvetica", "", 9)
    for c in spec.caveats:
        pdf.multi_cell(W, 4.8, _a(f"- {c}"), new_x="LMARGIN", new_y="NEXT")
    _h(pdf, "Render prompt (generated from these numbers)")
    pdf.set_font("Helvetica", "I", 8.5)
    pdf.multi_cell(W, 4.5, _a(spec.render_prompt), new_x="LMARGIN", new_y="NEXT")
    _h(pdf, "References")
    pdf.set_font("Helvetica", "", 8.5)
    for r in spec.references:
        pdf.multi_cell(W, 4.5, _a(f"- {r.title} {r.url}"), new_x="LMARGIN", new_y="NEXT", align="L")
    return bytes(pdf.output())


def _h(pdf, text: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 7, _a(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(220, 220, 220)
    pdf.line(15, pdf.get_y(), pdf.w - 15, pdf.get_y())
    pdf.ln(1)
    pdf.set_text_color(20, 20, 20)


def _row(pdf, W: float, cells: list[str], fracs: list[float]) -> None:
    pdf.set_font("Helvetica", "", 9)
    y0 = pdf.get_y()
    if y0 > pdf.h - 25:
        pdf.add_page()
        y0 = pdf.get_y()
    x = 15
    ys = []
    for i, (c, f) in enumerate(zip(cells, fracs)):
        pdf.set_xy(x, y0)
        pdf.set_font("Helvetica", "B" if i == 1 else "", 9)
        pdf.multi_cell(W * f, 5, _a(c))
        ys.append(pdf.get_y())
        x += W * f
    pdf.set_xy(15, max(ys))


def _chart(pdf, spec: ProductSpec, W: float) -> None:
    xs, ys = spec.series["x"], spec.series["y"]
    extra = list((spec.extra_series or {}).items())
    all_y = list(ys) + [v for _, s in extra for v in s]
    pdf.ln(3)
    h, w = 45, W
    x0, y0 = 15, pdf.get_y()
    if y0 + h + 12 > pdf.h - 15:
        pdf.add_page()
        y0 = pdf.get_y()
    lab = spec.series_labels or {"x": "x", "y": "y"}
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(W, 5, _a(f"{lab['y']} vs {lab['x']}"), new_x="LMARGIN", new_y="NEXT")
    y0 += 6
    pdf.set_draw_color(180, 180, 180)
    pdf.rect(x0, y0, w, h)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(0, min(all_y)), max(all_y) or 1

    def P(x, y):
        return x0 + (x - xmin) / ((xmax - xmin) or 1) * w, y0 + h - (y - ymin) / ((ymax - ymin) or 1) * h

    for series, color in [(ys, (37, 99, 235))] + [(s, (234, 88, 12)) for _, s in extra]:
        pdf.set_draw_color(*color)
        pdf.set_line_width(0.6)
        pts = [P(x, y) for x, y in zip(xs, series)]
        for a, b in zip(pts, pts[1:]):
            pdf.line(a[0], a[1], b[0], b[1])
    pdf.set_line_width(0.2)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(90, 90, 90)
    pdf.set_xy(x0, y0 + h + 0.5)
    pdf.cell(w / 2, 4, _a(f"{xmin:g}"))
    pdf.cell(w / 2, 4, _a(f"{xmax:g} {lab['x']}"), align="R")
    pdf.set_xy(x0 + 1, y0 + 1)
    pdf.cell(30, 4, _a(f"max {ymax:,.4g}"))
    if extra:
        pdf.set_xy(x0 + w - 70, y0 + 1)
        pdf.set_text_color(37, 99, 235)
        pdf.cell(35, 4, _a(lab["y"][:28]))
        pdf.set_text_color(234, 88, 12)
        pdf.cell(35, 4, _a(extra[0][0][:28]))
    pdf.set_text_color(20, 20, 20)
    pdf.set_y(y0 + h + 6)


def to_blender_script(spec: ProductSpec) -> str:
    """Generate an executable Blender Python script (bpy) from the ProductSpec."""
    from .blender_gen import generate_blender_script

    return generate_blender_script(spec)


def to_stl(spec: ProductSpec) -> str:
    """Generate an ASCII STL mesh representation from the ProductSpec."""
    from .mesh_gen import generate_stl

    return generate_stl(spec)


def to_threejs_html(spec: ProductSpec) -> str:
    """Generate a self-contained Three.js interactive 3D WebGL viewer HTML."""
    from .mesh_gen import generate_threejs_viewer_html

    return generate_threejs_viewer_html(spec)

