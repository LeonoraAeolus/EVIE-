r"""Turn the LaTeX a math model writes into numbers and SymPy expressions.

OpenMath-Nemotron answers in LaTeX and puts the final answer in \boxed{...}. This module
pulls the boxed answer out, splits the solution into steps, and converts simple LaTeX
(\frac, \sqrt, ^, \cdot, greek letters, subscripts, \times 10^{n}, units in \text{})
into SymPy so the cross-check can compare it with the Python calculator.
It deliberately handles the common cases only; anything it cannot parse is reported as
"inconclusive", never guessed.
"""
from __future__ import annotations

import re
from typing import Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

GREEK = ["alpha", "beta", "gamma", "delta", "epsilon", "varepsilon", "zeta", "eta", "theta", "kappa",
         "lambda", "mu", "nu", "xi", "rho", "sigma", "tau", "phi", "varphi", "chi", "psi", "omega",
         "Gamma", "Delta", "Theta", "Lambda", "Sigma", "Phi", "Psi", "Omega"]


# ----------------------------------------------------------------------------- text structure
def split_reasoning(text: str) -> tuple[str, str]:
    r"""Return (reasoning, solution). Reasoning models wrap their scratch work in <think>...</think>."""
    text = text or ""
    if "</think>" in text:
        head, _, tail = text.rpartition("</think>")
        return head.replace("<think>", "").strip(), tail.strip()
    if "<think>" in text:  # cut off mid-thought: everything is reasoning
        return text.replace("<think>", "").strip(), ""
    return "", text.strip()


def _match_brace(s: str, i: int) -> int:
    r"""s[i] == '{'; return index of the matching '}' (or -1)."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return -1


def boxed_answers(text: str) -> list[str]:
    r"""Contents of every \boxed{...} / \fbox{...}, outermost braces matched properly."""
    out = []
    for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text or ""):
        start = m.end() - 1
        end = _match_brace(text, start)
        if end > start:
            out.append(text[start + 1:end].strip())
    return out


def last_boxed(text: str) -> str | None:
    found = boxed_answers(text)
    return found[-1] if found else None


_STEP_HEAD = re.compile(r"^\s*(?:#+\s*|\*\*)?\s*(?:Step\s*\d+|\d+[.)])", re.I)


def split_steps(solution: str, max_steps: int = 14) -> list[str]:
    r"""Split a written solution into readable steps (numbered headings, else paragraphs)."""
    solution = (solution or "").strip()
    if not solution:
        return []
    lines = solution.splitlines()
    heads = [i for i, ln in enumerate(lines) if _STEP_HEAD.match(ln)]
    if len(heads) >= 2:
        chunks, cut = [], heads + [len(lines)]
        pre = "\n".join(lines[:heads[0]]).strip()
        if pre:
            chunks.append(pre)
        for a, b in zip(cut, cut[1:]):
            chunks.append("\n".join(lines[a:b]).strip())
    else:
        chunks = [c.strip() for c in re.split(r"\n\s*\n", solution) if c.strip()]
    if len(chunks) > max_steps:  # keep the first ones, fold the tail into the last step
        chunks = chunks[:max_steps - 1] + ["\n\n".join(chunks[max_steps - 1:])]
    return chunks


def display_equations(text: str, limit: int = 8) -> list[str]:
    r"""Display equations (\[...\] or $$...$$) in order, for the 'key equations' list."""
    eqs = re.findall(r"\\\[(.+?)\\\]|\$\$(.+?)\$\$", text or "", flags=re.S)
    flat = [" ".join((a or b).split()) for a, b in eqs]
    return [e for e in flat if e][:limit]


def to_markdown_math(text: str) -> str:
    r"""Streamlit renders $...$ and $$...$$; models often write \( \) and \[ \]."""
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: "\n$$" + m.group(1).strip() + "$$\n", text or "", flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: "$" + m.group(1).strip() + "$", text, flags=re.S)
    return text


# ----------------------------------------------------------------------------- LaTeX -> SymPy
def _replace_cmd_with_args(s: str, cmd: str, nargs: int, fmt) -> str:
    r"""Replace \cmd{a}{b} (nargs groups) using fmt(*args), innermost-safe by looping."""
    pat = "\\" + cmd
    guard = 0
    while guard < 200:
        guard += 1
        i = s.find(pat + "{")
        if i < 0:
            # also accept a space: \frac {a}{b}
            m = re.search(re.escape(pat) + r"\s+\{", s)
            if not m:
                break
            i = m.start()
        j = s.find("{", i)
        args, k = [], j
        for _ in range(nargs):
            while k < len(s) and s[k] == " ":
                k += 1
            if k >= len(s) or s[k] != "{":
                return s
            e = _match_brace(s, k)
            if e < 0:
                return s
            args.append(s[k + 1:e])
            k = e + 1
        s = s[:i] + fmt(*args) + s[k:]
    return s


def _clean(latex: str) -> str:
    s = latex.strip().strip("$")
    # subscripts written as text keep their letters: E_{\text{out}} -> E_{out}
    s = re.sub(r"_\s*\{\s*\\(?:text|mathrm|rm)\s*\{([A-Za-z0-9 ]+)\}\s*\}", lambda m: "_{" + m.group(1).replace(" ", "") + "}", s)
    s = re.sub(r"\\(?:text|mathrm|textrm|operatorname|mbox|mathit|mathbf)\s*\{([^{}]*)\}", r" \\TXT{\1}", s)
    s = re.sub(r"\\TXT\{\s*(?:[a-zA-Z°µΩ/·\^\-\s\d%]*)\}", " ", s)  # unit words: drop
    for junk in [r"\left", r"\right", r"\displaystyle", r"\!", r"\,", r"\;", r"\:", r"\quad", r"\qquad", "~"]:
        s = s.replace(junk, " ")
    s = s.replace(r"\%", " ").replace("%", " ").replace(r"\ ", " ").replace("°", " ")
    return s


def split_relation(latex: str) -> str:
    r"""'\tau = \frac{Q}{2\pi f_0} \approx 0.102' -> the right-most side ('0.102')."""
    s = latex
    for rel in [r"\approx", r"\simeq", r"\sim", r"\equiv", "\\to", "="]:
        if rel in s:
            s = s.split(rel)[-1]
    return s.strip()


def latex_to_sympy(latex: str, symbols: dict[str, Any] | None = None) -> sp.Expr:
    r"""Best-effort LaTeX -> SymPy. Raises ValueError when it cannot parse."""
    s = _clean(latex)
    s = _replace_cmd_with_args(s, "dfrac", 2, lambda a, b: f"(({a})/({b}))")
    s = _replace_cmd_with_args(s, "tfrac", 2, lambda a, b: f"(({a})/({b}))")
    s = _replace_cmd_with_args(s, "frac", 2, lambda a, b: f"(({a})/({b}))")
    s = re.sub(r"\\sqrt\s*\[([^\]]+)\]\s*\{", lambda m: f"\\ROOT{{{m.group(1)}}}{{", s)
    s = _replace_cmd_with_args(s, "ROOT", 2, lambda n, a: f"(({a})**(1/({n})))")
    s = _replace_cmd_with_args(s, "sqrt", 1, lambda a: f"sqrt({a})")
    s = re.sub(r"\\sqrt\s*(\w)", r"sqrt(\1)", s)
    s = re.sub(r"(\d)\s*(?:\\times|\\cdot|x)\s*10\s*\^\s*\{\s*([+-]?\d+)\s*\}", r"\1e\2", s)
    s = re.sub(r"(\d)\s*(?:\\times|\\cdot|x)\s*10\s*\^\s*([+-]?\d+)", r"\1e\2", s)
    s = re.sub(r"(?<![A-Za-z\\])e\s*\^\s*\{", r"exp{", s)
    s = s.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\div", "/")
    s = s.replace(r"\ln", " log").replace(r"\log", " log").replace(r"\exp", " exp")
    s = s.replace(r"\sin", " sin").replace(r"\cos", " cos").replace(r"\tan", " tan")
    s = s.replace(r"\pi", " pi ")
    for g in sorted(GREEK, key=len, reverse=True):
        s = re.sub(r"\\" + g + r"(?![A-Za-z])", " " + g + " ", s)
    # subscripts: x_{abc} / x_1 -> x_abc / x_1
    s = re.sub(r"([A-Za-z]+)\s*_\s*\{\s*([A-Za-z0-9]+)\s*\}", r"\1_\2", s)
    s = re.sub(r"([A-Za-z]+)\s*_\s*([A-Za-z0-9])", r"\1_\2", s)
    s = re.sub(r"\^\s*\{", "**{", s).replace("^", "**")
    s = s.replace("{", "(").replace("}", ")").replace("[", "(").replace("]", ")")
    if "\\" in s:
        raise ValueError(f"unsupported LaTeX in {latex!r}")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)  # 1,000 -> 1000
    local: dict[str, Any] = {"pi": sp.pi, "e": sp.E, "E": sp.E, "sqrt": sp.sqrt, "log": sp.log, "exp": sp.exp}
    for name in (symbols or {}):
        local[name] = sp.Symbol(name)
    for g in GREEK:
        local.setdefault(g, sp.Symbol(g))
    for name in set(re.findall(r"[A-Za-z]+_[A-Za-z0-9]+", s)):
        local.setdefault(name, sp.Symbol(name))
    try:
        return parse_expr(s, local_dict=local, transformations=_TRANSFORMS, evaluate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"could not parse {latex!r}: {exc}") from exc


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def evaluate(expr: sp.Expr, variables: dict[str, float] | None = None) -> float | None:
    r"""Numeric value of expr after substituting variables (names matched loosely: Q_1 ~ Q1)."""
    variables = variables or {}
    by_norm = {_norm(k): v for k, v in variables.items()}
    subs = {}
    for sym in getattr(expr, "free_symbols", set()):
        key = _norm(str(sym))
        if key in by_norm:
            subs[sym] = by_norm[key]
    val = expr.subs(subs) if subs else expr
    if getattr(val, "free_symbols", None):
        return None
    try:
        out = complex(sp.N(val))
    except Exception:  # noqa: BLE001
        return None
    if abs(out.imag) > 1e-9 * max(1.0, abs(out.real)):
        return None
    return float(out.real)


def parse_answer(latex: str | None, variables: dict[str, float] | None = None) -> dict[str, Any]:
    r"""Read a boxed answer. Returns {'value', 'expr', 'symbolic', 'error'}.

    'symbolic' is True when the answer is a formula in the problem's variables (it is then
    evaluated at the given values); False when the model boxed a plain number.
    """
    if not latex:
        return {"value": None, "expr": None, "symbolic": False, "error": "no \\boxed{} answer"}
    rhs = split_relation(latex)
    try:
        expr = latex_to_sympy(rhs, variables)
    except ValueError as exc:
        return {"value": None, "expr": None, "symbolic": False, "error": str(exc)}
    symbolic = bool(getattr(expr, "free_symbols", None))
    value = evaluate(expr, variables)
    err = None if value is not None else "answer has symbols the problem did not define"
    return {"value": value, "expr": str(expr), "symbolic": symbolic, "error": err}
