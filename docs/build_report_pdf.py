"""
Compiles `docs/Final_Project_Report.md` into an IEEE-styled HTML document (`docs/Final_Project_Report.html`)
with embedded Base64 high-resolution PNG figures and renders `Final_Project_Report.pdf` in the workspace root
via headless Microsoft Edge (`msedge --headless --print-to-pdf=...`).
"""

from __future__ import annotations

import base64
from pathlib import Path
import re
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parent.parent
MD_PATH = ROOT_DIR / "docs" / "Final_Project_Report.md"
HTML_PATH = ROOT_DIR / "docs" / "Final_Project_Report.html"
PDF_PATH = ROOT_DIR / "Final_Project_Report.pdf"
FIG_DIR = ROOT_DIR / "results" / "figures"


def latex_to_html_math(tex: str) -> str:
    """
    Converts LaTeX mathematical expressions into clean, self-contained HTML/Unicode math
    so headless Chromium/Edge renders every equation, matrix, subscript, and symbol
    offline with zero external CDN dependencies.
    """
    s = tex.strip()

    # Handle \tag{N}
    tag_match = re.search(r"\\tag\{([^}]+)\}", s)
    tag_html = ""
    if tag_match:
        tag_html = f'<span class="eq-tag">({tag_match.group(1)})</span>'
        s = re.sub(r"\\tag\{[^}]+\}", "", s).strip()

    # Handle bmatrix
    def _repl_bmatrix(m: re.Match) -> str:
        inner = m.group(1).strip()
        rows = [r.strip() for r in inner.split(r"\\") if r.strip()]
        row_html = []
        for r in rows:
            cols = [c.strip() for c in r.split("&")]
            col_cells = "".join(f"<td>{latex_to_html_math(c)}</td>" for c in cols)
            row_html.append(f"<tr>{col_cells}</tr>")
        return f'<table class="bmatrix">{"".join(row_html)}</table>'

    s = re.sub(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", _repl_bmatrix, s, flags=re.DOTALL)

    # Handle cases
    def _repl_cases(m: re.Match) -> str:
        inner = m.group(1).strip()
        rows = [r.strip() for r in inner.split(r"\\") if r.strip()]
        row_html = []
        for r in rows:
            cols = [c.strip() for c in r.split("&")]
            col_cells = "".join(f"<td>{latex_to_html_math(c)}</td>" for c in cols)
            row_html.append(f"<tr>{col_cells}</tr>")
        return f'<span class="cases-brace">{{</span><table class="cases-table">{"".join(row_html)}</table>'

    s = re.sub(r"\\begin\{cases\}(.*?)\\end\{cases\}", _repl_cases, s, flags=re.DOTALL)

    # Handle \underbrace{expr}_{sub}
    for _ in range(6):
        m_ub = re.search(r"\\underbrace\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}_\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", s)
        if not m_ub:
            break
        expr_inner = latex_to_html_math(m_ub.group(1))
        sub_inner = latex_to_html_math(m_ub.group(2))
        rep = f'<span class="underbrace"><span class="ub-expr">{expr_inner}</span><span class="ub-label">{sub_inner}</span></span>'
        s = s[: m_ub.start()] + rep + s[m_ub.end() :]

    # Handle \sqrt{...}
    for _ in range(5):
        m_sq = re.search(r"\\sqrt\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", s)
        if not m_sq:
            break
        inner = latex_to_html_math(m_sq.group(1))
        s = s[: m_sq.start()] + f'&radic;<span class="sqrt-bar">{inner}</span>' + s[m_sq.end() :]

    # Handle \frac{a}{b}
    for _ in range(8):
        m_fr = re.search(r"\\frac\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", s)
        if not m_fr:
            break
        num = latex_to_html_math(m_fr.group(1))
        den = latex_to_html_math(m_fr.group(2))
        rep = f'<span class="frac"><span class="frac-num">{num}</span><span class="frac-den">{den}</span></span>'
        s = s[: m_fr.start()] + rep + s[m_fr.end() :]

    # Handle \hat, \tilde, \bar, \underline, \overline, \mathbf, \boldsymbol, \mathcal, \mathbb, \text
    s = re.sub(r"\\hat\{\\(?:mathbf|boldsymbol)\{([^{}]+)\}\}", r"<b>\1&#770;</b>", s)
    s = re.sub(r"\\tilde\{\\(?:mathbf|boldsymbol)\{([^{}]+)\}\}", r"<b>\1&#771;</b>", s)
    s = re.sub(r"\\bar\{\\(?:mathbf|boldsymbol)\{([^{}]+)\}\}", r"<b>\1&#772;</b>", s)
    s = re.sub(r"\\hat\{([^{}]+)\}", r"\1&#770;", s)
    s = re.sub(r"\\tilde\{([^{}]+)\}", r"\1&#771;", s)
    s = re.sub(r"\\bar\{([^{}]+)\}", r"\1&#772;", s)
    s = re.sub(r"\\underline\{([^{}]+)\}", r"<u>\1</u>", s)
    s = re.sub(r"\\overline\{([^{}]+)\}", r'<span style="text-decoration:overline">\1</span>', s)

    for _ in range(4):
        s = re.sub(r"\\(?:mathbf|boldsymbol)\{([^{}]+)\}", r"<b>\1</b>", s)
        s = re.sub(r"\\mathcal\{([^{}]+)\}", r"<i>\1</i>", s)
        s = re.sub(r"\\mathbb\{R\}", "&#8477;", s)
        s = re.sub(r"\\mathbb\{I\}", "&#120128;", s)
        s = re.sub(r"\\text\{([^{}]+)\}", r'<span class="math-text">\1</span>', s)

    # Sizing & spacing commands
    s = re.sub(r"\\(?:left|right|Big|Bigg|big|bigg)\b", "", s)
    s = s.replace(r"\|", "&#8214;")
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "")
    s = s.replace(r"\qquad", " &nbsp;&nbsp;&nbsp;&nbsp; ").replace(r"\quad", " &nbsp;&nbsp; ")

    # Greek & Math symbols
    replacements = [
        (r"\alpha", "&alpha;"),
        (r"\beta", "&beta;"),
        (r"\gamma", "&gamma;"),
        (r"\delta", "&delta;"),
        (r"\Delta", "&Delta;"),
        (r"\epsilon", "&epsilon;"),
        (r"\varepsilon", "&varepsilon;"),
        (r"\eta", "&eta;"),
        (r"\theta", "&theta;"),
        (r"\boldsymbol{\theta}", "<b>&theta;</b>"),
        (r"\boldsymbol{\mu}", "<b>&mu;</b>"),
        (r"\boldsymbol{\Delta}", "<b>&Delta;</b>"),
        (r"\boldsymbol{\eta}", "<b>&eta;</b>"),
        (r"\boldsymbol{\xi}", "<b>&xi;</b>"),
        (r"\boldsymbol{\epsilon}", "<b>&epsilon;</b>"),
        (r"\mu", "&mu;"),
        (r"\pi", "&pi;"),
        (r"\rho", "&rho;"),
        (r"\sigma", "&sigma;"),
        (r"\tau", "&tau;"),
        (r"\xi", "&xi;"),
        (r"\prod", "&prod;"),
        (r"\sum", "&sum;"),
        (r"\in", "&isin;"),
        (r"\subset", "&sub;"),
        (r"\le", "&le;"),
        (r"\ge", "&ge;"),
        (r"\neq", "&ne;"),
        (r"\equiv", "&equiv;"),
        (r"\approx", "&approx;"),
        (r"\sim", "&sim;"),
        (r"\to", "&rarr;"),
        (r"\rightarrow", "&rarr;"),
        (r"\implies", "&rArr;"),
        (r"\times", "&times;"),
        (r"\cdot", "&middot;"),
        (r"\odot", "&#8857;"),
        (r"\pm", "&plusmn;"),
        (r"\infty", "&infin;"),
        (r"\top", "&#8868;"),
        (r"\perp", "&perp;"),
        (r"\parallel", "&par;"),
        (r"\partial", "&part;"),
        (r"\blacksquare", "&#9632;"),
        (r"\dots", "&hellip;"),
        (r"\downarrow", "&darr;"),
        (r"\uparrow", "&uarr;"),
        (r"\tanh", "tanh"),
        (r"\arccos", "arccos"),
        (r"\arcsin", "arcsin"),
        (r"\cos", "cos"),
        (r"\sin", "sin"),
        (r"\exp", "exp"),
        (r"\max", "max"),
        (r"\min", "min"),
        (r"\det", "det"),
    ]
    for k, v in replacements:
        s = s.replace(k, v)

    # Subscripts and superscripts: _{...}, ^{...}, _x, ^x
    for _ in range(4):
        s = re.sub(r"_\{([^{}]+)\}", r"<sub>\1</sub>", s)
        s = re.sub(r"\^\{([^{}]+)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_([a-zA-Z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([a-zA-Z0-9\+\-\*])", r"<sup>\1</sup>", s)
    s = s.replace(r"\{", "{").replace(r"\}", "}")

    if tag_html:
        return f'<span class="eq-content">{s}</span>{tag_html}'
    return s


def encode_image_b64(img_rel_path: str) -> str:
    fname = Path(img_rel_path).name
    candidate = FIG_DIR / fname
    if not candidate.exists():
        candidate = (MD_PATH.parent / img_rel_path).resolve()
    data = candidate.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def format_inline_md(text: str) -> str:
    # Inline math $...$
    def _repl_inline_math(m: re.Match) -> str:
        return f'<span class="math-inline">{latex_to_html_math(m.group(1))}</span>'

    text = re.sub(r"\$([^\$]+)\$", _repl_inline_math, text)
    # Bold **...**
    text = re.sub(r"\*\*([^\*]+)\*\*", r"<strong>\1</strong>", text)
    # Italic *...*
    text = re.sub(r"\*([^\*]+)\*", r"<em>\1</em>", text)
    # Inline code `...`
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


def convert_markdown_to_html(md_text: str) -> str:
    lines = md_text.splitlines()
    html_out = []
    in_list = None
    in_table = False
    table_rows = []
    i = 0
    n = len(lines)

    def _flush_list():
        nonlocal in_list
        if in_list == "ul":
            html_out.append("</ul>")
        elif in_list == "ol":
            html_out.append("</ol>")
        in_list = None

    def _flush_table():
        nonlocal in_table, table_rows
        if not in_table or not table_rows:
            in_table = False
            table_rows = []
            return
        html_out.append('<div class="table-wrap"><table>')
        header = table_rows[0]
        html_out.append("<thead><tr>")
        for cell in header:
            html_out.append(f"<th>{format_inline_md(cell.strip())}</th>")
        html_out.append("</tr></thead><tbody>")
        for row in table_rows[2:]:
            html_out.append("<tr>")
            for cell in row:
                html_out.append(f"<td>{format_inline_md(cell.strip())}</td>")
            html_out.append("</tr>")
        html_out.append("</tbody></table></div>")
        in_table = False
        table_rows = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Display math $$ ... $$
        if stripped.startswith("$$"):
            _flush_list()
            _flush_table()
            if stripped.endswith("$$") and len(stripped) > 4:
                math_body = stripped[2:-2]
                html_out.append(f'<div class="math-block">{latex_to_html_math(math_body)}</div>')
                i += 1
                continue
            else:
                math_lines = [stripped[2:]]
                i += 1
                while i < n and not lines[i].strip().endswith("$$"):
                    math_lines.append(lines[i])
                    i += 1
                if i < n:
                    math_lines.append(lines[i].strip()[:-2])
                    i += 1
                html_out.append(f'<div class="math-block">{latex_to_html_math(" ".join(math_lines))}</div>')
                continue

        # Table rows
        if stripped.startswith("|") and stripped.endswith("|"):
            _flush_list()
            in_table = True
            cells = [c for c in stripped[1:-1].split("|")]
            table_rows.append(cells)
            i += 1
            continue
        else:
            if in_table:
                _flush_table()

        # Horizontal rule
        if stripped == "---":
            _flush_list()
            html_out.append("<hr/>")
            i += 1
            continue

        # Image ![caption](path)
        img_match = re.match(r"^!\[([^\]]*)\]\(([^\)]+)\)$", stripped)
        if img_match:
            _flush_list()
            caption = img_match.group(1)
            img_path = img_match.group(2)
            b64_src = encode_image_b64(img_path)
            html_out.append(
                f'<figure class="report-fig"><img src="{b64_src}" alt="{caption}"/><figcaption>{format_inline_md(caption)}</figcaption></figure>'
            )
            i += 1
            continue

        # Headings
        if stripped.startswith("# "):
            _flush_list()
            html_out.append(f"<h1>{format_inline_md(stripped[2:])}</h1>")
            i += 1
            continue
        if stripped.startswith("## "):
            _flush_list()
            html_out.append(f"<h2>{format_inline_md(stripped[3:])}</h2>")
            i += 1
            continue
        if stripped.startswith("### "):
            _flush_list()
            html_out.append(f"<h3>{format_inline_md(stripped[4:])}</h3>")
            i += 1
            continue

        # Lists
        ul_match = re.match(r"^\s*\*\s+(.*)$", line)
        ol_match = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if ul_match:
            if in_list != "ul":
                _flush_list()
                html_out.append("<ul>")
                in_list = "ul"
            html_out.append(f"<li>{format_inline_md(ul_match.group(1))}</li>")
            i += 1
            continue
        if ol_match:
            if in_list != "ol":
                _flush_list()
                html_out.append("<ol>")
                in_list = "ol"
            html_out.append(f"<li>{format_inline_md(ol_match.group(1))}</li>")
            i += 1
            continue

        if not stripped:
            _flush_list()
            i += 1
            continue

        # Paragraph
        html_out.append(f"<p>{format_inline_md(stripped)}</p>")
        i += 1

    _flush_list()
    _flush_table()
    return "\n".join(html_out)


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    body_html = convert_markdown_to_html(md_text)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Constrained Graph Kinematics — Final Project Report</title>
<style>
  @page {{
    size: A4;
    margin: 14mm 14mm 14mm 14mm;
  }}
  body {{
    font-family: "Times New Roman", Times, Georgia, serif;
    font-size: 10pt;
    line-height: 1.42;
    color: #111827;
    max-width: 190mm;
    margin: 0 auto;
    text-align: justify;
  }}
  h1 {{
    font-size: 17pt;
    text-align: center;
    margin-bottom: 6px;
    line-height: 1.2;
    color: #0f172a;
  }}
  h2 {{
    font-size: 12pt;
    border-bottom: 1px solid #cbd5e1;
    padding-bottom: 2px;
    margin-top: 14px;
    margin-bottom: 6px;
    color: #0f172a;
    text-transform: uppercase;
    letter-spacing: 0.3px;
  }}
  h3 {{
    font-size: 10.5pt;
    font-style: italic;
    margin-top: 10px;
    margin-bottom: 4px;
    color: #1e293b;
  }}
  p {{
    margin: 5px 0;
  }}
  ul, ol {{
    margin: 4px 0 6px 18px;
    padding: 0;
  }}
  li {{
    margin-bottom: 3px;
  }}
  code {{
    font-family: "Consolas", "Courier New", monospace;
    font-size: 8.8pt;
    background: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
  }}
  hr {{
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 10px 0;
  }}
  .math-block {{
    display: flex;
    justify-content: center;
    align-items: center;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 6px 10px;
    margin: 7px 0;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.8pt;
    overflow-x: auto;
  }}
  .eq-content {{
    flex: 1;
    text-align: center;
  }}
  .eq-tag {{
    margin-left: 12px;
    font-weight: bold;
    color: #334155;
  }}
  .math-inline {{
    font-family: "Cambria Math", "Times New Roman", serif;
    white-space: nowrap;
  }}
  .frac {{
    display: inline-flex;
    flex-direction: column;
    vertical-align: middle;
    text-align: center;
    padding: 0 2px;
    font-size: 88%;
  }}
  .frac-num {{
    border-bottom: 1px solid #1e293b;
    padding: 0 2px;
  }}
  .frac-den {{
    padding: 0 2px;
  }}
  .sqrt-bar {{
    border-top: 1px solid #1e293b;
    padding-top: 1px;
  }}
  .bmatrix {{
    display: inline-table;
    vertical-align: middle;
    border-left: 1.5px solid #0f172a;
    border-right: 1.5px solid #0f172a;
    margin: 0 3px;
    border-collapse: collapse;
  }}
  .bmatrix td {{
    padding: 1px 5px;
    text-align: center;
    border: none;
  }}
  .cases-brace {{
    font-size: 20pt;
    vertical-align: middle;
    margin-right: 2px;
  }}
  .cases-table {{
    display: inline-table;
    vertical-align: middle;
    border-collapse: collapse;
  }}
  .cases-table td {{
    padding: 1px 6px;
    text-align: left;
    border: none;
  }}
  .table-wrap {{
    margin: 8px 0;
    overflow-x: auto;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.2pt;
  }}
  th, td {{
    border: 1px solid #94a3b8;
    padding: 3.5px 5px;
    text-align: center;
  }}
  th {{
    background-color: #e2e8f0;
    font-weight: bold;
    color: #0f172a;
  }}
  tr:nth-child(even) td {{
    background-color: #f8fafc;
  }}
  .report-fig {{
    text-align: center;
    margin: 10px 0;
    page-break-inside: avoid;
  }}
  .report-fig img {{
    max-width: 96%;
    max-height: 250px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
  }}
  .report-fig figcaption {{
    font-size: 8.5pt;
    font-style: italic;
    color: #334155;
    margin-top: 4px;
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""
    HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"Wrote HTML report to: {HTML_PATH} ({HTML_PATH.stat().st_size / 1024:.1f} KB)")

    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser_exe = None
    for c in edge_candidates:
        if Path(c).exists():
            browser_exe = c
            break

    if browser_exe is None:
        raise RuntimeError("Could not find Microsoft Edge or Google Chrome to print PDF.")

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        HTML_PATH.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)
    print(f"Compiled PDF report to: {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
