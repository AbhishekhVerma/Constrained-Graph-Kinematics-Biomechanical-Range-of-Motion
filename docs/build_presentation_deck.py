"""
Compiles `docs/final_presentation_slides.md` into a widescreen 16:9 visual presentation deck:
  1. `docs/Final_Presentation_Slides.html` (Interactive widescreen HTML5 slide deck with embedded Base64 figures)
  2. `Final_Presentation_Slides.pdf` (16:9 widescreen PDF presentation deck rendered via headless Microsoft Edge)
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from docs.build_report_pdf import (
    format_inline_md as format_inline,
    latex_to_html_math,
)


def render_table(table_lines: list[str]) -> str:
    if not table_lines:
        return ""
    rows = [[c for c in line.strip()[1:-1].split("|")] for line in table_lines]
    out = ['<div class="table-wrap"><table><thead><tr>']
    for cell in rows[0]:
        out.append(f"<th>{format_inline(cell.strip())}</th>")
    out.append("</tr></thead><tbody>")
    for r in rows[2:]:
        out.append("<tr>")
        for cell in r:
            out.append(f"<td>{format_inline(cell.strip())}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)

SLIDES_MD_PATH = ROOT_DIR / "docs" / "final_presentation_slides.md"
SLIDES_HTML_PATH = ROOT_DIR / "docs" / "Final_Presentation_Slides.html"
SLIDES_PDF_PATH = ROOT_DIR / "Final_Presentation_Slides.pdf"

_env_art = os.environ.get("ANTIGRAVITY_ARTIFACT_DIR")
if _env_art and Path(_env_art).exists():
    ARTIFACT_DIR = Path(_env_art)
else:
    _brain_root = Path.home() / ".gemini" / "antigravity" / "brain"
    if _brain_root.exists():
        _dirs = [d for d in _brain_root.iterdir() if d.is_dir() and not d.name.startswith(".")]
        ARTIFACT_DIR = max(_dirs, key=lambda d: d.stat().st_mtime) if _dirs else _brain_root
    else:
        ARTIFACT_DIR = ROOT_DIR / "results"


def build_slide_html_from_section(slide_md: str, slide_index: int, total_slides: int) -> str:
    lines = slide_md.strip().splitlines()
    title = f"Slide {slide_index}"
    subtitle = ""
    body_lines = []

    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            title = format_inline(s[3:].strip())
        elif s.startswith("### ") and not subtitle:
            subtitle = format_inline(s[4:].strip())
        elif s.startswith("#### Speaker Script"):
            # Stop before speaker script so the visual slide stays clean and uncluttered
            break
        else:
            body_lines.append(line)

    html_chunks = []
    fig_chunks = []
    in_list = False
    in_table = False
    table_lines = []
    in_code = False
    code_lines = []
    in_math = False
    math_lines = []

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            html_chunks.append("</ul>")
            in_list = False

    def close_table() -> None:
        nonlocal in_table, table_lines
        if in_table:
            html_chunks.append(render_table(table_lines))
            in_table = False
            table_lines = []

    for line in body_lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            close_list()
            close_table()
            if not in_code:
                in_code = True
                code_lines = []
            else:
                in_code = False
                code_text = "\n".join(code_lines).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html_chunks.append(f'<pre class="slide-code"><code>{code_text}</code></pre>')
            continue
        if in_code:
            code_lines.append(line)
            continue

        if stripped.startswith("$$"):
            close_list()
            close_table()
            if stripped.endswith("$$") and len(stripped) > 4:
                tex = stripped[2:-2].strip()
                html_chunks.append(f'<div class="slide-math">{latex_to_html_math(tex)}</div>')
                continue
            if not in_math:
                in_math = True
                math_lines = [stripped[2:]]
            else:
                in_math = False
                math_lines.append(stripped[:-2])
                tex = "\n".join(math_lines).strip()
                html_chunks.append(f'<div class="slide-math">{latex_to_html_math(tex)}</div>')
            continue
        if in_math:
            math_lines.append(line)
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            close_list()
            in_table = True
            table_lines.append(stripped)
            continue
        else:
            close_table()

        if stripped.startswith("!["):
            close_list()
            m_img = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
            if m_img:
                alt_text = m_img.group(1)
                rel_path = m_img.group(2)
                img_path = (ROOT_DIR / "docs" / rel_path).resolve()
                if img_path.exists():
                    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
                    fig_chunks.append(
                        f'<div class="slide-fig"><img src="data:image/png;base64,{b64}" alt="{alt_text}"/><div class="fig-cap">{format_inline(alt_text)}</div></div>'
                    )
            continue

        if not stripped:
            close_list()
            continue

        if stripped.startswith("* ") or stripped.startswith("- ") or re.match(r"^\d+\.\s+", stripped):
            if not in_list:
                html_chunks.append('<ul class="slide-list">')
                in_list = True
            item_text = re.sub(r"^(?:\*|-|\d+\.)\s+", "", stripped)
            html_chunks.append(f"<li>{format_inline(item_text)}</li>")
            continue

        close_list()
        html_chunks.append(f"<p>{format_inline(stripped)}</p>")

    close_list()
    close_table()

    content_html = "\n".join(html_chunks)
    figs_html = ""
    if fig_chunks:
        ncols = len(fig_chunks)
        figs_html = f'<div class="slide-figs-row cols-{ncols}">{"".join(fig_chunks)}</div>'

    return f"""
    <section class="slide">
      <div class="slide-header">
        <div class="slide-title-group">
          <div class="slide-kicker">FOUNDATIONS OF DATA SCIENCE (FDS) &bull; FINAL PRESENTATION &bull; NOV 17, 2026</div>
          <h2>{title}</h2>
          {f'<div class="slide-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        <div class="slide-badge">Slide {slide_index} / {total_slides}</div>
      </div>
      <div class="slide-body {'has-figs' if fig_chunks else 'no-figs'}">
        <div class="slide-content">
          {content_html}
        </div>
        {figs_html}
      </div>
      <div class="slide-footer">
        <span>Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry</span>
        <span>Abhishekh Verma &bull; Bhavya &bull; Vrushank &bull; Kapish</span>
      </div>
    </section>
    """


def main() -> None:
    md_text = SLIDES_MD_PATH.read_text(encoding="utf-8")
    raw_sections = re.split(r"\n---\n", md_text)
    slide_sections = [s for s in raw_sections if "## Slide " in s]

    total_slides = len(slide_sections)
    rendered_slides = []
    for idx, sec in enumerate(slide_sections, start=1):
        rendered_slides.append(build_slide_html_from_section(sec, idx, total_slides))

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Constrained Graph Kinematics — FDS Final Presentation Deck</title>
<style>
  @page {{
    size: 13.333in 7.5in;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    margin: 0;
    padding: 0;
    background: #0f172a;
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  .slide {{
    width: 13.333in;
    height: 7.5in;
    background: #ffffff;
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 0.28in 0.42in 0.22in 0.42in;
    position: relative;
    overflow: hidden;
  }}
  .slide-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    border-bottom: 2.5px solid #0f172a;
    padding-bottom: 6px;
    margin-bottom: 8px;
  }}
  .slide-kicker {{
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 0.06em;
    color: #0284c7;
    margin-bottom: 2px;
  }}
  .slide-header h2 {{
    font-size: 16.5pt;
    font-weight: 800;
    color: #0f172a;
    margin: 0;
    line-height: 1.15;
  }}
  .slide-subtitle {{
    font-size: 10.5pt;
    font-weight: 600;
    color: #334155;
    margin-top: 2px;
  }}
  .slide-badge {{
    background: #0f172a;
    color: #f8fafc;
    font-size: 9pt;
    font-weight: 700;
    padding: 4px 10px;
    border-radius: 5px;
    white-space: nowrap;
  }}
  .slide-body {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 6px;
    overflow: hidden;
  }}
  .slide-content {{
    font-size: 9.4pt;
    line-height: 1.32;
    color: #1e293b;
  }}
  .slide-list {{
    margin: 3px 0 5px 0;
    padding-left: 18px;
  }}
  .slide-list li {{
    margin-bottom: 3.5px;
  }}
  .slide-math {{
    background: #f8fafc;
    border-left: 3px solid #0284c7;
    padding: 4px 10px;
    margin: 4px 0;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.8pt;
    text-align: center;
  }}
  .slide-code {{
    background: #0f172a;
    color: #38bdf8;
    padding: 6px 10px;
    border-radius: 5px;
    font-family: Consolas, monospace;
    font-size: 8.5pt;
    margin: 3px 0;
  }}
  .table-wrap {{
    margin: 4px 0;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7.7pt;
  }}
  th, td {{
    border: 1px solid #cbd5e1;
    padding: 2.8px 5px;
    text-align: center;
  }}
  th {{
    background: #0f172a;
    color: #ffffff;
    font-weight: 700;
  }}
  tr:nth-child(even) td {{
    background: #f8fafc;
  }}
  .slide-figs-row {{
    display: flex;
    gap: 10px;
    align-items: center;
    justify-content: center;
    margin-top: auto;
    margin-bottom: 2px;
  }}
  .slide-figs-row.cols-1 .slide-fig img {{
    max-height: 3.55in;
    max-width: 11.8in;
  }}
  .slide-figs-row.cols-2 .slide-fig img {{
    max-height: 2.35in;
    max-width: 6.0in;
  }}
  .slide-figs-row.cols-3 .slide-fig img {{
    max-height: 1.95in;
    max-width: 3.95in;
  }}
  .slide-fig {{
    text-align: center;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 4px;
  }}
  .slide-fig img {{
    display: block;
    margin: 0 auto;
    object-fit: contain;
  }}
  .fig-cap {{
    font-size: 7.5pt;
    font-weight: 600;
    color: #475569;
    margin-top: 2px;
  }}
  .slide-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #cbd5e1;
    padding-top: 4px;
    font-size: 8pt;
    font-weight: 600;
    color: #64748b;
  }}
  .frac {{
    display: inline-block;
    vertical-align: middle;
    text-align: center;
    margin: 0 2px;
  }}
  .frac-num {{
    display: block;
    border-bottom: 1px solid #0f172a;
    padding: 0 2px;
  }}
  .frac-den {{
    display: block;
    padding: 0 2px;
  }}
  .underbrace {{
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    vertical-align: top;
  }}
  .ub-expr {{
    border-bottom: 1.5px solid #0284c7;
    padding-bottom: 1px;
  }}
  .ub-label {{
    font-size: 7.5pt;
    color: #0284c7;
  }}
  .bmatrix {{
    display: inline-table;
    vertical-align: middle;
    border-left: 1.5px solid #0f172a;
    border-right: 1.5px solid #0f172a;
    margin: 0 3px;
  }}
  .bmatrix td {{
    border: none;
    padding: 1px 4px;
  }}
</style>
</head>
<body>
{"".join(rendered_slides)}
</body>
</html>
"""

    SLIDES_HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"Wrote Widescreen HTML Presentation Deck -> {SLIDES_HTML_PATH} ({SLIDES_HTML_PATH.stat().st_size / 1024:.1f} KB)")

    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser_exe = next((c for c in edge_candidates if Path(c).exists()), None)
    if browser_exe is None:
        raise RuntimeError("Could not find Microsoft Edge or Chrome to compile presentation PDF.")

    cmd = [
        browser_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={SLIDES_PDF_PATH}",
        SLIDES_HTML_PATH.resolve().as_uri(),
    ]
    subprocess.run(cmd, check=True)
    print(f"Compiled Widescreen 16:9 Presentation PDF -> {SLIDES_PDF_PATH} ({SLIDES_PDF_PATH.stat().st_size / 1024:.1f} KB)")

    if ARTIFACT_DIR.exists():
        shutil.copy2(SLIDES_PDF_PATH, ARTIFACT_DIR / SLIDES_PDF_PATH.name)
        shutil.copy2(SLIDES_MD_PATH, ARTIFACT_DIR / SLIDES_MD_PATH.name)
        shutil.copy2(ROOT_DIR / "Final_Project_Report.pdf", ARTIFACT_DIR / "Final_Project_Report.pdf")
        shutil.copy2(ROOT_DIR / "docs" / "Final_Project_Report.md", ARTIFACT_DIR / "Final_Project_Report.md")
        for fig_file in (ROOT_DIR / "results" / "figures").glob("*.*"):
            shutil.copy2(fig_file, ARTIFACT_DIR / fig_file.name)


if __name__ == "__main__":
    main()
