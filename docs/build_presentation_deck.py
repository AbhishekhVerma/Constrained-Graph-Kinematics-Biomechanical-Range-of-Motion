"""
Compiles `docs/final_presentation_slides.md` into a widescreen 16:9 visual presentation deck:
  1. `docs/Final_Presentation_Slides.html` (Interactive widescreen HTML5 slide deck with embedded Base64 figures)
  2. `Final_Presentation_Slides.pdf` (16:9 widescreen PDF presentation deck rendered via headless Microsoft Edge)
  3. `docs/Final_Presentation_Slides.pdf` (Synced copy for presentation distribution)
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


def extract_balanced_braces(s: str, start: int) -> tuple[str, int]:
    """Extracts balanced { ... } content starting at `start` index."""
    if start >= len(s) or s[start] != "{":
        return "", start
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start + 1 : i], i + 1
    return s[start + 1 :], len(s)


def format_presentation_math(tex: str) -> str:
    """
    Robust recursive LaTeX-to-HTML math formatter supporting nested balanced braces
    for \\underbrace, \\frac, \\sqrt, \\begin{bmatrix}, Greek letters, and SO(3) symbols offline.
    """
    s = tex.strip()

    # Strip sizing commands first so they don't collide with operator names
    s = re.sub(r"\\(?:left|right|Big|Bigg|big|bigg)\b", "", s)

    # Handle bmatrix
    def _repl_bmatrix(m: re.Match) -> str:
        inner = m.group(1).strip()
        rows = [r.strip() for r in inner.split(r"\\") if r.strip()]
        row_html = []
        for r in rows:
            cols = [c.strip() for c in r.split("&")]
            col_cells = "".join(f"<td>{format_presentation_math(c)}</td>" for c in cols)
            row_html.append(f"<tr>{col_cells}</tr>")
        return f'<table class="bmatrix">{"".join(row_html)}</table>'

    s = re.sub(r"\\begin\{bmatrix\}(.*?)\\end\{bmatrix\}", _repl_bmatrix, s, flags=re.DOTALL)

    # Handle \underbrace{expr}_{label} with balanced braces
    while True:
        pos = s.find(r"\underbrace")
        if pos == -1:
            break
        expr_raw, next_idx = extract_balanced_braces(s, pos + len(r"\underbrace"))
        sub_raw = ""
        end_idx = next_idx
        if next_idx < len(s) and s[next_idx : next_idx + 2] == "_{":
            sub_raw, end_idx = extract_balanced_braces(s, next_idx + 1)
        expr_fmt = format_presentation_math(expr_raw)
        sub_fmt = format_presentation_math(sub_raw)
        rep = f'<span class="underbrace"><span class="ub-expr">{expr_fmt}</span><span class="ub-label">{sub_fmt}</span></span>'
        s = s[:pos] + rep + s[end_idx:]

    # Handle \frac{num}{den} with balanced braces
    while True:
        pos = s.find(r"\frac")
        if pos == -1:
            break
        num_raw, next_idx = extract_balanced_braces(s, pos + len(r"\frac"))
        den_raw, end_idx = extract_balanced_braces(s, next_idx)
        num_fmt = format_presentation_math(num_raw)
        den_fmt = format_presentation_math(den_raw)
        rep = f'<span class="frac"><span class="frac-num">{num_fmt}</span><span class="frac-den">{den_fmt}</span></span>'
        s = s[:pos] + rep + s[end_idx:]

    # Handle \sqrt{inner} with balanced braces
    while True:
        pos = s.find(r"\sqrt")
        if pos == -1:
            break
        inner_raw, end_idx = extract_balanced_braces(s, pos + len(r"\sqrt"))
        inner_fmt = format_presentation_math(inner_raw)
        rep = f'&radic;<span class="sqrt-bar">{inner_fmt}</span>'
        s = s[:pos] + rep + s[end_idx:]

    # Handle \text{...}, \mathbf{...}, \boldsymbol{...}, \mathcal{...}
    s = re.sub(r"\\text\{([^}]+)\}", r'<span class="math-text">\1</span>', s)
    s = re.sub(r"\\(?:mathbf|boldsymbol)\{([^}]+)\}", r"<b>\1</b>", s)
    s = re.sub(r"\\mathcal\{L\}", r"<i>L</i>", s)
    s = re.sub(r"\\mathcal\{N\}", r"<i>N</i>", s)
    s = re.sub(r"\\mathcal\{V\}", r"<i>V</i>", s)
    s = re.sub(r"\\mathcal\{([^}]+)\}", r"<i>\1</i>", s)
    s = re.sub(r"\\mathbb\{R\}", r"&#8477;", s)
    s = re.sub(r"\\mathbb\{I\}", r"&#120128;", s)

    # Accents: \hat, \tilde, \bar
    s = re.sub(r"\\hat\{<b>([^<]+)</b>\}", r"<b>\1&#770;</b>", s)
    s = re.sub(r"\\tilde\{<b>([^<]+)</b>\}", r"<b>\1&#771;</b>", s)
    s = re.sub(r"\\bar\{<b>([^<]+)</b>\}", r"<b>\1&#772;</b>", s)
    s = re.sub(r"\\hat\{([^}]+)\}", r"\1&#770;", s)
    s = re.sub(r"\\tilde\{([^}]+)\}", r"\1&#771;", s)
    s = re.sub(r"\\bar\{([^}]+)\}", r"\1&#772;", s)

    # Powers & Subscripts
    s = re.sub(r"\^\{?\\top\}?", r"<sup>T</sup>", s)
    s = re.sub(r"\^\{?\\circ\}?", r"&deg;", s)
    s = re.sub(r"\^\{([^}]+)\}", r"<sup>\1</sup>", s)
    s = re.sub(r"_\{([^}]+)\}", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([0-9a-zA-Z])", r"<sup>\1</sup>", s)
    s = re.sub(r"_([0-9a-zA-Z])", r"<sub>\1</sub>", s)

    # Greek letters
    greek = [
        (r"\alpha", "&alpha;"),
        (r"\beta", "&beta;"),
        (r"\gamma", "&gamma;"),
        (r"\delta", "&delta;"),
        (r"\Delta", "&Delta;"),
        (r"\epsilon", "&epsilon;"),
        (r"\varepsilon", "&varepsilon;"),
        (r"\theta", "&theta;"),
        (r"\mu", "&mu;"),
        (r"\sigma", "&sigma;"),
        (r"\rho", "&rho;"),
        (r"\tau", "&tau;"),
        (r"\xi", "&xi;"),
    ]
    for p, r_rep in greek:
        s = s.replace(p, r_rep)

    # Strict word-boundary operators to prevent partial collisions
    ops_word = [
        (r"\\le\b", "&le;"),
        (r"\\ge\b", "&ge;"),
        (r"\\in\b", "&isin;"),
        (r"\\to\b", "&rarr;"),
        (r"\\implies\b", "&rArr;"),
        (r"\\equiv\b", "&equiv;"),
        (r"\\times\b", "&times;"),
        (r"\\cdot\b", "&bull;"),
        (r"\\odot\b", "&odot;"),
        (r"\\pm\b", "&plusmn;"),
        (r"\\blacksquare\b", "&#9632;"),
        (r"\\sum\b", "&sum;"),
        (r"\\prod\b", "&prod;"),
        (r"\\det\b", "det"),
        (r"\\arccos\b", "arccos"),
        (r"\\tanh\b", "tanh"),
    ]
    for p, r_rep in ops_word:
        s = re.sub(p, r_rep, s)

    # Direct literal operators
    s = s.replace(r"\|", "&#8214;")
    s = s.replace(r"\%", "%")
    s = s.replace(r"\_", "_")
    s = s.replace(r"\,", " ")
    s = s.replace(r"\;", " ")
    s = s.replace(r"\!", "")
    s = s.replace(r"\qquad", " &nbsp;&nbsp;&nbsp;&nbsp; ")
    s = s.replace(r"\quad", " &nbsp;&nbsp; ")

    return s


def clean_presentation_inline(md_text: str) -> str:
    """Inlines markdown with clean mathematical formatting and character cleanup."""
    # Convert math blocks $...$
    def _repl_inline_math(m: re.Match) -> str:
        tex = m.group(1)
        return f'<span class="inline-math">{format_presentation_math(tex)}</span>'

    s = re.sub(r"\$([^\$]+)\$", _repl_inline_math, md_text)

    # Bold and italics
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)

    # General cleanup
    s = s.replace(r"\%", "%")
    s = s.replace(r"\_", "_")
    s = s.replace(r"^\circ", "&deg;")
    s = s.replace(r"^{\circ}", "&deg;")
    return s


def render_table(table_lines: list[str]) -> str:
    if not table_lines:
        return ""
    rows = [[c for c in line.strip()[1:-1].split("|")] for line in table_lines]
    out = ['<div class="table-wrap"><table><thead><tr>']
    for cell in rows[0]:
        out.append(f"<th>{clean_presentation_inline(cell.strip())}</th>")
    out.append("</tr></thead><tbody>")
    for r in rows[2:]:
        row_text = " ".join(r)
        row_class = ""
        if any(k in row_text for k in ["Teacher ST-GAE", "StudentKinematicMLP", "Distilled Student", "Teacher (Ours)"]):
            row_class = ' class="highlight-mint"'
        elif any(k in row_text for k in ["UnconstrainedStudent", "Corrupted Input"]):
            row_class = ' class="highlight-warning"'
        out.append(f"<tr{row_class}>")
        for cell in r:
            out.append(f"<td>{clean_presentation_inline(cell.strip())}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


SLIDES_MD_PATH = ROOT_DIR / "docs" / "final_presentation_slides.md"
SLIDES_HTML_PATH = ROOT_DIR / "docs" / "Final_Presentation_Slides.html"
SLIDES_PDF_PATH = ROOT_DIR / "Final_Presentation_Slides.pdf"
SLIDES_PDF_DOCS_PATH = ROOT_DIR / "docs" / "Final_Presentation_Slides.pdf"

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


def build_slide_2_objectives_html(title: str, subtitle: str, speaker: str, slide_index: int, total_slides: int) -> str:
    """
    Renders Slide 2 in the exact 2-column visual layout shown in the reference design:
      Column 1: 01, 03, 05
      Column 2: 02, 04
    with large modern numbering, crisp white titles, and thin horizontal divider lines.
    """
    kicker = "FOUNDATIONS OF DATA SCIENCE &bull; FINAL EVALUATION"
    if speaker:
        kicker += f" &bull; SPEAKER: {speaker.upper()}"
    badge = f"Slide {slide_index:02d} / {total_slides:02d}"

    objectives = [
        {
            "num": "01",
            "title": "Hybrid Real FreiHAND + Clinical<br/>MANO Benchmark",
            "desc": "32,560 real poses paired 50/50 with clinical MANO trajectories under 30%, 50%, 70% tool occlusion &amp; latex glove physics.",
        },
        {
            "num": "02",
            "title": "Zero-Bone-Error Differentiable Forward<br/>Kinematics",
            "desc": "Gram-Schmidt SO(3) + BiomechanicalRoMClamper + recursive FK guaranteeing 0.0000 mm bone error &amp; 100% validity.",
        },
        {
            "num": "03",
            "title": "Mask-Aware Spatio-Temporal Graph<br/>Autoencoder Teacher",
            "desc": "3-layer 4-head visibility-gated GAT + Bi-GRU across T=8; cuts occlusion error by 27.0% and motion jitter by 41.1%.",
        },
        {
            "num": "04",
            "title": "Knowledge Distillation into a Lightweight Edge<br/>Student",
            "desc": "Distills Teacher into 78.4K-param MLP (6.64&times; smaller, 137.3&times; fewer FLOPs) via 5-term geodesic + smoothness loss.",
        },
        {
            "num": "05",
            "title": "Real-Time Edge Deployment &amp; Live<br/>Telemetry Demo",
            "desc": "Sub-2 ms latency (&gt;500 FPS) on Apple Silicon CoreML &amp; x86 CPU; sustains 60 Hz live ARKit &amp; dual-hand webcam demo.",
        },
    ]

    col1_items = [objectives[0], objectives[2], objectives[4]]  # 01, 03, 05
    col2_items = [objectives[1], objectives[3]]                  # 02, 04

    def render_col(items: list[dict[str, str]]) -> str:
        html = []
        for it in items:
            html.append(f"""
            <div class="objective-item">
              <div class="obj-num">{it["num"]}</div>
              <div class="obj-content">
                <div class="obj-title">{it["title"]}</div>
                <div class="obj-desc">{it["desc"]}</div>
              </div>
            </div>
            """)
        return "".join(html)

    return f"""
    <section class="slide slide-objectives">
      <div class="slide-header">
        <div class="slide-title-group">
          <div class="slide-kicker">{kicker}</div>
          <h2 class="slide-title">{title}</h2>
          <div class="slide-title-bar"></div>
          {f'<div class="slide-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        <div class="slide-badge">{badge}</div>
      </div>
      <div class="slide-body">
        <div class="objectives-grid">
          <div class="objectives-col">
            {render_col(col1_items)}
          </div>
          <div class="objectives-col">
            {render_col(col2_items)}
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <span class="footer-title">Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry</span>
        <span class="footer-authors">Abhishekh Verma &bull; Bhavya &bull; Vrushank &bull; Kapish</span>
      </div>
    </section>
    """


def build_slide_4_architecture_html(
    title: str, subtitle: str, speaker: str, slide_index: int, total_slides: int, fig_base64: str
) -> str:
    """Renders Slide 4 with the block diagram framed above and 4 horizontal stage cards below."""
    kicker = "FOUNDATIONS OF DATA SCIENCE &bull; FINAL EVALUATION"
    if speaker:
        kicker += f" &bull; SPEAKER: {speaker.upper()}"
    badge = f"Slide {slide_index:02d} / {total_slides:02d}"

    return f"""
    <section class="slide slide-4">
      <div class="slide-header">
        <div class="slide-title-group">
          <div class="slide-kicker">{kicker}</div>
          <h2 class="slide-title">{title}</h2>
          <div class="slide-title-bar"></div>
          {f'<div class="slide-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        <div class="slide-badge">{badge}</div>
      </div>
      <div class="slide-body">
        <div class="slide-fig-arch">
          <img src="data:image/png;base64,{fig_base64}" alt="Figure 0: System Architecture Block Diagram"/>
          <div class="fig-cap">Figure 0: End-to-End Hybrid Ingestion, Teacher ST-GAE, Knowledge Distillation &amp; Kinematic Manifold Pipeline</div>
        </div>
        <div class="stages-strip">
          <div class="stage-card">
            <div class="stage-num">01</div>
            <div class="stage-title">Data Ingestion</div>
            <div class="stage-desc">Wrist-norm (100 mm) + 12% glove stiffness + 3D cylinder tool clipping &rarr; X &isin; &#8477;<sup>B&times;T&times;21&times;4</sup></div>
          </div>
          <div class="stage-card">
            <div class="stage-num">02</div>
            <div class="stage-title">Teacher ST-GAE</div>
            <div class="stage-desc">7D rest imputation + 3-layer visibility-gated GAT + Bi-GRU (T=8, 520K params)</div>
          </div>
          <div class="stage-card">
            <div class="stage-num">03</div>
            <div class="stage-title">Student Distillation</div>
            <div class="stage-desc">78.4K-param MLP supervised by 5-term geodesic SO(3) + temporal smoothness loss</div>
          </div>
          <div class="stage-card">
            <div class="stage-num">04</div>
            <div class="stage-title">Hard Kinematic Manifold</div>
            <div class="stage-desc">Gram-Schmidt SO(3) + tanh RoM Clamper + DifferentiableFK (0.0000 mm bone error)</div>
          </div>
        </div>
      </div>
      <div class="slide-footer">
        <span class="footer-title">Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry</span>
        <span class="footer-authors">Abhishekh Verma &bull; Bhavya &bull; Vrushank &bull; Kapish</span>
      </div>
    </section>
    """


def build_slide_7_theorem_html(title: str, subtitle: str, speaker: str, slide_index: int, total_slides: int) -> str:
    """Renders Slide 7 with hierarchical traversal and a dedicated Theorem 1 proof card."""
    kicker = "FOUNDATIONS OF DATA SCIENCE &bull; FINAL EVALUATION"
    if speaker:
        kicker += f" &bull; SPEAKER: {speaker.upper()}"
    badge = f"Slide {slide_index:02d} / {total_slides:02d}"

    return f"""
    <section class="slide slide-7">
      <div class="slide-header">
        <div class="slide-title-group">
          <div class="slide-kicker">{kicker}</div>
          <h2 class="slide-title">{title}</h2>
          <div class="slide-title-bar"></div>
          {f'<div class="slide-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        <div class="slide-badge">{badge}</div>
      </div>
      <div class="slide-body">
        <div class="slide-content">
          <ul class="slide-list">
            <li><strong>Hierarchical Directed Kinematic Tree Traversal (DifferentiableFK / EdgeFK):</strong>
              Hand skeleton is organized as a directed tree rooted at wrist <strong>P</strong><sub>0</sub> = (0, 0, 0) with parent map <code>p(j) = PARENT_INDICES[j]</code>:
              <div class="slide-code"><code>PARENT_INDICES = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]</code></div>
            </li>
            <li>Global orientation and 3D position of joint <em>j</em> are computed recursively from canonical rest bone vectors <strong>v</strong><sub>j</sub><sup>rest</sup> = L<sub>j</sub> <strong>u&#770;</strong><sub>j</sub><sup>rest</sup>:
              <div class="slide-math"><strong>R</strong><sub>j</sub><sup>global</sup> = <strong>R</strong><sub>p(j)</sub><sup>global</sup> <strong>R</strong><sub>j</sub><sup>local</sup>(&theta;<sub>j</sub><sup>clamp</sup>), &nbsp;&nbsp;&nbsp;&nbsp; <strong>P</strong><sub>j</sub> = <strong>P</strong><sub>p(j)</sub> + <strong>R</strong><sub>j</sub><sup>global</sup> <strong>v</strong><sub>j</sub><sup>rest</sup></div>
            </li>
          </ul>
        </div>
        <div class="theorem-card">
          <div class="theorem-kicker">MATHEMATICAL PROOF &bull; EXACT BONE-LENGTH INVARIANCE</div>
          <div class="theorem-claim">
            <strong>Theorem 1 (Isometric Invariance):</strong> For any arbitrary network weights <strong>W</strong>, sensor noise &sigma;, or severe tool occlusion, every predicted bone segment strictly satisfies &#8214;<strong>P</strong><sub>j</sub> &minus; <strong>P</strong><sub>p(j)</sub>&#8214;<sub>2</sub> &equiv; L<sub>j</sub>, guaranteeing <strong>&epsilon;<sub>bone</sub> &equiv; 0.0000 mm</strong> identically.
          </div>
          <div class="theorem-body">
            <div class="slide-math">&#8214;<strong>P</strong><sub>j</sub> &minus; <strong>P</strong><sub>p(j)</sub>&#8214;<sub>2</sub> = &#8214;<strong>R</strong><sub>j</sub><sup>global</sup> <strong>v</strong><sub>j</sub><sup>rest</sup>&#8214;<sub>2</sub> = &radic;<span class="sqrt-bar">(<strong>v</strong><sub>j</sub><sup>rest</sup>)<sup>T</sup> (<strong>R</strong><sub>j</sub><sup>global</sup>)<sup>T</sup> <strong>R</strong><sub>j</sub><sup>global</sup> <strong>v</strong><sub>j</sub><sup>rest</sup></span></div>
            <p class="theorem-step">Because <strong>R</strong><sub>j</sub><sup>global</sup> &isin; SO(3) is an orthogonal rotation matrix, (<strong>R</strong><sub>j</sub><sup>global</sup>)<sup>T</sup> <strong>R</strong><sub>j</sub><sup>global</sup> &equiv; <strong>I</strong><sub>3</sub>. Therefore:</p>
            <div class="slide-math">&#8214;<strong>P</strong><sub>j</sub> &minus; <strong>P</strong><sub>p(j)</sub>&#8214;<sub>2</sub> = &radic;<span class="sqrt-bar">(<strong>v</strong><sub>j</sub><sup>rest</sup>)<sup>T</sup> <strong>I</strong><sub>3</sub> <strong>v</strong><sub>j</sub><sup>rest</sup></span> = &#8214;<strong>v</strong><sub>j</sub><sup>rest</sup>&#8214;<sub>2</sub> = L<sub>j</sub> &nbsp;&rArr;&nbsp; <strong>&epsilon;<sub>bone</sub> &equiv; 0.0000 mm</strong> &nbsp; &#9632;</div>
          </div>
        </div>
        <div class="stat-pill-row">
          <div class="stat-pill"><strong>Bare-Metal EdgeFK:</strong> Pure NumPy buffer-reuse executes full 21-joint kinematics in <strong>0.0725 ms (13.79 kHz)</strong> on ARM64 Apple Silicon.</div>
        </div>
      </div>
      <div class="slide-footer">
        <span class="footer-title">Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry</span>
        <span class="footer-authors">Abhishekh Verma &bull; Bhavya &bull; Vrushank &bull; Kapish</span>
      </div>
    </section>
    """


def build_slide_html_from_section(slide_md: str, slide_index: int, total_slides: int) -> str:
    lines = slide_md.strip().splitlines()
    raw_title = f"Slide {slide_index}"
    subtitle = ""
    speaker = ""
    body_lines = []

    for line in lines:
        s = line.strip()
        if s.startswith("## "):
            raw_title = s[3:].strip()
        elif s.startswith("### ") and not subtitle:
            subtitle = clean_presentation_inline(s[4:].strip())
        elif s.startswith("#### Speaker Script"):
            # Stop before speaker script so the visual slide stays clean and uncluttered
            break
        else:
            body_lines.append(line)

    # Clean title and extract speaker name
    m_title = re.match(r"^Slide\s+\d+:\s*(.*?)(?:\s*\(([^)]+)\))?$", raw_title)
    if m_title:
        title_text = m_title.group(1).strip()
        if m_title.group(2):
            speaker = m_title.group(2).strip()
    else:
        title_text = raw_title.strip()

    title_html = clean_presentation_inline(title_text)

    # Slide 2: Dedicated Objectives 2-column layout matching reference image
    if slide_index == 2:
        return build_slide_2_objectives_html(title_html, subtitle, speaker, slide_index, total_slides)

    # Slide 4: Block Diagram showcase + 4 horizontal stage cards
    if slide_index == 4:
        img_p = ROOT_DIR / "results" / "figures" / "fig0_block_diagram.png"
        b64 = base64.b64encode(img_p.read_bytes()).decode("ascii") if img_p.exists() else ""
        return build_slide_4_architecture_html(title_html, subtitle, speaker, slide_index, total_slides, b64)

    # Slide 7: Differentiable Forward Kinematics & Theorem 1 Proof
    if slide_index == 7:
        return build_slide_7_theorem_html(title_html, subtitle, speaker, slide_index, total_slides)

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
                html_chunks.append(f'<div class="slide-math">{format_presentation_math(tex)}</div>')
                continue
            if not in_math:
                in_math = True
                math_lines = [stripped[2:]]
            else:
                in_math = False
                math_lines.append(stripped[:-2])
                tex = "\n".join(math_lines).strip()
                html_chunks.append(f'<div class="slide-math">{format_presentation_math(tex)}</div>')
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
                        f'<div class="slide-fig"><img src="data:image/png;base64,{b64}" alt="{alt_text}"/><div class="fig-cap">{clean_presentation_inline(alt_text)}</div></div>'
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
            html_chunks.append(f"<li>{clean_presentation_inline(item_text)}</li>")
            continue

        close_list()
        html_chunks.append(f"<p>{clean_presentation_inline(stripped)}</p>")

    close_list()
    close_table()

    content_html = "\n".join(html_chunks)
    figs_html = ""
    if fig_chunks:
        ncols = len(fig_chunks)
        figs_html = f'<div class="slide-figs-row cols-{ncols}">{"".join(fig_chunks)}</div>'

    # Specialized layout adaptations for visual balance
    body_inner_html = ""
    if slide_index == 12 and fig_chunks:
        # Split layout: 15-row MPJPE table on left, Figure 1 on right
        body_inner_html = f"""
        <div class="split-row split-s12">
          <div class="split-col-table">{content_html}</div>
          <div class="split-col-fig">{figs_html}</div>
        </div>
        """
    elif slide_index == 13 and fig_chunks:
        # Split layout: Figure 2 on left, Findings & Biomechanical Proof on right
        body_inner_html = f"""
        <div class="split-row split-s13">
          <div class="split-col-fig">{figs_html}</div>
          <div class="split-col-content">{content_html}</div>
        </div>
        """
    elif slide_index == 16 and fig_chunks:
        # Split layout: Live Demo HUD + Command on left, Controls + Conclusions on right
        body_inner_html = f"""
        <div class="split-row split-s16">
          <div class="split-col-fig">{figs_html}</div>
          <div class="split-col-content">{content_html}</div>
        </div>
        """
    else:
        body_inner_html = f"""
        <div class="slide-content">
          {content_html}
        </div>
        {figs_html}
        """

    kicker = "FOUNDATIONS OF DATA SCIENCE &bull; FINAL EVALUATION"
    if speaker:
        kicker += f" &bull; SPEAKER: {speaker.upper()}"
    badge = f"Slide {slide_index:02d} / {total_slides:02d}"

    return f"""
    <section class="slide slide-{slide_index}">
      <div class="slide-header">
        <div class="slide-title-group">
          <div class="slide-kicker">{kicker}</div>
          <h2 class="slide-title">{title_html}</h2>
          <div class="slide-title-bar"></div>
          {f'<div class="slide-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        <div class="slide-badge">{badge}</div>
      </div>
      <div class="slide-body {'has-figs' if fig_chunks else 'no-figs'}">
        {body_inner_html}
      </div>
      <div class="slide-footer">
        <span class="footer-title">Constrained Graph Kinematics: Biomechanical Range-of-Motion for Occluded Hand Gesture Telemetry</span>
        <span class="footer-authors">Abhishekh Verma &bull; Bhavya &bull; Vrushank &bull; Kapish</span>
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
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

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
    background: #020204;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #f1f5f9;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
    -webkit-font-smoothing: antialiased;
  }}
  .slide {{
    width: 13.333in;
    height: 7.5in;
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 0.36in 0.52in 0.24in 0.52in;
    position: relative;
    overflow: hidden;
    background-color: #030305;
    background-image:
      /* Vibrant golden amber / orange glow sweeping across the mid-right */
      radial-gradient(ellipse 64% 48% at 85% 50%, rgba(245, 158, 11, 0.46) 0%, rgba(217, 119, 6, 0.30) 28%, rgba(180, 83, 9, 0.08) 55%, transparent 76%),
      /* Deep electric cobalt blue aura rising on the lower-left */
      radial-gradient(ellipse 60% 55% at 18% 74%, rgba(37, 99, 235, 0.48) 0%, rgba(30, 64, 175, 0.30) 32%, rgba(2, 132, 199, 0.14) 58%, transparent 80%),
      /* Confluence glow connecting cobalt blue and warm amber */
      radial-gradient(ellipse 72% 38% at 52% 62%, rgba(59, 130, 246, 0.20) 0%, rgba(245, 158, 11, 0.14) 42%, transparent 72%),
      /* Dark vignette keeping top title area deep pitch black for high contrast */
      linear-gradient(180deg, #020204 0%, rgba(2, 2, 4, 0.88) 22%, transparent 48%);
    background-repeat: no-repeat;
  }}
  .slide-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 10px;
    position: relative;
    z-index: 2;
  }}
  .slide-title-group {{
    flex: 1;
  }}
  .slide-kicker {{
    font-size: 8.2pt;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #38bdf8;
    text-transform: uppercase;
    margin-bottom: 3px;
  }}
  .slide-title {{
    font-size: 19.5pt;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    line-height: 1.15;
    letter-spacing: 0.04em;
    text-transform: uppercase;
  }}
  .slide-title-bar {{
    width: 60px;
    height: 3.5px;
    background: #f59e0b;
    border-radius: 2px;
    margin-top: 5px;
    margin-bottom: 2px;
  }}
  .slide-subtitle {{
    font-size: 9.6pt;
    font-weight: 500;
    color: #94a3b8;
    margin-top: 4px;
    line-height: 1.3;
  }}
  .slide-badge {{
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #f1f5f9;
    font-size: 8.5pt;
    font-weight: 700;
    padding: 5px 12px;
    border-radius: 6px;
    white-space: nowrap;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
  }}
  .slide-body {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    gap: 7px;
    overflow: hidden;
    position: relative;
    z-index: 2;
  }}
  .slide-content {{
    font-size: 9.1pt;
    line-height: 1.36;
    color: #cbd5e1;
  }}
  .slide-content p {{
    margin: 3.5px 0;
    color: #cbd5e1;
  }}
  .slide-content strong {{
    color: #ffffff;
    font-weight: 700;
  }}
  .slide-list {{
    margin: 3px 0 5px 0;
    padding-left: 20px;
  }}
  .slide-list li {{
    margin-bottom: 3.5px;
    color: #cbd5e1;
  }}
  .slide-list li::marker {{
    color: #f59e0b;
  }}
  .slide-list li strong {{
    color: #ffffff;
  }}

  /* Slide 2: Objectives 2-Column Grid Layout */
  .objectives-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    column-gap: 55px;
    row-gap: 16px;
    margin-top: 14px;
    flex: 1;
  }}
  .objectives-col {{
    display: flex;
    flex-direction: column;
    gap: 18px;
  }}
  .objective-item {{
    display: flex;
    align-items: flex-start;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.18);
  }}
  .obj-num {{
    font-size: 26pt;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.05;
    min-width: 52px;
  }}
  .obj-content {{
    flex: 1;
  }}
  .obj-title {{
    font-size: 13pt;
    font-weight: 600;
    line-height: 1.32;
    color: #f8fafc;
  }}
  .obj-desc {{
    font-size: 8.6pt;
    font-weight: 400;
    color: #94a3b8;
    margin-top: 4px;
    line-height: 1.35;
  }}

  /* Slide 4: Block Diagram Frame & Horizontal 4-Stage Strip */
  .slide-fig-arch {{
    text-align: center;
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 6px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(37, 99, 235, 0.12);
  }}
  .slide-fig-arch img {{
    max-height: 2.85in;
    max-width: 100%;
    display: block;
    margin: 0 auto;
    object-fit: contain;
    border-radius: 4px;
  }}
  .stages-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 6px;
  }}
  .stage-card {{
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-top: 2.5px solid #f59e0b;
    border-radius: 6px;
    padding: 8px 10px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
  }}
  .stage-num {{
    font-size: 13pt;
    font-weight: 700;
    color: #f59e0b;
    margin-bottom: 2px;
  }}
  .stage-title {{
    font-size: 8.8pt;
    font-weight: 700;
    color: #f8fafc;
  }}
  .stage-desc {{
    font-size: 7.4pt;
    color: #94a3b8;
    margin-top: 2px;
    line-height: 1.3;
  }}

  /* Slide 7: Dedicated Theorem 1 Card */
  .theorem-card {{
    background: rgba(15, 23, 42, 0.75);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-left: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5), 0 0 15px rgba(245, 158, 11, 0.10);
  }}
  .theorem-kicker {{
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #f59e0b;
    text-transform: uppercase;
    margin-bottom: 4px;
  }}
  .theorem-claim {{
    font-size: 9pt;
    color: #f8fafc;
    line-height: 1.35;
    margin-bottom: 6px;
  }}
  .theorem-step {{
    font-size: 8.5pt;
    color: #94a3b8;
    margin: 4px 0;
  }}
  .stat-pill-row {{
    margin-top: 4px;
  }}
  .stat-pill {{
    display: inline-block;
    background: rgba(15, 23, 42, 0.80);
    border: 1px solid rgba(56, 189, 248, 0.35);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 8.4pt;
    color: #e2e8f0;
  }}
  .stat-pill strong {{
    color: #38bdf8;
  }}

  /* Split row layouts for complex slides */
  .split-row {{
    display: flex;
    gap: 18px;
    align-items: center;
    flex: 1;
    overflow: hidden;
  }}
  .split-s12 .split-col-table {{
    flex: 1.35;
  }}
  .split-s12 .split-col-fig {{
    flex: 1;
    text-align: center;
  }}
  .split-s12 .split-col-fig img {{
    max-height: 4.3in !important;
    max-width: 100% !important;
  }}
  .split-s13 .split-col-fig {{
    flex: 1.05;
    text-align: center;
  }}
  .split-s13 .split-col-fig img {{
    max-height: 4.4in !important;
    max-width: 100% !important;
  }}
  .split-s13 .split-col-content {{
    flex: 1.25;
  }}
  .split-s16 .split-col-fig {{
    flex: 1.15;
    text-align: center;
  }}
  .split-s16 .split-col-fig img {{
    max-height: 3.4in !important;
    max-width: 100% !important;
  }}
  .split-s16 .split-col-content {{
    flex: 1.2;
  }}

  /* Math Blocks */
  .slide-math {{
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-left: 3.5px solid #f59e0b;
    border-radius: 6px;
    padding: 5px 12px;
    margin: 4px 0;
    font-family: "Cambria Math", "Times New Roman", serif;
    font-size: 9.6pt;
    color: #f8fafc;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  }}

  /* Code Blocks */
  .slide-code {{
    background: rgba(10, 15, 30, 0.90);
    border: 1px solid rgba(56, 189, 248, 0.30);
    border-left: 3.5px solid #38bdf8;
    color: #38bdf8;
    padding: 6px 12px;
    border-radius: 6px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 8.5pt;
    margin: 4px 0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
  }}

  /* Sleek Dark Glass Tables */
  .table-wrap {{
    margin: 4px 0;
    background: rgba(15, 23, 42, 0.70);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), 0 0 15px rgba(37, 99, 235, 0.10);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7.4pt;
    color: #e2e8f0;
  }}
  th {{
    background: rgba(10, 15, 30, 0.95);
    color: #38bdf8;
    font-weight: 700;
    font-size: 7.6pt;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 4px 6px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-bottom: 2px solid #f59e0b;
    text-align: center;
  }}
  td {{
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 2.8px 5px;
    text-align: center;
    color: #cbd5e1;
  }}
  tr:nth-child(even) td {{
    background: rgba(255, 255, 255, 0.02);
  }}
  tr.highlight-mint td {{
    background: rgba(16, 185, 129, 0.18) !important;
    color: #f0fdf4 !important;
    font-weight: 600;
    border-top: 1px solid rgba(16, 185, 129, 0.40);
    border-bottom: 1px solid rgba(16, 185, 129, 0.40);
  }}
  tr.highlight-mint td strong {{
    color: #34d399 !important;
  }}
  tr.highlight-warning td {{
    background: rgba(245, 158, 11, 0.12) !important;
    color: #fef3c7 !important;
  }}

  /* Figures & Image Frames in Dark Glass Cards */
  .slide-figs-row {{
    display: flex;
    gap: 12px;
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
    max-width: 5.9in;
  }}
  .slide-figs-row.cols-3 .slide-fig img {{
    max-height: 1.95in;
    max-width: 3.85in;
  }}
  .slide-fig {{
    text-align: center;
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 8px;
    padding: 6px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6), 0 0 20px rgba(37, 99, 235, 0.12);
  }}
  .slide-fig img {{
    display: block;
    margin: 0 auto;
    object-fit: contain;
    border-radius: 5px;
  }}
  .fig-cap {{
    font-size: 7.6pt;
    font-weight: 600;
    color: #94a3b8;
    margin-top: 4px;
    letter-spacing: 0.02em;
  }}

  /* Footer */
  .slide-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid rgba(255, 255, 255, 0.12);
    padding-top: 6px;
    font-size: 7.8pt;
    font-weight: 500;
    color: #64748b;
    letter-spacing: 0.03em;
    position: relative;
    z-index: 2;
  }}
  .slide-footer .footer-title {{
    color: #94a3b8;
  }}
  .slide-footer .footer-authors {{
    color: #cbd5e1;
  }}

  /* Math typesetting helpers */
  .frac {{
    display: inline-block;
    vertical-align: middle;
    text-align: center;
    margin: 0 2px;
  }}
  .frac-num {{
    display: block;
    border-bottom: 1px solid rgba(255, 255, 255, 0.6);
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
    border-bottom: 1.5px solid #38bdf8;
    padding-bottom: 1px;
  }}
  .ub-label {{
    font-size: 7.4pt;
    color: #38bdf8;
  }}
  .bmatrix {{
    display: inline-table;
    vertical-align: middle;
    border-left: 1.5px solid #94a3b8;
    border-right: 1.5px solid #94a3b8;
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
<script>
  let currentSlide = 0;
  const slides = document.querySelectorAll('.slide');
  function showSlide(index) {{
    if (index >= 0 && index < slides.length) {{
      currentSlide = index;
      slides[currentSlide].scrollIntoView({{ behavior: 'smooth' }});
    }}
  }}
  document.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') {{
      e.preventDefault();
      showSlide(currentSlide + 1);
    }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
      e.preventDefault();
      showSlide(currentSlide - 1);
    }} else if (e.key === 'Home') {{
      e.preventDefault();
      showSlide(0);
    }} else if (e.key === 'End') {{
      e.preventDefault();
      showSlide(slides.length - 1);
    }} else if (e.key === 'f' || e.key === 'F') {{
      if (!document.fullscreenElement) {{
        document.documentElement.requestFullscreen().catch(()=>{{}});
      }} else {{
        document.exitFullscreen().catch(()=>{{}});
      }}
    }}
  }});
</script>
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

    # Also sync a copy directly into docs/ as requested
    shutil.copy2(SLIDES_PDF_PATH, SLIDES_PDF_DOCS_PATH)
    print(f"Synced copy -> {SLIDES_PDF_DOCS_PATH}")

    if ARTIFACT_DIR.exists():
        shutil.copy2(SLIDES_PDF_PATH, ARTIFACT_DIR / SLIDES_PDF_PATH.name)
        shutil.copy2(SLIDES_MD_PATH, ARTIFACT_DIR / SLIDES_MD_PATH.name)
        if (ROOT_DIR / "Final_Project_Report.pdf").exists():
            shutil.copy2(ROOT_DIR / "Final_Project_Report.pdf", ARTIFACT_DIR / "Final_Project_Report.pdf")
        if (ROOT_DIR / "docs" / "Final_Project_Report.md").exists():
            shutil.copy2(ROOT_DIR / "docs" / "Final_Project_Report.md", ARTIFACT_DIR / "Final_Project_Report.md")
        for fig_file in (ROOT_DIR / "results" / "figures").glob("*.*"):
            shutil.copy2(fig_file, ARTIFACT_DIR / fig_file.name)


if __name__ == "__main__":
    main()
