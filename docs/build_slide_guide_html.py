"""
Compiles `docs/Slide_By_Slide_Presentation_Guide.md` into a self-contained, responsive HTML guide with MathJax.
"""
from pathlib import Path
import re

ROOT_DIR = Path(__file__).resolve().parent.parent
MD_PATH = ROOT_DIR / "docs" / "Slide_By_Slide_Presentation_Guide.md"
HTML_PATH = ROOT_DIR / "docs" / "Slide_By_Slide_Presentation_Guide.html"

def md_to_html(md_text: str) -> str:
    # Convert tables
    def replace_table(match):
        lines = match.group(0).strip().splitlines()
        if len(lines) < 2:
            return match.group(0)
        headers = [c.strip() for c in lines[0].split('|')[1:-1]]
        html = ['<div class="table-wrap"><table><thead><tr>']
        for h in headers:
            html.append(f'<th>{h}</th>')
        html.append('</tr></thead><tbody>')
        for line in lines[2:]:
            cols = [c.strip() for c in line.split('|')[1:-1]]
            html.append('<tr>')
            for c in cols:
                html.append(f'<td>{c}</td>')
            html.append('</tr>')
        html.append('</tbody></table></div>')
        return "".join(html)

    # Tables pattern
    table_pattern = re.compile(r'((?:\|[^\n]+\|\n){2,})')
    content = table_pattern.sub(replace_table, md_text)

    # Headings
    content = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)

    # Blockquotes
    def replace_blockquote(match):
        lines = match.group(0).strip().splitlines()
        clean_lines = [re.sub(r'^>\s?', '', l) for l in lines]
        return f'<blockquote>{"<br/>".join(clean_lines)}</blockquote>'
    content = re.sub(r'((?:^>[^\n]*\n?)+)', replace_blockquote, content, flags=re.MULTILINE)

    # Code blocks
    content = re.sub(r'```([a-z]*)\n(.*?)```', r'<pre><code>\2</code></pre>', content, flags=re.DOTALL)

    # Lists
    def replace_list(match):
        items = match.group(0).strip().splitlines()
        lis = []
        for it in items:
            it_text = re.sub(r'^\s*[\*\-]\s+', '', it)
            lis.append(f'<li>{it_text}</li>')
        return f'<ul>{"".join(lis)}</ul>'
    content = re.sub(r'((?:^\s*[\*\-]\s+[^\n]+\n?)+)', replace_list, content, flags=re.MULTILINE)

    # Inline styles: bold, italic, code
    content = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content)
    content = re.sub(r'\*(.*?)\*', r'<em>\1</em>', content)
    content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)

    # HR
    content = re.sub(r'^---$', r'<hr/>', content, flags=re.MULTILINE)

    # Paragraphs
    paragraphs = content.split('\n\n')
    out_paras = []
    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        if any(p_str.startswith(tag) for tag in ['<h1', '<h2', '<h3', '<div', '<table', '<ul', '<blockquote', '<pre', '<hr']):
            out_paras.append(p_str)
        else:
            out_paras.append(f'<p>{p_str}</p>')

    return "\n".join(out_paras)

def main():
    md_text = MD_PATH.read_text(encoding='utf-8')
    body_html = md_to_html(md_text)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Slide-by-Slide Presentation & Defense Master Guide (16 Slides)</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }}
    }};
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --card-border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --emerald: #10b981;
      --amber: #f59e0b;
      --rose: #f43f5e;
      --purple: #a855f7;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.65;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1120px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 2.2rem;
      color: var(--accent);
      border-bottom: 2px solid var(--card-border);
      padding-bottom: 12px;
      margin-bottom: 8px;
    }}
    h2 {{
      font-size: 1.5rem;
      color: var(--text);
      margin-top: 36px;
      margin-bottom: 12px;
      padding-left: 10px;
      border-left: 4px solid var(--accent);
    }}
    h3 {{
      font-size: 1.15rem;
      color: var(--muted);
      margin-top: 18px;
      margin-bottom: 8px;
    }}
    p, li {{
      font-size: 0.98rem;
      color: #cbd5e1;
    }}
    strong {{ color: #ffffff; }}
    em {{ color: var(--accent); }}
    code {{
      background: #090d16;
      color: #38bdf8;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: "Cascadia Code", Consolas, Monaco, monospace;
      font-size: 0.88em;
    }}
    pre {{
      background: #090d16;
      padding: 14px 18px;
      border-radius: 8px;
      overflow-x: auto;
      border: 1px solid var(--card-border);
    }}
    pre code {{
      background: none;
      padding: 0;
      color: #a5f3fc;
    }}
    blockquote {{
      background: #172554;
      border-left: 4px solid #38bdf8;
      margin: 14px 0;
      padding: 12px 18px;
      border-radius: 0 8px 8px 0;
      color: #e0f2fe;
      font-style: italic;
    }}
    .table-wrap {{
      overflow-x: auto;
      margin: 16px 0;
      border: 1px solid var(--card-border);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
      text-align: left;
    }}
    th {{
      background: #1e293b;
      color: var(--accent);
      padding: 10px 14px;
      font-weight: 600;
      border-bottom: 1px solid var(--card-border);
    }}
    td {{
      padding: 9px 14px;
      border-bottom: 1px solid #1e293b;
      color: #cbd5e1;
    }}
    tr:nth-child(even) td {{
      background: #131c2e;
    }}
    hr {{
      border: 0;
      height: 1px;
      background: var(--card-border);
      margin: 32px 0;
    }}
    ul {{
      padding-left: 24px;
      margin: 8px 0;
    }}
    li {{
      margin-bottom: 6px;
    }}
  </style>
</head>
<body>
  <div class="container">
    {body_html}
  </div>
</body>
</html>
"""
    HTML_PATH.write_text(full_html, encoding='utf-8')
    print(f"Wrote updated HTML guide -> {HTML_PATH} ({HTML_PATH.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()
