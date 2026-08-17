import os
import re
import html
import json
from pathlib import Path
from typing import Dict, Any, Optional
import markdown as md_lib
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Preformatted
)
from backend.config import EXPORTS_DIR

class NoteExporter:
    @staticmethod
    def export_to_markdown(title: str, content: str, output_filename: Optional[str] = None) -> str:
        """
        1. Markdown (.md) Export
        """
        if not output_filename:
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:40]
            output_filename = f"{safe_title}.md"
        
        file_path = EXPORTS_DIR / output_filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return str(file_path)

    @staticmethod
    def markdown_to_latex(title: str, author: str, md_content: str, output_filename: Optional[str] = None) -> str:
        """
        2. Academic LaTeX (.tex) Export
        """
        if not output_filename:
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:40]
            output_filename = f"{safe_title}.tex"

        tex_lines = [
            r"\documentclass[11pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[margin=1in]{geometry}",
            r"\usepackage{amsmath,amssymb,amsfonts}",
            r"\usepackage{hyperref}",
            r"\usepackage{xcolor}",
            r"\usepackage{tcolorbox}",
            r"\usepackage{listings}",
            r"\usepackage{enumitem}",
            r"\usepackage{booktabs}",
            r"\usepackage{titlesec}",
            r"",
            r"\definecolor{primary}{RGB}{37, 99, 235}",
            r"\definecolor{codebg}{RGB}{245, 247, 250}",
            r"\definecolor{calloutbg}{RGB}{240, 249, 255}",
            r"\definecolor{calloutborder}{RGB}{14, 165, 233}",
            r"",
            r"\hypersetup{colorlinks=true, linkcolor=primary, urlcolor=primary}",
            r"\titleformat{\section}{\large\bfseries\color{primary}}{\thesection}{1em}{}[\titlerule]",
            r"\titleformat{\subsection}{\normalsize\bfseries\color{darkgray}}{\thesubsection}{1em}{}",
            r"",
            r"\newtcolorbox{calloutbox}[1][]{colback=calloutbg, colframe=calloutborder, fonttitle=\bfseries, title=#1, arc=3mm}",
            r"",
            f"\\title{{\\textbf{{{title}}}}}",
            f"\\author{{{author}}}",
            r"\date{\today}",
            r"",
            r"\begin{document}",
            r"\maketitle",
            r"\tableofcontents",
            r"\vspace{1cm}",
            r"\hrule",
            r"\vspace{0.5cm}",
            r""
        ]

        in_code_block = False
        code_buffer = []

        for line in md_content.split("\n"):
            line_str = line.strip()
            
            if line_str.startswith("```"):
                if in_code_block:
                    tex_lines.append(r"\begin{lstlisting}[backgroundcolor=\color{codebg}, basicstyle=\ttfamily\small, breaklines=true]")
                    tex_lines.extend(code_buffer)
                    tex_lines.append(r"\end{lstlisting}")
                    code_buffer = []
                    in_code_block = False
                else:
                    in_code_block = True
                continue

            if in_code_block:
                code_buffer.append(line)
                continue

            if line_str.startswith("# "):
                tex_lines.append(f"\\section*{{{line_str[2:].strip()}}}")
            elif line_str.startswith("## "):
                tex_lines.append(f"\\section{{{line_str[3:].strip()}}}")
            elif line_str.startswith("### "):
                tex_lines.append(f"\\subsection{{{line_str[4:].strip()}}}")
            elif line_str.startswith("#### "):
                tex_lines.append(f"\\subsubsection*{{{line_str[5:].strip()}}}")
            elif line_str.startswith("- ") or line_str.startswith("* "):
                clean_bullet = line_str[2:].replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
                clean_bullet = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', clean_bullet)
                tex_lines.append(f"\\begin{{itemize}}[noitemsep]\n  \\item {clean_bullet}\n\\end{{itemize}}")
            elif line_str.startswith("> "):
                clean_quote = line_str[2:].replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
                clean_quote = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', clean_quote)
                tex_lines.append(f"\\begin{{calloutbox}}[Key Takeaway]\n{clean_quote}\n\\end{{calloutbox}}")
            elif line_str:
                clean_p = line_str.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
                clean_p = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', clean_p)
                tex_lines.append(f"{clean_p}\n")
            else:
                tex_lines.append(r"\par")

        tex_lines.append(r"\end{document}")

        file_path = EXPORTS_DIR / output_filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(tex_lines))
        return str(file_path)

    @staticmethod
    def markdown_to_pdf(title: str, author: str, md_content: str, output_filename: Optional[str] = None) -> str:
        """
        3. Compiled PDF (.pdf) Export
        """
        if not output_filename:
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:40]
            output_filename = f"{safe_title}.pdf"

        file_path = str(EXPORTS_DIR / output_filename)
        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            rightMargin=45,
            leftMargin=45,
            topMargin=45,
            bottomMargin=45
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=22,
            leading=26,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=6
        )

        subtitle_style = ParagraphStyle(
            'DocSubTitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#64748b'),
            spaceAfter=15
        )

        h1_style = ParagraphStyle(
            'SectionH1',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=15,
            leading=19,
            textColor=colors.HexColor('#1e40af'),
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True
        )

        h2_style = ParagraphStyle(
            'SectionH2',
            parent=styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#334155'),
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True
        )

        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=6
        )

        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor('#1e293b'),
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=3
        )

        code_style = ParagraphStyle(
            'CodeBlock',
            fontName='Courier',
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor('#0f172a'),
            backColor=colors.HexColor('#f1f5f9'),
            borderPadding=6,
            spaceBefore=6,
            spaceAfter=6
        )

        story = []

        # Header Title
        story.append(Paragraph(html.escape(title), title_style))
        story.append(Paragraph(f"<b>Generated by YouTube NotebookLM</b> | {html.escape(author)}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceAfter=14))

        in_code = False
        code_lines = []

        for raw_line in md_content.split("\n"):
            line = raw_line.strip()
            
            if line.startswith("```"):
                if in_code:
                    code_text = html.escape("\n".join(code_lines))
                    story.append(Preformatted(code_text, code_style))
                    code_lines = []
                    in_code = False
                else:
                    in_code = True
                continue

            if in_code:
                code_lines.append(raw_line)
                continue

            if not line:
                story.append(Spacer(1, 4))
                continue

            processed = html.escape(line)
            processed = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', processed)
            processed = re.sub(r'`(.*?)`', r'<font name="Courier" color="#2563eb">\1</font>', processed)

            if line.startswith("# "):
                story.append(Paragraph(processed[2:], h1_style))
            elif line.startswith("## "):
                story.append(Paragraph(processed[3:], h1_style))
            elif line.startswith("### "):
                story.append(Paragraph(processed[4:], h2_style))
            elif line.startswith("- ") or line.startswith("* "):
                bullet_content = "&bull; " + processed[2:]
                story.append(Paragraph(bullet_content, bullet_style))
            elif line.startswith("> "):
                callout_data = [[Paragraph(f"<b>Takeaway:</b> {processed[2:]}", body_style)]]
                callout_table = Table(callout_data, colWidths=[500])
                callout_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eff6ff')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#60a5fa')),
                    ('LEFTPADDING', (0,0), (-1,-1), 10),
                    ('RIGHTPADDING', (0,0), (-1,-1), 10),
                    ('TOPPADDING', (0,0), (-1,-1), 6),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 6),
                ]))
                story.append(Spacer(1, 4))
                story.append(callout_table)
                story.append(Spacer(1, 4))
            else:
                story.append(Paragraph(processed, body_style))

        doc.build(story)
        return file_path

    @staticmethod
    def markdown_to_standalone_html(title: str, author: str, md_content: str, output_filename: Optional[str] = None) -> str:
        """
        4. Standalone Styled Web Document (.html) Export
        """
        if not output_filename:
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:40]
            output_filename = f"{safe_title}.html"

        body_html = md_lib.markdown(md_content, extensions=['fenced_code', 'tables'])

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html.escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #38bdf8;
      --border: #334155;
    }}
    body {{
      font-family: 'Inter', sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.7;
      padding: 40px 20px;
      margin: 0;
    }}
    .container {{
      max-width: 860px;
      margin: auto;
      background: var(--card-bg);
      padding: 40px;
      border-radius: 16px;
      border: 1px solid var(--border);
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }}
    h1, h2, h3, h4 {{ font-family: 'Outfit', sans-serif; color: #fff; margin-top: 24px; }}
    h1 {{ font-size: 28px; border-bottom: 2px solid var(--accent); padding-bottom: 10px; color: var(--accent); }}
    h2 {{ font-size: 20px; color: #93c5fd; border-bottom: 1px solid var(--border); padding-bottom: 6px; }}
    h3 {{ font-size: 16px; color: #cbd5e1; }}
    p {{ margin: 14px 0; color: #e2e8f0; font-size: 15px; }}
    blockquote {{
      background: rgba(56, 189, 248, 0.1);
      border-left: 4px solid var(--accent);
      padding: 12px 18px;
      border-radius: 0 8px 8px 0;
      margin: 18px 0;
      font-style: italic;
    }}
    pre {{
      background: #090d16;
      border: 1px solid var(--border);
      padding: 16px;
      border-radius: 10px;
      overflow-x: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 13.5px;
    }}
    code {{
      font-family: 'JetBrains Mono', monospace;
      background: rgba(255,255,255,0.08);
      padding: 2px 6px;
      border-radius: 4px;
      color: #38bdf8;
      font-size: 13px;
    }}
    ul, ol {{ margin-left: 24px; margin-bottom: 16px; }}
    li {{ margin-bottom: 6px; }}
    .header-meta {{
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 24px;
    }}
    .btn-print {{
      background: #3b82f6;
      color: #fff;
      border: none;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      float: right;
    }}
  </style>
</head>
<body>
  <div class="container">
    <button class="btn-print" onclick="window.print()">🖨️ Print / Save PDF</button>
    <div class="header-meta">YouTube NotebookLM &bull; {html.escape(author)} &bull; Exported Note</div>
    {body_html}
  </div>
</body>
</html>
"""
        file_path = EXPORTS_DIR / output_filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        return str(file_path)
