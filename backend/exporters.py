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
    def generate_latex(title: str, md_content: str, author: str = "YouTube NotebookLM") -> str:
        """Generates LaTeX document string."""
        path = NoteExporter.markdown_to_latex(title, author, md_content)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def generate_pdf(title: str, md_content: str, author: str = "YouTube NotebookLM") -> str:
        """Generates PDF and returns the filename."""
        path = NoteExporter.markdown_to_pdf(title, author, md_content)
        return Path(path).name

    @staticmethod
    def generate_html(title: str, md_content: str, author: str = "YouTube NotebookLM") -> str:
        """Generates HTML document string."""
        path = NoteExporter.markdown_to_standalone_html(title, author, md_content)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

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
    def _normalize_text_for_pdf(text: str) -> str:
        """Normalizes unicode special characters for ReportLab PDF rendering."""
        replacements = {
            '\u2011': '-', # non-breaking hyphen
            '\u2012': '-', # figure dash
            '\u2013': '-', # en dash
            '\u2014': ' -- ', # em dash
            '\u2015': ' -- ', # horizontal bar
            '\u2018': "'", # left single quote
            '\u2019': "'", # right single quote
            '\u201a': "'",
            '\u201b': "'",
            '\u201c': '"', # left double quote
            '\u201d': '"', # right double quote
            '\u201e': '"',
            '\u201f': '"',
            '\u00a0': ' ', # non-breaking space
            '\u2026': '...', # ellipsis
            '\u2212': '-', # minus sign
            '\u2192': '->', # right arrow
            '\u2190': '<-', # left arrow
            '\u2194': '<->', # left right arrow
            '\u2264': '<=', # less than or equal
            '\u2265': '>=', # greater than or equal
            '\u2260': '!=', # not equal
            '\u221e': 'inf', # infinity
            '\u2200': 'FORALL ', # forall
            '\u2203': 'EXISTS ', # exists
            '\u2208': ' IN ', # in
            '\u2229': ' AND ',
            '\u222a': ' OR ',
            '\u00d7': 'x', # multiplication
        }
        for char, repl in replacements.items():
            text = text.replace(char, repl)
        return text

    @staticmethod
    def markdown_to_pdf(title: str, author: str, md_content: str, output_filename: Optional[str] = None) -> str:
        """
        3. Compiled PDF (.pdf) Export with Full Unicode Normalization & Table Support
        """
        if not output_filename:
            safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:40]
            output_filename = f"{safe_title}.pdf"

        file_path = str(EXPORTS_DIR / output_filename)
        doc = SimpleDocTemplate(
            file_path,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=20,
            leading=24,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=4
        )

        subtitle_style = ParagraphStyle(
            'DocSubTitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor('#64748b'),
            spaceAfter=12
        )

        h1_style = ParagraphStyle(
            'SectionH1',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=14,
            leading=18,
            textColor=colors.HexColor('#1e40af'),
            spaceBefore=12,
            spaceAfter=4,
            keepWithNext=True
        )

        h2_style = ParagraphStyle(
            'SectionH2',
            parent=styles['Heading3'],
            fontName='Helvetica-Bold',
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor('#334155'),
            spaceBefore=8,
            spaceAfter=3,
            keepWithNext=True
        )

        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=5
        )

        bullet_style = ParagraphStyle(
            'Bullet',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#1e293b'),
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=2
        )

        table_header_style = ParagraphStyle(
            'TableHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor('#0f172a')
        )

        table_cell_style = ParagraphStyle(
            'TableCell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor('#334155')
        )

        code_style = ParagraphStyle(
            'CodeBlock',
            fontName='Courier',
            fontSize=8,
            leading=10.5,
            textColor=colors.HexColor('#0f172a'),
            backColor=colors.HexColor('#f1f5f9'),
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=4
        )

        math_style = ParagraphStyle(
            'MathBlock',
            fontName='Courier-Oblique',
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor('#1e3a8a'),
            backColor=colors.HexColor('#f0f9ff'),
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=4
        )

        story = []

        # Header Title
        clean_title = NoteExporter._normalize_text_for_pdf(title)
        clean_author = NoteExporter._normalize_text_for_pdf(author)
        story.append(Paragraph(html.escape(clean_title), title_style))
        story.append(Paragraph(f"<b>Generated by YouTube NotebookLM</b> | {html.escape(clean_author)}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#3b82f6'), spaceAfter=10))

        # Pre-normalize entire markdown content
        normalized_content = NoteExporter._normalize_text_for_pdf(md_content)
        lines = normalized_content.split("\n")
        
        in_code = False
        code_lines = []
        in_table = False
        table_rows = []

        def flush_table():
            nonlocal table_rows, in_table
            if not table_rows:
                in_table = False
                return
            
            # Format rows with Paragraphs
            formatted_data = []
            for r_idx, row in enumerate(table_rows):
                row_cells = []
                is_header = (r_idx == 0)
                st_to_use = table_header_style if is_header else table_cell_style
                for c in row:
                    cell_p = Paragraph(html.escape(c.strip()), st_to_use)
                    row_cells.append(cell_p)
                if row_cells:
                    formatted_data.append(row_cells)

            if formatted_data:
                col_count = max(len(r) for r in formatted_data)
                col_w = min(510 / col_count, 220)
                t = Table(formatted_data, colWidths=[col_w] * col_count)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#0f172a')),
                    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
                    ('TOPPADDING', (0,0), (-1,-1), 4),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                    ('LEFTPADDING', (0,0), (-1,-1), 5),
                    ('RIGHTPADDING', (0,0), (-1,-1), 5),
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 4))
            
            table_rows = []
            in_table = False

        for raw_line in lines:
            line = raw_line.strip()
            
            # Code block handling
            if line.startswith("```"):
                if in_table:
                    flush_table()
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

            # Markdown Table detection
            if line.startswith("|") and line.endswith("|"):
                # Check if it's separator row |---|---|
                if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                    continue
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if cells:
                    in_table = True
                    table_rows.append(cells)
                continue
            elif in_table:
                flush_table()

            if not line:
                story.append(Spacer(1, 3))
                continue

            # Math display block handling ($$ or \[ ... \])
            if (line.startswith("$$") and line.endswith("$$")) or line.startswith("\\[") or line.startswith("\\]") or line.startswith("\\["):
                clean_math = line.replace("$$", "").replace("\\[", "").replace("\\]", "").strip()
                if clean_math:
                    story.append(Preformatted(clean_math, math_style))
                continue

            # Horizontal rules
            if line in ["---", "***", "___"]:
                story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#cbd5e1'), spaceBefore=6, spaceAfter=6))
                continue

            # Parse line prefixes safely BEFORE html escaping
            if line.startswith("# "):
                content = line[2:].strip()
                escaped = html.escape(content)
                escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
                story.append(Paragraph(escaped, h1_style))
            elif line.startswith("## "):
                content = line[3:].strip()
                escaped = html.escape(content)
                escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
                story.append(Paragraph(escaped, h1_style))
            elif line.startswith("### "):
                content = line[4:].strip()
                escaped = html.escape(content)
                escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
                story.append(Paragraph(escaped, h2_style))
            elif line.startswith("#### "):
                content = line[5:].strip()
                escaped = html.escape(content)
                escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
                story.append(Paragraph(escaped, h2_style))
            elif line.startswith("- ") or line.startswith("* "):
                content = line[2:].strip()
                escaped = html.escape(content)
                escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
                escaped = re.sub(r'`(.*?)`', r'<font name="Courier" color="#2563eb">\1</font>', escaped)
                story.append(Paragraph(f"&bull; {escaped}", bullet_style))
            elif line.startswith("> "):
                content = line[2:].strip()
                escaped = html.escape(content)
                escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
                escaped = re.sub(r'`(.*?)`', r'<font name="Courier" color="#2563eb">\1</font>', escaped)
                callout_data = [[Paragraph(f"<b>Key Takeaway:</b> {escaped}", body_style)]]
                callout_table = Table(callout_data, colWidths=[510])
                callout_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#eff6ff')),
                    ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#60a5fa')),
                    ('LEFTPADDING', (0,0), (-1,-1), 8),
                    ('RIGHTPADDING', (0,0), (-1,-1), 8),
                    ('TOPPADDING', (0,0), (-1,-1), 5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ]))
                story.append(Spacer(1, 3))
                story.append(callout_table)
                story.append(Spacer(1, 3))
            else:
                escaped = html.escape(line)
                escaped = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', escaped)
                escaped = re.sub(r'\*(.*?)\*', r'<i>\1</i>', escaped)
                escaped = re.sub(r'`(.*?)`', r'<font name="Courier" color="#2563eb">\1</font>', escaped)
                story.append(Paragraph(escaped, body_style))

        if in_table:
            flush_table()

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

# Aliases for compatibility
NoteExporter.markdown_to_html = NoteExporter.markdown_to_standalone_html
Exporter = NoteExporter

