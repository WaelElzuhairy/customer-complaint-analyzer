"""Export the academic report Markdown to a styled PDF using xhtml2pdf."""

import re
import base64
from pathlib import Path

import markdown
from xhtml2pdf import pisa

BASE_DIR    = Path(__file__).resolve().parent.parent
REPORT_MD   = BASE_DIR / "report" / "report_draft.md"
REPORT_PDF  = BASE_DIR / "report" / "Customer_Complaint_Analyzer_Report.pdf"
RESULTS_DIR = BASE_DIR / "results"

# ── Read & convert Markdown ───────────────────────────────────────────────────
print("Reading report...")
md_text = REPORT_MD.read_text(encoding="utf-8")

# Remove the figure list section (we'll append inline images separately)
md_text_body = re.sub(
    r"\n---\n\n## List of Figures.*?(?=\n---\n\n## References)",
    "",
    md_text,
    flags=re.DOTALL,
)

md_conv  = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])
body_html = md_conv.convert(md_text_body)

# ── Embed figures as base64 ───────────────────────────────────────────────────
FIGURES = [
    ("Figure 1: Class distribution before and after balancing",
     "eda_balance_comparison.png"),
    ("Figure 2: Narrative length distribution with 128- and 256-token cutoffs",
     "eda_text_length.png"),
    ("Figure 3: Top TF-IDF features per class (logistic regression coefficients)",
     "baseline_top_features.png"),
    ("Figure 4: BiLSTM training dynamics — loss, Macro F1, and accuracy over 10 epochs",
     "bilstm_training_curves.png"),
    ("Figure 6: Grouped bar chart comparing Accuracy, Precision, Recall, Macro F1",
     "comparison_grouped_bar.png"),
    ("Figure 7: Per-class F1-score heatmap across all three models (test set)",
     "comparison_f1_heatmap.png"),
    ("Figure 8: Confusion matrix — TF-IDF + Logistic Regression",
     "confusion_matrix_baseline_product.png"),
    ("Figure 9: Confusion matrix — BiLSTM + Attention",
     "confusion_matrix_bilstm_attention_product.png"),
    ("Figure 10: Confusion matrix — DistilBERT (Fine-Tuned)",
     "confusion_matrix_distilbert_product.png"),
    ("Figure 11: Word clouds — most frequent terms per product category",
     "eda_word_clouds.png"),
    ("Figure 12: Issue group distribution (bar and pie)",
     "eda_issue_groups.png"),
    ("Figure 13: Complaint priority distribution",
     "eda_priority_distribution.png"),
]

figures_html = "<h2>Figures</h2>\n"
for caption, fname in FIGURES:
    fpath = RESULTS_DIR / fname
    if not fpath.exists():
        figures_html += f"<p><em>{caption} — [file not found]</em></p>\n"
        print(f"  SKIP: {fname}")
        continue
    data = base64.b64encode(fpath.read_bytes()).decode()
    figures_html += (
        f'<div class="figure">'
        f'<p class="fig-caption"><strong>{caption}</strong></p>'
        f'<img src="data:image/png;base64,{data}" style="max-width:100%;">'
        f'</div>\n'
    )
    print(f"  Embedded: {caption[:55]}")

# ── Full HTML document ────────────────────────────────────────────────────────
CSS = """
@page {
    size: A4;
    margin: 2.2cm 2cm 2.2cm 2cm;
    @frame footer {
        -pdf-frame-content: footer;
        bottom: 1cm;
        margin-left: 2cm;
        margin-right: 2cm;
        height: 1cm;
    }
}

body {
    font-family: Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #222;
}

h1 {
    font-size: 18pt;
    color: #1a3a6e;
    border-bottom: 2pt solid #1a3a6e;
    padding-bottom: 4pt;
    margin-top: 0;
}

h2 {
    font-size: 14pt;
    color: #1a3a6e;
    border-bottom: 0.5pt solid #aaa;
    padding-bottom: 2pt;
    margin-top: 18pt;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    color: #333;
    margin-top: 12pt;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    color: #555;
    margin-top: 8pt;
    page-break-after: avoid;
}

p { margin: 5pt 0; }

table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

th {
    background-color: #1a3a6e;
    color: #ffffff;
    padding: 5pt 7pt;
    text-align: left;
}

td {
    padding: 4pt 7pt;
    border: 0.5pt solid #ccc;
}

tr:nth-child(even) td { background-color: #eef1f8; }

code {
    font-family: Courier, monospace;
    font-size: 9pt;
    background: #f0f0f0;
    padding: 1pt 3pt;
}

pre {
    background: #f4f4f4;
    border-left: 3pt solid #1a3a6e;
    padding: 8pt;
    font-size: 8.5pt;
    white-space: pre-wrap;
    page-break-inside: avoid;
}

hr { border: 0.3pt solid #ccc; margin: 14pt 0; }

blockquote {
    border-left: 3pt solid #1a3a6e;
    margin: 8pt 0 8pt 12pt;
    padding: 4pt 10pt;
    color: #555;
    font-style: italic;
}

ul, ol { margin: 5pt 0 5pt 18pt; }
li { margin: 2pt 0; }

.figure {
    margin: 14pt 0;
    text-align: center;
    page-break-inside: avoid;
    page-break-before: auto;
}

.fig-caption {
    font-size: 9.5pt;
    color: #444;
    font-style: italic;
    margin-bottom: 5pt;
}

#footer { font-size: 8pt; color: #888; text-align: center; }
"""

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body>
{body_html}
<hr>
{figures_html}
<div id="footer">
    <pdf:pagenumber> / <pdf:pagecount>
</div>
</body>
</html>"""

# ── Render PDF ────────────────────────────────────────────────────────────────
print("\nRendering PDF (this may take 30-60s for embedded images)...")
with open(REPORT_PDF, "wb") as pdf_file:
    result = pisa.CreatePDF(
        full_html.encode("utf-8"),
        dest=pdf_file,
        encoding="utf-8",
    )

if result.err:
    print(f"PDF generation errors: {result.err}")
else:
    size_mb = REPORT_PDF.stat().st_size / 1_048_576
    print(f"\nPDF saved: {REPORT_PDF}")
    print(f"Size: {size_mb:.1f} MB")
