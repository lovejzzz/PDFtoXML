"""PDF rasterization: render each page to PNG at 300 DPI."""

import os
import sys

import fitz  # PyMuPDF

PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "charlie-parker-omnibook.pdf")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pages")
DPI = 300


def ingest_pdf(pdf_path: str = PDF_PATH, output_dir: str = OUTPUT_DIR, dpi: int = DPI) -> list[str]:
    """Rasterize all pages of a PDF to PNG files.

    Returns list of output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    paths = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(output_dir, f"page_{i + 1:03d}.png")
        pix.save(out_path)
        paths.append(out_path)

    doc.close()
    print(f"Rasterized {len(paths)} pages to {output_dir} at {dpi} DPI")
    return paths


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    ingest_pdf(pdf)
