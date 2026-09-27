"""Render a supplied PDF and summarize page/float locations without modifying it."""
from pathlib import Path
import re
import sys

import pymupdf

sys.stdout.reconfigure(encoding="utf-8")
source = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
document = pymupdf.open(source)
print(f"PDF: {source.name}; pages: {len(document)}")
for index, page in enumerate(document, 1):
    page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(
        destination / f"page-{index:02d}.png"
    )
    blocks = page.get_text("blocks")
    print(f"\nPAGE {index}")
    for block in blocks:
        if block[6] != 0:
            continue
        text = " ".join(block[4].split())
        if (block[1] < 130 or block[3] > 670
                or re.match(r"(?:Figure|Table)\s+\d+[:.]", text)
                or text.startswith(("2 Related", "3 Method", "4 Experiment"))):
            print(f"  ({block[0]:.0f},{block[1]:.0f}) {text[:700]}")
