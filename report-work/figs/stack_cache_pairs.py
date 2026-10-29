"""Re-layout the existing scientific figure without changing its plotted data.

Original graph pixels, including axes, bars, and value labels, are preserved.
Only the three panel positions and the shared legend layout change.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "teammate-cache-figs/cache-fig1-pairs-comparison-v3.png"
OUT = ROOT / "figs/report-figures"
image = mpimg.imread(SOURCE)
assert image.shape[:2] == (972, 4374)

labels = ["GPTCache", "GPTCache + reranker", "GPTCache + classifier",
          "LangCache", "cosine only", "cosine + reranker", "cosine + NLI"]
# Colors sampled from the solid bars in the original figure.
colors = ["#f2b134", "#17becf", "#b5cc18", "#9b4fd8", "#2c6fd6", "#e4572e", "#1b9e77"]

width, height, dpi = 1500, 2820, 200
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor="white")
for panel in range(3):
    # Both vertical boundaries lie in whitespace between the original panels.
    crop = image[:850, panel * 1458:(panel + 1) * 1458]
    top = 20 + panel * 890
    fig.figimage(crop, xo=21, yo=height - top - 850, origin="upper")

handles = [Patch(facecolor=c, edgecolor="#222222" if i == 6 else "none", label=l)
           for i, (c, l) in enumerate(zip(colors, labels))]
# Matplotlib fills legend columns first; reorder for the original row-wise order.
order = [0, 4, 1, 5, 2, 6, 3]
fig.legend(handles=[handles[i] for i in order], loc="lower center",
           bbox_to_anchor=(0.5, 0.015), ncol=4, frameon=False,
           fontsize=9.5, columnspacing=1.25, handlelength=1.5, labelspacing=0.9)
OUT.mkdir(parents=True, exist_ok=True)
stem = OUT / "5-03a-cache-pairs-comparison-vertical"
fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
fig.savefig(stem.with_suffix(".svg"), facecolor="white")
plt.close(fig)
print(stem.with_suffix(".png"))
