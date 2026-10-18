"""Figures for Chapter ch:walkthrough (2026-10-18): composed, without change, from the images the chain wrote into its own reports
(Biological_Physics/MethylPhys/doors/data/REPORT_EXAMPLES_V4/final). Nothing is drawn or computed here.
Writes docs/book/figures/part6/fig_p6_walk_overlays.{png,pdf} and fig_p6_walk_gauges.{png,pdf}."""
import os
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
F = os.path.join(REPO, "Biological_Physics/MethylPhys/doors/data/REPORT_EXAMPLES_V4/final") + "/"; O = os.path.join(REPO, "docs/book/figures/part6") + "/"
def col(sky, bar, w=1400):
    sky, bar = Image.open(F + sky).convert("RGB"), Image.open(F + bar).convert("RGB")
    sky = sky.resize((w, int(sky.height * w / sky.width))); bar = bar.resize((w, int(bar.height * w / bar.width)))
    S = Image.new("RGB", (w, sky.height + bar.height + 20), "white"); S.paste(sky, (0, 0)); S.paste(bar, (0, sky.height + 20)); return S
A = col("GSE235717_GSM7508429_changes_sky.png", "GSE235717_GSM7508429_gauge_cscore.png")
B = col("GSE235717_GSM7508429_PLANTED_CLUSTERED_changes_sky.png", "GSE235717_GSM7508429_PLANTED_CLUSTERED_gauge_cscore.png")
S = Image.new("RGB", (A.width + B.width + 60, max(A.height, B.height)), "white"); S.paste(A, (0, 0)); S.paste(B, (A.width + 60, 0))
S.save(O + "fig_p6_walk_overlays.png"); S.save(O + "fig_p6_walk_overlays.pdf", resolution=200)
ims = [Image.open(F + f).convert("RGB") for f in ("GSE235717_GSM7508429_gauge_met_a.png", "GSM4293895_CD8_gauge_iam_a.png")]; w = 2000
ims = [i.resize((w, int(i.height * w / i.width))) for i in ims]; G = Image.new("RGB", (w, sum(i.height for i in ims) + 30), "white"); y = 0
for i in ims: G.paste(i, (0, y)); y += i.height + 30
G.save(O + "fig_p6_walk_gauges.png"); G.save(O + "fig_p6_walk_gauges.pdf", resolution=200); print("written")
