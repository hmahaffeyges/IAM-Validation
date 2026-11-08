"""A still of a globe for the front page's atlas cards (2026-11-08, author: show the sequencing atlas as a globe too, like the viewer). Input: a
full-page screenshot of the live Globe Viewer taken by the committed page_shots.yml workflow (for example atlas/sphere.html?globe=seq-atlas&cell=
granulocytes). The globe is found as the largest connected region of pixels brighter than the page's dark background on the left side of the
page (the controls sit on the right), cropped square with a small margin and saved as a 720 x 720 JPEG. A picture of the viewer: nothing is read or
scored from it. Usage: python3 globe_still_crop.py SCREENSHOT.png OUT.jpg"""
import sys
import numpy as np
from PIL import Image
im = Image.open(sys.argv[1]).convert("RGB"); a = np.asarray(im).astype(int); H, W, _ = a.shape
bg = np.median(a[:, :40].reshape(-1, 3), axis=0)                       # the page background, from its left margin
lit = (np.abs(a - bg).sum(axis=2) > 60)
lit[:, int(W * 0.58):] = False                                          # the controls column
from scipy import ndimage
lab, n = ndimage.label(ndimage.binary_closing(lit, iterations=3)); big = np.argmax(np.bincount(lab.ravel())[1:]) + 1   # the globe: the largest lit region
ys, xs = np.where(lab == big); r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
d = max(r1 - r0, c1 - c0); cy, cx = (r0 + r1) / 2, (c0 + c1) / 2; m = int(d * 0.04); h = d / 2 + m
box = tuple(int(round(v)) for v in (cx - h, cy - h, cx + h, cy + h))
assert abs((r1 - r0) - (c1 - c0)) < 0.05 * d, f"not round: {r1 - r0} by {c1 - c0} pixels (the globe may be cut off by the page)"
crop = np.asarray(im.crop(box).convert("RGB")).copy(); yy, xx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]]
crop[(yy - (cy - box[1])) ** 2 + (xx - (cx - box[0])) ** 2 > (d / 2 + 1) ** 2] = bg.astype(np.uint8)   # only the globe: the page around it set to the background
out = Image.fromarray(crop)
out.resize((720, 720), Image.LANCZOS).save(sys.argv[2], quality=90)
print(f"globe {c1 - c0} x {r1 - r0} pixels at ({int(cx)}, {int(cy)}) -> {sys.argv[2]}")
