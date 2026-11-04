"""How each library's floor reaches each kind of unmarked site (development; diagnostic, nothing scored). Healthy pass 6 found that only about
half the floor rate measured at the floor sites shows up at the islands inside the atlas's own laboratory, while in plasma nearly all of it
does (slope 0.939). This measures where it goes. From each healthy input's whole-sample per-site table (NAME_whole_sites.csv.gz, written by
the one command with --dev-whole-sites in healthy pass 5) and its bundle's library floor (development.library_floor, unmarked side): at
sites the sample's own mix expects unmarked (expected level at most 0.01), the share of reads called methylated (read-weighted level),
by class: islands; domains; other unmarked sites. Per library, that rate over its floor is the share of the floor that class carries (1:
the class takes false calls at the floor sites' rate). The excess over the expected level is recorded too; it is near 0 inside the atlas's
laboratory because the expected level comes from donors of the same laboratory.
Usage: python3 floor_by_site_class.py LAND_DIR OUT.csv OUT.txt"""
import os, sys, glob, json, numpy as np, pandas as pd
from scipy.stats import spearmanr
LO = 0.01; rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "**", "*_whole_sites.csv.gz"), recursive=True)):
    g = os.path.basename(f).split("_")[0]; b = glob.glob(os.path.join(os.path.dirname(f), g + "_bundle.json"))
    if not b: continue
    fl = (((json.load(open(b[0])).get("development") or {}).get("library_floor") or {}).get("unmarked") or {}).get("rate")
    if fl is None: continue
    T = pd.read_csv(f, compression="gzip", usecols=["reads", "sample_level", "expected_level", "island", "domain"])
    T = T[T.expected_level <= LO]; T["cls"] = np.where(T.island, "islands", np.where(T.domain, "domains", "other unmarked"))
    for c, G in T.groupby("cls"):
        w = G.reads.to_numpy(float); rate = float(np.sum(w * G.sample_level) / w.sum()); ex = float(np.sum(w * (G.sample_level - G.expected_level)) / w.sum())
        rows.append(dict(gsm=g, cls=c, sites=len(G), floor=fl, rate=rate, excess=ex, share_of_floor=rate / fl if fl > 0 else np.nan))
D = pd.DataFrame(rows); D.to_csv(sys.argv[2], index=False)
L = [f"{D.gsm.nunique()} healthy inputs; sites the sample's own mix expects at most {LO} methylated"]
for c, G in D.groupby("cls"):
    L.append(f"{c}: median sites {int(G.sites.median())}; median rate {G.rate.median():.5f}; median excess over expected {G.excess.median():+.5f}; median share of the library's floor {G.share_of_floor.median():.3f} "
             f"(25 to 75 % {G.share_of_floor.quantile(.25):.3f} to {G.share_of_floor.quantile(.75):.3f}); rate against floor rho {spearmanr(G.floor, G.rate).correlation:+.3f}")
open(sys.argv[3], "w").write("\n".join(L) + "\n"); print("\n".join(L))
