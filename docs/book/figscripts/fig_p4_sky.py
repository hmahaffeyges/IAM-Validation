"""Part VI, chapter 'Tools from the sky': the five sky figures, drawn by the chain's own sky scripts.

    "fig_sky_cmb_vs_neutrophil", "fig_sky_genome_on_sphere", "fig_sky_what_it_sees", "fig_sky_cscore_unrolled"
        <- Biological_Physics/MethylPhys/reference_floors_v1/sky/make_sky_figs.py
    "fig_sky_cd"
        <- Biological_Physics/MethylPhys/reference_floors_v1/sky/make_cd_fig.py

This wrapper runs those scripts unchanged, with their inputs named, and writes the figures into docs/book/figures/part4/.
Inputs: sky_neut6_maps.npz and sky_neut6_stats.csv (the sky run of sky_neut6.py), cd_neut.csv (cd_neut.py), all in that folder;
and the CMB map cmb_planck2018_nside256.npz, which is not in the repository. If it is absent it is rebuilt here as the script's docstring
describes it: CAMB 2.0.4 lensed TT spectrum at the Planck 2018 TT,TE,EE+lowE+lensing best fit (Planck 2018 VI, Table 1:
ombh2 0.022383, omch2 0.12011, H0 67.32, tau 0.0543, ln(1e10 As) 3.0448, ns 0.96605), healpy synfast at NSIDE 256, seed 42, in uK.
A rebuilt map is one realisation of that sky; it need not match the original pixel for pixel.
Needs healpy and camb. Run: python docs/book/figscripts/fig_p4_sky.py
"""
import os, sys, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
BOOK, REPO = HERE.parent, HERE.parent.parent.parent
SKYDIR = REPO / "Biological_Physics" / "MethylPhys" / "reference_floors_v1" / "sky"
OUT = BOOK / "figures" / "part6"
CMBMAP = HERE / "_data" / "cmb_planck2018_nside256.npz"
NAMES = ("fig_sky_cmb_vs_neutrophil", "fig_sky_genome_on_sphere", "fig_sky_what_it_sees", "fig_sky_cscore_unrolled", "fig_sky_cd")


def build_cmb():
    import camb, healpy as hp
    p = camb.set_params(ombh2=0.022383, omch2=0.12011, H0=67.32, tau=0.0543, As=np.exp(3.0448) * 1e-10, ns=0.96605, lmax=800)
    cl = camb.get_results(p).get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=True)["total"][:, 0]
    np.random.seed(42)
    m = hp.synfast(cl, 256, lmax=768)
    CMBMAP.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CMBMAP, map=m.astype(np.float32), nside=256, note="rebuilt by docs/book/figscripts/fig_p4_sky.py")
    print(f"rebuilt {CMBMAP.relative_to(REPO)} (rms {m.std():.1f} uK)")


def run(script, extra, replace=()):
    src = (SKYDIR / script).read_text()
    for a, b in replace:
        assert a in src, (script, a)
        src = src.replace(a, b)
    g = {"__name__": "__main__", "__file__": str(SKYDIR / script)}
    g.update(extra)
    exec(compile(src, str(SKYDIR / script), "exec"), g)


if __name__ == "__main__":
    if not CMBMAP.exists():
        build_cmb()
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(OUT)                                    # the chain's scripts write into the current directory
    run("make_sky_figs.py", dict(SKY=str(SKYDIR / "sky_neut6_maps.npz"), STATS=str(SKYDIR / "sky_neut6_stats.csv"), CMB=str(CMBMAP)))
    run("make_cd_fig.py", {}, replace=[('pd.read_csv("cd_neut.csv")', f'pd.read_csv(r"{SKYDIR / "cd_neut.csv"}")')])
    for n in NAMES:
        assert (OUT / f"{n}.pdf").exists(), n
    print("wrote", ", ".join(f"figures/part4/{n}.pdf" for n in NAMES))
