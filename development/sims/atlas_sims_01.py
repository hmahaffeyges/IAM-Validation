"""Atlas v2 simulations of 2026-10-09: (1) lavage neutrophil recovery against the composition bars (DEV-ATLAS-COMMISSION-01; 11-cell
lung panel, NOT atlas_e), (2) whole-blood truth from a laboratory's own sorted cells (DEV-COMPOSITION-TRUTH-03).
Input: 60 atlas v2 posterior-draw blocks, s3://methylphys-data-945451304272-us-west-2-an/atlas_v2/blocks_v2/block_NNNNN_draws.npz,
chosen with numpy default_rng(7); sha256 of each in atlas_blocks_60.json (written on first run, checked afterwards).
Run: python3 atlas_sims_01.py [cache_dir]   (needs read access to the bucket the first time)"""
import os, sys, json, hashlib, numpy as np
from scipy.optimize import nnls
HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "_atlas_cache")
BUCKET = "methylphys-data-945451304272-us-west-2-an"; MAN = os.path.join(HERE, "atlas_blocks_60.json")
def blocks():
    os.makedirs(CACHE, exist_ok=True); ids = sorted(int(b) for b in np.random.default_rng(7).choice(700, 60, replace=False))
    man = json.load(open(MAN)) if os.path.exists(MAN) else {}; out = []
    for b in ids:
        p = os.path.join(CACHE, f"block_{b:05d}_draws.npz")
        if not os.path.exists(p):
            import boto3; boto3.client("s3").download_file(BUCKET, f"atlas_v2/blocks_v2/block_{b:05d}_draws.npz", p)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if str(b) in man: assert man[str(b)] == h, f"block {b} checksum differs"
        man[str(b)] = h; out.append(p)
    json.dump(man, open(MAN, "w"), indent=0); return out
def load():
    ps = blocks(); D = np.concatenate([np.load(p, allow_pickle=True)["draws"].astype(np.float32) for p in ps], axis=2)
    return D, np.load(ps[0], allow_pickle=True)["cells"].tolist()
LUNG = ["lung alveolar macrophages", "lung interstitial macrophages", "monocytes", "cd4 t cells", "cd8 t cells", "b cells", "nk cells",
        "neutrophils", "eosinophils", "lung alveolar epithelium", "lung bronchus epithelium"]
BLOOD6 = ["neutrophils", "monocytes", "b cells", "cd4 t cells", "nk cells", "cd8 t cells"]
def panel(D, cells, names, k):
    ix = [cells.index(c) for c in names]; Mu = D[:, ix, :].mean(0); ok = np.isfinite(Mu).all(0)
    inf = np.argsort(-Mu[:, ok].std(0))[:k]; return Mu[:, ok][:, inf], D[:, ix, :][:, :, ok][:, :, inf]
def bal_truth(r, neu):
    lym = r.uniform(0.04, 0.20); eos = r.uniform(0, 0.02); epi = r.uniform(0, 0.03); mac = 1 - neu - lym - eos - epi
    f = np.zeros(len(LUNG)); f[0], f[1], f[2] = mac * 0.85, mac * 0.10, mac * 0.05
    f[3:7] = lym * r.dirichlet([5, 3, 1.5, 1]); f[7], f[8], f[9], f[10] = neu, eos, epi / 2, epi / 2; return f
def lavage(Mu, Dd, sd_arr, sd_bio, sd_lab=0.0, shift=0.0, n=300, seed=21):
    r = np.random.default_rng(seed); labo = r.normal(0, sd_lab, Mu.shape[1]) if sd_lab else 0.0; e = []
    for _ in range(n):
        neu = r.uniform(0.005, 0.12); f = bal_truth(r, neu)
        person = Dd[r.integers(Dd.shape[0])] + r.normal(0, sd_bio, Mu.shape) if sd_bio else Dd[r.integers(Dd.shape[0])]
        b = f @ person; b = np.clip(b - shift * b + labo + r.normal(0, sd_arr, Mu.shape[1]), 0, 1)
        w, _ = nnls(Mu.T, b); w /= w.sum(); e.append((w[7] - neu, w[7] - r.binomial(400, neu) / 400))
    e = np.array(e); return np.abs(e[:, 0]).mean(), np.abs(e[:, 1]).mean(), int((np.abs(e[:, 1]) <= 0.05).sum()), n
def own_cell(Db, sd_arr, lab_template, seed=21):
    r = np.random.default_rng(seed); err = []
    people = [Db[r.integers(Db.shape[0])] + r.normal(0, 0.02, Db.shape[1:]) for _ in range(30)]
    fr = [r.dirichlet([12, 2, 1.5, 3.5, 1.2, 2.2]) for _ in range(30)]; pur = [r.uniform(0.93, 0.99, 6) for _ in range(30)]
    srt = [np.clip(pur[i][:, None] * (people[i] + r.normal(0, sd_arr, people[i].shape)) + (1 - pur[i][:, None]) * (fr[i] @ people[i]), 0, 1) for i in range(30)]
    T = np.mean(srt, axis=0)
    for i in range(30):
        wb = np.clip(fr[i] @ people[i] + r.normal(0, sd_arr, people[i].shape[1]), 0, 1)
        w, _ = nnls((T if lab_template else srt[i]).T, wb); w /= w.sum(); err.append(w[0] - fr[i][0])
    err = np.array(err); return np.abs(err).mean(), np.abs(err).max(), err.mean()
if __name__ == "__main__":
    D, cells = load(); print("loci", D.shape[2])
    Mu, Dd = panel(D, cells, LUNG, 8000)
    print("LAVAGE (lung panel): conditions | MAE vs truth | MAE vs count | within 0.05")
    for lab, kw in (("array 0.02", dict(sd_arr=0.02, sd_bio=0.0)), ("array 0.04 person 0.02", dict(sd_arr=0.04, sd_bio=0.02)),
                    ("array 0.06 person 0.04", dict(sd_arr=0.06, sd_bio=0.04)), ("+ lab offset 0.06", dict(sd_arr=0.04, sd_bio=0.02, sd_lab=0.06)),
                    ("+ channel shift 0.10", dict(sd_arr=0.04, sd_bio=0.02, shift=0.10))):
        a, b, c, n = lavage(Mu, Dd, **kw); print(f"  {lab:24s} {a:.4f} {b:.4f} {c}/{n}")
    _, Db = panel(D, cells, BLOOD6, 6000)
    print("OWN-CELL TRUTH: array SD | own six arrays MAE max bias | lab template MAE max bias")
    for sd in (0.02, 0.04, 0.06):
        o = own_cell(Db, sd, False); t = own_cell(Db, sd, True)
        print(f"  {sd:.2f}  {o[0]:.4f} {o[1]:.4f} {o[2]:+.4f} | {t[0]:.4f} {t[1]:.4f} {t[2]:+.4f}")
