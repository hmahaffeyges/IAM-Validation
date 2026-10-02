import sys, json, camb, numpy as np
v=sys.argv[1]; zs=[0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0]
# Run A posterior means (Level 2): ombh2 0.02217, omch2 0.11994, H0 67.161, tau 0.0537, lnAs 3.0407, ns 0.9630
p=camb.set_params(H0=67.161, ombh2=0.02217, omch2=0.11994, tau=0.0537, As=np.exp(3.0407)*1e-10, ns=0.9630, mnu=0.06, WantTransfer=True, redshifts=zs, kmax=2.0)
p.NonLinear=camb.model.NonLinear_none
r=camb.get_results(p); s8=r.get_sigma8()[::-1]; fs8=r.get_fsigma8()[::-1]
json.dump(dict(version=camb.__version__, z=zs, s8=s8.tolist(), fs8=fs8.tolist()), open(f"growth_{v}.json","w")); print(v, camb.__version__, s8[0])
