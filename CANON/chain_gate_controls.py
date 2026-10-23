#!/usr/bin/env python3
"""Controls for the chain gate (lesson C8: a check is switched on only if every control gives its stated answer). Builds a throwaway repo
tree and feeds the gate the mistakes it must refuse, plus one correct test it must accept:
 C1 the DEV-SECONDLAB-01 pattern: a synthetic script with '# NO CHAIN:' and no chain import                -> REFUSE
 C2 a script outside the chain typing the line's numbers                                                     -> REFUSE
 C3 a result file with no chain stamp                                                                        -> REFUSE
 C4 a stamped result edited by hand afterwards                                                               -> REFUSE
 C5 a stamped result from a chain version that was never committed                                          -> REFUSE
 C6 a record citing scratch/                                                                                 -> REFUSE
 C7 a correct test: chain-importing synthetic, stamped result, committed chain version                      -> ACCEPT
 C8 a commissioning note adding a side script as evidence                                                   -> REFUSE
 C9 a commissioning note adding chain code                                                                  -> ACCEPT
 C10 a test script calling chain stages directly, not the one command (the METAA-ATLAS-SEAL-01 pattern)    -> REFUSE
 C11 a test script that runs the one command                                                                -> ACCEPT
 C12 a website picture script calling the globe code directly                                               -> REFUSE
 C13 a test script downloading a .pat file itself (the lesson C34 pattern), even though it runs the one command -> REFUSE
 C14 a test script letting the chain fetch the .pat (--geo-pat)                                            -> ACCEPT
 C15 a script that makes its .pat from raw reads (wgbstools bam2pat) and counts it with the one command       -> ACCEPT
 C16 the same, counting it with chain pieces instead of the one command                                       -> REFUSE"""
import os, sys, json, tempfile, hashlib, importlib.util
H = os.path.dirname(os.path.abspath(__file__)); spec = importlib.util.spec_from_file_location("cg", os.path.join(H, "chain_gate.py")); CG = importlib.util.module_from_spec(spec); spec.loader.exec_module(CG)
def tree():
    r = tempfile.mkdtemp(); os.makedirs(os.path.join(r, CG.CHAIN)); os.makedirs(os.path.join(r, CG.DOORS, "data", "DEV_X_01"))
    open(os.path.join(r, CG.CHAIN, "stage_q2_selfref.py"), "w").write("X = 1\n"); return r
def w(r, p, s): open(os.path.join(r, p), "w").write(s)
def stamped(o, sha):
    o = dict(o); o["chain_stamp"] = {"chain_sha": sha, "payload_sha256": hashlib.sha256(CG._canon(o)).hexdigest()}; return o
NOTE = CG.DOORS + "DEV_X_01.md"; SYN = "**Synthetic test:** DEV_X_01/t.py, DEV_X_01/out.json"; OUT = CG.DOORS + "data/DEV_X_01/out.json"; T = CG.DOORS + "data/DEV_X_01/t.py"
GOOD = "import subprocess, sys\nsubprocess.run([sys.executable, 'chain/MethylPhys_Interface/run_sample.py', '--pat', 'S.pat.gz'])\n"
def run(r, changed, added_note=(SYN,), known=None):
    return CG.check(changed, {NOTE: list(added_note)}, root=r, known=known if known is not None else {CG.chain_sha(r)})
res = {}
r = tree(); w(r, NOTE, SYN + "\n"); w(r, T, "# NO CHAIN: gemBS\nimport pysam\n"); w(r, OUT, json.dumps(stamped({"v": 1}, CG.chain_sha(r))))
res["C1 NO CHAIN script"] = run(r, [NOTE, T, OUT])
r = tree(); w(r, "Biological_Physics/MethylPhys/doors/data/DEV_X_01/side.py", "FIRM, LOOSE = 0.0298, 0.0555\n")
res["C2 line numbers typed"] = run(r, ["Biological_Physics/MethylPhys/doors/data/DEV_X_01/side.py"], ())
r = tree(); w(r, NOTE, SYN + "\n"); w(r, T, GOOD); w(r, OUT, json.dumps({"v": 1}))
res["C3 unstamped result"] = run(r, [NOTE, T, OUT])
r = tree(); w(r, NOTE, SYN + "\n"); w(r, T, GOOD); o = stamped({"v": 1}, CG.chain_sha(r)); o["v"] = 2; w(r, OUT, json.dumps(o))
res["C4 hand-edited result"] = run(r, [NOTE, T, OUT])
r = tree(); w(r, NOTE, SYN + "\n"); w(r, T, GOOD); w(r, OUT, json.dumps(stamped({"v": 1}, "f" * 64)))
res["C5 uncommitted chain"] = run(r, [NOTE, T, OUT])
r = tree(); w(r, NOTE, SYN + "\nsee scratch/try.py\n"); w(r, T, GOOD); w(r, OUT, json.dumps(stamped({"v": 1}, CG.chain_sha(r))))
res["C6 scratch cited"] = run(r, [NOTE, T, OUT])
r = tree(); w(r, NOTE, SYN + "\n"); w(r, T, GOOD); w(r, OUT, json.dumps(stamped({"v": 1}, CG.chain_sha(r))))
res["C7 correct test"] = run(r, [NOTE, T, OUT])
r = tree(); CN = CG.DOORS + "COMMISSIONING_NOTE_X.md"; w(r, CN, "Evidence: data/X/side_reader.py\n")
res["C8 commissioning cites side code"] = CG.check([CN], {CN: ["Evidence: data/X/side_reader.py"]}, root=r, known={CG.chain_sha(r)})
r = tree(); w(r, CN, "Evidence: chain/conductor_v3.py\n")
res["C9 commissioning cites chain"] = CG.check([CN], {CN: ["Evidence: chain/conductor_v3.py"]}, root=r, known={CG.chain_sha(r)})
r = tree(); S1 = CG.DOORS + "data/DEV_X_01/read.py"; w(r, S1, "import sys\nimport stage_metaa_atlas as SA\nfrom stage_1_idat_calibration import calibrate_idat_to_beta\n")
res["C10 chain pieces, no one command"] = CG.check([S1], {}, root=r, known={CG.chain_sha(r)})
r = tree(); w(r, S1, "import subprocess\nimport conductor_v3 as C3   # planting rule only\nsubprocess.run(['python3', 'chain/MethylPhys_Interface/run_sample.py', '--betas', 'b.csv'])\n")
res["C11 runs the one command"] = CG.check([S1], {}, root=r, known={CG.chain_sha(r)})
r = tree(); os.makedirs(os.path.join(r, "website/art")); S2 = "website/art/make_x.py"; w(r, S2, "import report_globe_v4 as G\n")
res["C12 website picture from globe code"] = CG.check([S2], {}, root=r, known={CG.chain_sha(r)})
r = tree(); w(r, S1, "import subprocess, urllib.request, re\nn = re.findall(r'GSM1_[^<]*\\.pat\\.gz', urllib.request.urlopen('https://x/').read().decode())[0]\nsubprocess.run(['python3', 'chain/MethylPhys_Interface/run_sample.py', '--pat', n])\n")
res["C13 own .pat download"] = CG.check([S1], {}, root=r, known={CG.chain_sha(r)})
r = tree(); w(r, S1, "import subprocess\nsubprocess.run(['python3', 'chain/MethylPhys_Interface/run_sample.py', '--geo-pat', 'GSM1'])   # the .pat.gz is fetched by the chain\n")
res["C14 chain fetches the .pat"] = CG.check([S1], {}, root=r, known={CG.chain_sha(r)})
r = tree(); w(r, S1, "import subprocess\nsubprocess.run('curl -o g.fa.gz https://x/ && wgbstools bam2pat s.bam -o pat', shell=True)\nsubprocess.run(['python3', 'chain/MethylPhys_Interface/run_sample.py', '--pat', 'pat/s.pat.gz', '--dev-count-only'])\n")
res["C15 .pat made from reads, counted by the one command"] = CG.check([S1], {}, root=r, known={CG.chain_sha(r)})
r = tree(); w(r, S1, "import subprocess\nsubprocess.run('curl -o g.fa.gz https://x/ && wgbstools bam2pat s.bam -o pat', shell=True)\nimport stage_q2_selfref as Q2\nQ2.count_pat(open('pat/s.pat.gz'), {})\n")
res["C16 .pat made from reads, counted by chain pieces"] = CG.check([S1], {}, root=r, known={CG.chain_sha(r)})
want = {k: (k.startswith(("C7", "C9", "C11", "C14", "C15"))) for k in res}
bad = [k for k, F in res.items() if (len(F) == 0) != want[k]]
for k, F in res.items(): print(f"  {k}: {'ACCEPTED' if not F else 'REFUSED'} (want {'ACCEPT' if want[k] else 'REFUSE'})" + ("" if not F else f" - {F[0][-110:]}"))
print("chain_gate_controls: " + (f"ok ({len(res)} of {len(res)})" if not bad else "FAILED: " + ", ".join(bad))); sys.exit(1 if bad else 0)
