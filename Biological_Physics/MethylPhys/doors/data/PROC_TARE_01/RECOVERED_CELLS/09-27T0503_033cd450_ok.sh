cd iamrepo/Biological_Physics/MethylPhys/kit && python3 -c "
L=open('PROC_TARE_01.py').read().split('\n')
if L[1].startswith('# INSTRUMENT-TEST: measures a CANDIDATE per-array tare (SNP probes)'): del L[1]
open('PROC_TARE_01.py','w').write('\n'.join(L)); print(L[:2])"