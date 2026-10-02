#!/bin/bash
# Build CAMB 1.5.8 twice from source: with the Level 2 equations.f90 (iam_dual_sector = .true.) and with the switch off; same everything else.
set -e
E=/home/ubuntu/data/chains/env/bin; for i in $(seq 1 90); do [ -x $E/gfortran ] && [ -x $E/pip ] && break; sleep 60; done
export PATH=$E:$PATH; W=/home/ubuntu/data/l2growth; mkdir -p $W && cd $W
for v in on off; do
  if [ ! -d camb_$v ]; then
    $E/git clone -q --branch 1.5.8 --depth 1 https://github.com/cmbant/CAMB.git camb_$v && (cd camb_$v && $E/git submodule update --init --recursive -q)
    cp $OLDPWD/equations_iam_level2.f90 camb_$v/fortran/equations.f90
    [ $v = off ] && sed -i 's/logical  :: iam_dual_sector = .true./logical  :: iam_dual_sector = .false./' camb_$v/fortran/equations.f90
    grep -c "iam_dual_sector = .$( [ $v = on ] && echo true || echo false )." camb_$v/fortran/equations.f90
    $E/python -m venv --system-site-packages venv_$v && venv_$v/bin/pip install -q ./camb_$v 2>&1 | tail -2
  fi
done
cd $OLDPWD
for v in on off; do /home/ubuntu/data/l2growth/venv_$v/bin/python growth_out.py $v; done
/home/ubuntu/data/chains/env/bin/python - <<'PY'
import json
a=json.load(open("growth_on.json")); b=json.load(open("growth_off.json"))
print("camb versions", a["version"], b["version"])
print(" z     D_on/D_off   fs8_on/fs8_off   sigma8(0) on/off")
for i,z in enumerate(a["z"]): print("%5.2f  %.5f      %.5f"%(z, a["s8"][i]/b["s8"][i], a["fs8"][i]/b["fs8"][i]))
print("sigma8(0): %.5f / %.5f = %.5f"%(a["s8"][0], b["s8"][0], a["s8"][0]/b["s8"][0]))
PY
echo DONE
