#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% import ===================================================================
from pathlib import Path
import os
import subprocess
import shlex
import sys

if '__file__' not in locals():
    __file__ = 'this.py'

# %% Set Paths ================================================================
PROJ_ROOT = Path.home() / 'TVT'
DATA_ROOT = PROJ_ROOT / 'data' / 'dog_EmotionalContingency'

# %% Collect run sciprt =======================================================
script_dirs = DATA_ROOT.glob('*TVT*')

run_script = []
for scdir in script_dirs:
    sc_f = scdir / 'DLC_training.sh'
    if not sc_f.is_file():
        print(f"{scdir} does not have DLC_training.sh")
        continue
    
    out_csv = list(DATA_ROOT.glob(
        scdir.name.split('-')[0] + 'DLC_*_filtered.csv'))
    if len(out_csv) > 0:
        continue

    out_f = scdir / 'nohup_DLC_training.out'
    if out_f.is_file():
        print(f"{scdir} is runninng!?")
        continue

    run_script.append(sc_f)


# %% Run DLC_train.sh =========================================================

for sc_f in run_script:
    scdir = sc_f.parent
    out_csv = list(DATA_ROOT.glob(
        scdir.name.split('-')[0] + 'DLC_*_filtered.csv'))
    if len(out_csv) > 0:
        continue

    out_f = scdir / 'nohup_DLC_training.out'
    if out_f.is_file():
        print(f"{scdir} is runninng!?")
        continue
    
    print(f"Run {scdir}")
    sys.stdout.flush()
    with open(out_f, 'w') as fd:
        cmd = shlex.split("sh DLC_training.sh")
        pr = subprocess.Popen(shlex.split("sh DLC_training.sh"), cwd=scdir,
            stdout=fd, stderr=fd)
        pr.wait()
    
    pr.terminate()

