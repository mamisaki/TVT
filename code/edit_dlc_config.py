#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% import ===================================================================
from pathlib import Path, PurePath
from datetime import timedelta
import re
import sys
import time

# %%
PROJ_ROOT =  Path.home() / 'TVT'
DATA_ROOT = PROJ_ROOT / 'data'

def find_dlc_conf(dd):
    for pp in dd.glob('*'):
        if pp.is_dir():
            find_dlc_conf(pp)
            continue

        if pp.suffix == '.yaml':
            print(f"Fix path in {pp}")
            with open(pp) as fd:
                C = fd.read()

            C = re.sub('{APP_ROOT}/data', '{DATA_ROOT}', C)
            with open(pp, 'w') as fd:
                fd.write(C)

find_dlc_conf(DATA_ROOT)



# %%
