#!/usr/bin/env python
# -*- coding: utf-8 -*-


# %% import ===================================================================
from pathlib import Path, PurePath
import os
import argparse
import pandas as pd
import numpy as np
import csv


# %% 

# %% __main__ =================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog = 'filter_thermo_export', description = 'Filter TVT export')
    parser.add_argument('filename')
    parser.add_argument('--min_temp', default=32)
    parser.add_argument('--min_dSD', default=2)

    args = parser.parse_args()
    in_f = Path(args.filename)
    min_temp = args.min_temp
    min_dSD = args.min_dSD
    #in_f = Path('/Users/mmisaki/TVT/data/dog_EmotionalContingency/FLIR6567_tracking_points.csv')

    track_df = pd.read_csv(in_f, header=[1, 2], index_col=0)
    with open(in_f, 'r') as fd:
        head = fd.readline()

    PointNames = [col for col in track_df.columns.levels[0]
                  if len(col) and 'Unnamed' not in col]

    cols = pd.MultiIndex.from_product([[''], ['time_ms', 'marker']])
    cols = cols.append(
        pd.MultiIndex.from_product(
            [PointNames, ['x', 'y', 'temp', 'temp_lpf']]))
    track_df.columns = cols

    # --- filter data ---
    for point in PointNames:
        x = track_df[point].x.values
        y = track_df[point].y.values
        temp = track_df[point].temp.values
        temp_lpf = track_df[point].temp_lpf.values

        rm_mask = temp < min_temp
        dmean = np.nanmean(np.abs(temp-temp_lpf))
        dsd = np.nanstd(np.abs(temp-temp_lpf))
        rm_mask |= np.abs(temp-temp_lpf) > (dmean+dsd * min_dSD)

        x[rm_mask] = np.nan
        y[rm_mask] = np.nan
        temp[rm_mask] = np.nan

        track_df.loc[:, (point, 'x')] = x
        track_df.loc[:, (point, 'y')] = y
        track_df.loc[:, (point, 'temp')] = temp

    save_f = in_f.parent / (in_f.stem + '_filtered.csv')
    with open(save_f, 'w') as fd:
        print(head, file=fd)
        fd.write(track_df.to_csv(quoting=csv.QUOTE_NONNUMERIC,
                                 encoding='cp932'))

