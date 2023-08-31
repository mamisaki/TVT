#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% import ===================================================================
from pathlib import Path, PurePath
from datetime import timedelta
import sys
import time

import numpy as np
import pandas as pd
import DLC_GUI
from PySide6.QtWidgets import QApplication


# %% Read data list ===================================================================
PROJ_ROOT =  Path.home() / 'TVT'
DATA_ROOT = PROJ_ROOT / 'data' / 'dog_ObjectPermanency'
OUT_DIR = PROJ_ROOT / 'config_dogObjPerm_filtered'
if not OUT_DIR.is_dir():
    OUT_DIR.mkdir()

dlist_f = DATA_ROOT / 'dataList.csv'
dlist = pd.read_csv(dlist_f)


# %% Process ===================================================================
OVERWRIE = True

app = QApplication(sys.argv)

lh_thresh=0.975
ons_cols = [cc for cc in dlist.columns if cc[1] == '_']
for idx, row in dlist.iterrows():
    if np.all(pd.isnull(row[ons_cols])):
        continue

    fname = row.FileName
    status_fname = OUT_DIR / f'{fname}_filtered_state.pkl'
    if status_fname.is_file() and not OVERWRIE:
        continue
    
    print(f"Process {fname} ...")
    st =time.time()

    win = DLC_GUI.ViewWindow(batchmode=True)
    model = win.model

    dlc_trackling_fs = sorted(list(
        DATA_ROOT.glob(f"{fname}DLC_*_filtered.csv")))
    if len(dlc_trackling_fs) == 0:
        continue
    dlc_trackling_f = dlc_trackling_fs[-1]

    # Load videoData
    video_f = DATA_ROOT / f"{fname}.mp4"
    model.openVideoFile(fileName=video_f)

    # Load DLC config
    DLC_dirs = list(DATA_ROOT.glob(f"{fname}-DLCGUI-*"))
    DLC_dir = sorted(DLC_dirs)[-1]
    dlc_conf_f = DLC_dir / 'config_rel.yaml'
    model.dlci.config_path = dlc_conf_f

    # Set time_marker
    fps = model.videoData.frame_rate
    NFrames = model.videoData.duration_frame

    video_start_fr = None
    for trial in range(1, 7):
        S_disappear = row[f"{trial}_on"]
        S_appear = row[f"{trial}_off"]
        if pd.isnull(S_disappear) or pd.isnull(S_appear):
            continue

        h, m, s, fr = [float(v) for v in S_disappear.split(':')]
        S_disappear = m*60+s+fr*1/fps
        h, m, s, fr = [float(v) for v in S_appear.split(':')]
        S_appear = m*60+s+fr*1/fps
        Start = S_disappear-2.0
        End = S_appear+2.0

        Start_fr = max(0, int(np.round(Start * fps)))
        S_disappear_fr = int(np.round(S_disappear * fps))
        S_appear_fr = int(np.round(S_appear * fps))
        End_fr = min(NFrames, int(np.round(End * fps)))
        if trial == 1:
            video_start_fr = Start_fr

        model.time_marker[Start_fr] = f'T{trial}_start'
        model.time_marker[S_disappear_fr] = f'T{trial}_disappear'
        model.time_marker[S_appear_fr] = f'T{trial}_appear'
        model.time_marker[End_fr] = f'T{trial}_end'

    # Load tracking_point
    # Adjust lh_thresh
    track_df = pd.read_csv(dlc_trackling_f, header=[1, 2], index_col=0)
    PointNames = [col for col in track_df.columns.levels[0]
                    if len(col) and 'Unnamed' not in col]

    _lh_thresh = lh_thresh
    for point in PointNames:
        lh = track_df[point].likelihood.values
        if (lh > _lh_thresh).sum() < 1000:
            _lh_thresh = np.percentile(lh, 100 - 100*1000/len(lh))

    model.load_tracking(dlc_trackling_f, lh_thresh=_lh_thresh)
    model.videoData.show_frame(video_start_fr)

    # export
    fname_export = fname + '_tracking_points_filtered.csv'
    fname_export = model.videoData.filename.parent / fname_export
    model.export_roi_data(fname=fname_export, realod_data=False)

    # Save status file
    model.save_status(fname=status_fname)

    win.close()
    del win
    et = timedelta(seconds=time.time() - st)
    print(f"    done (took {et})")

# %%
