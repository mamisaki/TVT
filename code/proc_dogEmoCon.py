#!/usr/bin/env python
# -*- coding: utf-8 -*-

# %% import ===================================================================
from pathlib import Path, PurePath
from datetime import timedelta
import sys
import time
import psutil
import gc

import numpy as np
import pandas as pd
import TVT
from PySide6.QtWidgets import QApplication


# %% Read data list ===========================================================
PROJ_ROOT =  Path.home() / 'TVT'
DATA_ROOT = PROJ_ROOT / 'data' / 'dog_EmotionalContingency'
OUT_DIR = PROJ_ROOT / 'config_dogEmoCon_filtered'
if not OUT_DIR.is_dir():
    OUT_DIR.mkdir()

dlist_f = DATA_ROOT / 'dataList.csv'
dlist = pd.read_csv(dlist_f)


# %% Process ===================================================================
OVERWRITE = False
UPDATE_STATUS = False

app = QApplication(sys.argv)

min_temp = 32
min_dSD = 1.5
lh_thresh = 0.975

N_files = pd.notnull(dlist.FLIR_NR1).sum() + pd.notnull(dlist.FLIR_NR2).sum()
dsk = psutil.disk_usage(str(DATA_ROOT))
dsk_free_gb = dsk.free / 1000000000 
extract_temp_file = True  # dsk_free_gb > N_files * 10

for idx, row in dlist.iterrows():
    FLIR_NRs = [int(row.FLIR_NR1)]
    if pd.notnull(row.FLIR_NR2):
        FLIR_NRs.append(int(row.FLIR_NR2))
    
    for ses, FLIR_NR in enumerate(FLIR_NRs):
        out_fname = DATA_ROOT / f"FLIR{FLIR_NR}_tracking_points_filtered.csv"

        if ses == 0:
            S_ON = row.S1_ON
            S_OFF = row.S1_OFF
        elif ses == 1:
            S_ON = row.S2_ON
            S_OFF = row.S2_OFF
        if pd.isnull(S_ON):
            print(f"FLIR {FLIR_NR}: no onset/offset times")
            if out_fname.is_file():
                out_fname.unlink()
            continue

        # Check DLC tracking
        dlc_trackling_fs = sorted(list(
        DATA_ROOT.glob(f"FLIR{FLIR_NR}_thermoDLC_*_filtered.csv")))
        if len(dlc_trackling_fs) == 0:
            if out_fname.is_file():
                out_fname.unlink()
            continue
        dlc_trackling_f = dlc_trackling_fs[-1]

        track_df = pd.read_csv(dlc_trackling_f, header=[1, 2], index_col=0)
        PointNames = [col for col in track_df.columns.levels[0]
                    if len(col) and 'Unnamed' not in col]
        _lh_thresh = lh_thresh
        for point in PointNames:
            lh = track_df[point].likelihood.values
            if (lh > _lh_thresh).sum() < 1000:
                _lh_thresh = np.percentile(lh, 100 - 100*1000/len(lh))

        if _lh_thresh < 0.9:
            print(f"FLIR {FLIR_NR}: DLC performance was not good (lh = {_lh_thresh})")
            if out_fname.is_file():
                out_fname.unlink()
            continue
        
        if out_fname.is_file() and not OVERWRITE:
            continue

        # Open TVT
        st = time.time()
        win = TVT.MainWindow(batchmode=True,
                             extract_temp_file=extract_temp_file)
        tvmodel = win.model
        radius = int(row.Radius)

        print(f"Process {FLIR_NR} ...")

        # Read temprature data
        status_fname = OUT_DIR / f'FLIR{FLIR_NR}_filtered_state.pkl'
        if not status_fname.is_file() or UPDATE_STATUS:
            # Load thermalData
            csq_f = DATA_ROOT / f"FLIR{FLIR_NR}.csq"
            tvmodel.openThermalFile(fileName=csq_f)

            # Load videoData
            video_f = DATA_ROOT / f"FLIR{FLIR_NR}_thermo.mp4"
            tvmodel.openVideoFile(fileName=video_f)
            tvmodel.on_sync = True

            # Load DLC config
            DLC_dirs = list(DATA_ROOT.glob(
                f"FLIR{FLIR_NR}_thermo-TVT-*"))
            DLC_dir = sorted(DLC_dirs)[-1]
            dlc_conf_f = DLC_dir / 'config_rel.yaml'
            tvmodel.dlci.config_path = dlc_conf_f

            # Set time_marker
            fps = tvmodel.thermalData.frame_rate
            NFrames = tvmodel.thermalData.duration_frame

            # S_ON
            m, s = [float(v) for v in S_ON.split(':')]
            t_sec = m*60 + s
            S1_ON_fr = int(np.round(t_sec * fps))
            tvmodel.time_marker[S1_ON_fr] = 'S1_ON'
            S1_START_fr = int(S1_ON_fr - 120 * fps)
            tvmodel.time_marker[S1_START_fr] = 'S1_START'

            # S_OFF
            m, s = [float(v) for v in S_OFF.split(':')]
            t_sec = m*60 + s
            S1_OFF_fr = int(np.round(t_sec * fps))
            tvmodel.time_marker[S1_OFF_fr] = 'S1_OFF'
            S1_END_fr = int(S1_OFF_fr + 120 * fps)
            S1_END_fr = min(S1_END_fr, NFrames-1)
            tvmodel.time_marker[S1_END_fr] = 'S1_END'

            if pd.isnull(row.FLIR_NR2):
                S_ON = row.S2_ON
                S_OFF = row.S2_OFF
                # S_ON
                m, s = [float(v) for v in S_ON.split(':')]
                t_sec = m*60 + s
                S2_ON_fr = int(np.round(t_sec * fps))
                tvmodel.time_marker[S2_ON_fr] = 'S2_ON'
                S2_START_fr = int(S2_ON_fr - 120 * fps)
                tvmodel.time_marker[S2_START_fr] = 'S2_START'

                # S_OFF
                m, s = [float(v) for v in S_OFF.split(':')]
                t_sec = m*60 + s
                S2_OFF_fr = int(np.round(t_sec * fps))
                tvmodel.time_marker[S2_OFF_fr] = 'S2_OFF'
                S2_END_fr = int(S2_OFF_fr + 120 * fps)
                S2_END_fr = min(S2_END_fr, NFrames-1)
                tvmodel.time_marker[S2_END_fr] = 'S2_END'

            # Load tracking_point
            tvmodel.load_tracking(dlc_trackling_f, lh_thresh=_lh_thresh)

            # Set tracking_point properties
            for point in tvmodel.tracking_point.keys():
                tvmodel.tracking_point[point].radius = radius
                tvmodel.tracking_point[point].aggfunc = 'max'
                tvmodel.tracking_mark[point]['aggfunc'] = tvmodel.tracking_point[point].aggfunc
                tvmodel.tracking_mark[point]['rad'] = tvmodel.tracking_point[point].radius

                # Erase < S1_START
                tvmodel.tracking_point[point].x[:S1_START_fr] = np.nan
                tvmodel.tracking_point[point].y[:S1_START_fr] = np.nan
                tvmodel.tracking_point[point].value_ts[:S1_START_fr] = np.nan
                if pd.notnull(row.FLIR_NR2):
                    # Erase > S1_END
                    tvmodel.tracking_point[point].x[S1_END_fr:] = np.nan
                    tvmodel.tracking_point[point].y[S1_END_fr:] = np.nan
                    tvmodel.tracking_point[point].value_ts[S1_END_fr:] = np.nan
                else:
                    if S1_END_fr < S2_START_fr:
                        tvmodel.tracking_point[point].x[S1_END_fr:S2_START_fr] = np.nan
                        tvmodel.tracking_point[point].y[S1_END_fr:S2_START_fr] = np.nan
                        tvmodel.tracking_point[point].value_ts[S1_END_fr:S2_START_fr] = np.nan
                    
                    tvmodel.tracking_point[point].x[S2_END_fr:] = np.nan
                    tvmodel.tracking_point[point].y[S2_END_fr:] = np.nan
                    tvmodel.tracking_point[point].value_ts[S2_END_fr:] = np.nan

                tvmodel.tracking_point[point].update_all_values()
            
            # Save temperature data file
            if extract_temp_file:
                temp_f = tvmodel.thermalData.thermal_data_reader.thermaldata_npy_f
                if not temp_f.is_file():
                    tvmodel.thermalData.save_temperature_frames()

            tvmodel.thermalData.show_frame(S1_ON_fr)

            # Save status file
            tvmodel.save_status(fname=status_fname)
        else:
            # Load saved state
            tvmodel.load_status(fname=status_fname)

        # Update point parametera
        for point in tvmodel.tracking_point.keys():
            tvmodel.tracking_point[point].radius = radius
            tvmodel.tracking_point[point].aggfunc = 'max'
            tvmodel.tracking_mark[point]['aggfunc'] = tvmodel.tracking_point[point].aggfunc
            tvmodel.tracking_mark[point]['rad'] = tvmodel.tracking_point[point].radius
            tvmodel.tracking_point[point].update_all_values()

        # filter temperature series
        tvmodel.lpf = 0.05
        for point in tvmodel.tracking_point.keys():
            temp = tvmodel.tracking_point[point].value_ts

            si = 1.0 / tvmodel.tracking_point[point].frequency
            xi0 = np.argwhere(np.logical_not(np.isnan(temp))).ravel()
            y0 = temp[xi0]
            temp_lpf = np.ones(len(temp)) * np.nan
            temp_lpf[np.min(xi0):np.max(xi0)+1] = \
                tvmodel.InterpLPF(y0, xi0, si, tvmodel.lpf)

            rm_mask = temp < min_temp
            dmean = np.nanmean(np.abs(temp-temp_lpf))
            dsd = np.nanstd(np.abs(temp-temp_lpf))
            rm_mask |= np.abs(temp-temp_lpf) > (dmean+dsd * min_dSD)

            tvmodel.tracking_point[point].x[rm_mask] = np.nan
            tvmodel.tracking_point[point].y[rm_mask] = np.nan
            tvmodel.tracking_point[point].value_ts[rm_mask] = np.nan

        tvmodel.save_status(fname=status_fname)
        
        # export
        tvmodel.export_roi_data(fname=out_fname, realod_data=False)

        win.close()
        del win
        gc.collect()
        
        et = timedelta(seconds=time.time() - st)
        print(f"    done (took {et})")


# %%
