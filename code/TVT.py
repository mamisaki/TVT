#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thermal Video Tracking App

Model-View architecture
Model classes :
    ThermalVideoModel :
        Model class.
    TrackingPoint :
        Tracking point data class.
    ThermalDataMovie :
        Thermal data movie model class.

View classe:
    MainWindow:
        Mian GUI window.

    TVTDisplayImage:
        Display image class.
"""


# %% import ===================================================================
from pathlib import Path, PurePath
import sys
import os
from datetime import datetime, timedelta
import re
from functools import partial
import platform
import pickle
import shutil
import time
import gc
import traceback
import json

import numpy as np
import cv2

# import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import csv
import pandas as pd
from scipy import interpolate
from scipy.fft import ifft, fft, fftfreq

from PySide6.QtCore import Qt, QObject, QTimer
from PySide6.QtCore import Signal as pyqtSignal

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFrame,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QDialog,
    QLabel,
    QSizePolicy,
    QPushButton,
    QStyle,
    QSlider,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QSplitter,
    QGroupBox,
    QDoubleSpinBox,
    QLineEdit,
    QDialogButtonBox,
    QProgressDialog,
    QInputDialog,
)

from PySide6.QtGui import QImage, QPixmap, QAction

# https://matplotlib.org/3.1.0/gallery/user_interfaces/embedding_in_qt_sgskip.html
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT

import imageio

from dlc_interface import DLCinter
from data_movie import DataMovie, DisplayImage
from csq_reader import CSQ_READER


# %% Default values ===========================================================
tracking_point_radius_default = 6
tracking_point_pen_color_default = "darkRed"
Aggfuncs = ["mean", "median", "min", "max"]
tracking_point_aggfunc_default = "mean"

qt_global_colors = [
    "red",
    "green",
    "blue",
    "cyan",
    "magenta",
    "yellow",
    "darkRed",
    "darkGreen",
    "darkBlue",
    "darkCyan",
    "darkMagenta",
    "darkYellow",
    "black",
    "white",
    "darkGray",
    "gray",
    "lightGray",
]
pen_color_rgb = {
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "yellow": "#ffff00",
    "darkRed": "#800000",
    "darkGreen": "#008000",
    "darkBlue": "#000080",
    "darkCyan": "#008080",
    "darkMagenta": "#800080",
    "darkYellow": "#808000",
    "black": "#000000",
    "white": "#ffffff",
    "darkGray": "#808080",
    "gray": "#a0a0a4",
    "lightGray": "#c0c0c0",
}

themro_cmap = cv2.COLORMAP_JET

if "__file__" not in locals():
    __file__ = "./this.py"

APP_ROOT = Path(__file__).absolute().parent.parent

OS = platform.system()


# %% TrackingPoint class ====================================================
class TrackingPoint:
    """Tracking point data class.
    Each point is an instance of TrackingPoint class.
    The class handles frame-wise point seqence, aggrgate values in the ROI, and
    sequence of values at a point.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, dataMovie, name="", x=np.nan, y=np.nan):
        self.dataMovie = dataMovie
        self.name = name
        self.frequency = self.dataMovie.frame_rate
        data_length = self.dataMovie.duration_frame
        self.x = np.ones(data_length) * x
        self.y = np.ones(data_length) * y
        self.radius = np.ones(data_length, dtype=int) * tracking_point_radius_default
        self.aggfunc = tracking_point_aggfunc_default
        self.value_ts = np.ones(data_length) * np.nan

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_position(self, x, y, frame_indices=None, update_frames=[]):
        if frame_indices is None:
            frame_indices = [self.dataMovie.frame_position]
            if type(update_frames) is bool and update_frames:
                update_frames = frame_indices

        self.x[frame_indices] = x
        self.y[frame_indices] = y

        if hasattr(self.dataMovie, "get_rois_dataseries") and len(update_frames) > 0:
            self.value_ts[update_frames] = self.get_value(
                update_frames, force_update=True
            )

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_current_position(self):
        current_frame = self.dataMovie.frame_position
        x = self.x[current_frame]
        y = self.y[current_frame]
        return x, y

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_value(self, frame_indices, force_update=False):
        # Update values
        xyt = np.ndarray([0, 3])
        radii = []
        for t in frame_indices:
            if not force_update and not np.isnan(self.value_ts[t]):
                continue
            x = self.x[t]
            y = self.y[t]
            xyt = np.concatenate([xyt, [[x, y, t]]], axis=0)
            radii.append(self.radius[t])

        if len(xyt):
            val = self.dataMovie.get_rois_dataseries(
                [xyt], [radii], aggfunc=[self.aggfunc]
            )[0]
            self.value_ts[frame_indices] = val

        val = self.value_ts[frame_indices]

        return val

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_all_values(self):
        xyt = np.array(
            [
                xyt
                for xyt in zip(self.x, self.y, np.arange(len(self.x)))
                if ~np.any(np.isnan(xyt))
            ]
        )
        vals = self.dataMovie.get_rois_dataseries(
            [xyt], [self.radius], aggfunc=[self.aggfunc]
        )
        if vals is None:
            return

        self.value_ts[:] = np.nan
        self.value_ts[xyt[:, 2].astype(int)] = vals[0]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def resampled_pos(self, common_time_ms):
        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_save_params(self):
        settings = {}
        settings["dataMovie.filename"] = self.dataMovie.filename
        save_params = ["aggfunc", "radius", "value_ts", "x", "y"]
        for param in save_params:
            settings[param] = getattr(self, param)

        return settings


# %% VideoDataMovie class =====================================================
class VideoDataMovie(DataMovie):
    """
    Video data movie class
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, parent, dispImg, UI_objs):
        super().__init__(parent, dispImg, UI_objs)
        self.videoCap = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open(self, filename):
        if self.loaded:
            self.unload()

        self.videoCap = cv2.VideoCapture(str(filename))
        self.frame_rate = self.videoCap.get(cv2.CAP_PROP_FPS)
        self.duration_frame = int(self.videoCap.get(cv2.CAP_PROP_FRAME_COUNT))

        super(VideoDataMovie, self).open(filename)

        # Reset sync status
        self.model.sync_video_thermal(False)
        if self.paired_data.loaded:
            self.model.main_win.syncVideoBtn.setEnabled(True)

        # --- Set point transformation matrix ---
        frameData = self.dispImg.frameData  # Get framedata
        # Check margin
        xmean = frameData.mean(axis=(0, 2))
        ymean = frameData.mean(axis=(1, 2))

        xedge = np.argwhere(np.abs(np.diff(xmean)) > 50).ravel()
        yedge = np.argwhere(np.abs(np.diff(ymean)) > 50).ravel()
        if len(xedge):
            xshift = xedge[0] + 1
            xscale = np.diff(xedge)[0] / len(xmean)
        else:
            xshift = 0
            xscale = 1

        if len(yedge) > 0:
            yshift = yedge[0] + 1
            yscale = np.diff(yedge)[0] / len(ymean)
        else:
            yshift = 0
            yscale = 1

        shiftMtx = np.eye(3)
        shiftMtx[0, 2] = xshift
        shiftMtx[1, 2] = yshift

        scaleMtx = np.eye(3)
        scaleMtx[0, 0] = xscale
        scaleMtx[1, 1] = yscale

        # Coordinate transformation matrix of thermo to video image
        self.shift_scale_Mtx = np.dot(shiftMtx, scaleMtx)
        self.dispImg.shift_scale_Mtx = self.shift_scale_Mtx

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_frame(self, frame_idx):
        """Read one frame

        Options
        -------
        frame_idx: integer
            Frame index to read. If frame == None and time == None,
            the next frame is read.

        Returns
        -------
        success : bool
            Read success.
        frame_data : array
            Frame image data.
        frame_time:
            Frame position in time (sec).
        """

        if not self.loaded:
            return

        # Set videoCap frame position
        if frame_idx != self.frame_position + 1:
            self.videoCap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # --- read frame ---
        success, frame_data = self.videoCap.read()
        if success:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

        self.frame_position = int(self.videoCap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame_time = self.videoCap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        return success, frame_data, frame_time

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def unload(self):
        super(VideoDataMovie, self).unload()

        # Reset sync status
        self.model.sync_video_thermal(False)
        if self.paired_data.loaded:
            self.model.main_win.syncVideoBtn.setEnabled(False)


# %% ThermalDataMovie class ===================================================
class ThermalDataMovie(DataMovie):
    """Thermal data movie class"""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, parent, dispImg, UI_objs, extract_temp_file=False):
        super(ThermalDataMovie, self).__init__(parent, dispImg, UI_objs)
        self.extract_temp_file = extract_temp_file
        self.thermal_data_reader = None
        self.file_path = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open(self, filename):
        if self.loaded:
            self.unload()

        filename = Path(filename)
        self.file_path = filename
        self.thermal_data_reader = None

        if filename.suffix == ".csq":
            # --- Open and read csq file ----------------------------------
            # Open progress dialog
            progressDlg = QProgressDialog(
                "Reading thermal data ...", "Cancel", 0, 100, self.model.main_win
            )
            progressDlg.setWindowTitle("Reading thermal data")
            progressDlg.setWindowModality(Qt.WindowModal)
            progressDlg.resize(400, 89)
            progressDlg.show()

            try:
                cr = CSQ_READER(
                    filename,
                    progressDlg=progressDlg,
                    extract_temp_file=self.extract_temp_file,
                )
            except Exception as e:
                print(e)
                return

            if self.thermal_data_reader is not None:
                del self.thermal_data_reader
            self.thermal_data_reader = cr

            # Close progress dialog
            progressDlg.close()
            self.thermal_data_reader.progressDlg = None

        self.frame_rate = self.thermal_data_reader.FrameRate
        self.frame_rate = int(self.frame_rate * 100) / 100

        self.duration_frame = self.thermal_data_reader.Count
        super(ThermalDataMovie, self).open(filename)

        # Reset video sync
        self.model.common_time_ms = 0
        self.model.common_duration_ms = (self.duration_frame / self.frame_rate) * 1000
        self.model.main_win.positionSlider.blockSignals(True)
        self.model.main_win.positionSlider.setRange(
            0, int(self.model.common_duration_ms)
        )
        self.model.main_win.positionSlider.setValue(int(self.model.common_time_ms))
        self.model.main_win.positionSlider.setEnabled(True)
        self.model.main_win.positionSlider.blockSignals(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_frame(self, frame_idx):
        """Read one frame

        Options
        -------
        frame_idx: integer
            Frame index to read. If frame == None and time == None,
            the next frame is read.

        Returns
        -------
        success : bool
            Read success.
        frame_data : array
            Frame image data.
        frame_time:
            Frame position in time (sec).
        """

        if not self.loaded:
            return

        try:
            # --- read frame ---
            frame_data = self.thermal_data_reader.getFramebyIdx(frame_idx)
            frame_time = frame_idx * (1.0 / self.frame_rate)
            success = True

        except Exception:
            success = False
            frame_data = None
            frame_time = None

        return success, frame_data, frame_time

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def export_movie(self, fname=None):
        # --- Set saving filenamr ---------------------------------------------
        if fname is None:
            gray_fname = self.filename.parent / (
                self.filename.stem + "_thermo_gray.mp4"
            )
            color_fname = self.filename.parent / (
                self.filename.stem + "_thermo_color.mp4"
            )
        else:
            dst_dir = Path(fname).parent
            fname = Path(fname).stem
            gray_fname = dst_dir / (fname + "_gray.mp4")
            color_fname = dst_dir / (fname + "_color.mp4")

        # --- Ask parameters --------------------------------------------------
        low_perc = 15
        high_perc = 99.9

        # --- Convert thermal data to a video image ---------------------------
        frame_indices = np.arange(self.duration_frame, dtype=int)

        # Open progress dialog
        progressDlg = QProgressDialog(
            "Convert thermal data into a video file ...",
            "Cancel",
            0,
            len(frame_indices),
            self.model.main_win,
        )
        progressDlg.setWindowTitle("Export thermal data as a video file")
        progressDlg.setWindowModality(Qt.WindowModal)
        progressDlg.resize(400, 89)
        progressDlg.show()

        thermal_data_array0 = self.thermal_data_reader._get_thermal_data([0])
        arr_shape = [
            len(frame_indices),
            thermal_data_array0.shape[1],
            thermal_data_array0.shape[2],
        ]

        gray_img_data = np.empty(arr_shape, dtype=np.uint8)
        color_img_data = np.empty([*arr_shape, 3], dtype=np.uint8)
        for ii in frame_indices:
            progressDlg.setValue(ii)
            progressDlg.setLabelText(
                "Convert thermal data into a video file ..."
                f" ({ii+1}/{len(frame_indices)})"
            )
            progressDlg.repaint()

            # Scale frame-by-frame
            frame = self.thermal_data_reader._get_thermal_data(
                [ii], update_buffer=False
            )[0, :, :]
            low, high = np.percentile(frame.ravel(), [low_perc, high_perc])
            gray_frame = (frame - low) / (high - low)
            gray_frame *= 255
            gray_frame[gray_frame < 0] = 0
            gray_frame[gray_frame > 255] = 255

            gray_img_data[ii, :, :] = gray_frame.astype(np.uint8)
            color_frame = cv2.applyColorMap(
                255 - gray_frame.astype(np.uint8), themro_cmap
            )
            color_img_data[ii, :, :, :] = color_frame

            if progressDlg.wasCanceled():
                return

        progressDlg.setValue(np.round(int(len(frame_indices) * 1.1)))
        progressDlg.setLabelText("Save movie file ...")
        progressDlg.repaint()

        """
        progressDlg.repaint()
        low, med, high = np.percentile(thermal_data_array.ravel(), [5, 50, 95])
        val_range = max(med-low, high-med)

        gray_data = (np.tanh((thermal_data_array - med) / val_range) + 1) / 2
        gray_data = (gray_data * 255).astype(np.uint8)
        """

        # --- Save as movie file ----------------------------------------------
        imageio.mimwrite(gray_fname, gray_img_data, fps=self.frame_rate)
        imageio.mimwrite(color_fname, color_img_data, fps=self.frame_rate)

        """
        N_frames, h, w = thermal_data_array.shape
        process = (
            ffmpeg.input('pipe:', format='rawvideo', pix_fmt='gray8',
                         s=f'{w}x{h}')
            .output(str(fname), pix_fmt='yuv420p', vcodec='libx264',
                    r=self.frame_rate)
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        progressDlg.setLabelText('Encoding movie ...')
        for ii, frame in enumerate(gray_data):
            progressDlg.setValue(len(frame_indices) + ii*0.1)
            progressDlg.repaint()

            process.stdin.write(frame.tobytes())
            process.stderr.read()

        process.stdin.close()
        process.wait()
        """

        progressDlg.close()

        return gray_fname

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_frame(self, frame_idx=None, common_time_ms=None, sync_update=True):
        super(ThermalDataMovie, self).show_frame(frame_idx, common_time_ms, sync_update)

        if self.model.main_win.plot_timeline is not None:
            xpos = self.model.main_win.plot_xvals[self.frame_position]
            if self.model.main_win.plot_timeline.get_xdata()[0] != xpos:
                if len(self.model.main_win.plot_line) == 0:
                    self.model.main_win.plot_ax.set_ylim([0, 1])
                self.model.main_win.plot_timeline.set_xdata([xpos, xpos])
                self.model.main_win.roi_plot_canvas.draw()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_rois_dataseries(self, points_ts, rads, aggfunc):
        """Get ROI values time series from the data image time series

        Parameters
        ----------
        points_ts : List of 2D arrays
            List of 2D array [x, y, t] for each tracking points.
        rads : List of float
            List of ROI radius in pixels.
        aggfunc : List of string
            List of value aggregation function names.

        Returns
        -------
        values: List of 1D array
            List of sampled value time series.

        """

        if not self.loaded:
            return np.nan

        # Initialize return values
        values = []
        for xyt in points_ts:
            values.append(np.ones(len(xyt)) * np.nan)

        # --- Set reading frames ----------------------------------------------
        read_frames = np.empty(0, dtype=int)
        for xyt in points_ts:
            mask = ~(np.any(np.isnan(xyt[:, :2].astype(np.float64)), axis=1))
            read_frames = np.concatenate((read_frames, xyt[mask, 2]))
            read_frames = np.unique(read_frames)

        num_reading_frames = len(read_frames)
        if num_reading_frames == 0:
            return values

        # --- Read data -------------------------------------------------------
        show_progress = num_reading_frames > 50
        if show_progress:
            progressDlg = QProgressDialog(
                "Reading thermal data ...",
                "Cancel",
                0,
                num_reading_frames,
                self.model.main_win,
            )
            progressDlg.setWindowTitle("Reading thermal data")
            progressDlg.setWindowModality(Qt.WindowModal)
            progressDlg.resize(250, 89)
            progressDlg.show()
            prog_n = 0

        for ii, frmIdx in enumerate(read_frames.astype(int)):
            for jj, xyt in enumerate(points_ts):
                rad = rads[jj][ii]
                aggf = aggfunc[jj]

                p_idx = np.argwhere(xyt[:, 2] == frmIdx).ravel()
                if len(p_idx) == 0:
                    continue

                if np.any(np.isnan(xyt[p_idx[0], :2])):
                    continue

                cx, cy, frmIdx = xyt[p_idx[0], :]
                val = self.thermal_data_reader.getCircleROIData(
                    frmIdx, cx, cy, rad, aggf
                )
                values[jj][p_idx[0]] = val

            if show_progress:
                if progressDlg.wasCanceled():
                    break
                prog_n += 1
                progressDlg.setValue(prog_n)
                progressDlg.repaint()

        if show_progress:
            progressDlg.close()

        return values

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_temperature_frames(self, update=False):
        self.thermal_data_reader.saveTempFrames(update=update)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def unload(self):
        super(ThermalDataMovie, self).unload()

        # Reset sync status
        self.model.sync_video_thermal(False)
        if self.paired_data.loaded:
            self.model.main_win.syncVideoBtn.setEnabled(False)


# %% TVTDisplayImage class ====================================================
class TVTDisplayImage(DisplayImage):

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, *args, **kwargs):
        super(TVTDisplayImage, self).__init__(*args, **kwargs)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_pixmap(self):
        super(TVTDisplayImage, self).set_pixmap()
        if self.frameData is not None and self.frameData.ndim != 3:
            self.parent.thermal_clim_max_spbx.blockSignals(True)
            self.parent.thermal_clim_min_spbx.blockSignals(True)
            self.parent.thermal_clim_max_spbx.setValue(self.cmax)
            self.parent.thermal_clim_min_spbx.setValue(self.cmin)
            self.parent.thermal_clim_max_spbx.blockSignals(False)
            self.parent.thermal_clim_min_spbx.blockSignals(False)

            val = self.frameData[self.point_mark_xy[1], self.point_mark_xy[0]]
            pos_txt = self.parent.thermalPositionLab.text()
            pos_txt = re.sub("].*", "], Temp. {:.3f} °C".format(val), pos_txt)
            self.parent.thermalPositionLab.setText(pos_txt)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mouseDoubleClickEvent(self, e):
        if not self.parent.model.thermalData.loaded:
            return

        super(TVTDisplayImage, self).mouseDoubleClickEvent(e)


# %% Model class : ThermalVideoModel ==========================================
class ThermalVideoModel(QObject):
    """Model class : ThermalVideoModel"""

    move_point_signal = pyqtSignal()
    select_point_ui_signal = pyqtSignal(str)
    edit_point_signal = pyqtSignal(str)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(
        self, main_win, batchmode=False, extract_temp_file=None, save_interval=10
    ):
        super(ThermalVideoModel, self).__init__(parent=main_win)

        # main_win object
        self.main_win = main_win
        self.DATA_ROOT = APP_ROOT / "data"

        if extract_temp_file is None:
            extract_temp_file = False

        self.CONF_DIR = Path.home() / ".TVT"
        if not self.CONF_DIR.is_dir():
            self.CONF_DIR.mkdir()

        self.conf_f = self.CONF_DIR / "TVT_conf.json"
        if self.conf_f.is_file():
            try:
                with open(self.conf_f, "r") as fd:
                    conf = json.load(fd)

                for k, v in conf.items():
                    if k in ("DATA_ROOT",):
                        v = Path(v)
                    setattr(self, k, v)
            except Exception:
                pass

        # --- DataMovie objects -----------------------------------------------
        # Thermal data
        if self.main_win is not None:
            thermal_UI_objs = {
                "frFwdBtn": self.main_win.thermalFrameFwdBtn,
                "frBkwBtn": self.main_win.thermalFrameBkwBtn,
                "skipFwdBtn": self.main_win.thermalSkipFwdBtn,
                "skipBkwBtn": self.main_win.thermalSkipBkwBtn,
                "framePosSpBox": self.main_win.thermalFramePosSpBox,
                "framePosLab": self.main_win.thermalFramePosLab,
                "positionLabel": self.main_win.thermalPositionLab,
                "syncBtn": self.main_win.syncVideoBtn,
            }
            self.thermalData = ThermalDataMovie(
                self,
                self.main_win.thermalDispImg,
                thermal_UI_objs,
                extract_temp_file=extract_temp_file,
            )

            video_UI_objs = {
                "frFwdBtn": self.main_win.videoFrameFwdBtn,
                "frBkwBtn": self.main_win.videoFrameBkwBtn,
                "skipFwdBtn": self.main_win.videoSkipFwdBtn,
                "skipBkwBtn": self.main_win.videoSkipBkwBtn,
                "framePosSpBox": self.main_win.videoFramePosSpBox,
                "framePosLab": self.main_win.videoFramePosLab,
                "positionLabel": self.main_win.videoPositionLab,
                "syncBtn": self.main_win.syncVideoBtn,
            }
        self.videoData = VideoDataMovie(self, self.main_win.videoDispImg, video_UI_objs)

        # Set pair
        self.thermalData.paired_data = self.videoData
        self.videoData.paired_data = self.thermalData

        # Point marker (black dot) position
        self.point_mark_xy = [0, 0]
        if self.main_win is not None:
            self.main_win.videoDispImg.point_mark_xy = self.point_mark_xy
            self.main_win.thermalDispImg.point_mark_xy = self.point_mark_xy

        # --- Tracking point --------------------------------------------------
        self.tracking_mark = dict()  # tracking point marks on display
        if self.main_win is not None:
            self.main_win.videoDispImg.tracking_mark = self.tracking_mark
            self.main_win.thermalDispImg.tracking_mark = self.tracking_mark
        self.editRange = "current"

        # self.lpf = 0  # Hz, 0 == No filter

        # --- Common time (ms), movie parameters ------------------------------
        self.common_time_ms = 0
        self.on_sync = False

        # timer for movie play
        if self.main_win is not None:
            self.play_timer = QTimer(self)
            self.play_timer.setSingleShot(True)
            self.play_timer.timeout.connect(self.play_update)
            self.play_frame_interval_ms = np.inf

        # --- Time marker -----------------------------------------------------
        self.time_marker = {}
        self.tracking_point = dict()  # tracking point temperatures values
        self.editRange = "current"

        # --- DeepLabCut interface --------------------------------------------
        self.dlci = DLCinter(self.DATA_ROOT, ui_parent=self.main_win)

        # --- Connect signals -------------------------------------------------
        self.move_point_signal.connect(self.update_dispImg)
        self.select_point_ui_signal.connect(self.select_point_ui)
        self.edit_point_signal.connect(self.edit_tracking_point)

        # --- Load last working status ----------------------------------------
        self.loaded_state_f = None
        self.tmp_state_f = APP_ROOT / "config" / "tmp_working_state.pkl"
        self.num_saved_setting_hist = 5
        last_state_f = APP_ROOT / "config" / "TVT_last_working_state-0.pkl"
        if not last_state_f.parent.is_dir():
            os.makedirs(last_state_f.parent)

        if not batchmode:
            self.save_timer = QTimer()
            self.save_timer.setSingleShot(True)
            self.save_timer.timeout.connect(self.save_tmp_status)
            self.save_tmp_wait = save_interval  # seconds
            self.save_timer.start(self.save_tmp_wait * 1000)

            if self.tmp_state_f.is_file():
                ret = QMessageBox.question(
                    self.main_win,
                    "Recover state",
                    "Recover the last aborted state?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if ret == QMessageBox.Yes:
                    self.load_status(fname=self.tmp_state_f)
            elif last_state_f.is_file():
                ret = QMessageBox.question(
                    self.main_win,
                    "Load last state",
                    "Retrieve the last working state?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if ret == QMessageBox.Yes:
                    self.load_status(fname=last_state_f)

    # --- Data file handling --------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def openThermalFile(self, *args, fileName=None, **kwargs):
        if fileName is None:
            stdir = self.DATA_ROOT
            filterStr = "Thermal data (*.csq);;All files (*.*)"
            fileName, _ = QFileDialog.getOpenFileName(
                self.main_win, "Open thermal file", str(stdir), filterStr, None
            )  # , QFileDialog.DontUseNativeDialog)

            if fileName == "":
                return

        fileName = Path(fileName)

        self.thermalData.open(fileName)

        # Reset
        self.time_marker = {}
        self.videoData.paired_data = self.thermalData

        # Point marker (black dot) position
        self.point_mark_xy = [0, 0]
        if self.main_win is not None:
            self.main_win.videoDispImg.point_mark_xy = self.point_mark_xy
            self.main_win.thermalDispImg.point_mark_xy = self.point_mark_xy

        # Tracking point
        self.tracking_mark = dict()  # tracking point marks on display
        if self.main_win is not None:
            self.main_win.videoDispImg.tracking_mark = self.tracking_mark
            self.main_win.thermalDispImg.tracking_mark = self.tracking_mark

            self.main_win.unloadThermalDataBtn.setText(
                f"Unload {Path(self.thermalData.filename).name}"
            )
            self.main_win.unloadThermalDataBtn.setEnabled(True)
            self.main_win.exportThermalDataVideoBtn.setEnabled(True)

            del self.main_win.plot_ax
            self.main_win.roi_plot_canvas.figure.clear()
            self.main_win.plot_ax = self.main_win.roi_plot_canvas.figure.subplots(1, 1)
            self.main_win.plot_xvals = None
            self.main_win.plot_line = {}
            # self.main_win.plot_line_lpf = {}
            self.main_win.plot_timeline = None
            self.main_win.plot_marker_line = {}
            self.main_win.roi_plot_canvas.setEnabled(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_thermal_clim(self, *args, **kwargs):
        if not self.thermalData.loaded:
            return

        cmax = self.main_win.thermal_clim_max_spbx.value()
        cmin = self.main_win.thermal_clim_min_spbx.value()

        if (
            hasattr(self.main_win.thermalDispImg, "cmax")
            and cmax != self.main_win.thermalDispImg.cmax
        ):
            if cmax <= cmin:
                cmax = cmin + 0.1
                self.main_win.thermal_clim_max_spbx.blockSignals(True)
                self.main_win.thermal_clim_max_spbx.setValue(cmax)
                self.main_win.thermal_clim_max_spbx.blockSignals(False)

        if (
            hasattr(self.main_win.thermalDispImg, "cmin")
            and cmin != self.main_win.thermalDispImg.cmin
        ):
            if cmin >= cmax:
                cmin = cmax - 0.1
                self.main_win.thermal_clim_min_spbx.blockSignals(True)
                self.main_win.thermal_clim_min_spbx.setValue(cmin)
                self.main_win.thermal_clim_min_spbx.blockSignals(False)

        # No pixmap update but set clim
        if (
            hasattr(self.main_win.thermalDispImg, "cmin")
            and cmin == self.main_win.thermalDispImg.cmin
            and cmax == self.main_win.thermalDispImg.cmax
        ):
            # Fix checkbox
            if (
                self.main_win.thermal_clim_fix_chbx.checkState()
                != Qt.CheckState.Unchecked
            ):
                self.main_win.thermalDispImg.clim = [cmin, cmax]
            else:
                self.main_win.thermalDispImg.clim = None

            return

        self.main_win.thermalDispImg.clim = [cmin, cmax]
        self.main_win.thermalDispImg.set_pixmap()
        if self.main_win.thermal_clim_fix_chbx.checkState() == Qt.CheckState.Unchecked:
            self.main_win.thermalDispImg.clim = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def unloadThermalData(self):
        self.thermalData.unload()

        self.time_marker = {}
        self.tracking_mark = dict()  # tracking point marks on display
        if self.main_win is not None:
            self.main_win.unloadThermalDataBtn.setText("---")
            self.main_win.unloadThermalDataBtn.setEnabled(False)
            self.main_win.exportThermalDataVideoBtn.setEnabled(False)
            self.main_win.roi_ctrl_grpbx.setEnabled(False)

            del self.main_win.plot_ax
            self.main_win.roi_plot_canvas.figure.clear()
            self.main_win.plot_ax = self.main_win.roi_plot_canvas.figure.subplots(1, 1)
            self.main_win.plot_xvals = None
            self.main_win.plot_line = {}
            # self.main_win.plot_line_lpf = {}
            self.main_win.plot_timeline = None
            self.main_win.plot_marker_line = {}
            self.main_win.roi_plot_canvas.setEnabled(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def exportThermalDataVideo(self):
        # Export thermal data as a video file
        fileName = self.thermalData.export_movie()

        if fileName is None:
            # Canceled
            return

        # Load exported video file
        ret = QMessageBox.question(
            self.main_win,
            "Load exported video?",
            "Load the exported thermal data video as video data?",
            QMessageBox.Yes | QMessageBox.No,
            defaultButton=QMessageBox.Yes,
        )

        if ret == QMessageBox.No:
            return

        self.openVideoFile(fileName=fileName)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def openVideoFile(self, *args, fileName=None, **kwargs):
        if fileName is None:
            stdir = self.DATA_ROOT
            fileName, _ = QFileDialog.getOpenFileName(
                self.main_win,
                "Open Movie",
                str(stdir),
                "movie files (*.mp4 *.avi);; all (*.*)",
                None,
            )

            if fileName == "":
                return

        fileName = Path(fileName)

        if not str(fileName.absolute()).startswith(str(self.DATA_ROOT.absolute())):
            # the data file is not in the DATA_ROOT
            msgBox = QMessageBox()
            msgBox.setText(
                f"The video file is not located under {self.DATA_ROOT}."
                f" Would you like to copy it there ({self.DATA_ROOT})?"
            )
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msgBox.setDefaultButton(QMessageBox.Yes)
            ret = msgBox.exec()
            if ret == QMessageBox.Yes:
                destination = self.DATA_ROOT / fileName.name
                shutil.copy(fileName, destination)
                fileName = destination  # Update filePath to the new location

        self.videoData.open(fileName)
        self.main_win.unloadVideoDataBtn.setText(f"Unload {Path(fileName).name}")

        self.main_win.unloadVideoDataBtn.setEnabled(True)

        # Sync video to thermo if framerate and number of frames are same
        if not self.thermalData.loaded:
            return

        fr_diff = np.abs(self.videoData.frame_rate - self.thermalData.frame_rate)
        if (
            fr_diff > 0.001
            or self.videoData.duration_frame != self.thermalData.duration_frame
        ):
            return

        self.videoData.show_frame(self.thermalData.frame_position)
        self.sync_video_thermal(True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def unloadVideoData(self):
        self.videoData.unload()
        self.main_win.unloadVideoDataBtn.setText("Unload")
        self.main_win.unloadVideoDataBtn.setEnabled(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def sync_video_thermal(self, on_sync):
        self.on_sync = on_sync

        # --- Off synch ---
        if not self.on_sync:
            self.main_win.syncVideoBtn.setText("Sync video to thermo")
            if self.main_win.syncVideoBtn.isChecked():
                self.main_win.syncVideoBtn.blockSignals(True)
                self.main_win.syncVideoBtn.setChecked(False)
                self.main_win.syncVideoBtn.blockSignals(False)
            return

        # --- On synch ---
        # Check if both data are loaded
        if not self.videoData.loaded or not self.thermalData.loaded:
            # Either one is not loaded
            self.on_sync = False
            self.main_win.syncVideoBtn.blockSignals(True)
            self.main_win.syncVideoBtn.setChecked(False)
            self.main_win.syncVideoBtn.setText("Sync video to thermo")
            self.main_win.syncVideoBtn.blockSignals(False)
            return

        # Set time offest
        t_thermal_ms = 1000 * (
            self.thermalData.frame_position / self.thermalData.frame_rate
        )
        t_video_ms = 1000 * (self.videoData.frame_position / self.videoData.frame_rate)
        self.videoData.shift_from_refTime = t_video_ms - t_thermal_ms

        # Set common time range
        dur_thermal_ms = (
            self.thermalData.duration_frame / self.thermalData.frame_rate
        ) * 1000
        dur_video_ms = (
            self.videoData.duration_frame / self.videoData.frame_rate
        ) * 1000
        comt_min = int(min(0, -self.videoData.shift_from_refTime))
        comt_max = int(
            max(dur_thermal_ms, dur_video_ms - self.videoData.shift_from_refTime)
        )

        self.common_time_ms = t_thermal_ms

        # Set slider
        self.main_win.positionSlider.blockSignals(True)
        self.main_win.positionSlider.setRange(comt_min, comt_max)
        self.main_win.positionSlider.setValue(int(self.common_time_ms))
        self.main_win.positionSlider.blockSignals(False)
        self.set_common_time_label()

        # Set main_win UIs
        if not self.main_win.syncVideoBtn.isChecked():
            self.main_win.syncVideoBtn.blockSignals(True)
            self.main_win.syncVideoBtn.setChecked(True)
            self.main_win.syncVideoBtn.blockSignals(False)

        self.main_win.syncVideoBtn.setText("Unsync video to thermo")

    # --- Common movie control functions --------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_video_frame(self, frame, caller=None):
        if self.videoData.loaded and self.on_sync:
            if caller != self.videoData:
                self.videoData.show_frame(frame_idx=frame, sync_update=False)
        if caller is None:
            caller.blockSignals(True)
            caller.setValue(frame)
            caller.blockSignals(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_common_time(self, time_ms=None, caller=None):
        if time_ms is None:
            time_ms = self.main_win.positionSlider.value()
        else:
            self.main_win.positionSlider.blockSignals(True)
            self.main_win.positionSlider.setValue(int(time_ms))
            self.main_win.positionSlider.blockSignals(False)

        self.common_time_ms = time_ms

        # Show frame
        if self.thermalData.loaded:
            if caller != self.thermalData:
                self.thermalData.show_frame(common_time_ms=time_ms, sync_update=False)

        if self.videoData.loaded and self.on_sync:
            if caller != self.videoData:
                self.videoData.show_frame(common_time_ms=time_ms, sync_update=False)

        self.plot_timecourse()

        # Set time label
        self.set_common_time_label()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_common_time_label(self):
        # Set common position text
        tstr = str(timedelta(seconds=self.common_time_ms / 1000))
        if "." in tstr:
            tstr = tstr[:-4]
        else:
            tstr += ".00"

        self.main_win.commonPosisionLab.setText(tstr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def commonSkipFwd(self):
        max_t = self.main_win.positionSlider.maximum()
        common_time_ms = min(self.common_time_ms + 1000, max_t)
        if common_time_ms == self.common_time_ms:
            return

        self.set_common_time(common_time_ms)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def commonSkipBkw(self):
        common_time_ms = max(self.common_time_ms - 1000, 0)

        if common_time_ms == 0:
            return

        self.set_common_time(common_time_ms)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def play(self):
        # Get frame interval
        self.play_frame_interval_ms = np.inf
        if self.videoData.loaded:
            self.play_frame_interval_ms = 1000 / self.videoData.frame_rate

        if self.thermalData.loaded:
            self.play_frame_interval_ms = min(
                1000 / self.thermalData.frame_rate, self.play_frame_interval_ms
            )

        if np.isinf(self.play_frame_interval_ms):
            return

        self.play_timer.start(0)
        self.main_win.playBtn.setIcon(
            self.main_win.style().standardIcon(QStyle.SP_MediaPause)
        )
        self.main_win.playBtn.clicked.disconnect(self.play)
        self.main_win.playBtn.clicked.connect(self.pause)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def play_update(self):
        st = time.time()
        # End time of the movie
        max_t = self.main_win.positionSlider.maximum()
        if self.common_time_ms + self.play_frame_interval_ms > max_t:
            self.pause()
            return

        # Increment movie
        self.common_time_ms += self.play_frame_interval_ms
        self.set_common_time(self.common_time_ms)
        update_time = time.time() - st

        # Schedule the next frame
        interval = self.play_frame_interval_ms - update_time * 1000
        self.play_timer.start(max(0, int(interval)))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def pause(self):
        self.play_timer.stop()
        self.main_win.playBtn.setIcon(
            self.main_win.style().standardIcon(QStyle.SP_MediaPlay)
        )
        self.main_win.playBtn.clicked.disconnect(self.pause)
        self.main_win.playBtn.clicked.connect(self.play)

    # --- Tracking point click callbacks --------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_editRange(self, selectIdx):
        edRngs = ["current", "Mark<", "<Mark", "0<", ">End"]
        self.editRange = edRngs[selectIdx]

        if selectIdx != self.main_win.roi_editRange_cmbbx.currentIndex():
            self.main_win.roi_editRange_cmbbx.blockSignals(True)
            self.main_win.roi_editRange_cmbbx.setCurrentIndex(selectIdx)
            self.main_win.roi_editRange_cmbbx.blockSignals(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_dispImg(self):
        if self.videoData.loaded:
            self.main_win.videoDispImg.set_pixmap()

        if self.thermalData.loaded:
            self.main_win.thermalDispImg.set_pixmap()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def select_point_ui(self, point_name=None, update_plot=True):
        """chnage tracking point
        If point_name is not in self.tracking_mark, the point was deleted.
        """
        if point_name is None:
            point_name = self.main_win.roi_idx_cmbbx.currentText()

        # block signals to main_win
        self.main_win.roi_idx_cmbbx.blockSignals(True)
        self.main_win.roi_name_ledit.blockSignals(True)
        self.main_win.roi_x_spbx.blockSignals(True)
        self.main_win.roi_y_spbx.blockSignals(True)
        self.main_win.roi_rad_spbx.blockSignals(True)
        self.main_win.roi_aggfunc_cmbbx.blockSignals(True)
        self.main_win.roi_color_cmbbx.blockSignals(True)

        if point_name in self.tracking_mark:
            frame = self.thermalData.frame_position

            # Set main_win values
            self.main_win.roi_idx_cmbbx.setCurrentText(point_name)
            self.main_win.roi_name_ledit.setText(point_name)
            x = self.tracking_mark[point_name]["x"]
            y = self.tracking_mark[point_name]["y"]
            if np.isnan(x) or np.isnan(y):
                self.main_win.roi_x_spbx.setValue(-1)
                self.main_win.roi_y_spbx.setValue(-1)
            else:
                self.main_win.roi_x_spbx.setValue(int(x))
                self.main_win.roi_y_spbx.setValue(int(y))
            self.main_win.roi_rad_spbx.setValue(
                self.tracking_point[point_name].radius[frame]
            )
            self.main_win.roi_aggfunc_cmbbx.setCurrentText(
                self.tracking_point[point_name].aggfunc
            )
            self.main_win.roi_color_cmbbx.setCurrentText(
                self.tracking_mark[point_name]["pen_color"]
            )

            val = self.tracking_point[point_name].get_value([frame])[0]

            if np.isnan(val):
                self.main_win.roi_val_lab.setText("Temp. ----- °C")
            else:
                self.main_win.roi_val_lab.setText(f"Temp. {val:.2f} °C")
        else:
            # point_name is deleted. Reset
            self.main_win.roi_name_ledit.setText("")
            self.main_win.roi_x_spbx.setValue(-1)
            self.main_win.roi_y_spbx.setValue(-1)
            self.main_win.roi_rad_spbx.setValue(tracking_point_radius_default)
            self.main_win.roi_aggfunc_cmbbx.setCurrentText(
                tracking_point_aggfunc_default
            )
            self.main_win.roi_color_cmbbx.setCurrentText(
                tracking_point_pen_color_default
            )
            self.main_win.roi_val_lab.setText("Temp. ----- °C")

        # unblock signals
        self.main_win.roi_idx_cmbbx.blockSignals(False)
        self.main_win.roi_name_ledit.blockSignals(False)
        self.main_win.roi_x_spbx.blockSignals(False)
        self.main_win.roi_y_spbx.blockSignals(False)
        self.main_win.roi_rad_spbx.blockSignals(False)
        self.main_win.roi_aggfunc_cmbbx.blockSignals(False)
        self.main_win.roi_color_cmbbx.blockSignals(False)

        if update_plot:
            self.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def edit_tracking_point(self, point_name):
        if not self.thermalData.loaded:
            return

        # --- Set max position (frame can be larger than the image) -----------
        if self.thermalData.loaded:
            xmax = self.thermalData.dispImg.frame_w - 1
            ymax = self.thermalData.dispImg.frame_h - 1

        if self.videoData.loaded:
            xmax = max(self.videoData.dispImg.frame_w - 1, xmax)
            ymax = max(self.videoData.dispImg.frame_h - 1, ymax)

        if self.main_win.roi_x_spbx.maximum() != xmax:
            self.main_win.roi_x_spbx.setMaximum(xmax)

        if self.main_win.roi_y_spbx.maximum() != ymax:
            self.main_win.roi_y_spbx.setMaximum(ymax)

        if point_name not in self.tracking_mark.keys():
            # point_name is deleted
            del self.tracking_point[point_name]
        else:
            # --- Check properties in self.tracking_mark[k] -------------------
            x = self.tracking_mark[point_name]["x"]
            y = self.tracking_mark[point_name]["y"]

            # Set position values in main_win
            if x > xmax:
                x = xmax
                self.main_win.roi_x_spbx.blockSignals(True)
                self.main_win.roi_x_spbx.setValue(x)
                self.main_win.roi_x_spbx.blockSignals(False)

            if y > ymax:
                y = ymax
                self.main_win.roi_y_spbx.blockSignals(True)
                self.main_win.roi_y_spbx.setValue(y)
                self.main_win.roi_y_spbx.blockSignals(False)

            # --- Edit tracking point time series -----------------------------
            if point_name in self.tracking_point:
                # Edit existing trcking point time series
                self.tracking_point[point_name].set_position(x, y, update_frames=True)
            else:
                # Make a new tracking point time series for thermalData
                self.tracking_point[point_name] = TrackingPoint(self.thermalData, x, y)

                # Set tracking mark properties with tracking_point object
                self.tracking_mark[point_name]["name"] = point_name
                self.tracking_mark[point_name]["rad"] = self.tracking_point[
                    point_name
                ].radius[self.thermalData.frame_position]
                self.tracking_mark[point_name]["aggfunc"] = self.tracking_point[
                    point_name
                ].aggfunc
                self.tracking_mark[point_name][
                    "pen_color"
                ] = tracking_point_pen_color_default

                # Set the point positions
                frame_indices = np.arange(
                    self.thermalData.frame_position, self.thermalData.duration_frame
                )
                self.tracking_point[point_name].set_position(
                    x,
                    y,
                    frame_indices=frame_indices,
                    update_frames=[self.thermalData.frame_position],
                )

        # update display
        self.update_dispImg()

        # Set tracking points control widgets
        self.main_win.roi_idx_cmbbx.clear()
        if len(self.tracking_mark) == 0:
            self.main_win.roi_ctrl_grpbx.setEnabled(False)
            self.main_win.roi_export_btn.setEnabled(False)
            self.main_win.roi_plot_canvas.setEnabled(False)
        else:
            self.main_win.roi_ctrl_grpbx.setEnabled(True)
            self.main_win.roi_export_btn.setEnabled(True)
            self.main_win.roi_plot_canvas.setEnabled(True)

            self.main_win.roi_idx_cmbbx.blockSignals(True)
            self.main_win.roi_idx_cmbbx.addItems(list(self.tracking_point.keys()))
            self.main_win.roi_idx_cmbbx.blockSignals(False)

        self.select_point_ui(point_name)

        # Reset edit range
        self.set_editRange(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def edit_point_property(self, *args):
        frame = self.thermalData.frame_position
        point_name = self.main_win.roi_idx_cmbbx.currentText()
        edit_name = self.main_win.roi_name_ledit.text()
        x = self.main_win.roi_x_spbx.value()
        y = self.main_win.roi_y_spbx.value()
        if x < 0 and y < 0:
            x = np.nan
            y = np.nan
        elif x < 0:
            x = y
            self.main_win.roi_x_spbx.blockSignals(True)
            self.main_win.roi_x_spbx.setValue(x)
            self.main_win.roi_x_spbx.blockSignals(False)
        elif y < 0:
            y = x
            self.main_win.roi_y_spbx.blockSignals(True)
            self.main_win.roi_y_spbx.setValue(y)
            self.main_win.roi_y_spbx.blockSignals(False)

        rad = self.main_win.roi_rad_spbx.value()
        aggfunc = self.main_win.roi_aggfunc_cmbbx.currentText()
        col = self.main_win.roi_color_cmbbx.currentText()

        self.tracking_mark[point_name]["pen_color"] = col

        # Name edit
        if edit_name != point_name:
            self.tracking_point[edit_name] = self.tracking_point.pop(point_name)
            self.tracking_mark[edit_name] = self.tracking_mark.pop(point_name)
            point_name = edit_name

            # update main_win list
            self.main_win.roi_idx_cmbbx.clear()
            self.main_win.roi_idx_cmbbx.addItems(
                sorted(list(self.tracking_point.keys()))
            )

        # Radius change
        if rad != self.tracking_point[point_name].radius[frame]:
            self.tracking_point[point_name].radius[frame] = rad
            self.tracking_mark[point_name]["rad"] = rad
            # Reset tracking values
            self.tracking_point[point_name].value_ts[frame] = np.nan

        # aggfunc change
        if aggfunc != self.tracking_point[point_name].aggfunc:
            self.tracking_point[point_name].aggfunc = aggfunc
            # Reset tracking values
            self.tracking_point[point_name].value_ts[:] = np.nan

        # Position change
        current_x, current_y = self.tracking_point[point_name].get_current_position()
        if x != current_x or y != current_y:
            self.tracking_mark[point_name]["x"] = x
            self.tracking_mark[point_name]["y"] = y

            # Reset tracking values from the current frame
            Nframes = len(self.tracking_point[point_name].value_ts)

            if self.editRange == "current":
                frame_indices = [frame]

            elif self.editRange == "Mark<":
                markFrames = np.unique(list(self.time_marker.keys()))
                fromFrame = markFrames[markFrames < frame]
                if len(fromFrame):
                    fromFrame = fromFrame[-1] + 1
                else:
                    fromFrame = 0

                if frame in self.time_marker:
                    toFrame = frame  # Not include the current frame
                else:
                    toFrame = frame + 1  # include the current frame

                frame_indices = np.arange(fromFrame, toFrame)

            elif self.editRange == "<Mark":
                markFrames = np.unique(list(self.time_marker.keys()))
                toFrame = markFrames[markFrames > frame]
                if len(toFrame):
                    toFrame = toFrame[0]
                else:
                    toFrame = Nframes
                toFrame = min(toFrame, Nframes)

                if frame in self.time_marker:
                    fromFrame = frame + 1  # Not include the current frame
                else:
                    fromFrame = frame  # include the current frame

                frame_indices = np.arange(fromFrame, toFrame)

            elif self.editRange == "0<":
                if frame in self.time_marker:
                    toFrame = frame  # Not include the current frame
                else:
                    toFrame = frame + 1  # include the current frame
                frame_indices = np.arange(0, toFrame)

            elif self.editRange == ">End":
                if frame in self.time_marker:
                    fromFrame = frame + 1  # Not include the current frame
                else:
                    fromFrame = frame  # include the current frame
                frame_indices = np.arange(fromFrame, Nframes)

            self.tracking_point[point_name].value_ts[frame_indices] = np.nan

            # update position and value
            self.tracking_point[point_name].set_position(
                x, y, frame_indices=frame_indices, update_frames=[frame]
            )

        # update display
        self.update_dispImg()
        self.select_point_ui(point_name)

        self.plot_timecourse()
        self.set_editRange(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def apply_radius_all(self, *args):
        point_name = self.main_win.roi_idx_cmbbx.currentText()
        rad = self.main_win.roi_rad_spbx.value()
        val_reset_frames = self.tracking_point[point_name].radius != rad
        self.tracking_point[point_name].radius[:] = rad
        self.tracking_point[point_name].value_ts[val_reset_frames] = np.nan

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def erase_point(self):
        """Erase current (and later) tracking position"""
        point_name = self.main_win.roi_idx_cmbbx.currentText()

        frame = self.thermalData.frame_position
        Nframes = len(self.tracking_point[point_name].value_ts)

        if self.editRange == "current":
            frame_indices = [frame]

        elif self.editRange == "Mark<":
            markFrames = np.unique(list(self.time_marker.keys()))
            fromFrames = markFrames[markFrames < frame]
            if len(fromFrames):
                fromFrame = fromFrames[-1] + 1
            else:
                fromFrame = 0
            toFrame = frame  # Not include the current frame
            frame_indices = np.arange(fromFrame, toFrame)

        elif self.editRange == "<Mark":
            markFrames = np.unique(list(self.time_marker.keys()))
            toFrames = markFrames[markFrames > frame]
            if len(toFrames):
                toFrame = toFrames[0]
            else:
                toFrame = Nframes

            toFrame = min(toFrame, Nframes)
            fromFrame = frame + 1  # Not include the current frame
            frame_indices = np.arange(fromFrame, toFrame)

        elif self.editRange == "0<":
            toFrame = frame  # Not include the current frame
            frame_indices = np.arange(0, toFrame)

        elif self.editRange == ">End":
            fromFrame = frame + 1  # Not include the current frame
            frame_indices = np.arange(fromFrame, Nframes)

        self.tracking_point[point_name].set_position(
            np.nan, np.nan, frame_indices=frame_indices, update_frames=frame_indices
        )

        self.tracking_mark[point_name]["x"] = np.nan
        self.tracking_mark[point_name]["y"] = np.nan

        self.update_dispImg()
        self.select_point_ui(point_name)
        self.plot_timecourse()

        self.set_editRange(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def delete_point(self, point_name=None, ask_confirm=True):
        """Delete tracking point time series"""

        if point_name is None:
            point_name = self.main_win.roi_idx_cmbbx.currentText()

        if ask_confirm:
            # Confirm delete
            confMsg = (
                f"Are you sure to delete the point '{point_name}'?\n"
                + f"All time-seriese data for '{point_name}' will be deleted."
            )
            rep = QMessageBox.question(
                self.main_win,
                "Confirm delete",
                confMsg,
                QMessageBox.Yes,
                QMessageBox.No,
            )
            if rep == QMessageBox.No:
                return

        del self.tracking_mark[point_name]
        self.edit_tracking_point(point_name)
        self.plot_timecourse()

    # --- Time marker control functions ---------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def add_marker(self):
        if not self.thermalData.loaded:
            msgBox = QMessageBox(self.main_win)
            msgBox.setWindowModality(Qt.WindowModal)
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(
                "No thermal data is loaded!"
                "\nMakers can be set only for a thermal data frame"
            )
            msgBox.setWindowTitle("Error")
            msgBox.exec()
            return

        # Check marker name
        marker_name = self.main_win.tmark_name_cmbbx.currentText()
        if len(marker_name) == 0:
            msgBox = QMessageBox(self.main_win)
            msgBox.setWindowModality(Qt.WindowModal)
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText("Marker name is empty!")
            msgBox.setWindowTitle("Error")
            msgBox.exec()
            return

        # Put a marker at the current thermal frame
        frmIdx = self.thermalData.frame_position
        self.time_marker[frmIdx] = marker_name
        self.show_marker()

        # Set maker list
        marker_list = np.unique(list(self.time_marker.values()))
        marker_list = [""] + list(marker_list)
        self.main_win.tmark_name_cmbbx.clear()
        self.main_win.tmark_name_cmbbx.addItems(marker_list)
        self.main_win.tmark_name_cmbbx.setCurrentText(marker_name)

        # Plot time mark
        self.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def del_marker(self):
        frmIdx = self.thermalData.frame_position
        if frmIdx in self.time_marker:
            del self.time_marker[frmIdx]

            # Set maker list
            marker_list = np.unique(list(self.time_marker.values()))
            marker_list = [""] + list(marker_list)
            self.main_win.tmark_name_cmbbx.clear()
            self.main_win.tmark_name_cmbbx.addItems(marker_list)
            self.main_win.tmark_name_cmbbx.setCurrentText("")
            self.show_marker()

            self.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def jump_marker(self, shift):
        if len(self.time_marker) == 0:
            return

        marker_positions = np.array(sorted(self.time_marker.keys()))
        current = self.thermalData.frame_position
        if shift > 0:  # Forward
            if marker_positions[-1] <= current:
                # No marker later than the current
                return
            jumpFrame = int(marker_positions[marker_positions > current][0])

        elif shift < 0:  # Backward
            if marker_positions[0] >= current:
                # No maker ealier than the current
                return
            jumpFrame = int(marker_positions[marker_positions < current][-1])

        marker_name = self.time_marker[jumpFrame]
        self.main_win.tmark_name_cmbbx.setCurrentText(marker_name)
        self.thermalData.show_frame(jumpFrame)

        # Set position slider
        self.main_win.positionSlider.blockSignals(True)
        self.main_win.positionSlider.setValue(int(self.common_time_ms))
        self.main_win.positionSlider.blockSignals(False)

        # update plot timeline
        self.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_marker(self):
        if not self.thermalData.loaded:
            return

        thermalFrmIdx = self.thermalData.frame_position
        if thermalFrmIdx in self.time_marker:
            marker_name = self.time_marker[thermalFrmIdx]
            self.main_win.thermalMakerLab.setStyleSheet("background:red; color:white;")
            self.main_win.thermalMakerLab.setText(marker_name)
            self.main_win.tmark_name_cmbbx.setCurrentText(marker_name)

            if self.videoData.loaded and self.on_sync:
                self.main_win.videoMakerLab.setStyleSheet(
                    "background:red; color:white;"
                )
                self.main_win.videoMakerLab.setText(marker_name)
        else:
            self.main_win.thermalMakerLab.setStyleSheet(
                "background:black; color:white;"
            )
            self.main_win.thermalMakerLab.setText("")
            self.main_win.tmark_name_cmbbx.setCurrentText("")

            self.main_win.videoMakerLab.setStyleSheet("background:black; color:white;")
            self.main_win.videoMakerLab.setText("")

    # --- Thermal tracking time course plot -----------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def plot_onclick(self, event):
        if self.main_win.plot_xvals is None:
            return

        xpos = event.xdata
        if xpos is not None:
            self.set_common_time(time_ms=xpos * 1000)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def plot_timecourse(
        self,
        plot_all_points=False,
        update_all_data=False,
        update_plot=False,
        *args,
        **kwargs,
    ):

        # self.lpf = self.main_win.roi_LPF_thresh_spbx.value()

        # --- Set xvals in time -----------------------------------------------
        if self.main_win.plot_xvals is None:
            xvals = np.arange(0, self.thermalData.duration_frame)
            if len(xvals) == 0:
                self.main_win.plot_ax.cla()
                self.main_win.plot_xvals = None
                return

            xunit = 1.0 / self.videoData.frame_rate
            self.main_win.plot_xvals = xvals * xunit
            xvals = self.main_win.plot_xvals
            self.main_win.plot_ax.set_xlim(xvals[0] - xunit, xvals[-1] + xunit)

            # Set xtick label
            xticks = self.main_win.plot_ax.get_xticks()
            xtick_labs = []
            for xt in xticks:
                if xt < 0:
                    xtick_labs.append("")
                else:
                    tstr = str(timedelta(seconds=xt))
                    tstr = re.sub(r"^0+:", "", tstr)
                    tstr = re.sub(r"^0", "", tstr)
                    tstr = re.sub(r"\..+$", "", tstr)
                    xtick_labs.append(tstr)

            self.main_win.plot_ax.set_xticks(xticks, xtick_labs)
            self.main_win.plot_ax.set_xlim(xvals[0] - xunit, xvals[-1] + xunit)

        # --- Time line -------------------------------------------------------
        xpos = self.main_win.plot_xvals[self.thermalData.frame_position]
        if self.main_win.plot_timeline is None:
            self.main_win.plot_timeline = self.main_win.plot_ax.axvline(
                xpos, color="k", ls=":", lw=1
            )
        else:
            self.main_win.plot_timeline.set_xdata([xpos, xpos])

        # --- Marker line -----------------------------------------------------
        for frame in self.time_marker.keys():
            tx = self.main_win.plot_xvals[frame]
            self.main_win.plot_marker_line[frame] = self.main_win.plot_ax.axvline(
                tx, color="r", lw=1
            )

        # Delete marker
        if hasattr(self.main_win, "plot_marker_line"):
            rm_marker = np.setdiff1d(
                list(self.main_win.plot_marker_line.keys()),
                list(self.time_marker.keys()),
            )
            if len(rm_marker):
                for rmfrm in rm_marker:
                    self.main_win.plot_marker_line[rmfrm].remove()
                    del self.main_win.plot_marker_line[rmfrm]

        # --- Check value update ----------------------------------------------
        # Select points
        all_points = list(self.tracking_point.keys())
        if plot_all_points:
            Points = all_points
        elif len(all_points) == 0:
            # Delete all points
            Points = []
        else:
            point = self.main_win.roi_idx_cmbbx.currentText()
            if point == "":
                return
            Points = [point]

        # Check point list update
        update_point_list = False | update_plot
        rm_lines = []
        for line in self.main_win.plot_line.keys():
            if line not in all_points:
                rm_lines.append(line)
                # if hasattr(self.main_win, 'plot_line_lpf') and \
                #         line in self.main_win.plot_line_lpf:
                #     rm_lines.append(self.main_win.plot_line_lpf[line])

        if len(rm_lines):
            update_point_list = True
            for line in rm_lines:
                if (
                    hasattr(self.main_win, "plot_line")
                    and line in self.main_win.plot_line
                ):
                    del self.main_win.plot_line[line]

                # if hasattr(self.main_win, 'plot_line_lpf') and \
                #         line in self.main_win.plot_line_lpf:
                #     self.main_win.plot_line_lpf[line].remove()
                #     del self.main_win.plot_line_lpf[line]

        # Check color change
        update_color = False
        for point in Points:
            if point not in self.main_win.plot_line:
                continue

            cur_col = self.main_win.plot_line[point].get_color()
            col = pen_color_rgb[self.tracking_mark[point]["pen_color"]]
            if col == "#ffffff":  # white
                col = "#000000"

            if col != cur_col:
                update_color = True
                break

        # -- Update value --
        if update_all_data:
            # Update tracking_point values
            for point in Points:
                self.tracking_point[point].update_all_values()
        elif self.main_win.roi_online_plot_chbx.checkState() != Qt.CheckState.Unchecked:
            # Update current data
            for point_name in Points:
                self.tracking_point[point].get_value(
                    [self.thermalData.frame_position], force_update=True
                )
        elif not update_color:
            self.main_win.roi_plot_canvas.draw()
            return

        # -- Plot line --------------------------------------------------------
        for point in Points:
            col = pen_color_rgb[self.tracking_mark[point]["pen_color"]]
            if col == "#ffffff":  # white
                col = "#000000"
                ls = "--"
            else:
                ls = "-"

            # si = 1.0 / self.tracking_point[point].frequency
            if point not in self.main_win.plot_line:  # or \
                # point not in self.main_win.plot_line_lpf:

                if point in self.main_win.plot_line:
                    self.main_win.plot_line[point].remove()
                    del self.main_win.plot_line[point]

                # if point in self.main_win.plot_line_lpf:
                #     del self.main_win.plot_line_lpf[point]

                # Create lines
                self.main_win.plot_line[point] = self.main_win.plot_ax.plot(
                    self.main_win.plot_xvals,
                    self.tracking_point[point].value_ts,
                    ls,
                    lw=1,
                    color=col,
                    label=point,
                )[0]

                # xi0 = np.argwhere(
                #     np.logical_not(
                #         np.isnan(self.tracking_point[point].value_ts))).ravel()
                # if self.lpf > 0.0 and len(xi0) > 1:
                #     y0 = self.tracking_point[point].value_ts[xi0]
                #     lpf_ts = np.ones(
                #         len(self.tracking_point[point].value_ts)) * np.nan
                #     lpf_ts[np.min(xi0):np.max(xi0)+1] = \
                #         self.InterpLPF(y0, xi0, si, self.lpf)
                # else:
                #     lpf_ts = self.tracking_point[point].value_ts

                # self.main_win.plot_line_lpf[point] = \
                #     self.main_win.plot_ax.plot(
                #         self.main_win.plot_xvals,
                #         lpf_ts, ':', lw=2, color='k')[0]

                # update_point_list = True
            else:
                if update_color:
                    self.main_win.plot_line[point].set_color(col)
                    self.main_win.plot_line[point].set_ls(ls)
                    # if point in self.main_win.plot_line_lpf:
                    #     self.main_win.plot_line_lpf[point].set_color(col)
                else:
                    self.main_win.plot_line[point].set_ydata(
                        self.tracking_point[point].value_ts.copy()
                    )

                    # xi0 = np.argwhere(
                    #     np.logical_not(
                    #         np.isnan(self.tracking_point[point].value_ts))
                    #     ).ravel()
                    # if len(xi0) > 100:
                    #     y0 = self.tracking_point[point].value_ts[xi0]
                    #     lpf_ts = np.ones(
                    #         len(self.tracking_point[point].value_ts)
                    #         ) * np.nan
                    #     lpf_ts[np.min(xi0):np.max(xi0)+1] = \
                    #         self.InterpLPF(y0, xi0, si, self.lpf)
                    #     if point in self.main_win.plot_line_lpf:
                    #         self.main_win.plot_line_lpf[point
                    #                                     ].set_ydata(lpf_ts)

        self.main_win.plot_ax.relim()
        self.main_win.plot_ax.autoscale_view()

        # -- legend --
        if update_color or update_point_list:
            if self.main_win.plot_ax.get_legend() is not None:
                self.main_win.plot_ax.get_legend().remove()

        if len(self.main_win.plot_line):
            self.main_win.plot_ax.legend(bbox_to_anchor=(1, 1), loc="upper left")

        # -- draw --
        self.main_win.roi_plot_canvas.draw()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def InterpLPF(self, y0, xi0, si=1, lpf=None):
        """
        Interpolates a signal in a regular grid with linear interpolation.
        Then, FFT low-pass filter is applied to the interpolated signal.

        Parameters
        ----------
        y0 : array
            Sample values.
        xi0 : int array
            Indices of samples in y0.
        si : float
            Sampling interval (s).
        lpf : float
            Low-pass filtering frequency threshold.

        Returns
        -------
        yo :
            Filtered interpolated array in [min(x0), max(x1)] range.


        """

        xi = np.arange(np.min(xi0), np.max(xi0) + 1)
        y = interpolate.interp1d(xi0, y0, kind="linear")(xi)
        fy = fft(y)
        freq = fftfreq(len(y), d=si)
        fy[np.abs(freq) > lpf] = 0
        yo = ifft(fy).real

        return yo

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def jump_extrema_point(self, minmax="min"):
        if len(self.tracking_point) == 0:
            return

        point_name = self.main_win.roi_idx_cmbbx.currentText()
        if np.all(np.isnan(self.tracking_point[point_name].value_ts)):
            return

        if minmax == "max":
            jumpFrame = np.nanargmax(self.tracking_point[point_name].value_ts)
        elif minmax == "min":
            jumpFrame = np.nanargmin(self.tracking_point[point_name].value_ts)

        self.thermalData.show_frame(jumpFrame)

        # Set position slider
        self.main_win.positionSlider.blockSignals(True)
        self.main_win.positionSlider.setValue(int(self.common_time_ms))
        self.main_win.positionSlider.blockSignals(False)

        # update plot timeline
        self.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def export_roi_data(self, fname=None, realod_data=None, **kwargs):
        if len(self.tracking_point) == 0:
            return

        if fname is None:
            # Set file name
            stdir = self.thermalData.filename.parent
            initial_name = stdir / (
                self.thermalData.filename.stem + "_tracking_points.csv"
            )
            fname, _ = QFileDialog.getSaveFileName(
                self.main_win,
                "Export data filename",
                str(initial_name),
                "csv (*.csv);;all (*.*)",
                None,
            )
            if fname == "":
                return

        ext = Path(fname).suffix
        if ext != ".csv":
            fname += ".csv."

        if realod_data is None:
            # Ask if reload the data
            ret = QMessageBox.question(
                self.main_win,
                "Reload the tempertature values",
                "Reload the tempertature values?",
                QMessageBox.Yes | QMessageBox.No,
                defaultButton=QMessageBox.No,
            )
            if ret == QMessageBox.Yes:
                realod_data = True
            else:
                realod_data = False

        Points = list(self.tracking_point.keys())
        if realod_data:
            for point in Points:
                self.tracking_point[point].update_all_values()

        # Initialize saving data frame
        cols = pd.MultiIndex.from_product([[""], ["time_ms", "marker"]])
        cols = cols.append(
            pd.MultiIndex.from_product([Points, ["x", "y", "radius", "temp"]])
        )
        saveData = pd.DataFrame(columns=cols)
        saveData.index.name = "frame"

        # Time millisec
        frame_per_msec = 1000 / self.thermalData.frame_rate
        saveData[("", "time_ms")] = (
            np.arange(self.thermalData.duration_frame) * frame_per_msec
        )

        # Time marker
        for fridx, val in self.time_marker.items():
            saveData.loc[fridx, ("", "marker")] = val

        # x, y, temp for each marker
        for point in Points:
            saveData.loc[:, (point, "x")] = self.tracking_point[point].x
            saveData.loc[:, (point, "y")] = self.tracking_point[point].y
            saveData.loc[:, (point, "radius")] = self.tracking_point[point].radius
            temp = self.tracking_point[point].value_ts
            saveData.loc[:, (point, "temp")] = temp

            # si = 1.0 / self.tracking_point[point].frequency
            # xi0 = np.argwhere(np.logical_not(np.isnan(temp))).ravel()
            # y0 = temp[xi0]
            # lpf_ts = np.ones(len(temp)) * np.nan
            # lpf_ts[np.min(xi0):np.max(xi0)+1] = \
            #     self.InterpLPF(y0, xi0, si, self.lpf)
            # saveData.loc[:, (point, 'temp_lpf')] = lpf_ts

        # Save as csv
        saveData.to_csv(fname, quoting=csv.QUOTE_NONNUMERIC)

        # Append point properties
        point_property = {}
        for point in self.tracking_point.keys():
            point_property[point] = {
                "aggfunc": self.tracking_point[point].aggfunc,
                "color": self.tracking_mark[point]["pen_color"],
            }

        with open(fname, "r") as fd:
            C = fd.read()

        C = f"# TVT export,{str(point_property)}\n" + C
        with open(fname, "w") as fd:
            fd.write(C)

    # --- DeepLabCut interface ------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dlc_call(self, call, opt=None):
        if call != "batch_run_training":
            # Check if the video is loaded
            if not hasattr(self, "videoData") or not self.videoData.loaded:
                self.main_win.error_MessageBox("No video data is loaded.")
                return

        # Check if the video is loaded
        if not hasattr(self, "videoData") or not self.videoData.loaded:
            self.main_win.error_MessageBox("No video data is loaded.")
            return

        elif not Path(self.videoData.filename).is_file():
            self.main_win.error_MessageBox(
                "Not found the video file," + f" {self.videoData.filename}."
            )
            return

        if call == "boot_dlc_gui":
            self.dlci.boot_dlc_gui()

        elif call == "new_project":
            proj_name = self.videoData.filename.stem
            experimenter_name = "TVT"
            video_files = [str(self.videoData.filename)]
            work_dir = self.videoData.filename.parent
            copy_videos = False
            self.dlci.new_project(
                proj_name, experimenter_name, video_files, work_dir, copy_videos
            )

        elif call == "load_config":
            video_name = Path(self.videoData.filename).stem
            dirs = [
                str(dd)
                for dd in self.videoData.filename.parent.glob(video_name + "*")
                if dd.is_dir()
            ]
            if len(dirs):
                st_dir = sorted(dirs)[-1]
            else:
                st_dir = self.videoData.filename.parent
            conf_file, _ = QFileDialog.getOpenFileName(
                self.main_win,
                "DLC config",
                str(st_dir),
                "config yaml files (config_*.yaml);;yaml (*.yaml)",
                None,
            )

            if conf_file == "":
                return

            self.dlci.config_path = conf_file

        elif call == "edit_config":
            if (
                self.dlci.config_path is None
                or not Path(self.dlci.config_path).is_file()
            ):
                return

            default_values = {}
            # default_values = {'bodyparts': ['LEYE', 'MID', 'REYE', 'NOSE'],
            #                   'dotsize': 6}
            self.dlci.edit_config(
                self.main_win.ui_edit_config, default_values=default_values
            )

        elif call == "extract_frames":
            # Wait message box
            msgBox = self.main_win.waitDialog(
                title="TVT DLC call",
                msgTxt="Extracting training frames. Please wait.",
                modal=True,
                parent=self.main_win,
            )
            msgBox.show()

            self.dlci.extract_frames(edit_gui_fn=self.main_win.ui_edit_config)
            msgBox.close()

        elif call == "label_frames":
            self.dlci.label_frames(edit_gui_fn=self.main_win.ui_edit_config)

        elif call == "check_labels":
            self.dlci.check_labels()

            # message box
            task = self.dlci.get_config()["Task"]
            labImgPath = Path(self.dlci.get_config()["project_path"])
            labImgPath /= Path("labeled-data") / f"{task}_labeled"
            msgBox = QMessageBox(self.main_win)
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText(
                f"Labeled images are saved in\n {labImgPath}\n"
                "Check them with an image viewer."
            )
            msgBox.setWindowTitle("DLC call")
            msgBox.exec()

        elif call == "create_training_dataset":
            self.dlci.create_training_dataset(num_shuffles=1)

        elif call == "train_network":
            self.dlci.train_network(
                proc_type=opt,
                analyze_videos=[self.videoData.filename],
                ui_edit_config=self.main_win.ui_edit_config,
            )

        elif call == "show_training_progress":
            self.dlci.show_training_progress()

        elif call == "kill_training_process":
            self.dlci.kill_training_process()

        elif call == "evaluate_network":
            self.dlci.evaluate_network()

        elif call == "analyze_videos":
            """
            Filename of the analysis result does not discriminate iterations,
            while it has the number of training repaet.
            If you want to update the result with another training iteration
            having the same number of repeat, you need to delete the existing
            files.
            """
            if not self.dlci.check_config_file():
                return

            res_fs = self.dlci.find_analysis_results(self.videoData.filename)
            if len(res_fs):
                # Confirm delete
                confirmMsg = "Overwrite the existing result files?"
                rep = QMessageBox.question(
                    self.main_win,
                    "Confirm delete",
                    confirmMsg,
                    QMessageBox.Yes,
                    QMessageBox.No,
                )
                if rep == QMessageBox.No:
                    return

                # Delte result files
                for ff in res_fs:
                    ff.unlink()

            self.dlci.analyze_videos(self.videoData.filename)

        elif call == "filterpredictions":
            self.dlci.filterpredictions(self.videoData.filename)

        elif call == "plot_trajectories":
            self.dlci.plot_trajectories(self.videoData.filename, filtered=opt)

        elif call == "create_labeled_video":
            self.dlci.create_labeled_video(self.videoData.filename, filtered=opt)

        elif call == "extract_outlier_frames":
            self.dlci.extract_outlier_frames(self.videoData.filename)

        elif call == "refine_labels":
            self.dlci.refine_labels()

        elif call == "merge_datasets":
            self.dlci.merge_datasets()

        elif call == "boot_dlc_gui":
            self.dlci.boot_dlc_gui()

        elif call == "batch_run_training":
            # Select data directry
            if self.DATA_ROOT.is_dir():
                stdir = self.DATA_ROOT.parent
            else:
                stdir = APP_ROOT
            data_dir = QFileDialog.getExistingDirectory(
                self.main_win,
                "Select data directory",
                str(stdir),
                QFileDialog.ShowDirsOnly,
            )
            if data_dir == "":
                return

            # Ask if overwrite
            ret = QMessageBox.question(
                self.main_win,
                "Batch run",
                "Overwrite (re-train) the existing results?",
                QMessageBox.No | QMessageBox.Yes,
            )
            overwrite = ret == QMessageBox.Yes

            self.dlci.batch_run_training(data_dir, overwrite)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_tracking(self, fileName=None, lh_thresh=None, **kwargs):
        if not self.videoData.loaded:
            self.main_win.error_MessageBox("No video data is set.")
            return

        if not self.thermalData.loaded:
            self.main_win.error_MessageBox("No thermal data is set.")
            return

        if not self.on_sync:
            self.main_win.error_MessageBox("Video must be synced to the thermal movie.")
            return

        # --- Load file -------------------------------------------------------
        if fileName is None:
            stdir = self.videoData.filename.parent
            fileName, _ = QFileDialog.getOpenFileName(
                self.main_win,
                "Open tracking file",
                str(stdir),
                "csv files (*.csv)",
                None,
            )
            if fileName == "":
                return

        track_df = pd.read_csv(fileName, header=[1, 2], index_col=0)
        with open(fileName, "r") as fd:
            head = fd.readline()

        if "TVT export" in head:
            cols = [
                col
                for col in track_df.columns
                if len(col[0]) and "Unnamed" not in col[0]
            ]
            cols = pd.MultiIndex.from_tuples(cols)
            point_property = eval(
                ",".join([p for p in head.rstrip().split(",")[1:] if len(p) > 0])
            )
        else:
            cols = track_df.columns
            point_property = {}

        if len(track_df.index) == self.thermalData.duration_frame:
            data_time = "thermo"
        elif len(track_df.index) == self.videoData.duration_frame:
            data_time = "video"
        else:
            errmsg = f"Loaded data length, {len(track_df.index)}"
            errmsg += " does not match either thermal or video frame length."
            self.main_win.error_MessageBox(errmsg)
            return

        PointNames = [
            col
            for col in track_df.columns.levels[0]
            if len(col) and "Unnamed" not in col
        ]

        if "likelihood" in track_df[PointNames[0]].columns:
            if lh_thresh is None:
                max_lh = track_df[PointNames[0]].likelihood.max()
                lh_thresh, ok = QInputDialog.getDouble(
                    self.main_win,
                    "Likelihood",
                    "Likelihood threshold:",
                    value=min(0.95, max_lh),
                    minValue=0.0,
                    maxValue=max_lh,
                    decimals=2,
                )
                if not ok:
                    return

        # --- Read data -------------------------------------------------------
        thermo_times = [
            self.thermalData.get_comtime_from_frame(frmIdx)
            for frmIdx in range(self.thermalData.duration_frame)
        ]
        res_track_df = pd.DataFrame(
            index=np.arange(self.thermalData.duration_frame), columns=cols
        )

        for point in PointNames:
            x = track_df[point].x.values
            y = track_df[point].y.values
            if "likelihood" in track_df[point].columns:
                lh = track_df[point].likelihood.values
            if "temp" in track_df[point].columns:
                temp = track_df[point].temp.values
            if "radius" in track_df[point].columns:
                radius = track_df[point].radius.values

            if data_time == "video":
                # Resample tracking points in thermal video timings
                video_times = [
                    self.videoData.get_comtime_from_frame(frmIdx)
                    for frmIdx in track_df.index
                ]
                interp_x = interpolate.interp1d(video_times, x)
                x = interp_x(thermo_times)

                interp_y = interpolate.interp1d(video_times, y)
                y = interp_y(thermo_times)

                if "likelihood" in track_df[point].columns:
                    interp_lh = interpolate.interp1d(video_times, lh)
                    lh = interp_lh(thermo_times)

                if "temp" in track_df[point].columns:
                    interp_temp = interpolate.interp1d(video_times, temp)
                    temp = interp_temp(thermo_times)

                if "radius" in track_df[point].columns:
                    interp_radius = interpolate.interp1d(video_times, radius)
                    radius = interp_radius(thermo_times).astype(int)
                    radius[radius == 0] = 1

            res_track_df.loc[:, (point, "x")] = x
            res_track_df.loc[:, (point, "y")] = y
            if "likelihood" in track_df[point].columns:
                res_track_df.loc[:, (point, "likelihood")] = lh
            if "temp" in track_df[point].columns:
                res_track_df.loc[:, (point, "temp")] = temp
            if "radius" in track_df[point].columns:
                res_track_df.loc[:, (point, "radius")] = radius

        # --- Set tracking_points ---------------------------------------------
        currentFrm = self.thermalData.frame_position
        for point in PointNames:
            frm_mask = np.ones(len(res_track_df), dtype=bool)
            if "likelihood" in res_track_df[point].columns:
                lh = res_track_df[point].likelihood.values
                frm_mask &= lh >= lh_thresh
            frm_mask &= pd.notnull(res_track_df[point].x).values

            valid_x = res_track_df[point].x.values[frm_mask]
            valid_y = res_track_df[point].y.values[frm_mask]
            valid_frms = np.argwhere(frm_mask).ravel()

            if currentFrm in valid_frms:
                xp = valid_x[np.argwhere(valid_frms == currentFrm).ravel()[0]]
                yp = valid_y[np.argwhere(valid_frms == currentFrm).ravel()[0]]
            else:
                xp = np.nan
                yp = np.nan

            # Delete if point with the same name exists
            if point in self.tracking_mark:
                del self.tracking_mark[point]
            if point in self.tracking_point:
                del self.tracking_point[point]

            # Create a point mark display
            self.tracking_mark[point] = {"x": xp, "y": yp}
            self.main_win.thermalDispImg.tracking_mark = self.tracking_mark
            self.main_win.videoDispImg.tracking_mark = self.tracking_mark

            # Create self.tracking_point[point]
            self.edit_point_signal.emit(point)

            # Reset x, y to the read values
            self.tracking_point[point].x[:] = np.nan
            self.tracking_point[point].y[:] = np.nan
            self.tracking_point[point].value_ts[:] = np.nan
            self.tracking_point[point].set_position(
                valid_x, valid_y, frame_indices=valid_frms
            )

            if "temp" in res_track_df[point].columns:
                self.tracking_point[point].value_ts = res_track_df[point]["temp"].values

            if "radius" in res_track_df[point].columns:
                self.tracking_point[point].radisu = res_track_df[point]["radius"].values

            if point in point_property:
                if (
                    "radius" not in res_track_df[point].columns
                    and "radius" in point_property[point]
                ):
                    self.tracking_point[point].radius[:] = point_property[point][
                        "radius"
                    ]

                self.tracking_point[point].aggfunc = point_property[point]["aggfunc"]
                self.tracking_mark[point]["aggfunc"] = point_property[point]["aggfunc"]
                self.tracking_mark[point]["rad"] = self.tracking_point[point].radius[
                    currentFrm
                ]
                if "color" in point_property[point]:
                    self.tracking_mark[point]["pen_color"] = point_property[point][
                        "color"
                    ]
                self.edit_point_signal.emit(point)

        # --- Read marker -----------------------------------------------------
        if "marker" in track_df.columns.get_level_values(1):
            marker = track_df.iloc[:, 1]
            marker = marker[pd.notnull(marker)]
            if len(marker):
                for frame, val in marker.items():
                    self.time_marker[frame] = val

                self.show_marker()

        # Update thermal data time-series plot
        self.plot_timecourse()

    # --- Save/Load working status --------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_status(self, fname=None, **kwargs):
        # --- Filename setup ---
        if fname is None:
            stdir = self.DATA_ROOT / "work_state"
            video_name = self.videoData.file_path.stem
            if video_name is not None:
                dtstr = datetime.now().strftime("%Y%m%d%H%M")
                stdir = stdir / f"{video_name}_working_state_{dtstr}.pkl"

            if self.loaded_state_f is not None:
                stdir = self.loaded_state_f

            fname, _ = QFileDialog.getSaveFileName(
                self.main_win,
                "Save setting filename",
                str(stdir),
                "pkl (*.pkl);;all (*.*)",
                None,
            )
            if fname == "":
                return

            fname = Path(fname)
            if fname.suffix != ".pkl":
                fname = Path(str(fname) + ".pkl")

            self.loaded_state_f = fname
        else:
            if not fname.parent.is_dir():
                os.makedirs(fname.parent)

        # --- Extract saving parameter values for the model object ---
        settings = {}
        saving_params = [
            "on_sync",
            "time_marker",
            "thermalData",
            "videoData",
            "tracking_point",
            "tracking_mark",
            "dlci",
            "DATA_ROOT",
        ]
        for param in saving_params:
            if not hasattr(self, param):
                continue

            obj = getattr(self, param)
            if hasattr(obj, "get_save_params"):
                obj_params = obj.get_save_params()
                if obj_params is not None:
                    settings[param] = obj_params
                continue

            if type(obj) is dict:
                for k, dobj in obj.items():
                    if param not in settings:
                        settings[param] = {}
                    if hasattr(dobj, "get_save_params"):
                        settings[param][k] = dobj.get_save_params()
                    else:
                        settings[param][k] = dobj
                continue

            if callable(obj):
                continue

            try:
                pickle.dumps(obj)
                settings[param] = obj
                continue
            except Exception:
                errmsg = f"{param} cannot be saved.\n"
                self.main_win.error_MessageBox(errmsg)
                pass

        if self.main_win.roi_idx_cmbbx.count() > 0:
            settings["current_point_name"] = self.main_win.roi_idx_cmbbx.currentText()

        # thermal_clim
        thermal_clim_fix = self.main_win.thermal_clim_fix_chbx.checkState()
        settings["thermal_clim_fix"] = thermal_clim_fix
        if thermal_clim_fix != Qt.CheckState.Unchecked:
            settings["thermal_clim_min"] = self.main_win.thermal_clim_min_spbx.value()
            settings["thermal_clim_max"] = self.main_win.thermal_clim_max_spbx.value()

        # --- Convert Path to relative to DATA_ROOT ---
        def path_to_rel(param):
            if type(param) is dict:
                for k, v in param.items():
                    if k == "DATA_ROOT":
                        param[k] = Path(v)
                    else:
                        param[k] = path_to_rel(v)
            else:
                if isinstance(param, Path):
                    if param.is_relative_to(self.DATA_ROOT):
                        param = PurePath(param.relative_to(self.DATA_ROOT))

            return param

        settings = path_to_rel(settings)

        # --- Save in pickle ---
        with open(fname, "wb") as fd:
            pickle.dump(settings, fd)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def shift_save_setting_fname(self, fname):
        """If fname exists, rename it with incrementing the file number."""

        if fname.is_file():
            fn, save_num = fname.stem.split("-")
            if int(save_num) == self.num_saved_setting_hist - 1:
                fname.unlink()
                return

            save_num = int(save_num) + 1
            mv_fname = fn + f"-{save_num}" + fname.suffix
            mv_fname = fname.parent / mv_fname
            self.shift_save_setting_fname(mv_fname)
            fname.rename(mv_fname)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_status(self, fname=None, **kwargs):
        if fname is None:
            stdir = self.DATA_ROOT / "work_state"
            fname, _ = QFileDialog.getOpenFileName(
                self.main_win, "Open state file", str(stdir), "pickle (*.pkl)", None
            )  # , QFileDialog.DontUseNativeDialog)
            if fname == "":
                return

        with open(fname, "rb") as fd:
            settings = pickle.load(fd)

        if fname != APP_ROOT / "config" / "TVT_last_working_state-0.pkl":
            self.loaded_state_f = Path(fname)

        # Load DATA_ROOT
        if "DATA_ROOT" in settings:
            if Path(settings["DATA_ROOT"]).is_dir():
                self.DATA_ROOT = Path(settings["DATA_ROOT"])
                self.dlci.DATA_ROOT = self.DATA_ROOT
            del settings["DATA_ROOT"]

        # Load thermalData
        if "thermalData" in settings:
            fname = self.DATA_ROOT / settings["thermalData"]["filename"]
            if not fname.is_file():
                self.openThermalFile()
                fname = self.thermalData.filename
                if isinstance(fname, Path):
                    self.DATA_ROOT = self.thermalData.filename.parent

            if isinstance(fname, Path) and fname.is_file():
                self.openThermalFile(fileName=fname)
                frame_position = settings["thermalData"]["frame_position"]
                self.main_win.thermal_clim_fix_chbx.blockSignals(True)
                self.main_win.thermal_clim_fix_chbx.setChecked(False)
                self.main_win.thermal_clim_fix_chbx.blockSignals(False)
                self.thermalData.show_frame(frame_idx=frame_position)

            del settings["thermalData"]

        # thermal_clim
        if "thermal_clim_fix" in settings:
            self.main_win.thermal_clim_fix_chbx.blockSignals(True)
            if hasattr(settings["thermal_clim_fix"], "value"):
                settings["thermal_clim_fix"] = settings["thermal_clim_fix"].value
            self.main_win.thermal_clim_fix_chbx.setChecked(settings["thermal_clim_fix"])
            self.main_win.thermal_clim_fix_chbx.blockSignals(False)
            if settings["thermal_clim_fix"] > 0:
                self.main_win.thermal_clim_min_spbx.blockSignals(True)
                self.main_win.thermal_clim_max_spbx.blockSignals(True)
                self.main_win.thermal_clim_min_spbx.setValue(
                    settings["thermal_clim_min"]
                )
                self.main_win.thermal_clim_max_spbx.setValue(
                    settings["thermal_clim_max"]
                )
                self.main_win.thermal_clim_min_spbx.blockSignals(False)
                self.main_win.thermal_clim_max_spbx.blockSignals(False)
            self.set_thermal_clim()

            del settings["thermal_clim_fix"]
            if "thermal_clim_min" in settings:
                del settings["thermal_clim_min"]
            if "thermal_clim_max" in settings:
                del settings["thermal_clim_max"]

        # Load videoData
        if "videoData" in settings:
            fname = self.DATA_ROOT / settings["videoData"]["filename"]
            if not fname.is_file():
                self.openVideoFile()
            else:
                self.openVideoFile(fileName=fname)

            fname = self.thermalData.filename
            if fname and fname.is_file():
                frame_position = settings["videoData"]["frame_position"]
                self.videoData.show_frame(frame_idx=frame_position)

            del settings["videoData"]

        if "on_sync" in settings:
            self.sync_video_thermal(settings["on_sync"])
            del settings["on_sync"]

        # Load time_marker
        if "time_marker" in settings:
            if self.thermalData.loaded or self.videoData.loaded:
                self.time_marker = settings["time_marker"]
                NFrames = self.thermalData.duration_frame
                frs = np.array(list(self.time_marker.keys()))
                frs = frs[frs >= NFrames]
                for fr in frs:
                    del self.time_marker[fr]

                self.show_marker()
            del settings["time_marker"]

        # Load DLC config
        if "dlci" in settings:
            self.dlci.config_path = self.DATA_ROOT / settings["dlci"]["_config_path"]
            del settings["dlci"]

        # Load tracking_point
        if "tracking_point" in settings:
            for lab, dobj in settings["tracking_point"].items():
                dmovie_fname = str(self.DATA_ROOT / dobj["dataMovie.filename"])
                if (
                    dmovie_fname == str(self.thermalData.filename)
                    and self.thermalData.loaded
                ):
                    dataMovie = self.thermalData
                elif (
                    dmovie_fname == str(self.videoData.filename)
                    and self.videoData.loaded
                ):
                    dataMovie = self.videoData
                else:
                    if dmovie_fname.suffix == ".csq":
                        dataMovie = self.thermalData
                    else:
                        dataMovie = self.videoData

                self.tracking_point[lab] = TrackingPoint(dataMovie)
                for k, v in dobj.items():
                    if hasattr(self.tracking_point[lab], k):
                        setattr(self.tracking_point[lab], k, v)
                        if k == "radius" and type(v) is int:
                            self.tracking_point[lab].radius = (
                                np.ones_like(self.tracking_point[lab].x, dtype=int) * v
                            )

            if len(self.tracking_point):
                for point_name, tm in settings["tracking_mark"].items():
                    if point_name in self.tracking_point:
                        self.tracking_mark[point_name] = tm

                    self.main_win.thermalDispImg.tracking_mark = self.tracking_mark
                    self.main_win.videoDispImg.tracking_mark = self.tracking_mark

                if "tracking_mark" in settings:
                    del settings["tracking_mark"]

                if "current_point_name" in settings:
                    point_name = settings["current_point_name"]
                else:
                    point_name = list(self.tracking_point.keys())[0]

                self.edit_tracking_point(point_name)
                if "current_point_name" in settings:
                    del settings["current_point_name"]

            if "tracking_point" in settings:
                del settings["tracking_point"]

        # Set lpf
        # if 'lpf' in settings:
        #     self.lpf = settings['lpf']
        #     self.main_win.roi_LPF_thresh_spbx.setValue(self.lpf)
        #     del settings['lpf']

        # Load other parameters
        for param, obj in settings.items():
            if hasattr(self, param):
                setattr(self, param, obj)

        gc.collect()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_tmp_status(self, timer=True):
        if self.thermalData.file_path is not None:
            video_name = self.thermalData.file_path.stem
            save_f = self.DATA_ROOT / "work_state" / f"{video_name}_working_state.pkl"
            self.save_status(fname=save_f)

        if timer:
            self.save_timer.start(self.save_tmp_wait * 1000)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_data_root(self, data_dir=None):
        if data_dir is None:
            if self.DATA_ROOT.is_dir():
                stdir = self.DATA_ROOT.parent
            else:
                stdir = APP_ROOT
            data_dir = QFileDialog.getExistingDirectory(
                self.main_win,
                "Select data directory",
                str(stdir),
                QFileDialog.ShowDirsOnly,
            )
            if data_dir == "":
                return

        self.DATA_ROOT = Path(data_dir)
        self.dlci.DATA_ROOT = self.DATA_ROOT

        conf = {"DATA_ROOT": str(self.DATA_ROOT)}
        with open(self.conf_f, "w") as fd:
            json.dump(conf, fd)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_focus(self):
        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def focus_filter(self):
        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def remove_temprature_outlier(self):
        pass


# %% View class : MainWindow ==================================================
class MainWindow(QMainWindow):
    """View class : MainWindow"""

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, parent=None, batchmode=False, extract_temp_file=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Thermal Video Tracking")

        # Create main_win objects
        # main_win objects is refered in the model init.
        # This must be made before making the model object.
        self.init_ui_objects()

        # Init the model class
        self.model = ThermalVideoModel(
            main_win=self, batchmode=batchmode, extract_temp_file=extract_temp_file
        )

        # Connect signals
        self.connect_signal_handlers()

        # Layout
        self.set_layout()

        # Set menu
        self.make_menu_objects()

        self.setFocusPolicy(Qt.StrongFocus)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def init_ui_objects(self):
        """Initialize UI objects"""

        # --- Thermal image widget --------------------------------------------
        # Load thermal data button
        self.loadThermalDataBtn = QPushButton("Load thermal data")
        self.loadThermalDataBtn.setStyleSheet("background:#8ED5EC; color:black;")
        # Unload thermal data button
        self.unloadThermalDataBtn = QPushButton("Unload")
        self.unloadThermalDataBtn.setEnabled(False)
        self.unloadThermalDataBtn.setStyleSheet("background:#FFF3F0; color:black;")
        # Export thermal data as video data button
        self.exportThermalDataVideoBtn = QPushButton("Export as video")
        self.exportThermalDataVideoBtn.setEnabled(False)
        self.exportThermalDataVideoBtn.setStyleSheet("background:#F7F0A8; color:black;")

        # Thermal display image
        self.thermalDispImg = TVTDisplayImage(self, cmap=themro_cmap)

        # Marker label
        self.thermalMakerLab = QLabel()
        self.thermalMakerLab.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.thermalMakerLab.setStyleSheet("background:black; color:white;")

        # Thermal position text
        self.thermalPositionLab = QLabel("00:00/00:00 [0/0 frames]")
        self.thermalPositionLab.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Maximum
        )

        # thermal skip forward button
        self.thermalSkipFwdBtn = QPushButton()
        self.thermalSkipFwdBtn.setEnabled(False)
        self.thermalSkipFwdBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSkipForward)
        )

        # skip backward button
        self.thermalSkipBkwBtn = QPushButton()
        self.thermalSkipBkwBtn.setEnabled(False)
        self.thermalSkipBkwBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSkipBackward)
        )

        # thermal frame backward button
        self.thermalFrameBkwBtn = QPushButton()
        self.thermalFrameBkwBtn.setEnabled(False)
        self.thermalFrameBkwBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSeekBackward)
        )

        # thermal frame forward button
        self.thermalFrameFwdBtn = QPushButton()
        self.thermalFrameFwdBtn.setEnabled(False)
        self.thermalFrameFwdBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSeekForward)
        )

        # Frame position
        self.thermalFramePosLab = QLabel("frame:")
        self.thermalFramePosSpBox = QSpinBox()
        self.thermalFramePosLab.setEnabled(False)
        self.thermalFramePosSpBox.setEnabled(False)

        # --- Video image widget ----------------------------------------------
        # Load video data button
        self.loadVideoDataBtn = QPushButton("Load video data")
        self.loadVideoDataBtn.setStyleSheet("background:#8ED5EC; color:black;")

        # Unload video data button
        self.unloadVideoDataBtn = QPushButton("Unload")
        self.unloadVideoDataBtn.setEnabled(False)
        self.unloadVideoDataBtn.setStyleSheet("background:#FFF3F0; color:black;")

        # Video display image
        self.videoDispImg = TVTDisplayImage(self)

        # Marker label
        self.videoMakerLab = QLabel()
        self.videoMakerLab.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.videoMakerLab.setStyleSheet("background:black; color:white;")

        # Video position text
        self.videoPositionLab = QLabel("00:00/00:00 [0/0 frames] | Thermal 00:00")
        self.videoPositionLab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Video skip forward button
        self.videoSkipFwdBtn = QPushButton()
        self.videoSkipFwdBtn.setEnabled(False)
        self.videoSkipFwdBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSkipForward)
        )

        # Video skip backward button
        self.videoSkipBkwBtn = QPushButton()
        self.videoSkipBkwBtn.setEnabled(False)
        self.videoSkipBkwBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSkipBackward)
        )

        # Video frame forward button
        self.videoFrameFwdBtn = QPushButton()
        self.videoFrameFwdBtn.setEnabled(False)
        self.videoFrameFwdBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSeekForward)
        )

        # Video frame backward button
        self.videoFrameBkwBtn = QPushButton()
        self.videoFrameBkwBtn.setEnabled(False)
        self.videoFrameBkwBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSeekBackward)
        )

        # Frame position
        self.videoFramePosLab = QLabel("frame:")
        self.videoFramePosSpBox = QSpinBox()
        self.videoFramePosLab.setEnabled(False)
        self.videoFramePosSpBox.setEnabled(False)

        # synch button
        self.syncVideoBtn = QPushButton("Sync video to thermal")
        self.syncVideoBtn.setCheckable(True)
        self.syncVideoBtn.setEnabled(False)

        # --- Common control widget -------------------------------------------
        # play button
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        # common skip forward button
        self.commonSkipFwdBtn = QPushButton()
        self.commonSkipFwdBtn.setEnabled(False)
        self.commonSkipFwdBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSkipForward)
        )

        # common skip backward button
        self.commonSkipBkwBtn = QPushButton()
        self.commonSkipBkwBtn.setEnabled(False)
        self.commonSkipBkwBtn.setIcon(
            self.style().standardIcon(QStyle.SP_MediaSkipBackward)
        )

        # position slider
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.setEnabled(False)

        # Common position text
        self.commonPosisionLab = QLabel("00:00.000/00:00.000")
        self.commonPosisionLab.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.commonPosisionLab.setEnabled(False)

        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # --- Thermal cmap edit widget ----------------------------------------
        self.cmap_grpbx = QGroupBox("Color")
        self.thermal_cbar_lab = QLabel()
        colorbar = np.reshape(np.arange(0, 256, dtype=np.uint8), [-1, 1])
        colorbar = np.tile(colorbar, 10).copy()
        frame = cv2.applyColorMap(colorbar, themro_cmap)
        bytesPerLine = 3 * colorbar.shape[1]
        qimg = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            bytesPerLine,
            QImage.Format_RGB888,
        )
        self.cbar_pix = QPixmap.fromImage(qimg)

        self.thermal_clim_max_spbx = QDoubleSpinBox()
        self.thermal_clim_max_spbx.setSingleStep(0.1)
        self.thermal_clim_min_spbx = QDoubleSpinBox()
        self.thermal_clim_min_spbx.setSingleStep(0.1)
        self.thermal_clim_fix_chbx = QCheckBox("fix")

        # --- Marker widgets --------------------------------------------------
        self.tmark_grpbx = QGroupBox("Time marker")
        self.tmark_add_btn = QPushButton("Add")
        self.tmark_name_cmbbx = QComboBox()
        self.tmark_name_cmbbx.setEditable(True)
        self.tmark_del_btn = QPushButton("Delete")
        self.tmark_jumpNext_btn = QPushButton("Jump next")
        self.tmark_jumpPrev_btn = QPushButton("Jump previous")

        # --- Tracking point edit widgets -------------------------------------
        self.roi_ctrl_grpbx = QGroupBox("Tracking points (shift+double-click to add)")
        self.roi_ctrl_grpbx.setEnabled(False)

        self.roi_idx_cmbbx = QComboBox()
        self.roi_idx_cmbbx.setEditable(False)

        self.roi_name_ledit = QLineEdit()
        self.roi_showName_chbx = QCheckBox("Show name")
        self.roi_showName_chbx.setChecked(True)

        self.roi_x_spbx = QSpinBox()
        self.roi_x_spbx.setMinimum(-1)

        self.roi_y_spbx = QSpinBox()
        self.roi_y_spbx.setMinimum(-1)
        self.roi_erase_btn = QPushButton("Erase")

        self.roi_rad_spbx = QSpinBox()
        self.roi_rad_spbx.setMinimum(1)
        self.roi_rad_spbx.setMaximum(10000)

        self.roi_rad_applyAll_btn = QPushButton("Apply all")

        self.roi_editRange_cmbbx = QComboBox()
        self.roi_editRange_cmbbx.setEditable(False)
        self.roi_editRange_cmbbx.addItems(
            [
                "Current",
                "PrevMark -> Current",
                "Current -> NextMark",
                "0 -> Current",
                "Current -> End",
                "All",
            ]
        )

        self.roi_color_cmbbx = QComboBox()
        self.roi_color_cmbbx.setEditable(False)
        self.roi_color_cmbbx.addItems(qt_global_colors)

        self.roi_aggfunc_cmbbx = QComboBox()
        self.roi_aggfunc_cmbbx.setEditable(False)
        self.roi_aggfunc_cmbbx.addItems(Aggfuncs)

        self.roi_val_lab = QLabel("Temp. {:.2f} °C".format(0))
        self.roi_jump_min_btn = QPushButton("to min")
        self.roi_jump_max_btn = QPushButton("to max")

        self.roi_online_plot_chbx = QCheckBox("Online plot")
        self.roi_online_plot_chbx.setChecked(True)
        self.roi_plot_btn = QPushButton("Plot all")
        self.roi_plot_btn.setStyleSheet("background:#7fbfff; color:black;")

        # self.roi_LPF_lb = QLabel('Low-pass filter (Hz)')
        # self.roi_LPF_thresh_spbx = QDoubleSpinBox()
        # self.roi_LPF_thresh_spbx.setDecimals(5)
        # self.roi_LPF_thresh_spbx.setSingleStep(0.001)
        # self.roi_LPF_thresh_spbx.setValue(0.0)
        # self.roi_LPF_thresh_spbx.setMinimum(0.0)

        self.roi_delete_btn = QPushButton("Delete this point")
        self.roi_delete_btn.setFixedHeight(18)
        self.roi_delete_btn.setStyleSheet("background:#ff7f7f; color:black;")

        self.roi_load_btn = QPushButton("Load tracking data")
        self.roi_load_btn.setStyleSheet("background:#7fbfff; color:black;")

        self.roi_export_btn = QPushButton("Export tracking data")
        self.roi_export_btn.setStyleSheet("background:#7fffbf; color:black;")
        self.roi_export_btn.setEnabled(False)

        # --- Plot panel ------------------------------------------------------
        self.roi_plot_canvas = FigureCanvas(Figure())
        self.roi_plot_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.plot_ax = self.roi_plot_canvas.figure.subplots(1, 1)
        self.roi_plot_canvas.figure.subplots_adjust(
            left=0.08, bottom=0.24, right=0.938, top=0.94
        )
        self.roi_plot_canvas.start_event_loop(0.005)
        self.plot_xvals = None
        self.plot_line = {}
        # self.plot_line_lpf = {}
        self.plot_timeline = None
        self.plot_marker_line = {}
        self.roi_plot_canvas.setEnabled(False)

        # Create the navigation toolbar and add it to the layout
        self.toolbar = NavigationToolbar2QT(self.roi_plot_canvas, self)

        # --- error label ---
        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def connect_signal_handlers(self):
        """Connect signal handlers for the UI objects
        movie control buttons (*FwdBtn, *BkwBtn, ect.) are connected at
        initializing the model class.
        """

        # Thermal load/unload
        self.loadThermalDataBtn.clicked.connect(self.model.openThermalFile)
        self.unloadThermalDataBtn.clicked.connect(self.model.unloadThermalData)
        self.exportThermalDataVideoBtn.clicked.connect(
            self.model.exportThermalDataVideo
        )

        # Video load/unload
        self.loadVideoDataBtn.clicked.connect(self.model.openVideoFile)
        self.unloadVideoDataBtn.clicked.connect(self.model.unloadVideoData)
        self.syncVideoBtn.clicked.connect(self.model.sync_video_thermal)

        # Thermal clim
        self.thermal_clim_max_spbx.editingFinished.connect(self.model.set_thermal_clim)
        self.thermal_clim_min_spbx.editingFinished.connect(self.model.set_thermal_clim)
        self.thermal_clim_fix_chbx.stateChanged.connect(self.model.set_thermal_clim)

        # Common time controls
        self.playBtn.clicked.connect(self.model.play)
        self.commonSkipFwdBtn.clicked.connect(self.model.commonSkipFwd)
        self.commonSkipBkwBtn.clicked.connect(self.model.commonSkipBkw)
        self.positionSlider.sliderReleased.connect(self.model.set_common_time)

        # Marker
        self.tmark_add_btn.clicked.connect(self.model.add_marker)
        self.tmark_del_btn.clicked.connect(self.model.del_marker)
        self.tmark_jumpNext_btn.clicked.connect(lambda: self.model.jump_marker(1))
        self.tmark_jumpPrev_btn.clicked.connect(lambda: self.model.jump_marker(-1))

        # Tracking point controls
        self.roi_idx_cmbbx.currentTextChanged.connect(self.model.select_point_ui)
        self.roi_name_ledit.returnPressed.connect(self.model.edit_point_property)
        self.roi_showName_chbx.stateChanged.connect(self.model.edit_point_property)
        self.roi_x_spbx.valueChanged.connect(self.model.edit_point_property)
        self.roi_y_spbx.valueChanged.connect(self.model.edit_point_property)
        self.roi_rad_spbx.valueChanged.connect(self.model.edit_point_property)
        self.roi_rad_applyAll_btn.clicked.connect(self.model.apply_radius_all)
        self.roi_editRange_cmbbx.currentIndexChanged.connect(self.model.set_editRange)
        self.roi_color_cmbbx.currentIndexChanged.connect(self.model.edit_point_property)
        self.roi_aggfunc_cmbbx.currentIndexChanged.connect(
            self.model.edit_point_property
        )
        self.roi_erase_btn.clicked.connect(self.model.erase_point)
        self.roi_delete_btn.clicked.connect(
            lambda: self.model.delete_point(point_name=None, ask_confirm=True)
        )
        self.roi_plot_btn.clicked.connect(
            partial(self.model.plot_timecourse, update_all_data=True)
        )
        self.roi_jump_min_btn.clicked.connect(
            partial(self.model.jump_extrema_point, minmax="min")
        )
        self.roi_jump_max_btn.clicked.connect(
            partial(self.model.jump_extrema_point, minmax="max")
        )

        # self.roi_LPF_thresh_spbx.valueChanged.connect(
        #     partial(self.model.plot_timecourse,
        #             update_plot=True))

        self.roi_load_btn.clicked.connect(
            partial(self.model.load_tracking, fileName=None)
        )
        self.roi_export_btn.clicked.connect(lambda state: self.model.export_roi_data())

        # Time-course plot
        self.roi_plot_canvas.mpl_connect("button_press_event", self.model.plot_onclick)

    # -------------------------------------------------------------------------
    def set_layout(self):

        # --- Layout each panel -----------------------------------------------
        # --- Time marker layout ---
        self.tmark_grpbx.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.tmark_grpbx.setFixedHeight(200)
        tmarkLayout = QGridLayout(self.tmark_grpbx)
        tmarkLayout.addWidget(QLabel("Name:"), 0, 0)
        tmarkLayout.addWidget(self.tmark_name_cmbbx, 0, 1)
        tmarkLayout.addWidget(self.tmark_add_btn, 1, 0, 1, 2)
        tmarkLayout.addWidget(self.tmark_del_btn, 2, 0, 1, 2)
        tmarkLayout.addWidget(self.tmark_jumpNext_btn, 3, 0, 1, 2)
        tmarkLayout.addWidget(self.tmark_jumpPrev_btn, 4, 0, 1, 2)

        # --- Thermal color map layout ---
        self.cmap_grpbx.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.cmap_grpbx.setMaximumWidth(75)
        cmapLayout = QVBoxLayout(self.cmap_grpbx)
        cmapLayout.setContentsMargins(0, 0, 0, 0)
        cmapLayout.addWidget(self.thermal_clim_max_spbx)
        self.thermal_cbar_lab.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Ignored)
        self.thermal_cbar_lab.setMaximumWidth(20)
        cmapLayout.addWidget(self.thermal_cbar_lab)
        cmapLayout.addWidget(self.thermal_clim_min_spbx)
        cmapLayout.addWidget(self.thermal_clim_fix_chbx)
        self.thermal_cbar_lab.resizeEvent = self.thermal_cbar_lab_resizeEvent

        # --- Tracking point control layout ---
        self.roi_ctrl_grpbx.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        roiCtrlLayout = QGridLayout(self.roi_ctrl_grpbx)
        roiCtrlLayout.addWidget(QLabel("Point:"), 0, 0)
        roiCtrlLayout.addWidget(self.roi_idx_cmbbx, 0, 1, 1, 2)
        roiCtrlLayout.addWidget(QLabel("Name:"), 1, 0)
        roiCtrlLayout.addWidget(self.roi_name_ledit, 1, 1, 1, 1)
        roiCtrlLayout.addWidget(self.roi_showName_chbx, 1, 2, 1, 1)
        roiCtrlLayout.addWidget(QLabel("Color:"), 2, 0)
        roiCtrlLayout.addWidget(self.roi_color_cmbbx, 2, 1, 1, 2)
        roiCtrlLayout.addWidget(QLabel("Edit range:"), 3, 0)
        roiCtrlLayout.addWidget(self.roi_editRange_cmbbx, 3, 1, 1, 2)
        roiCtrlLayout.addWidget(QLabel("x:"), 4, 0)
        roiCtrlLayout.addWidget(self.roi_x_spbx, 4, 1)
        roiCtrlLayout.addWidget(QLabel("y:"), 5, 0)
        roiCtrlLayout.addWidget(self.roi_y_spbx, 5, 1)
        roiCtrlLayout.addWidget(self.roi_erase_btn, 5, 2)
        roiCtrlLayout.addWidget(QLabel("Radius:"), 6, 0)
        roiCtrlLayout.addWidget(self.roi_rad_spbx, 6, 1)
        roiCtrlLayout.addWidget(self.roi_rad_applyAll_btn, 6, 2)
        roiCtrlLayout.addWidget(QLabel("Aggregation:"), 7, 0)
        roiCtrlLayout.addWidget(self.roi_aggfunc_cmbbx, 7, 1, 1, 2)
        roiCtrlLayout.addWidget(self.roi_val_lab, 8, 0, 1, 1)
        roiCtrlLayout.addWidget(self.roi_jump_min_btn, 8, 1, 1, 1)
        roiCtrlLayout.addWidget(self.roi_jump_max_btn, 8, 2, 1, 1)
        roiCtrlLayout.addWidget(self.roi_online_plot_chbx, 9, 0, 1, 2)
        roiCtrlLayout.addWidget(self.roi_plot_btn, 9, 2, 1, 1)
        # roiCtrlLayout.addWidget(self.roi_LPF_lb, 10, 0, 1, 2)
        # roiCtrlLayout.addWidget(self.roi_LPF_thresh_spbx, 10, 2, 1, 1)
        roiCtrlLayout.addWidget(self.roi_delete_btn, 11, 2, 1, 1)
        self.roi_ctrl_grpbx.resize(self.roi_ctrl_grpbx.sizeHint())

        # --- Place the frames ---
        thermalCtrlFrame = QFrame()
        thermalCtrlLayout = QVBoxLayout(thermalCtrlFrame)

        thermalCtrlUpperLayout = QHBoxLayout()
        thermalCtrlUpperLayout.addWidget(self.tmark_grpbx, alignment=Qt.AlignTop)
        thermalCtrlUpperLayout.addWidget(self.cmap_grpbx)
        thermalCtrlLayout.addLayout(thermalCtrlUpperLayout)
        thermalCtrlLayout.addWidget(self.roi_ctrl_grpbx)

        thermalCtrlLayout.addWidget(self.roi_load_btn)
        thermalCtrlLayout.addWidget(self.roi_export_btn)

        # --- Thermal image widgets layout ---
        thermalFrame = QFrame()
        thermalFrame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        thermalLayout = QVBoxLayout(thermalFrame)
        thermalLayout.setContentsMargins(0, 0, 0, 0)

        thermalOpenLayout = QHBoxLayout()
        thermalOpenLayout.addWidget(self.loadThermalDataBtn)
        thermalOpenLayout.addWidget(self.unloadThermalDataBtn)
        thermalOpenLayout.addWidget(self.exportThermalDataVideoBtn)
        thermalLayout.addLayout(thermalOpenLayout)

        thermalLayout.addWidget(self.thermalDispImg)
        self.thermalMakerLab.setFixedHeight(15)
        self.thermalMakerLab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        thermalLayout.addWidget(self.thermalMakerLab)
        thermalLayout.addWidget(self.thermalPositionLab)

        thermalCtrlLayout = QHBoxLayout()
        thermalCtrlLayout.addStretch()
        thermalCtrlLayout.addWidget(self.thermalSkipBkwBtn)
        thermalCtrlLayout.addWidget(self.thermalFrameBkwBtn)
        thermalCtrlLayout.addWidget(self.thermalFrameFwdBtn)
        thermalCtrlLayout.addWidget(self.thermalSkipFwdBtn)
        thermalCtrlLayout.addWidget(self.thermalFramePosLab)
        thermalCtrlLayout.addWidget(self.thermalFramePosSpBox)
        thermalCtrlLayout.addStretch()
        thermalLayout.addLayout(thermalCtrlLayout)

        # --- Video image widgets layout ---
        videoFrame = QFrame()
        videoFrame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        videoLayout = QVBoxLayout(videoFrame)
        videoLayout.setContentsMargins(0, 0, 0, 0)

        videoOpenLayout = QHBoxLayout()
        videoOpenLayout.addWidget(self.loadVideoDataBtn)
        videoOpenLayout.addWidget(self.unloadVideoDataBtn)
        videoLayout.addLayout(videoOpenLayout)

        videoLayout.addWidget(self.videoDispImg)
        self.videoMakerLab.setFixedHeight(15)
        self.videoMakerLab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        videoLayout.addWidget(self.videoMakerLab)
        videoLayout.addWidget(self.videoPositionLab)

        videoCtrlLayout = QHBoxLayout()
        videoCtrlLayout.addWidget(self.syncVideoBtn)
        videoCtrlLayout.addStretch()
        videoCtrlLayout.addWidget(self.videoSkipBkwBtn)
        videoCtrlLayout.addWidget(self.videoFrameBkwBtn)
        videoCtrlLayout.addWidget(self.videoFrameFwdBtn)
        videoCtrlLayout.addWidget(self.videoSkipFwdBtn)
        videoCtrlLayout.addWidget(self.videoFramePosLab)
        videoCtrlLayout.addWidget(self.videoFramePosSpBox)
        videoCtrlLayout.addStretch()
        videoLayout.addLayout(videoCtrlLayout)

        # --- Video and thermal images layouts ---
        imageSplitter = QSplitter(Qt.Horizontal)
        imageSplitter.addWidget(thermalCtrlFrame)
        imageSplitter.addWidget(thermalFrame)
        imageSplitter.addWidget(videoFrame)
        imageSplitter.setSizes([240, 640, 640])

        # --- Common movie control layout ---
        controlLayout = QHBoxLayout()
        controlLayout.addWidget(self.commonSkipBkwBtn)
        controlLayout.addWidget(self.playBtn)
        controlLayout.addWidget(self.commonSkipFwdBtn)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(self.commonPosisionLab)

        # --- Layout all ------------------------------------------------------
        upperWidgets = QWidget(self)
        upperLayout = QVBoxLayout(upperWidgets)
        upperLayout.addWidget(imageSplitter)
        upperLayout.addLayout(controlLayout)

        # layout.addWidget(self.errorLabel)

        # --- Create a central (base) widget for window contents ---
        centWid = QSplitter(Qt.Vertical)
        centWid.setStyleSheet("QSplitter::handle {background-color: #eaeaea;}")
        centWid.addWidget(upperWidgets)

        # --- Add roi_plot_canvas ---
        centWid.addWidget(self.roi_plot_canvas)
        centWid.addWidget(self.toolbar)
        self.roi_plot_canvas.adjustSize()
        self.errorLabel.setFixedHeight(15)
        centWid.addWidget(self.errorLabel)

        self.setCentralWidget(centWid)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def make_menu_objects(self):
        # --- Menu bar ---
        menuBar = self.menuBar()

        # -- File menu --
        fileMenu = menuBar.addMenu("&File")

        # Load
        loadSettingAction = QAction("&Load woking state", self)
        loadSettingAction.setShortcut("Ctrl+L")
        loadSettingAction.setStatusTip("Load woking state")
        loadSettingAction.triggered.connect(partial(self.model.load_status, fname=None))
        fileMenu.addAction(loadSettingAction)

        # Save
        saveSettingAction = QAction("&Save woking state", self)
        saveSettingAction.setShortcut("Ctrl+S")
        saveSettingAction.setStatusTip("Save woking state")
        saveSettingAction.triggered.connect(partial(self.model.save_status, fname=None))
        fileMenu.addAction(saveSettingAction)

        # Set DATA_ROOT
        setDataRootAction = QAction("&Set data root", self)
        setDataRootAction.setShortcut("Ctrl+D")
        setDataRootAction.setStatusTip("Load woking state")
        setDataRootAction.triggered.connect(
            partial(self.model.set_data_root, data_dir=None)
        )
        fileMenu.addAction(setDataRootAction)

        # Exit
        exitAction = QAction("&Exit", self)
        exitAction.setShortcut("Ctrl+Q")
        exitAction.setStatusTip("Exit application")
        exitAction.triggered.connect(self.exitCall)
        fileMenu.addAction(exitAction)

        # -- DLC menu --
        dlcMenu = menuBar.addMenu("&DLC")

        # -- I --
        action = QAction("New project", self)
        action.setStatusTip("Create a new DeepLabCut project")
        action.triggered.connect(partial(self.model.dlc_call, "new_project"))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction("Load config", self)
        action.setStatusTip("Load existing DeepLabCut configuraton")
        action.triggered.connect(partial(self.model.dlc_call, "load_config"))
        dlcMenu.addAction(action)

        # -- II --
        action = QAction("Edit configuration", self)
        action.setStatusTip("Edit DeepLabCut configuration")
        action.triggered.connect(partial(self.model.dlc_call, "edit_config"))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        # -- Boot deeplabcut GUI ---
        action = QAction("deeplabcut GUI", self)
        action.setStatusTip("Boot deeplabcut GUI application")
        action.triggered.connect(partial(self.model.dlc_call, "boot_dlc_gui"))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction("Make a training script", self)
        action.setStatusTip("Make a command script for DeepLabCut network training")
        action.triggered.connect(
            partial(self.model.dlc_call, "train_network", "prepare_script")
        )
        dlcMenu.addAction(action)

        action = QAction("Run training backgroud", self)
        action.setStatusTip(
            "Create a command script for DeepLabCut network training and run"
            + " it in the background"
        )
        action.triggered.connect(
            partial(self.model.dlc_call, "train_network", "run_subprocess")
        )
        dlcMenu.addAction(action)

        action = QAction("Show training progress", self)
        action.setStatusTip(
            "Show the progress of the training running in the background."
        )
        action.triggered.connect(partial(self.model.dlc_call, "show_training_progress"))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction("Kill training process", self)
        action.setStatusTip(
            "Show the progress of the training running in the background."
        )
        action.triggered.connect(partial(self.model.dlc_call, "kill_training_process"))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction("Analyze video", self)
        action.setStatusTip("Analyze video by DeepLabCut")
        action.triggered.connect(partial(self.model.dlc_call, "analyze_videos"))
        dlcMenu.addAction(action)

        action = QAction("Filter prediction", self)
        action.setStatusTip("Filter prediction by DeepLabCut")
        action.triggered.connect(partial(self.model.dlc_call, "filterpredictions"))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction("Run all training scripts in batch mode", self)
        action.setStatusTip(
            "Run all training scripts in a data directory sequentially."
        )
        action.triggered.connect(partial(self.model.dlc_call, "batch_run_training"))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        # -- XI --
        action = QAction("Load tracking positions", self)
        action.setStatusTip("Load positions tracked by DeepLabCut")
        action.triggered.connect(self.model.load_tracking)
        dlcMenu.addAction(action)

        # -- Filter menu --
        filterMenu = menuBar.addMenu("Filter")

        # Show Focus level
        showFocusAction = QAction("Show Image Focus Level", self)
        showFocusAction.setStatusTip("Show blurring level")
        showFocusAction.triggered.connect(self.model.show_focus)
        filterMenu.addAction(showFocusAction)

        # Filter Focus
        focusFilterAction = QAction("Image Focus filter", self)
        focusFilterAction.setStatusTip("Filtering blurred frames")
        focusFilterAction.triggered.connect(self.model.focus_filter)
        filterMenu.addAction(focusFilterAction)

        # Filter Temprature outlier
        tempOutlierFilterAction = QAction("Temperature outlier", self)
        tempOutlierFilterAction.setStatusTip("Remove temperature outliers")
        tempOutlierFilterAction.triggered.connect(self.model.remove_temprature_outlier)
        filterMenu.addAction(tempOutlierFilterAction)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def thermal_cbar_lab_resizeEvent(self, ev):
        width = self.thermal_cbar_lab.width()
        height = self.thermal_cbar_lab.height()
        pix = self.cbar_pix.scaled(width, height, Qt.IgnoreAspectRatio)
        self.thermal_cbar_lab.setPixmap(pix)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_edit_config(self, config_data, title="Edit DLC configuration"):

        # Dialog to set proj_name, experimenter_name, work_dir, copy_videos

        class EditConfigDlg(QDialog):
            def __init__(self, parent, config_data):
                super(EditConfigDlg, self).__init__(parent)
                self.setWindowTitle(title)

                # root layout
                vbox = QVBoxLayout(self)

                # Place ui
                self.ui_ctrls = {}
                for k, val in config_data.items():
                    self.place_ui(self, k, val, vbox)

                # OK, Cancel button
                self.buttons = QDialogButtonBox(
                    QDialogButtonBox.Ok | QDialogButtonBox.Cancel, Qt.Horizontal, self
                )
                vbox.addWidget(self.buttons)
                self.buttons.accepted.connect(self.accept)
                self.buttons.rejected.connect(self.reject)
                self.setLayout(vbox)
                self.resize(400, 200)

            def place_ui(self, parent, label, value, layout):
                if label == "video_sets":
                    # video_sets is set by dlci.add_video
                    return

                hbox = QHBoxLayout()
                hbox.addWidget(QLabel(f"{label}:"))
                self.ui_ctrls[label] = QLineEdit(str(value))
                hbox.addWidget(self.ui_ctrls[label])
                layout.addLayout(hbox)

        dlg = EditConfigDlg(self, config_data)
        res = dlg.exec()
        if res == 0:
            return None

        for k in config_data.keys():
            if k in dlg.ui_ctrls:
                orig_val = config_data[k]
                ed_val = dlg.ui_ctrls[k].text()
                if type(orig_val) is str:
                    config_data[k] = ed_val
                else:
                    config_data[k] = eval(ed_val)

        return config_data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def msg_dlg(self, msg, title="ThermalVideoTracking"):
        msgBox = QMessageBox(self)
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(msg)
        msgBox.setWindowTitle(title)
        msgBox.setStyleSheet("QLabel{max-height:720 px;}")
        msgBox.exec()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def error_MessageBox(self, errmsg, title="Error in ThermalVideoTracking"):
        msgBox = QMessageBox(self)
        msgBox.setWindowModality(Qt.WindowModal)
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(errmsg)
        msgBox.setWindowTitle(title)
        msgBox.exec()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class waitDialog(QDialog):
        def __init__(self, title="TVT", msgTxt="", modal=True, parent=None):
            super().__init__(parent)

            self.setWindowTitle(title)
            self.setModal(modal)
            vBoxLayout = QVBoxLayout(self)

            # message text
            self.msgTxt = QLabel(msgTxt)
            vBoxLayout.addWidget(self.msgTxt)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keyPressEvent(self, event):
        if not self.model.thermalData.loaded:
            return

        key = event.key()
        if key == Qt.Key_Right or key == Qt.Key_Period:
            if event.modifiers() & Qt.ControlModifier:
                self.model.jump_marker(1)
            elif event.modifiers() & Qt.ShiftModifier:
                # '>' 1 second forward
                self.model.thermalData.skip_fwd()
            else:
                self.model.thermalData.show_frame(frame_idx=None)

        elif key == Qt.Key_Greater:
            # '>' 1 second forward
            self.model.thermalData.skip_fwd()

        elif key == Qt.Key_Left or key == Qt.Key_Comma:
            if event.modifiers() & Qt.ControlModifier:
                self.model.jump_marker(-1)
            elif event.modifiers() & Qt.ShiftModifier:
                # '<' 1 second backward
                self.model.thermalData.skip_bkw()
            else:
                self.model.thermalData.prev_frame()

        elif key == Qt.Key_Less:
            # '<' 1 second backward
            self.model.thermalData.skip_bkw()

        elif key == Qt.Key_Down or key == Qt.Key_L:
            self.model.jump_extrema_point(minmax="min")

        elif key == Qt.Key_Up or key == Qt.Key_H:
            self.model.jump_extrema_point(minmax="max")

        elif key in (Qt.Key_V, Qt.Key_Space):
            self.playBtn.clicked.emit()

        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.model.erase_point()

        event.accept()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def exitCall(self):
        self.close()
        sys.exit(app.exec_())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def closeEvent(self, event):
        if self.model.thermalData.loaded or self.model.videoData.loaded:
            # Save working setting
            fname = APP_ROOT / "config" / "TVT_last_working_state-0.pkl"
            if not fname.parent.is_dir():
                fname.parent.mkdir()

            self.model.shift_save_setting_fname(fname)
            self.model.save_status(fname)
            self.model.save_tmp_status(timer=False)

        if self.model.CONF_DIR.is_dir():
            for rmf in self.model.CONF_DIR.glob("*.fff"):
                rmf.unlink()

            conf = {"DATA_ROOT": str(self.model.DATA_ROOT)}
            with open(self.model.conf_f, "w") as fd:
                json.dump(conf, fd)

        self.deleteLater()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mouseDoubleClickEvent(self, e):
        """For debug"""
        print(self.width(), self.height())


# %%
def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("error catched!:")
    print("error message:\n", tb)


# %% main =====================================================================
if __name__ == "__main__":
    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    win = MainWindow()

    # --- Set initial window size ---
    win.resize(1440, 920)
    win.move(0, 0)
    win.show()
    ret = app.exec()
    sys.exit(ret)
