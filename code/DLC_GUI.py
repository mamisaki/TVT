#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" GUI for DeepLabCut

The code follows the model view architecture
Model class: DLC_GUI
    Core functions
View class: ViewWindow
    User interface class. QMainWindow of PySide6.
View Model class: ViewModel
    Handling user inputs from the ViewWindow, calling functions in
    DLC_Interface_Model, and updating a view in the ViewWindow.

INstall note
------------
Install miniconda 3
    https://www.anaconda.com/distribution/

Create TVT environment
    sudo apt install exiftool git build-essential libgtk-3-dev -y
    cd
    git clone https://github.com/mamisaki/TVT.git
    cd ~/TVT
    conda env create -f TVT_linux.yaml

    Activate the environment
        conda activate TVT

"""


# %% import ===================================================================
from pathlib import Path, PurePath
import os
import sys
from datetime import timedelta
from functools import partial
import pickle
import platform
import re
import shutil
import time
import json
import traceback
from datetime import datetime
import gc

import numpy as np
import pandas as pd
import csv
import cv2

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (FigureCanvas,
                                               NavigationToolbar2QT)

from PySide6.QtCore import Qt, QObject, QTimer
from PySide6.QtCore import Signal as pyqtSignal

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QDialog, QFrame, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSizePolicy, QPushButton,
    QStyle, QSlider, QComboBox, QSpinBox, QSplitter, QGroupBox,
    QLineEdit, QDialogButtonBox, QSpacerItem, QInputDialog, QCheckBox)

from PySide6.QtGui import QAction

from data_movie import DataMovie, DisplayImage
from dlc_interface import DLCinter


# %% Default values ===========================================================
tracking_point_radius_default = 2
tracking_point_pen_color_default = 'darkRed'
plot_kind = ['position', 'angle']

qt_global_colors = ['darkRed', 'darkGreen', 'darkBlue', 'darkCyan',
                    'darkMagenta', 'darkYellow', 'red', 'green', 'blue',
                    'cyan', 'magenta', 'yellow', 'black', 'white', 'darkGray',
                    'gray', 'lightGray']
pen_color_rgb = {'darkRed': '#800000',
                 'darkGreen': '#008000',
                 'darkBlue': '#000080',
                 'darkCyan': '#008080',
                 'darkMagenta': '#800080',
                 'darkYellow': '#808000',
                 'red': '#ff0000',
                 'green': '#00ff00',
                 'blue': '#0000ff',
                 'cyan': '#00ffff',
                 'magenta': '#ff00ff',
                 'yellow': '#ffff00',
                 'black': '#000000',
                 'white': '#ffffff',
                 'darkGray': '#808080',
                 'gray': '#a0a0a4',
                 'lightGray': '#c0c0c0'}

if '__file__' not in locals():
    __file__ = './this.py'

APP_ROOT = Path(__file__).absolute().parent.parent

OS = platform.system()


# %% TrackingPoint class ======================================================
class TrackingPoint():
    """ Tracking point data class.
    Each point is an instance of TrackingPoint class.
    The class handles frame-wise point seqence.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, dataMovie, name='', x=np.nan, y=np.nan):
        self.dataMovie = dataMovie
        self.name = name
        self.frequency = self.dataMovie.frame_rate
        data_length = self.dataMovie.duration_frame
        self.x = np.ones(data_length) * x
        self.y = np.ones(data_length) * y
        self.radius = tracking_point_radius_default
        self.value_ts = np.ones(data_length) * np.nan

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_position(self, x, y, frame_indices=None, update_frames=[]):
        if frame_indices is None:
            frame_indices = [self.dataMovie.frame_position]
            if type(update_frames) is bool and update_frames:
                update_frames = frame_indices

        self.x[frame_indices] = x
        self.y[frame_indices] = y

        if hasattr(self.dataMovie, 'get_rois_dataseries') \
                and len(update_frames) > 0:
            self.value_ts[update_frames] = self.get_value(update_frames,
                                                          force_update=True)

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
        for t in frame_indices:
            if not force_update and not np.isnan(self.value_ts[t]):
                continue
            x = self.x[t]
            y = self.y[t]
            xyt = np.concatenate([xyt, [[x, y, t]]], axis=0)

        val = self.value_ts[frame_indices]

        return val

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_save_params(self):
        settings = {}
        settings['dataMovie.filename'] = self.dataMovie.filename
        save_params = ['radius', 'value_ts', 'x', 'y']
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
        self.file_path = None

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open(self, filename):
        if self.loaded:
            self.unload()

        self.file_path = Path(filename)
        self.videoCap = cv2.VideoCapture(str(filename))
        self.frame_rate = self.videoCap.get(cv2.CAP_PROP_FPS)
        self.duration_frame = \
            int(self.videoCap.get(cv2.CAP_PROP_FRAME_COUNT))

        super(VideoDataMovie, self).open(filename)

        self.model.common_duration_ms = (self.duration_frame /
                                         self.frame_rate) * 1000
        self.model.main_win.positionSlider.blockSignals(True)
        self.model.main_win.positionSlider.setRange(
            0, int(self.model.common_duration_ms))
        self.model.main_win.positionSlider.setValue(
            int(self.model.common_time_ms))
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

        # Set videoCap frame position
        if frame_idx != self.frame_position+1:
            self.videoCap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        # --- read frame ---
        success, frame_data = self.videoCap.read()
        if success:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_BGR2RGB)

        self.frame_position = \
            int(self.videoCap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame_time = self.videoCap.get(cv2.CAP_PROP_POS_MSEC)/1000

        return success, frame_data, frame_time

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_frame(self, frame_idx=None, common_time_ms=None,
                   sync_update=True):
        super(VideoDataMovie, self).show_frame(
            frame_idx, common_time_ms, sync_update)

        if self.model.main_win.plot_xvals is not None:
            xpos = self.model.main_win.plot_xvals[self.frame_position]
            for pp, tl in self.model.main_win.plot_timeline.items():
                if tl.get_xdata()[0] != xpos:
                    if pp not in self.model.main_win.plot_line or \
                            len(self.model.main_win.plot_line[pp]) == 0:
                        self.model.main_win.plot_ax[pp].set_ylim([0, 1])
                    tl.set_xdata([xpos, xpos])

        self.model.main_win.roi_plot_canvas.draw()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_frame_from_comtime(self, common_time_ms):
        ms_per_frame = 1000 / self.frame_rate
        return int(np.round(common_time_ms / ms_per_frame))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_comtime_from_frame(self, frame_idx):
        common_time_ms = 1000 * (frame_idx / self.frame_rate)
        return common_time_ms


# %% Model class : DLC_Interface_Model ========================================
class DLC_GUI(QObject):
    """ Model class
    """

    move_point_signal = pyqtSignal()
    select_point_ui_signal = pyqtSignal(str)
    edit_point_signal = pyqtSignal(str)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, main_win, batchmode=False, save_interval=10):
        super(DLC_GUI, self).__init__(parent=main_win)

        # View model
        self.main_win = main_win
        self.tracking_point_pen_color_default = \
            tracking_point_pen_color_default
        self.DATA_ROOT = APP_ROOT / 'data'

        # --- Video data ------------------------------------------------------
        self.video_UI_objs = {
            'frFwdBtn': self.main_win.videoFrameFwdBtn,
            'frBkwBtn': self.main_win.videoFrameBkwBtn,
            'skipFwdBtn': self.main_win.videoSkipFwdBtn,
            'skipBkwBtn': self.main_win.videoSkipBkwBtn,
            'framePosSpBox': self.main_win.videoFramePosSpBox,
            'positionLabel': self.main_win.videoPositionLab}
        self.videoData = VideoDataMovie(self, self.main_win.videoDispImg,
                                        self.video_UI_objs)

        # Point marker (black dot) position
        self.point_mark_xy = [0, 0]
        self.main_win.videoDispImg.point_mark_xy = self.point_mark_xy

        self.CONF_DIR = Path.home() / '.TVT'
        if not self.CONF_DIR.is_dir():
            self.CONF_DIR.mkdir()

        self.conf_f = self.CONF_DIR / 'DLCGUI_conf.json'
        if self.conf_f.is_file():
            try:
                with open(self.conf_f, 'r') as fd:
                    conf = json.load(fd)

                for k, v in conf.items():
                    if k in ('DATA_ROOT',):
                        v = Path(v)
                    setattr(self, k, v)
            except Exception:
                pass

        # --- movie parameters ------------------------------
        self.common_time_ms = 0
        self.on_sync = False

        # timer for movie play
        self.play_timer = QTimer(self)
        self.play_timer.setSingleShot(True)
        self.play_timer.timeout.connect(self.play_update)
        self.play_frame_interval_ms = np.inf

        # --- Tracking point --------------------------------------------------
        self.tracking_mark = dict()  # tracking point marks on display
        self.main_win.videoDispImg.tracking_mark = self.tracking_mark
        self.editRange = 'current'

        # --- Time marker -----------------------------------------------------
        self.time_marker = {}
        self.tracking_point = dict()  # tracking point temperatures values
        self.editRange = 'current'

        # --- DeepLabCut interface --------------------------------------------
        self.dlci = DLCinter(self.DATA_ROOT, ui_parent=self.main_win)

        # --- Connect signals -------------------------------------------------
        self.move_point_signal.connect(self.update_dispImg)
        self.select_point_ui_signal.connect(self.select_point_ui)
        self.edit_point_signal.connect(self.edit_tracking_point)

        # --- Load last working status ----------------------------------------
        self.loaded_state_f = None
        self.num_saved_setting_hist = 5
        last_state_f = APP_ROOT / 'config' / 'DLCGUI_last_working_state-0.pkl'
        if not last_state_f.parent.is_dir():
            os.makedirs(last_state_f.parent)

        if not batchmode:
            self.save_timer = QTimer()
            self.save_timer.setSingleShot(True)
            self.save_timer.timeout.connect(self.save_tmp_status)
            self.save_tmp_wait = save_interval  # seconds
            self.save_timer.start(self.save_tmp_wait * 1000)

            if last_state_f.is_file():
                ret = QMessageBox.question(self.main_win, "Load last state",
                                           "Retrieve the last working state?",
                                           QMessageBox.Yes | QMessageBox.No)
                if ret == QMessageBox.Yes:
                    self.load_status(fname=last_state_f)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def openVideoFile(self, *args, fileName=None, **kwargs):
        if fileName is None:
            stdir = self.DATA_ROOT
            fileName, _ = QFileDialog.getOpenFileName(
                self.main_win, "Open Movie", str(stdir),
                "movie files (*.mp4 *.avi);; all (*.*)", None,
                QFileDialog.DontUseNativeDialog)

            if fileName == '':
                return

        fileName = Path(fileName)

        if not str(fileName.absolute()).startswith(
                str(self.DATA_ROOT.absolute())):
            # the data file is not in the DATA_ROOT
            msgBox = QMessageBox()
            msgBox.setText(
                f"The video file is not located under {self.DATA_ROOT}."
                f" Would you like to copy it there ({self.DATA_ROOT})?")
            msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msgBox.setDefaultButton(QMessageBox.Yes)
            ret = msgBox.exec()
            if ret == QMessageBox.Yes:
                destination = self.DATA_ROOT / fileName.name
                shutil.copy(fileName, destination)
                fileName = destination  # Update filePath to the new location

        # extention must be lower case
        if fileName.suffix != fileName.suffix.lower():
            fileName0 = fileName
            fileName = fileName.parent / (fileName.stem +
                                          fileName.suffix.lower())
            fileName0.rename(fileName)

        self.videoData.open(fileName)
        self.main_win.unloadVideoDataBtn.setText(
            f"Unload {Path(fileName).name}")

        self.main_win.unloadVideoDataBtn.setEnabled(True)
        self.main_win.positionSlider.setEnabled(True)
        self.main_win.tmark_grpbx.setEnabled(True)
        self.main_win.roi_ctrl_grpbx.setEnabled(True)
        self.main_win.roi_plot_canvas.setEnabled(True)
        [ax.cla() for ax in self.main_win.plot_ax.values()]

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def unloadVideoData(self):
        self.videoData.unload()
        self.main_win.unloadVideoDataBtn.setText('Unload')
        self.main_win.unloadVideoDataBtn.setEnabled(False)

        # Reset time marker and tracking points
        self.time_marker = {}
        self.tracking_points = {}

        # Reset videoData
        del self.videoData
        self.videoData = VideoDataMovie(
            self, self.main_win.videoDispImg, self.video_UI_objs)

        # Reset dlci
        del self.dlci
        self.dlci = DLCinter(self.DATA_ROOT,
                             ui_parent=self.main_win)

        # Disable UIs
        self.main_win.positionSlider.setEnabled(False)
        self.main_win.tmark_grpbx.setEnabled(False)
        self.main_win.roi_ctrl_grpbx.setEnabled(False)
        self.main_win.roi_plot_canvas.setEnabled(False)
        [ax.cla() for ax in self.main_win.plot_ax.values()]

        self.plot_xvals = None
        self.plot_line = {}
        self.plot_timeline = {}
        self.plot_marker_line = {}

    # --- Common movie control functions --------------------------------------
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
        self.videoData.show_frame(common_time_ms=time_ms)

        # update plot timeline
        self.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def play(self):
        if not self.videoData.loaded:
            return

        # Get frame interval
        self.play_frame_interval_ms = 1000/self.videoData.frame_rate

        self.play_timer.start(0)
        self.main_win.playBtn.setIcon(
                self.main_win.style().standardIcon(QStyle.SP_MediaPause))
        self.main_win.playBtn.clicked.disconnect(self.play)
        self.main_win.playBtn.clicked.connect(self.pause)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def play_update(self):
        st = time.time()
        # End time of the movie
        max_t = self.main_win.positionSlider.maximum()
        if self.common_time_ms+self.play_frame_interval_ms > max_t:
            self.pause()
            return

        # Increment movie
        self.common_time_ms += self.play_frame_interval_ms
        self.set_common_time(self.common_time_ms)
        update_time = time.time() - st

        # Schedule the next frame
        interval = self.play_frame_interval_ms - update_time*1000
        self.play_timer.start(max(0, int(interval)))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def pause(self):
        self.play_timer.stop()
        self.main_win.playBtn.setIcon(
                self.main_win.style().standardIcon(QStyle.SP_MediaPlay))
        self.main_win.playBtn.clicked.disconnect(self.pause)
        self.main_win.playBtn.clicked.connect(self.play)

    # --- Tracking point click callbacks --------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_editRange(self, selectIdx):
        edRngs = ['current', 'Mark<', '<Mark', '0<', '>End']
        self.editRange = edRngs[selectIdx]

        if selectIdx != self.main_win.roi_editRange_cmbbx.currentIndex():
            self.main_win.roi_editRange_cmbbx.blockSignals(True)
            self.main_win.roi_editRange_cmbbx.setCurrentIndex(selectIdx)
            self.main_win.roi_editRange_cmbbx.blockSignals(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def update_dispImg(self):
        if self.videoData.loaded:
            self.main_win.videoDispImg.set_pixmap()

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
        self.main_win.roi_color_cmbbx.blockSignals(True)

        if point_name in self.tracking_mark:
            # Set main_win values
            self.main_win.roi_idx_cmbbx.setCurrentText(point_name)
            self.main_win.roi_name_ledit.setText(point_name)
            x = self.tracking_mark[point_name]['x']
            y = self.tracking_mark[point_name]['y']
            if np.isnan(x) or np.isnan(y):
                self.main_win.roi_x_spbx.setValue(-1)
                self.main_win.roi_y_spbx.setValue(-1)
            else:
                self.main_win.roi_x_spbx.setValue(int(x))
                self.main_win.roi_y_spbx.setValue(int(y))
            self.main_win.roi_rad_spbx.setValue(
                    self.tracking_point[point_name].radius)
            self.main_win.roi_color_cmbbx.setCurrentText(
                    self.tracking_mark[point_name]['pen_color'])
        else:
            # point_nameis deleted. Reset
            self.main_win.roi_name_ledit.setText('')
            self.main_win.roi_x_spbx.setValue(-1)
            self.main_win.roi_y_spbx.setValue(-1)
            self.main_win.roi_rad_spbx.setValue(tracking_point_radius_default)
            self.main_win.roi_color_cmbbx.setCurrentText(
                self.tracking_point_pen_color_default)

        # unblock signals
        self.main_win.roi_idx_cmbbx.blockSignals(False)
        self.main_win.roi_name_ledit.blockSignals(False)
        self.main_win.roi_x_spbx.blockSignals(False)
        self.main_win.roi_y_spbx.blockSignals(False)
        self.main_win.roi_rad_spbx.blockSignals(False)
        self.main_win.roi_color_cmbbx.blockSignals(False)

        if update_plot:
            self.plot_timecourse(update_val=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def edit_tracking_point(self, point_name):
        if not self.videoData.loaded:
            return

        # --- Set max position (frame can be larger than the image) -----------
        xmax = self.videoData.dispImg.frame_w-1
        ymax = self.videoData.dispImg.frame_h-1

        if self.main_win.roi_x_spbx.maximum() != xmax:
            self.main_win.roi_x_spbx.setMaximum(xmax)

        if self.main_win.roi_y_spbx.maximum() != ymax:
            self.main_win.roi_y_spbx.setMaximum(ymax)

        if point_name not in self.tracking_mark.keys():
            # point_name is deleted
            del self.tracking_point[point_name]
        else:
            # --- Check properties in self.tracking_mark[k] -------------------
            x = self.tracking_mark[point_name]['x']
            y = self.tracking_mark[point_name]['y']

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
                self.tracking_point[point_name].set_position(
                        x, y, update_frames=True)
            else:
                # Make a new tracking point time series
                self.tracking_point[point_name] = \
                    TrackingPoint(self.videoData, x, y)

                # Set tracking mark properties with tracking_point object
                self.tracking_mark[point_name]['name'] = point_name
                self.tracking_mark[point_name]['rad'] = \
                    self.tracking_point[point_name].radius
                self.tracking_mark[point_name]['pen_color'] = \
                    self.tracking_point_pen_color_default
                pcols = list(pen_color_rgb.keys())
                cidx = pcols.index(self.tracking_point_pen_color_default)
                cidx = (cidx+1) % len(pcols)
                self.tracking_point_pen_color_default = pcols[cidx]

                # Set the point positions
                frame_indices = np.arange(self.videoData.frame_position,
                                          self.videoData.duration_frame)
                self.tracking_point[point_name].set_position(
                        x, y, frame_indices,
                        update_frames=[self.videoData.frame_position])

        # update display
        self.update_dispImg()

        # Set tracking points control widgets
        self.main_win.roi_idx_cmbbx.clear()
        if len(self.tracking_mark) == 0:
            self.main_win.roi_ctrl_grpbx.setEnabled(False)
            # self.main_win.roi_export_btn.setEnabled(False)
            self.main_win.roi_plot_canvas.setEnabled(False)
        else:
            self.main_win.roi_ctrl_grpbx.setEnabled(True)
            # self.main_win.roi_export_btn.setEnabled(True)
            self.main_win.roi_plot_canvas.setEnabled(True)

            self.main_win.roi_idx_cmbbx.blockSignals(True)
            self.main_win.roi_idx_cmbbx.addItems(
                list(self.tracking_point.keys()))
            self.main_win.roi_idx_cmbbx.blockSignals(False)

        self.select_point_ui(point_name)

        # Reset edit range
        self.set_editRange(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def edit_point_property(self, *args):
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
        col = self.main_win.roi_color_cmbbx.currentText()

        self.tracking_mark[point_name]['pen_color'] = col

        # Name edit
        if edit_name != point_name:
            self.tracking_point[edit_name] = \
                self.tracking_point.pop(point_name)
            self.tracking_mark[edit_name] = \
                self.tracking_mark.pop(point_name)
            point_name = edit_name

            # update main_win list
            self.main_win.roi_idx_cmbbx.clear()
            self.main_win.roi_idx_cmbbx.addItems(
                    sorted(list(self.tracking_point.keys())))

        # Radius change
        if rad != self.tracking_point[point_name].radius:
            self.tracking_point[point_name].radius = rad
            self.tracking_mark[point_name]['rad'] = rad
            # Reset tracking values
            self.tracking_point[point_name].value_ts[:] = np.nan

        # Position change
        current_x, current_y = \
            self.tracking_point[point_name].get_current_position()
        if x != current_x or y != current_y:
            self.tracking_mark[point_name]['x'] = x
            self.tracking_mark[point_name]['y'] = y

            # Reset tracking values from the current frame
            frame = self.videoData.frame_position
            Nframes = len(self.tracking_point[point_name].value_ts)

            if self.editRange == 'current':
                frame_indices = [frame]

            elif self.editRange == 'Mark<':
                markFrames = np.unique(list(self.time_marker.keys()))
                fromFrame = markFrames[markFrames < frame]
                if len(fromFrame):
                    fromFrame = fromFrame[-1]+1
                else:
                    fromFrame = 0

                if frame in self.time_marker:
                    toFrame = frame  # Not include the current frame
                else:
                    toFrame = frame+1  # include the current frame

                frame_indices = np.arange(fromFrame, toFrame)

            elif self.editRange == '<Mark':
                markFrames = np.unique(list(self.time_marker.keys()))
                toFrame = markFrames[markFrames > frame]
                if len(toFrame):
                    toFrame = toFrame[0]
                else:
                    toFrame = Nframes
                toFrame = min(toFrame, Nframes)

                if frame in self.time_marker:
                    fromFrame = frame+1  # Not include the current frame
                else:
                    fromFrame = frame  # include the current frame

                frame_indices = np.arange(fromFrame, toFrame)

            elif self.editRange == '0<':
                if frame in self.time_marker:
                    toFrame = frame  # Not include the current frame
                else:
                    toFrame = frame+1  # include the current frame
                frame_indices = np.arange(0, toFrame)

            elif self.editRange == '>End':
                if frame in self.time_marker:
                    fromFrame = frame+1  # Not include the current frame
                else:
                    fromFrame = frame  # include the current frame
                frame_indices = np.arange(fromFrame, Nframes)

            self.tracking_point[point_name].value_ts[frame_indices] = np.nan

            # update position and value
            self.tracking_point[point_name].set_position(
                x, y, frame_indices=frame_indices, update_frames=[frame])

        # update display
        self.update_dispImg()
        self.select_point_ui(point_name)

        self.plot_timecourse(update_val=True)
        self.set_editRange(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def erase_point(self):
        """Erase current (and later) tracking position
        """
        point_name = self.main_win.roi_idx_cmbbx.currentText()

        frame = self.videoData.frame_position
        Nframes = len(self.tracking_point[point_name].value_ts)

        if self.editRange == 'current':
            frame_indices = [frame]

        elif self.editRange == 'Mark<':
            markFrames = np.unique(list(self.time_marker.keys()))
            fromFrames = markFrames[markFrames < frame]
            if len(fromFrames):
                fromFrame = fromFrames[-1]+1
            else:
                fromFrame = 0
            toFrame = frame  # Not include the current frame
            frame_indices = np.arange(fromFrame, toFrame)

        elif self.editRange == '<Mark':
            markFrames = np.unique(list(self.time_marker.keys()))
            toFrames = markFrames[markFrames > frame]
            if len(toFrames):
                toFrame = toFrames[0]
            else:
                toFrame = Nframes

            toFrame = min(toFrame, Nframes)
            fromFrame = frame+1  # Not include the current frame
            frame_indices = np.arange(fromFrame, toFrame)

        elif self.editRange == '0<':
            toFrame = frame  # Not include the current frame
            frame_indices = np.arange(0, toFrame)

        elif self.editRange == '>End':
            fromFrame = frame+1  # Not include the current frame
            frame_indices = np.arange(fromFrame, Nframes)

        self.tracking_point[point_name].set_position(
                np.nan, np.nan, frame_indices=frame_indices,
                update_frames=frame_indices)

        self.tracking_mark[point_name]['x'] = np.nan
        self.tracking_mark[point_name]['y'] = np.nan

        self.update_dispImg()
        self.select_point_ui(point_name)
        self.plot_timecourse(update_val=True)

        self.set_editRange(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def delete_point(self, point_name=None, ask_confirm=True):
        """Delete tracking point time series
        """

        if point_name is None:
            point_name = self.main_win.roi_idx_cmbbx.currentText()

        if ask_confirm:
            # Confirm delete
            confMsg = f"Are you sure to delete the point '{point_name}'?\n" + \
                f"All time-seriese data for '{point_name}' will be deleted."
            rep = QMessageBox.question(
                self.main_win, 'Confirm delete', confMsg, QMessageBox.Yes,
                QMessageBox.No)
            if rep == QMessageBox.No:
                return

        del self.tracking_mark[point_name]
        self.edit_tracking_point(point_name)
        self.plot_timecourse(update_val=True)

    # --- Time marker control functions ---------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def add_marker(self):
        if not self.videoData.loaded:
            msgBox = QMessageBox(self.main_win)
            msgBox.setWindowModality(Qt.WindowModal)
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText('No thermal data is loaded!'
                           '\nMakers can be set only for a thermal data frame')
            msgBox.setWindowTitle('Error')
            msgBox.exec()
            return

        # Check marker name
        marker_name = self.main_win.tmark_name_cmbbx.currentText()
        if len(marker_name) == 0:
            msgBox = QMessageBox(self.main_win)
            msgBox.setWindowModality(Qt.WindowModal)
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText('Marker name is empty!')
            msgBox.setWindowTitle('Error')
            msgBox.exec()
            return

        # Put a marker at the current thermal frame
        frmIdx = self.videoData.frame_position
        self.time_marker[frmIdx] = marker_name
        self.show_marker()

        # Set maker list
        marker_list = np.unique(list(self.time_marker.values()))
        marker_list = [''] + list(marker_list)
        self.main_win.tmark_name_cmbbx.clear()
        self.main_win.tmark_name_cmbbx.addItems(marker_list)
        self.main_win.tmark_name_cmbbx.setCurrentText(marker_name)

        # Plot time mark
        self.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def del_marker(self):
        frmIdx = self.videoData.frame_position
        if frmIdx in self.time_marker:
            del self.time_marker[frmIdx]

            # Set maker list
            marker_list = np.unique(list(self.time_marker.values()))
            marker_list = [''] + list(marker_list)
            self.main_win.tmark_name_cmbbx.clear()
            self.main_win.tmark_name_cmbbx.addItems(marker_list)
            self.main_win.tmark_name_cmbbx.setCurrentText('')
            self.show_marker()

            self.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def jump_marker(self, shift):
        if len(self.time_marker) == 0:
            return

        marker_positions = np.array(sorted(self.time_marker.keys()))
        current = self.videoData.frame_position
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
        self.videoData.show_frame(jumpFrame)

        # Set position slider
        self.main_win.positionSlider.blockSignals(True)
        self.main_win.positionSlider.setValue(self.common_time_ms)
        self.main_win.positionSlider.blockSignals(False)

        # update plot timeline
        self.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_marker(self):
        if not self.videoData.loaded:
            return

        videoFrmIdx = self.videoData.frame_position
        if videoFrmIdx in self.time_marker:
            marker_name = self.time_marker[videoFrmIdx]
            self.main_win.videoMakerLab.setStyleSheet(
                    "background:red; color:white;")
            self.main_win.videoMakerLab.setText(marker_name)
            self.main_win.tmark_name_cmbbx.setCurrentText(marker_name)

            if self.videoData.loaded and self.on_sync:
                self.main_win.videoMakerLab.setStyleSheet(
                        "background:red; color:white;")
                self.main_win.videoMakerLab.setText(marker_name)
        else:
            self.main_win.videoMakerLab.setStyleSheet(
                    "background:black; color:white;")
            self.main_win.videoMakerLab.setText('')
            self.main_win.tmark_name_cmbbx.setCurrentText('')

            self.main_win.videoMakerLab.setStyleSheet(
                    "background:black; color:white;")
            self.main_win.videoMakerLab.setText('')

    # --- Tacking time course plot --------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def plot_onclick(self, event):
        if self.main_win.plot_xvals is None:
            return

        xpos = event.xdata
        if xpos is not None:
            self.set_common_time(time_ms=xpos*1000)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def plot_timecourse(self, update_val=False, plot_all=False, *args,
                        **kwargs):

        # --- Set xvals in time -----------------------------------------------
        if self.main_win.plot_xvals is None:
            xvals = np.arange(0, self.videoData.duration_frame)
            if len(xvals) == 0:
                [ax.cla() for ax in self.main_win.plot_ax.values()]
                self.main_win.plot_xvals = None
                return

            xunit = 1.0 / self.videoData.frame_rate
            self.main_win.plot_xvals = xvals * xunit
            xvals = self.main_win.plot_xvals

            # Set xtick label
            tick_intv = max(np.round((xvals.max()/20) / 15) * 15, 15)
            xticks = np.arange(0, xvals.max(), tick_intv)
            xtick_labs = []
            for xt in xticks:
                if xt < 0:
                    xtick_labs.append('')
                else:
                    tstr = str(timedelta(seconds=xt))
                    tstr = re.sub(r'^0+:', '', tstr)
                    tstr = re.sub(r'^0', '', tstr)
                    tstr = re.sub(r'\..+$', '', tstr)
                    xtick_labs.append(tstr)

            self.main_win.plot_ax['x'].set_xticks(xticks, [])
            self.main_win.plot_ax['y'].set_xticks(xticks, xtick_labs)
            for ax in self.main_win.plot_ax.values():
                ax.set_xlim(xvals[0]-xunit, xvals[-1]+xunit)
            [self.main_win.plot_ax[pp].set_ylabel(pp) for pp in ('x', 'y')]

        # --- Time line -------------------------------------------------------
        xpos = self.main_win.plot_xvals[self.videoData.frame_position]
        for pp in ('x', 'y'):
            if pp not in self.main_win.plot_timeline:
                self.main_win.plot_timeline[pp] = \
                    self.main_win.plot_ax[pp].axvline(
                        xpos, color='k', ls=':', lw=1)
            else:
                self.main_win.plot_timeline[pp].set_xdata([xpos, xpos])

        # --- Marker line -----------------------------------------------------
        for frame in self.time_marker.keys():
            tx = self.main_win.plot_xvals[frame]
            self.main_win.plot_marker_line[frame] = \
                [self.main_win.plot_ax[pp].axvline(tx, color='r', lw=1)
                 for pp in ('x', 'y')]

        # Delete marker
        if hasattr(self.main_win, 'plot_marker_line'):
            rm_marker = np.setdiff1d(
                list(self.main_win.plot_marker_line.keys()),
                list(self.time_marker.keys()))
            if len(rm_marker):
                for rmfrm in rm_marker:
                    for rmln in self.main_win.plot_marker_line[rmfrm]:
                        [ln.remove() for ln in rmln]
                    self.main_win.plot_marker_line[rmfrm].remove()
                    del self.main_win.plot_marker_line[rmfrm]

        if not update_val:
            self.main_win.roi_plot_canvas.draw()
            return

        # --- Check point list update -----------------------------------------
        # Select points
        all_points = list(self.tracking_point.keys())
        if len(all_points) == 0:
            # Delete all points
            Points = []
        elif plot_all:
            Points = all_points
        else:
            Points = self.main_win.roi_idx_cmbbx.currentText()
            if Points == '':
                return
            Points = [Points]

        # Check point list update
        rm_lines = []
        if len(self.main_win.plot_line):
            for line in self.main_win.plot_line['x'].keys():
                if '_'.join(line.split('_')[:-1]) not in all_points:
                    rm_lines.append(line)

        if len(rm_lines):
            for line in rm_lines:
                for pp in ('x', 'y'):
                    # self.main_win.plot_ax[pp].lines.remove(
                    #     self.main_win.plot_line[pp][line])
                    self.main_win.plot_line[pp][line].remove()
                    del self.main_win.plot_line[pp][line]

        # -- Plot line --------------------------------------------------------
        for pp in ('x', 'y'):
            for point in Points:
                col = pen_color_rgb[self.tracking_mark[point]['pen_color']]
                if col == '#ffffff':  # white
                    col = '#0f0f0f'

                plab = f"{point}"
                val = getattr(self.tracking_point[point], pp)
                if pp not in self.main_win.plot_line:
                    self.main_win.plot_line[pp] = {}
                if plab not in self.main_win.plot_line[pp]:
                    # Create lines
                    self.main_win.plot_line[pp][plab] = \
                        self.main_win.plot_ax[pp].plot(
                            self.main_win.plot_xvals, val, '-', color=col,
                            label=plab)[0]
                else:
                    self.main_win.plot_line[pp][plab].set_color(col)
                    self.main_win.plot_line[pp][plab].set_ls('-')
                    self.main_win.plot_line[pp][plab].set_ydata(val)

            self.main_win.plot_ax[pp].relim()
            self.main_win.plot_ax[pp].autoscale_view()

        # -- legend --
        if self.main_win.plot_ax['x'].get_legend() is not None:
            self.main_win.plot_ax['x'].get_legend().remove()

        if len(self.main_win.plot_line['x']):
            self.main_win.plot_ax['x'].legend(bbox_to_anchor=(1.001, 1),
                                              loc='upper left')

        # -- draw --
        self.main_win.roi_plot_canvas.draw()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def export_roi_data(self, fname=None, **kwargs):
        if len(self.tracking_point) == 0 and len(self.time_marker) == 0:
            return

        if fname is None:
            # Set file name
            stdir = self.videoData.filename.parent
            initial_name = stdir / (self.videoData.filename.stem +
                                    '_tracking_points.csv')
            fname, _ = QFileDialog.getSaveFileName(
                    self.main_win, "Export data filename", str(initial_name),
                    "csv (*.csv);;all (*.*)", None,
                    QFileDialog.DontUseNativeDialog)
            if fname == '':
                return

        ext = Path(fname).suffix
        if ext != '.csv':
            fname += '.csv.'

        # Initialize saving data frame
        Points = list(self.tracking_point.keys())

        cols = pd.MultiIndex.from_product([[''], ['time_ms', 'marker']])
        cols = cols.append(pd.MultiIndex.from_product([Points, ['x', 'y']]))
        saveData = pd.DataFrame(columns=cols)
        saveData.index.name = 'frame'

        # Time millisec
        frame_per_msec = 1000 / self.videoData.frame_rate
        saveData[('', 'time_ms')] = \
            np.arange(self.videoData.duration_frame) * frame_per_msec

        # Time marker
        for fridx, val in self.time_marker.items():
            saveData.loc[fridx, ('', 'marker')] = val

        # x, y, temp for each marker
        for point in Points:
            saveData.loc[:, (point, 'x')] = self.tracking_point[point].x
            saveData.loc[:, (point, 'y')] = self.tracking_point[point].y

        # Save as csv
        # Get point properties
        point_property = {}
        for point in self.tracking_point.keys():
            point_property[point] = {
                'radius': self.tracking_point[point].radius,
                'color': self.tracking_mark[point]['pen_color']}

        with open(fname, 'w') as fd:
            print(f'DLCGUI export,{str(point_property)}', file=fd)
            fd.write(saveData.to_csv(quoting=csv.QUOTE_NONNUMERIC,
                                     encoding='cp932'))

    # --- DeepLabCut interface ------------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def dlc_call(self, call, opt=None):
        if call not in ('batch_run_training', 'boot_dlc_gui'):
            # Check if the video is loaded
            if not hasattr(self, 'videoData') or not self.videoData.loaded:
                self.main_win.error_MessageBox("No video data is loaded.")
                return

            elif not Path(self.videoData.filename).is_file():
                self.main_win.error_MessageBox(
                    "Not found the video file," +
                    f" {self.videoData.filename}.")
                return

        if call == 'new_project':
            proj_name = self.videoData.filename.stem
            experimenter_name = 'DLCGUI'
            video_files = [str(self.videoData.filename)]
            work_dir = self.videoData.filename.parent
            copy_videos = False
            self.dlci.new_project(proj_name, experimenter_name, video_files,
                                  work_dir, copy_videos)

        elif call == 'load_config':
            video_name = Path(self.videoData.filename).stem
            dirs = [str(dd) for dd in
                    self.videoData.filename.parent.glob(video_name + '*')
                    if dd.is_dir()]
            if len(dirs):
                st_dir = sorted(dirs)[-1]
            else:
                st_dir = self.videoData.filename.parent
            conf_file, _ = QFileDialog.getOpenFileName(
                    self.main_win, "DLC config", str(st_dir),
                    "config yaml files (config_*.yaml);;yaml (*.yaml)",
                    None, QFileDialog.DontUseNativeDialog)

            if conf_file == '':
                return

            self.dlci.config_path = conf_file

        elif call == 'edit_config':
            if self.dlci.config_path is None or \
                    not Path(self.dlci.config_path).is_file():
                return

            default_values = {}
            # default_values = {'bodyparts': ['LEYE', 'MID', 'REYE', 'NOSE'],
            #                   'dotsize': 6}
            self.dlci.edit_config(self.main_win.ui_edit_config,
                                  default_values=default_values)

        elif call == 'extract_frames':
            # Wait message box
            msgBox = self.main_win.waitDialog(
                title="TVT DLC call",
                msgTxt="Extracting training frames. Please wait.",
                modal=True, parent=self.main_win)
            msgBox.show()

            self.dlci.extract_frames(edit_gui_fn=self.main_win.ui_edit_config)
            msgBox.close()

        elif call == 'label_frames':
            self.dlci.label_frames(edit_gui_fn=self.main_win.ui_edit_config)

        elif call == 'check_labels':
            self.dlci.check_labels()

            # message box
            task = self.dlci.get_config()['Task']
            labImgPath = Path(self.dlci.get_config()['project_path'])
            labImgPath /= Path('labeled-data') / f"{task}_labeled"
            msgBox = QMessageBox(self.main_win)
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText(f"Labeled images are saved in\n {labImgPath}\n"
                           "Check them with an image viewer.")
            msgBox.setWindowTitle("DLC call")
            msgBox.exec()

        elif call == 'create_training_dataset':
            self.dlci.create_training_dataset(num_shuffles=1)

        elif call == 'train_network':
            self.dlci.train_network(
                proc_type=opt,
                analyze_videos=[self.videoData.filename],
                ui_edit_config=self.main_win.ui_edit_config)

        elif call == 'show_training_progress':
            self.dlci.show_training_progress()

        elif call == 'kill_training_process':
            self.dlci.kill_training_process()

        elif call == 'evaluate_network':
            self.dlci.evaluate_network()

        elif call == 'analyze_videos':
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
                confirmMsg = 'Overwrite the existing result files?'
                rep = QMessageBox.question(self.main_win, 'Confirm delete',
                                           confirmMsg, QMessageBox.Yes,
                                           QMessageBox.No)
                if rep == QMessageBox.No:
                    return

                # Delte result files
                for ff in res_fs:
                    ff.unlink()

            self.dlci.analyze_videos(self.videoData.filename)

        elif call == 'filterpredictions':
            self.dlci.filterpredictions(self.videoData.filename)

        elif call == 'plot_trajectories':
            self.dlci.plot_trajectories(self.videoData.filename, filtered=opt)

        elif call == 'create_labeled_video':
            self.dlci.create_labeled_video(self.videoData.filename,
                                           filtered=opt)

        elif call == 'extract_outlier_frames':
            self.dlci.extract_outlier_frames(self.videoData.filename)

        elif call == 'refine_labels':
            self.dlci.refine_labels()

        elif call == 'merge_datasets':
            self.dlci.merge_datasets()

        elif call == 'boot_dlc_gui':
            self.dlci.boot_dlc_gui()

        elif call == 'batch_run_training':
            # Select data directry
            if self.DATA_ROOT.is_dir():
                stdir = self.DATA_ROOT.parent
            else:
                stdir = APP_ROOT
            data_dir = QFileDialog.getExistingDirectory(
                self.main_win, "Select data directory",
                str(stdir), QFileDialog.ShowDirsOnly)
            if data_dir == '':
                return

            # Ask if overwrite
            ret = QMessageBox.question(
                self.main_win, "Batch run",
                "Overwrite (re-train) the existing results?",
                QMessageBox.No | QMessageBox.Yes)
            overwrite = (ret == QMessageBox.Yes)

            self.dlci.batch_run_training(data_dir, overwrite)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_tracking(self, fileName=None, lh_thresh=None, **kwargs):
        if not self.videoData.loaded:
            self.main_win.error_MessageBox("No video data is set.")
            return

        # --- Load file -------------------------------------------------------
        if fileName is None:
            stdir = self.videoData.filename.parent
            fileName, _ = QFileDialog.getOpenFileName(
                    self.main_win, "Open tracking file", str(stdir),
                    "csv files (*.csv)", None, QFileDialog.DontUseNativeDialog)

            if fileName == '':
                return

        track_df = pd.read_csv(fileName, header=[1, 2], index_col=0)
        with open(fileName, 'r') as fd:
            head = fd.readline()

        if 'DLCGUI export' in head:
            cols = [col for col in track_df.columns
                    if len(col[0]) and 'Unnamed' not in col[0]]
            if len(cols):
                cols = pd.MultiIndex.from_tuples(cols)
            point_property = eval(head.rstrip().replace('DLCGUI export,', ''))
        else:
            cols = track_df.columns
            point_property = {}

        if len(track_df.index) != self.videoData.duration_frame:
            errmsg = f"Loaded data length, {len(track_df.index)}"
            errmsg += " does not match either thermal or video frame length."
            self.main_win.error_MessageBox(errmsg)
            return

        PointNames = [col for col in track_df.columns.levels[0]
                      if len(col) and 'Unnamed' not in col]

        if len(PointNames) and 'likelihood' in track_df[PointNames[0]].columns:
            if lh_thresh is None:
                lh_thresh, ok = QInputDialog.getDouble(
                    self.main_win, 'Likelihood',
                    'Likelihood threshold:', value=0.9,
                    minValue=0.0, maxValue=1.0,
                    decimals=2)
                if not ok:
                    return

        # --- Read data and set tracking_points -------------------------------
        currentFrm = self.videoData.frame_position
        for point in PointNames:
            frm_mask = np.ones(len(track_df), dtype=bool)
            if 'likelihood' in track_df[point].columns:
                lh = track_df[point].likelihood.values
                frm_mask &= (lh >= lh_thresh)
            frm_mask &= pd.notnull(track_df[point].x).values

            valid_x = track_df[point].x.values[frm_mask]
            valid_y = track_df[point].y.values[frm_mask]
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
            self.tracking_mark[point] = {'x': xp, 'y': yp}
            self.main_win.videoDispImg.tracking_mark = self.tracking_mark

            # self.tracking_point[point] is created in the signal handelr
            self.edit_point_signal.emit(point)

            # Reset x, y to the read values
            self.tracking_point[point].x[:] = np.nan
            self.tracking_point[point].y[:] = np.nan
            self.tracking_point[point].set_position(valid_x, valid_y,
                                                    valid_frms)

            if point in point_property:
                self.tracking_point[point].radius = \
                    point_property[point]['radius']
                self.tracking_mark[point]['rad'] = \
                    point_property[point]['radius']
                if 'color' in point_property[point]:
                    self.tracking_mark[point]['pen_color'] = \
                        point_property[point]['color']
                self.edit_point_signal.emit(point)

        # --- Read marker -----------------------------------------------------
        if 'marker' in track_df.columns.get_level_values(1):
            marker = track_df.iloc[:, 1]
            marker = marker[pd.notnull(marker)]
            if len(marker):
                for frame, val in marker.items():
                    self.time_marker[frame] = val

                self.show_marker()

        # Update time-series plot
        self.plot_timecourse(update_val=True, plot_all=True)

    # --- Save/Load working status --------------------------------------------
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_status(self, fname=None, **kwargs):
        # --- Filename setup ---
        if fname is None:
            stdir = self.DATA_ROOT / 'work_state'
            video_name = self.videoData.file_path.stem
            if video_name is not None:
                dtstr = datetime.now().strftime("%Y%m%d%H%M")
                stdir = stdir / f"{video_name}_working_state_{dtstr}.pkl"

            if self.loaded_state_f is not None:
                stdir = self.loaded_state_f

            fname, _ = QFileDialog.getSaveFileName(
                    self.main_win, "Save setting filename", str(stdir),
                    "pkl (*.pkl);;all (*.*)", None,
                    QFileDialog.DontUseNativeDialog)
            if fname == '':
                return

            fname = Path(fname)
            if fname.suffix != '.pkl':
                fname = Path(str(fname) + '.pkl')

            self.loaded_state_f = fname
        else:
            if not fname.parent.is_dir():
                os.makedirs(fname.parent)

        # --- Extract saving parameter values for the model object ---
        settings = {}
        saving_params = ['time_marker', 'videoData', 'tracking_point',
                         'tracking_mark', 'dlci']
        for param in saving_params:
            if not hasattr(self, param):
                continue

            obj = getattr(self, param)
            if hasattr(obj, 'get_save_params'):
                obj_params = obj.get_save_params()
                if obj_params is not None:
                    settings[param] = obj_params
                continue

            if type(obj) is dict:
                for k, dobj in obj.items():
                    if param not in settings:
                        settings[param] = {}
                    if hasattr(dobj, 'get_save_params'):
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
            settings['current_point_name'] = \
                self.main_win.roi_idx_cmbbx.currentText()

        # --- Convert Path to relative to DATA_ROOT ---
        def path_to_rel(param):
            if type(param) is dict:
                for k, v in param.items():
                    if k == 'DATA_ROOT':
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
        with open(fname, 'wb') as fd:
            pickle.dump(settings, fd)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def shift_save_setting_fname(self, fname):
        """If fname exists, rename it with incrementing the file number.
        """

        if fname.is_file():
            fn, save_num = fname.stem.split('-')
            if int(save_num) == self.num_saved_setting_hist-1:
                fname.unlink()
                return

            save_num = int(save_num)+1
            mv_fname = fn + f"-{save_num}" + fname.suffix
            mv_fname = fname.parent / mv_fname
            self.shift_save_setting_fname(mv_fname)
            fname.rename(mv_fname)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def load_status(self, fname=None, **kwargs):
        if fname is None:
            stdir = self.DATA_ROOT / 'work_state'
            fname, _ = QFileDialog.getOpenFileName(
                self.main_win, "Open state file", str(stdir),
                "pickle (*.pkl)", None, QFileDialog.DontUseNativeDialog)
            if fname == '':
                return

        with open(fname, 'rb') as fd:
            settings = pickle.load(fd)

        if fname != APP_ROOT / 'config' / \
                'DLCGUI_last_working_state-0.pkl':
            self.loaded_state_f = Path(fname)

        # Load videoData
        if 'videoData' in settings:
            fname = self.DATA_ROOT / settings['videoData']['filename']
            frame_position = settings['videoData']['frame_position']
            if fname.is_file():
                self.openVideoFile(fileName=fname)
                self.videoData.show_frame(frame_idx=frame_position)

            del settings['videoData']

        # Load time_marker
        if 'time_marker' in settings:
            if self.videoData.loaded:
                self.time_marker = settings['time_marker']
                NFrames = self.videoData.duration_frame
                frs = np.array(list(self.time_marker.keys()))
                frs = frs[frs >= NFrames]
                for fr in frs:
                    del self.time_marker[fr]

                self.show_marker()
            del settings['time_marker']

        # Load DLC config
        if 'dlci' in settings:
            self.dlci.config_path = self.DATA_ROOT / \
                settings['dlci']['_config_path']
            del settings['dlci']

        # Load tracking_point
        if 'tracking_point' in settings:
            for lab, dobj in settings['tracking_point'].items():
                dmovie_fname = str(self.DATA_ROOT / dobj['dataMovie.filename'])
                if dmovie_fname == str(self.videoData.filename) \
                        and self.videoData.loaded:
                    dataMovie = self.videoData
                else:
                    continue

                self.tracking_point[lab] = TrackingPoint(dataMovie)
                for k, v in dobj.items():
                    if hasattr(self.tracking_point[lab], k):
                        setattr(self.tracking_point[lab], k, v)

            if len(self.tracking_point):
                for point_name, tm in settings['tracking_mark'].items():
                    if point_name in self.tracking_point:
                        self.tracking_mark[point_name] = tm

                    self.main_win.videoDispImg.tracking_mark = \
                        self.tracking_mark

                del settings['tracking_mark']

                if 'current_point_name' in settings:
                    point_name = settings['current_point_name']
                else:
                    point_name = list(self.tracking_point.keys())[0]

                self.edit_tracking_point(point_name)
                if 'current_point_name' in settings:
                    del settings['current_point_name']

            del settings['tracking_point']

            # Update time-series plot
            self.plot_timecourse(update_val=True, plot_all=True)

        # Load other parameters
        for param, obj in settings.items():
            if hasattr(self, param):
                setattr(self, param, obj)

        gc.collect()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_tmp_status(self, timer=True):
        if self.videoData.file_path is not None:
            video_name = self.videoData.file_path.stem
            save_f = self.DATA_ROOT / 'work_state' / \
                f"{video_name}_working_state.pkl"
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
                self.main_win, "Select data directory",
                str(stdir), QFileDialog.ShowDirsOnly)
            if data_dir == '':
                return

        self.DATA_ROOT = Path(data_dir)
        self.dlci.DATA_ROOT = self.DATA_ROOT

        conf = {'DATA_ROOT': str(self.DATA_ROOT)}
        with open(self.conf_f, 'w') as fd:
            json.dump(conf, fd)


# %% View class : ViewWindow ==================================================
class ViewWindow(QMainWindow):
    """ View class
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, parent=None, batchmode=False):
        super(ViewWindow, self).__init__(parent)
        self.setWindowTitle("DLC Video GUI")

        # Initialize viewer objects
        self.init_ui_objects()

        # Connect slot
        self.model = DLC_GUI(main_win=self, batchmode=batchmode)

        # Connect signals
        self.connect_signal_handlers()

        # Layout widgets
        self.set_layout()

        # Make menu items
        self.make_menu_objects()

        self.setFocusPolicy(Qt.ClickFocus)

    # -------------------------------------------------------------------------
    def init_ui_objects(self):

        # --- Video image widget ----------------------------------------------

        # Load video image button
        self.loadVideoDataBtn = QPushButton('Load video data')
        self.loadVideoDataBtn.setStyleSheet("background:#8ED5EC; color:black;")

        # Unload video data button
        self.unloadVideoDataBtn = QPushButton('Unload')
        self.unloadVideoDataBtn.setEnabled(False)
        self.unloadVideoDataBtn.setStyleSheet(
            "background:#FFF3F0; color:black;")

        # Video image display
        self.videoDispImg = DisplayImage(frame_w=640, frame_h=480, parent=self)

        # Marker label
        self.videoMakerLab = QLabel()
        self.videoMakerLab.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.videoMakerLab.setStyleSheet("background:black; color:white;")

        # Video position text
        self.videoPositionLab = QLabel('00:00/00:00 [0/0 frames]')
        self.videoPositionLab.setSizePolicy(QSizePolicy.Preferred,
                                            QSizePolicy.Maximum)

        # Video skip forward button
        self.videoSkipFwdBtn = QPushButton()
        self.videoSkipFwdBtn.setEnabled(False)
        self.videoSkipFwdBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaSkipForward))

        # Video skip backward button
        self.videoSkipBkwBtn = QPushButton()
        self.videoSkipBkwBtn.setEnabled(False)
        self.videoSkipBkwBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaSkipBackward))

        # Video frame forward button
        self.videoFrameFwdBtn = QPushButton()
        self.videoFrameFwdBtn.setEnabled(False)
        self.videoFrameFwdBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaSeekForward))

        # Video frame backward button
        self.videoFrameBkwBtn = QPushButton()
        self.videoFrameBkwBtn.setEnabled(False)
        self.videoFrameBkwBtn.setIcon(
                self.style().standardIcon(QStyle.SP_MediaSeekBackward))

        # Frame position
        self.videoFramePosLab = QLabel('frame:')
        self.videoFramePosSpBox = QSpinBox()
        self.videoFramePosLab.setEnabled(False)
        self.videoFramePosSpBox.setEnabled(False)

        # Play button
        self.playBtn = QPushButton()
        self.playBtn.setEnabled(False)
        self.playBtn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        # Position slider
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 1)
        self.positionSlider.setEnabled(False)

        # --- Time marker widgets ---------------------------------------------
        self.tmark_grpbx = QGroupBox('Time Marker')
        self.tmark_grpbx.setEnabled(False)
        self.tmark_add_btn = QPushButton('Add')
        self.tmark_name_cmbbx = QComboBox()
        self.tmark_name_cmbbx.setEditable(True)
        self.tmark_del_btn = QPushButton('Delete')
        self.tmark_jumpNext_btn = QPushButton('Jump next')
        self.tmark_jumpPrev_btn = QPushButton('Jump previous')

        # --- Tracking point edit widgets -------------------------------------
        self.roi_ctrl_grpbx = QGroupBox("Tracking Points")
        self.roi_ctrl_grpbx.setEnabled(False)

        self.roi_idx_cmbbx = QComboBox()
        self.roi_idx_cmbbx.setEditable(False)

        self.roi_name_ledit = QLineEdit()
        self.roi_showName_chbx = QCheckBox('Show name')
        self.roi_showName_chbx.setChecked(True)

        self.roi_x_spbx = QSpinBox()
        self.roi_x_spbx.setMinimum(-1)

        self.roi_y_spbx = QSpinBox()
        self.roi_y_spbx.setMinimum(-1)

        self.roi_rad_spbx = QSpinBox()
        self.roi_rad_spbx.setMinimum(1)

        self.roi_editRange_cmbbx = QComboBox()
        self.roi_editRange_cmbbx.setEditable(False)
        self.roi_editRange_cmbbx.addItems(['Current',
                                           'PrevMark -> Current',
                                           'Current -> NextMark',
                                           '0 -> Current',
                                           'Current -> End'])

        self.roi_color_cmbbx = QComboBox()
        self.roi_color_cmbbx.setEditable(False)
        self.roi_color_cmbbx.addItems(qt_global_colors)

        self.roi_erase_btn = QPushButton('Erase')

        self.roi_delete_btn = QPushButton('Delete this point')
        self.roi_delete_btn.setFixedHeight(18)
        self.roi_delete_btn.setStyleSheet("background:#ff7f7f; color:black;")

        self.roi_load_btn = QPushButton('Load tracking data')
        self.roi_load_btn.setStyleSheet("background:#7fbfff; color:black;")

        self.roi_export_btn = QPushButton('Export tracking data')
        self.roi_export_btn.setStyleSheet("background:#7fffbf; color:black;")
        # self.roi_export_btn.setEnabled(False)

        # --- Plot widgets ----------------------------------------------------
        self.point_plot_cmbbx = QComboBox()
        self.point_plot_cmbbx.setEditable(False)
        self.point_plot_cmbbx.addItems(['Position'])

        self.roi_plot_canvas = FigureCanvas(Figure())
        self.roi_plot_canvas.setSizePolicy(QSizePolicy.Expanding,
                                           QSizePolicy.Expanding)
        self.plot_ax = {'xy'[ii]: ax
                        for ii, ax in enumerate(
                            self.roi_plot_canvas.figure.subplots(2, 1))}
        [self.plot_ax[pp].set_ylabel(pp) for pp in ('x', 'y')]
        self.roi_plot_canvas.figure.subplots_adjust(
                left=0.05, bottom=0.24, right=0.9, top=0.94)
        self.roi_plot_canvas.start_event_loop(0.005)
        self.plot_xvals = None
        self.plot_line = {}
        self.plot_timeline = {}
        self.plot_marker_line = {}
        self.roi_plot_canvas.setEnabled(False)

        # Create the navigation toolbar and add it to the layout
        self.toolbar = NavigationToolbar2QT(self.roi_plot_canvas, self)

        # --- error label ---
        self.errorLabel = QLabel()
        self.errorLabel.setSizePolicy(QSizePolicy.Preferred,
                                      QSizePolicy.Maximum)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def connect_signal_handlers(self):
        # Video load/unload
        self.loadVideoDataBtn.clicked.connect(self.model.openVideoFile)
        self.unloadVideoDataBtn.clicked.connect(self.model.unloadVideoData)

        # Play/slider
        self.playBtn.clicked.connect(self.model.play)
        self.positionSlider.sliderReleased.connect(self.model.set_common_time)

        # Marker
        self.tmark_add_btn.clicked.connect(self.model.add_marker)
        self.tmark_del_btn.clicked.connect(self.model.del_marker)
        self.tmark_jumpNext_btn.clicked.connect(
                lambda: self.model.jump_marker(1))
        self.tmark_jumpPrev_btn.clicked.connect(
                lambda: self.model.jump_marker(-1))

        # Tracking point controls
        self.roi_idx_cmbbx.currentTextChanged.connect(
                self.model.select_point_ui)
        self.roi_name_ledit.returnPressed.connect(
                self.model.edit_point_property)
        self.roi_showName_chbx.stateChanged.connect(
                self.model.edit_point_property)
        self.roi_x_spbx.valueChanged.connect(self.model.edit_point_property)
        self.roi_y_spbx.valueChanged.connect(self.model.edit_point_property)
        self.roi_rad_spbx.valueChanged.connect(self.model.edit_point_property)
        self.roi_editRange_cmbbx.currentIndexChanged.connect(
                self.model.set_editRange)
        self.roi_color_cmbbx.currentIndexChanged.connect(
                self.model.edit_point_property)
        self.roi_erase_btn.clicked.connect(self.model.erase_point)
        self.roi_delete_btn.clicked.connect(
                lambda: self.model.delete_point(point_name=None,
                                                ask_confirm=True))
        self.roi_load_btn.clicked.connect(partial(self.model.load_tracking,
                                                  fileName=None))

        self.roi_export_btn.clicked.connect(
            lambda state: self.model.export_roi_data())

        # Time-course plot
        self.roi_plot_canvas.mpl_connect('button_press_event',
                                         self.model.plot_onclick)

    # -------------------------------------------------------------------------
    def set_layout(self):

        # --- Video image widgets layout --------------------------------------
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
        self.videoMakerLab.setSizePolicy(QSizePolicy.Expanding,
                                         QSizePolicy.Fixed)
        videoLayout.addWidget(self.videoMakerLab)
        videoLayout.addWidget(self.videoPositionLab)

        videoCtrlLayout = QHBoxLayout()
        videoCtrlLayout.addStretch()
        videoCtrlLayout.addWidget(self.videoSkipBkwBtn)
        videoCtrlLayout.addWidget(self.videoFrameBkwBtn)
        videoCtrlLayout.addWidget(self.playBtn)
        videoCtrlLayout.addWidget(self.videoFrameFwdBtn)
        videoCtrlLayout.addWidget(self.videoSkipFwdBtn)
        videoCtrlLayout.addWidget(self.videoFramePosSpBox)
        videoCtrlLayout.addStretch()
        videoLayout.addLayout(videoCtrlLayout)

        # --- Time Marker layout ---
        self.tmark_grpbx.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Fixed)
        self.tmark_grpbx.setFixedHeight(200)
        tmarkLayout = QGridLayout(self.tmark_grpbx)
        tmarkLayout.setColumnStretch(0, 1)
        tmarkLayout.setColumnStretch(1, 5)
        tmarkLayout.addWidget(QLabel('Name:'), 0, 0)
        tmarkLayout.addWidget(self.tmark_name_cmbbx, 0, 1)
        tmarkLayout.addWidget(self.tmark_add_btn, 1, 0, 1, 2)
        tmarkLayout.addWidget(self.tmark_del_btn, 2, 0, 1, 2)
        tmarkLayout.addWidget(self.tmark_jumpNext_btn, 3, 0, 1, 2)
        tmarkLayout.addWidget(self.tmark_jumpPrev_btn, 4, 0, 1, 2)

        # --- Tracking point control layout ---
        self.roi_ctrl_grpbx.setSizePolicy(QSizePolicy.Expanding,
                                          QSizePolicy.Fixed)
        roiCtrlLayout = QGridLayout(self.roi_ctrl_grpbx)
        roiCtrlLayout.addWidget(QLabel('Point:'), 0, 0)
        roiCtrlLayout.addWidget(self.roi_idx_cmbbx, 0, 1, 1, 2)
        roiCtrlLayout.addWidget(QLabel('Name:'), 1, 0)
        roiCtrlLayout.addWidget(self.roi_name_ledit, 1, 1, 1, 2)
        roiCtrlLayout.addWidget(QLabel('Color:'), 2, 0)
        roiCtrlLayout.addWidget(self.roi_color_cmbbx, 2, 1, 1, 2)
        roiCtrlLayout.addWidget(QLabel('Edit range:'), 3, 0)
        roiCtrlLayout.addWidget(self.roi_editRange_cmbbx, 3, 1, 1, 2)
        roiCtrlLayout.addWidget(QLabel('x:'), 4, 0)
        roiCtrlLayout.addWidget(self.roi_x_spbx, 4, 1)
        roiCtrlLayout.addWidget(QLabel('y:'), 5, 0)
        roiCtrlLayout.addWidget(self.roi_y_spbx, 5, 1)
        roiCtrlLayout.addWidget(self.roi_erase_btn, 5, 2)
        roiCtrlLayout.addWidget(QLabel('Radius:'), 6, 0)
        roiCtrlLayout.addWidget(self.roi_rad_spbx, 6, 1, 1, 2)
        roiCtrlLayout.addWidget(self.roi_showName_chbx, 7, 0, 1, 3)
        roiCtrlLayout.addWidget(self.roi_delete_btn, 8, 0, 1, 3)
        self.roi_ctrl_grpbx.resize(self.roi_ctrl_grpbx.sizeHint())

        # --- Place the maker and point edit control frames -------------------
        editCtrlFrame = QFrame()
        editCtrlFrameLayout = QVBoxLayout(editCtrlFrame)
        editCtrlFrameLayout.addWidget(self.tmark_grpbx)
        editCtrlFrameLayout.addWidget(self.roi_ctrl_grpbx)

        editCtrlFrameLayout.addWidget(self.roi_load_btn)
        editCtrlFrameLayout.addWidget(self.roi_export_btn)

        editCtrlFrameLayout.addStretch()

        # Select plot
        selPlotLayout = QHBoxLayout()
        selPlotLayout.addWidget(QLabel('Plot:'))
        selPlotLayout.addWidget(self.point_plot_cmbbx)
        editCtrlFrameLayout.addLayout(selPlotLayout)

        selPlotLayout.addStretch()
        editCtrlFrame.setFixedWidth(260)

        # --- Layout all ------------------------------------------------------
        # Create a central (base) widget for window contents
        centWid = QSplitter(Qt.Vertical)
        centWid.setStyleSheet(
                "QSplitter::handle {background-color: #eaeaea;}")

        upperFrame0 = QFrame()
        hlayout = QHBoxLayout(upperFrame0)
        hlayout.addWidget(editCtrlFrame)
        hlayout.addWidget(videoFrame)

        upperFrame1 = QFrame()
        vlayout = QVBoxLayout(upperFrame1)
        vlayout.addWidget(upperFrame0)

        slider_hlayout = QHBoxLayout()
        self.sli_hspace0 = QSpacerItem(12, 1, QSizePolicy.Minimum,
                                       QSizePolicy.Minimum)
        slider_hlayout.addSpacerItem(self.sli_hspace0)
        slider_hlayout.addWidget(self.positionSlider)
        self.sli_hspace1 = QSpacerItem(70, 1, QSizePolicy.Minimum,
                                       QSizePolicy.Minimum)
        slider_hlayout.addSpacerItem(self.sli_hspace1)
        vlayout.addLayout(slider_hlayout)

        centWid.addWidget(upperFrame1)
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
        fileMenu = menuBar.addMenu('&File')

        # Load status
        loadSettingAction = QAction('&Load working state', self)
        loadSettingAction.setShortcut('Ctrl+L')
        loadSettingAction.setStatusTip('Load working state')
        loadSettingAction.triggered.connect(partial(self.model.load_status,
                                                    fname=None))
        fileMenu.addAction(loadSettingAction)

        # Save status
        saveSettingAction = QAction('&Save working state', self)
        saveSettingAction.setShortcut('Ctrl+S')
        saveSettingAction.setStatusTip('Save working state')
        saveSettingAction.triggered.connect(partial(self.model.save_status,
                                                    fname=None))
        fileMenu.addAction(saveSettingAction)

        # Set DATA_ROOT
        setDataRootAction = QAction('&Set data root', self)
        setDataRootAction.setShortcut('Ctrl+D')
        setDataRootAction.setStatusTip('Load working state')
        setDataRootAction.triggered.connect(
            partial(self.model.set_data_root, data_dir=None))
        fileMenu.addAction(setDataRootAction)

        fileMenu.addSeparator()

        # Load video file
        loadVideoAction = QAction('&Load video data', self)
        loadVideoAction.setShortcut('Ctrl+N')
        loadVideoAction.setStatusTip('Load video data')
        loadVideoAction.triggered.connect(
            self.model.openVideoFile)
        fileMenu.addAction(loadVideoAction)

        fileMenu.addSeparator()

        # Exit
        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)
        fileMenu.addAction(exitAction)

        # -- DLC menu --
        dlcMenu = menuBar.addMenu('&DLC')

        # -- I --
        action = QAction('New project', self)
        action.setStatusTip('Create a new DeepLabCut project')
        action.triggered.connect(partial(self.model.dlc_call, 'new_project'))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction('Load config', self)
        action.setStatusTip('Load existing DeepLabCut configuraton')
        action.triggered.connect(partial(self.model.dlc_call, 'load_config'))
        dlcMenu.addAction(action)

        # -- II --
        action = QAction('Edit configuration', self)
        action.setStatusTip('Edit DeepLabCut configuration')
        action.triggered.connect(partial(self.model.dlc_call, 'edit_config'))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        # -- Boot deeplabcut GUI ---
        action = QAction('deeplabcut GUI', self)
        action.setStatusTip('Boot deeplabcut GUI application')
        action.triggered.connect(partial(self.model.dlc_call, 'boot_dlc_gui'))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction('Make a training script', self)
        action.setStatusTip(
            'Make a command script for DeepLabCut network training')
        action.triggered.connect(partial(self.model.dlc_call,
                                         'train_network', 'prepare_script'))
        dlcMenu.addAction(action)

        action = QAction('Run training backgroud', self)
        action.setStatusTip(
            'Create a command script for DeepLabCut network training and run' +
            ' it in the background')
        action.triggered.connect(partial(self.model.dlc_call,
                                         'train_network', 'run_subprocess'))
        dlcMenu.addAction(action)

        action = QAction('Show training progress', self)
        action.setStatusTip(
            'Show the progress of the training running in the background.')
        action.triggered.connect(
            partial(self.model.dlc_call, 'show_training_progress'))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction('Kill training process', self)
        action.setStatusTip(
            'Show the progress of the training running in the background.')
        action.triggered.connect(
            partial(self.model.dlc_call, 'kill_training_process'))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction('Analyze video', self)
        action.setStatusTip('Analyze video by DeepLabCut')
        action.triggered.connect(partial(self.model.dlc_call,
                                         'analyze_videos'))
        dlcMenu.addAction(action)

        action = QAction('Filter prediction', self)
        action.setStatusTip('Filter prediction by DeepLabCut')
        action.triggered.connect(
                partial(self.model.dlc_call, 'filterpredictions'))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        action = QAction('Run all training scripts in batch mode', self)
        action.setStatusTip(
            'Run all training scripts in a data directory sequentially.')
        action.triggered.connect(partial(self.model.dlc_call,
                                         'batch_run_training'))
        dlcMenu.addAction(action)

        dlcMenu.addSeparator()

        # -- XI --
        action = QAction('Load tracking positions', self)
        action.setStatusTip('Load positions tracked by DeepLabCut')
        action.triggered.connect(self.model.load_tracking)
        dlcMenu.addAction(action)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_edit_config(self, config_data, title='Edit DLC configuration'):

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
                        QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                        Qt.Horizontal, self)
                vbox.addWidget(self.buttons)
                self.buttons.accepted.connect(self.accept)
                self.buttons.rejected.connect(self.reject)
                self.setLayout(vbox)
                self.resize(400, 200)

            def place_ui(self, parent, label, value, layout):
                if label == 'video_sets':
                    # video_sets is set by dlci.add_video
                    return

                hbox = QHBoxLayout()
                hbox.addWidget(QLabel(f'{label}:'))
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
    def msg_dlg(self, msg, title='DLC GUI'):
        msgBox = QMessageBox(self)
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(msg)
        msgBox.setWindowTitle(title)
        msgBox.setStyleSheet("QLabel{max-height:720 px;}")
        msgBox.exec()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def error_MessageBox(self, errmsg, title='Error in DLC-GUI'):
        msgBox = QMessageBox(self)
        msgBox.setWindowModality(Qt.WindowModal)
        msgBox.setIcon(QMessageBox.Critical)
        msgBox.setText(errmsg)
        msgBox.setWindowTitle(title)
        msgBox.exec()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    class waitDialog(QDialog):
        def __init__(self, title='TVT', msgTxt='', modal=True, parent=None):
            super().__init__(parent)

            self.setWindowTitle(title)
            self.setModal(modal)
            vBoxLayout = QVBoxLayout(self)

            # message text
            self.msgTxt = QLabel(msgTxt)
            vBoxLayout.addWidget(self.msgTxt)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def keyPressEvent(self, event):
        if not self.model.videoData.loaded:
            return

        key = event.key()
        if key == Qt.Key_Right or key == Qt.Key_Period:
            if event.modifiers() & Qt.ControlModifier:
                self.model.jump_marker(1)
            elif event.modifiers() & Qt.ShiftModifier:
                # '>' 1 second forward
                self.model.videoData.skip_fwd()
            else:
                self.model.videoData.show_frame(frame_idx=None)

        elif key == Qt.Key_Greater:
            # '>' 1 second forward
            self.model.videoData.skip_fwd()

        elif key == Qt.Key_Left or key == Qt.Key_Comma:
            if event.modifiers() & Qt.ControlModifier:
                self.model.jump_marker(-1)
            elif event.modifiers() & Qt.ShiftModifier:
                # '<' 1 second backward
                self.model.videoData.skip_bkw()
            else:
                self.model.videoData.prev_frame()

        elif key == Qt.Key_Less:
            # '<' 1 second backward
            self.model.videoData.skip_bkw()

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
        if self.model.videoData.loaded:
            # Save working setting
            fname = APP_ROOT / 'config' / \
                'DLCGUI_last_working_state-0.pkl'
            if not fname.parent.is_dir():
                fname.parent.mkdir()

            self.model.shift_save_setting_fname(fname)
            self.model.save_status(fname)
            self.model.save_tmp_status(timer=False)

        if self.model.CONF_DIR.is_dir():
            for rmf in self.model.CONF_DIR.glob('*.fff'):
                rmf.unlink()

            conf = {'DATA_ROOT': str(self.model.DATA_ROOT)}
            with open(self.model.conf_f, 'w') as fd:
                json.dump(conf, fd)

        self.deleteLater()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mouseDoubleClickEvent(self, e):
        """For debug
        """
        print(self.width(), self.height())


# %%
def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("error catched!:")
    print("error message:\n", tb)


# %% main =====================================================================
if __name__ == '__main__':
    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    win = ViewWindow()

    win.resize(890, 830)
    win.move(0, 0)
    win.show()
    ret = app.exec()
    sys.exit(ret)
