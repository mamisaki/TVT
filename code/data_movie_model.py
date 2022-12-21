# -*- coding: utf-8 -*-
""" Data movie model class
"""


# %% import ===================================================================
from functools import partial
from pathlib import Path
from datetime import timedelta
import sys
import pickle

import numpy as np
import cv2
import imageio

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QProgressDialog

from csq_reader import CSQ_READER


# %% DataMovie class ==========================================================
class DataMovie():
    """Base class of VideoDataMovie and ThermalDataMovie
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, parent, dispImg, UI_objs):
        self.model = parent
        self.dispImg = dispImg

        # set UI objects
        for k, v in UI_objs.items():
            setattr(self, k, v)

        # Data properties
        self.filename = ''
        self.frame_rate = -1
        self.duration_frame = -1
        self.duration_t_str = ''
        self.loaded = False
        self.paired_data = None
        self.shift_scale_Mtx = None  # coordinate shifting and scaling matrix

        # Time shift from the common time of video and thermal data
        # N.B. comtime is thermal time, so this is always 0 for thermal data
        self.shift_from_thermoTime = 0

        # Movie position
        self.frame_position = -1

        # Connect callback
        self.frFwdBtn.clicked.connect(
                partial(self.show_frame, **{'frame_idx': None}))
        self.frBkwBtn.clicked.connect(self.prev_frame)
        self.skipFwdBtn.clicked.connect(self.skip_fwd)
        self.skipBkwBtn.clicked.connect(self.skip_bkw)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open(self, filename):
        self.filename = Path(filename)
        duration_t = timedelta(
                seconds=self.duration_frame/self.frame_rate)
        duration_t_str = str(duration_t).split('.')
        self.duration_t_str = duration_t_str[0] + '.' + duration_t_str[1][:3]

        # Enable control buttons
        self.ui_setEnabled(True)
        if self.paired_data is not None and self.paired_data.loaded:
            self.syncBtn.setEnabled(True)

        self.loaded = True

        # Reset sync status
        self.model.sync_video_thermal(False)
        if self.paired_data.loaded:
            self.model.main_win.syncVideoBtn.setEnabled(True)

        self.frame_position = -1
        self.show_frame(0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_frame(self, frame_idx):
        """Dummy class
        """
        pass

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def ui_setEnabled(self, enabled=True):
        for btn in ('frFwdBtn', 'frBkwBtn', 'skipFwdBtn', 'skipBkwBtn'):
            if getattr(self, btn) is not None:
                getattr(self, btn).setEnabled(enabled)

        # Common controls
        for btn in ('playBtn', 'commonSkipFwdBtn',
                    'commonSkipBkwBtn', 'positionSlider',
                    'commonPosisionLab'):
            if getattr(self.model.main_win, btn) is not None:
                getattr(self.model.main_win, btn).setEnabled(enabled)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_frame_from_comtime(self, common_time_ms):
        if self.shift_from_thermoTime is not None:
            local_time_ms = common_time_ms + self.shift_from_thermoTime
        else:
            local_time_ms = common_time_ms

        ms_per_frame = 1000 / self.frame_rate
        return int(np.round(local_time_ms / ms_per_frame))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_comtime_from_frame(self, frame_idx):
        common_time_ms = 1000 * (frame_idx / self.frame_rate)
        if self.shift_from_thermoTime is not None:
            common_time_ms -= self.shift_from_thermoTime

        return common_time_ms

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_frame(self, frame_idx=None, common_time_ms=None,
                   sync_update=True):
        """ Show video frame

        Option
        ------
        frame_idx: integer
            Frame index to read. If frame_idx == None and time == None,
            the next frame is shown.
        common_time_ms: float
            Time point to read in the common time. If self.on_sync is not True,
            this is ignored.
        sync_update: bool
            Update paired_data frame
        """

        if not self.loaded:
            return

        # --- Set reading frame index ---
        if common_time_ms is not None:
            frame_idx = self.get_frame_from_comtime(common_time_ms)
        elif frame_idx is None:
            frame_idx = self.frame_position + 1

        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= self.duration_frame:
            frame_idx = self.duration_frame-1

        if frame_idx == self.frame_position:
            # No need to update
            return

        # --- Read a frame data ---
        success, frame_data, frame_time = self.read_frame(frame_idx)
        if not success:
            sys.stderr.write(f"failed to read frame {frame_idx}.\n")
            sys.stderr.flush()
            return

        self.frame_position = frame_idx

        # --- Set tracking point positions ---
        for point_name in self.model.tracking_point:
            x, y = self.model.tracking_point[point_name].get_current_position()
            self.model.tracking_mark[point_name]['x'] = x
            self.model.tracking_mark[point_name]['y'] = y

        self.model.select_point_ui(update_plot=False)

        # -- Update linked UI ---
        # Time info label
        if self.positionLabel is not None:
            pos_t_str = str(timedelta(seconds=frame_time))
            if '.' in pos_t_str:
                pos_t_str = pos_t_str[:-4]
            else:
                pos_t_str += '.00'

            pos_txt = '{}/{} [{}/{} frame: {:.2f} Hz, {}x{}]'.format(
                    pos_t_str, self.duration_t_str,
                    self.frame_position+1, self.duration_frame,
                    self.frame_rate, frame_data.shape[1], frame_data.shape[0])
            self.positionLabel.setText(pos_txt)

        # --- Show the frame ---
        self.dispImg.set_frame(frame_data)

        # --- common time ---
        if self.model.on_sync and sync_update:
            self.model.common_time_ms = self.get_comtime_from_frame(frame_idx)
            self.model.set_common_time(self.model.common_time_ms, caller=self)

        # --- update marker bar ---
        self.model.show_marker()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def prev_frame(self):
        if self.frame_position == 0:
            return

        self.show_frame(self.frame_position-1)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def skip_fwd(self):
        # 1 sec forward
        frame = int(np.round(self.frame_position + self.frame_rate))
        if frame >= self.duration_frame:
            frame = self.duration_frame-1

        self.show_frame(frame)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def skip_bkw(self):
        # 1 sec rewind
        frame = int(np.round(self.frame_position - self.frame_rate))
        if frame < 0:
            frame = 0

        self.show_frame(frame)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def unload(self):
        self.filename = ''
        self.frame_rate = -1
        self.duration_frame = -1
        self.frame_position = 0
        self.duration_t_str = ''

        self.frameIdx = -1
        self.commonTimeSec = None
        self.frameCommonTimeSec = []  # commonTimeSec for each frame

        self.ui_setEnabled(False)
        self.syncBtn.setEnabled(False)

        # Delete tracking_point linked to this instance
        del_points = []
        for k, obj in self.model.tracking_point.items():
            if obj.dataMovie == self:
                del_points.append(k)

        for k in del_points:
            self.model.delete_point(point_name=k, ask_confirm=False)

        self.loaded = False
        self.dispImg.clear()

        # Reset sync status
        self.model.sync_video_thermal(False)
        if self.paired_data.loaded:
            self.model.main_win.syncVideoBtn.setEnabled(False)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_save_params(self):
        if not self.loaded:
            return None

        saving_params = ['filename', 'frame_position', 'shift_from_thermoTime']
        settings = {}
        for param in saving_params:
            obj = getattr(self, param)

            try:
                pickle.dumps(obj)
                settings[param] = obj
                continue
            except Exception:
                errmsg = f"{param} cannot be saved.\n"
                sys.stderr.write(errmsg)
                pass

        return settings


# %% VideoDataMovie class =====================================================
class VideoDataMovie(DataMovie):
    """Video data movie class
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
        self.duration_frame = \
            int(self.videoCap.get(cv2.CAP_PROP_FRAME_COUNT))

        super(VideoDataMovie, self).open(filename)

        # --- Set point transformation matrix ---
        frameData = self.dispImg.frameData  # Get framedata
        # Check margin
        xmean = frameData.mean(axis=(0, 2))
        ymean = frameData.mean(axis=(1, 2))

        xedge = np.argwhere(np.abs(np.diff(xmean)) > 50).ravel()
        yedge = np.argwhere(np.abs(np.diff(ymean)) > 50).ravel()
        if len(xedge):
            xshift = xedge[0] + 1
            xscale = np.diff(xedge)[0]/len(xmean)
        else:
            xshift = 0
            xscale = 1

        if len(yedge) > 0:
            yshift = yedge[0] + 1
            yscale = np.diff(yedge)[0]/len(ymean)
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


# %% ThermalDataMovie class ===================================================
class ThermalDataMovie(DataMovie):
    """ Thermal data movie class
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, parent, dispImg, UI_objs):
        super(ThermalDataMovie, self).__init__(parent, dispImg, UI_objs)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open(self, filename):
        if self.loaded:
            self.unload()

        filename = Path(filename)
        self.thermal_data_reader = None

        if filename.suffix == '.csq':
            # --- Open and read csq file ----------------------------------
            # Open progress dialog
            progressDlg = QProgressDialog('Reading thermal data ...',
                                          'Cancel', 0, 100,
                                          self.model.main_win)
            progressDlg.setWindowTitle("Reading thermal data")
            progressDlg.setWindowModality(Qt.WindowModal)
            progressDlg.resize(400, 89)
            progressDlg.show()

            try:
                cr = CSQ_READER(filename, progressDlg=progressDlg)
            except Exception as e:
                print(e)
                return

            self.thermal_data_reader = cr

            # Close progress dialog
            progressDlg.close()
            self.thermal_data_reader.progressDlg = None

        self.frame_rate = self.thermal_data_reader.FrameRate
        self.frame_rate = int(self.frame_rate * 100)/100

        self.duration_frame = self.thermal_data_reader.Count
        super(ThermalDataMovie, self).open(filename)

        # Reset video sync
        self.model.common_time_ms = 0
        self.model.common_duration_ms = (self.duration_frame /
                                         self.frame_rate) * 1000
        self.model.main_win.positionSlider.blockSignals(True)
        self.model.main_win.positionSlider.setRange(
            0, self.model.common_duration_ms)
        self.model.main_win.positionSlider.setValue(self.model.common_time_ms)
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
            frame_time = frame_idx * (1.0/self.frame_rate)
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
            fname = self.filename.parent / (self.filename.stem + '_thermo.mp4')
        else:
            fname = Path(fname)

        # --- Get frame data --------------------------------------------------
        frame_indices = np.arange(self.duration_frame, dtype=np.int)

        # Open progress dialog
        progressDlg = QProgressDialog('Reading thermal data ...',
                                      'Cancel', 0, len(frame_indices)*1.1,
                                      self.model.main_win)
        progressDlg.setWindowTitle("Export thermal data as video file")
        progressDlg.setWindowModality(Qt.WindowModal)
        progressDlg.resize(400, 89)
        progressDlg.show()

        thermal_data_array = \
            self.thermal_data_reader._get_thermal_data(frame_indices,
                                                       progressDlg=progressDlg)
        if thermal_data_array is None:
            # Cnaceled
            progressDlg.close()
            return None

        # Get value range
        progressDlg.setLabelText('Convert to gray image ...')
        gray_data = np.empty_like(thermal_data_array, dtype=np.uint8)
        for ii, frame in enumerate(thermal_data_array):
            progressDlg.setValue(len(frame_indices) + ii*0.1)
            progressDlg.repaint()
            low, high = np.percentile(frame.ravel(), [2.5, 97.5])
            gray_frame = np.tanh((frame - low) / (high - low) - 0.5) + 0.5
            gray_frame *= 255
            gray_frame[gray_frame < 0] = 0
            gray_frame[gray_frame > 255] = 255
            gray_data[ii, :, :] = gray_frame.astype(np.uint8)

        progressDlg.setValue(len(frame_indices)*1.1)
        progressDlg.setLabelText('Save movie file ...')
        progressDlg.repaint()

        """
        progressDlg.repaint()
        low, med, high = np.percentile(thermal_data_array.ravel(), [5, 50, 95])
        val_range = max(med-low, high-med)

        gray_data = (np.tanh((thermal_data_array - med) / val_range) + 1) / 2
        gray_data = (gray_data * 255).astype(np.uint8)
        """

        # --- Save as movie file ----------------------------------------------
        imageio.mimwrite(fname, gray_data, fps=self.frame_rate)

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

        return fname

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_frame(self, frame_idx=None, common_time_ms=None,
                   sync_update=True):
        super(ThermalDataMovie, self).show_frame(
            frame_idx, common_time_ms, sync_update)

        if self.model.main_win.plot_timeline is None:
            self.model.plot_timecourse()
        else:
            xpos = self.model.main_win.plot_xvals[self.frame_position]
            if self.model.main_win.plot_timeline.get_xdata()[0] != xpos:
                self.model.plot_timecourse()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_rois_dataseries(self, points_ts, rads, aggfuncs):
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
        read_frames = np.empty(0, dtype=np.int)
        for xyt in points_ts:
            mask = ~(np.any(np.isnan(xyt[:, :2].astype(np.float)), axis=1))
            read_frames = np.concatenate((read_frames, xyt[mask, 2]))
            read_frames = np.unique(read_frames)

        num_reading_frames = len(read_frames)
        if num_reading_frames == 0:
            return values

        # --- Read data -------------------------------------------------------
        show_progress = num_reading_frames > 50
        if show_progress:
            progressDlg = QProgressDialog('Reading thermal data ...',
                                          'Cancel', 0, num_reading_frames,
                                          self.model.main_win)
            progressDlg.setWindowTitle("Reading thermal data")
            progressDlg.setWindowModality(Qt.WindowModal)
            progressDlg.resize(250, 89)
            progressDlg.show()
            prog_n = 0

        for ii, frmIdx in enumerate(read_frames):
            for jj, xyt in enumerate(points_ts):
                rad = rads[jj]
                aggfunc = aggfuncs[jj]

                p_idx = np.argwhere(xyt[:, 2] == frmIdx).ravel()
                if len(p_idx) == 0:
                    continue

                if np.any(np.isnan(xyt[p_idx[0], :2])):
                    continue

                cx, cy, frmIdx = xyt[p_idx[0], :]
                val = self.thermal_data_reader.getCircleROIData(
                            frmIdx, cx, cy, rad, aggfunc)
                values[jj][p_idx[0]] = val

            if show_progress:
                if progressDlg.wasCanceled():
                    return None
                prog_n += 1
                progressDlg.setValue(prog_n)
                progressDlg.repaint()

        if show_progress:
            progressDlg.close()

        return values
