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

from PySide6.QtCore import Qt, QPoint
from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap


# %% DataMovie class ==========================================================
class DataMovie():
    """ Base class of DataMovie
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

        # Movie position
        self.frame_position = -1
        self.mstime_position = -1

        # Connect callback
        self.frFwdBtn.clicked.connect(
                partial(self.show_frame, **{'frame_idx': None}))
        self.frBkwBtn.clicked.connect(self.prev_frame)
        self.skipFwdBtn.clicked.connect(self.skip_fwd)
        self.skipBkwBtn.clicked.connect(self.skip_bkw)

        # Time shift from the common time of video and thermal data
        # N.B. comtime is thermal time, so this is always 0 for thermal data
        self.shift_from_refTime = 0

        self.on_sync = False

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open(self, filename):
        self.filename = Path(filename)
        duration_t = timedelta(
                seconds=self.duration_frame/self.frame_rate)
        duration_t_str = str(duration_t)

        # Enable control buttons
        self.ui_setEnabled(True)
        if self.paired_data is not None and self.paired_data.loaded:
            if hasattr(self, 'syncBtn'):
                self.syncBtn.setEnabled(True)

        self.loaded = True

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
            if hasattr(self.model.main_win, btn):
                getattr(self.model.main_win, btn).setEnabled(enabled)

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
                pos_t_str = pos_t_str
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
        self.mstime_position = self.get_comtime_from_frame(frame_idx)
        if sync_update:
            self.model.common_time_ms = self.mstime_position
            self.model.main_win.positionSlider.blockSignals(True)
            self.model.main_win.positionSlider.setValue(self.mstime_position)
            self.model.main_win.positionSlider.blockSignals(False)
            if self.model.on_sync:
                self.model.set_common_time(self.model.common_time_ms,
                                           caller=self)

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
    def get_frame_from_comtime(self, common_time_ms):
        local_time_ms = common_time_ms + self.shift_from_refTime
        ms_per_frame = 1000 / self.frame_rate
        return int(np.round(local_time_ms / ms_per_frame))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_comtime_from_frame(self, frame_idx):
        common_time_ms = 1000 * (frame_idx / self.frame_rate)
        common_time_ms -= self.shift_from_refTime

        return common_time_ms

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
        if hasattr(self, 'syncBtn'):
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

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_save_params(self, saving_params=['filename', 'frame_position']):
        if not self.loaded:
            return None

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


# %% DisplayImage class =======================================================
class DisplayImage(QLabel):
    """ Display image class
    View class for the movie images.
    Handling display image; show a frame image and handling click event
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, parent, frame_w=640, frame_h=480, cmap=cv2.COLORMAP_JET,
                 clim=None):
        super().__init__(parent)

        self.parent = parent  # QMainWindow class object
        self.cmap = cmap
        self.clim = clim

        self.frame_w = frame_w
        self.frame_h = frame_h
        self.frameData = None

        self.zoom_factor = 1.0
        self.pan_start = QPoint(0, 0)
        self.current_pan = QPoint(0, 0)

        self.tracking_mark = None
        # self.tracking_mark will be the reference to
        # self.parent.model.tracking_mark
        self.point_mark_xy = None
        self.shift_scale_Mtx = None

        self.moving = None
        self.panning = False
        self.zoomc_imx = -1
        self.zoomc_imy = -1
        self.disp_x0 = 0
        self.disp_y0 = 0

        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background:rgba(0, 0, 0, 255);")
        self.resize(640, 480)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_frame(self, frameData):
        self.frameData = frameData
        self.frame_h, self.frame_w = self.frameData.shape[:2]
        self.set_pixmap()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_pixmap(self):
        if self.frameData is None:
            return

        # Make QImage
        frame = self.frameData
        if frame.ndim != 3:
            if self.clim is None:
                self.cmin, self.cmax = np.min(frame), np.max(frame)
            else:
                self.cmin, self.cmax = self.clim

            frame = (frame - self.cmin) / (self.cmax - self.cmin)
            frame[frame < 0] = 0
            frame[frame > 1.0] = 1.0
            frame *= 255
            frame = 255-frame
            frame = cv2.applyColorMap(frame.astype(np.uint8), self.cmap)
        else:
            frame = frame.astype(np.uint8)

        bytesPerLine = 3 * self.frame_w
        qimg = QImage(frame.flatten(), self.frame_w, self.frame_h,
                      bytesPerLine, QImage.Format_RGB888)

        # --- Paint on qimg ---
        painter = QPainter()
        painter.begin(qimg)

        # Draw online point
        painter.setPen(QPen(Qt.black, 4, Qt.SolidLine))
        xp, yp = self.point_mark_xy
        if self.shift_scale_Mtx is not None:
            xp, yp = np.dot(self.shift_scale_Mtx, [xp, yp, 1])[:2]
        painter.drawPoint(xp, yp)

        # Draw measure points
        for point_name in self.tracking_mark.keys():
            x = self.tracking_mark[point_name]['x']
            y = self.tracking_mark[point_name]['y']
            if np.isnan(x) or np.isnan(y):
                continue

            if self.shift_scale_Mtx is not None:
                x, y = np.dot(self.shift_scale_Mtx, [x, y, 1])[:2]

            pen_color = self.tracking_mark[point_name]['pen_color']
            rad = self.tracking_mark[point_name]['rad']

            painter.setPen(eval('Qt.{}'.format(pen_color)))
            painter.drawEllipse(x-rad, y-rad, rad*2, rad*2)
            if hasattr(self.parent, 'roi_showName_chbx'):
                if self.parent.roi_showName_chbx.checkState() > 0:
                    painter.drawText(QPoint(x, y), point_name)
            else:
                painter.drawText(QPoint(x, y), point_name)

        painter.end()

        pix = QPixmap.fromImage(qimg)
        # Apply zoom and pan transformations
        if self.zoom_factor > 1:
            # Crop
            self.disp_x0 = max(int(np.round(self.zoomc_imx -
                               (self.frame_w/2)/self.zoom_factor)), 0)
            self.disp_y0 = max(int(np.round(self.zoomc_imy -
                               (self.frame_h/2)/self.zoom_factor)), 0)
            crop_w = min(int(np.round(self.frame_w / self.zoom_factor)),
                         self.frame_w)
            crop_h = min(int(np.round(self.frame_h / self.zoom_factor)),
                         self.frame_h)
            pix = pix.copy(self.disp_x0, self.disp_y0, crop_w, crop_h)

            self.zoomc_imx = int(np.round(crop_w / 2 + self.disp_x0))
            self.zoomc_imy = int(np.round(crop_h / 2 + self.disp_y0))
        else:
            self.zoomc_imx = self.frame_w // 2
            self.zoomc_imy = self.frame_h // 2
            self.disp_x0 = 0
            self.disp_y0 = 0

        aspect = self.frame_w / self.frame_h
        map_w = self.width()
        map_h = self.height()
        if map_w > map_h * aspect:
            map_w = map_h * aspect
        elif map_h > map_w / aspect:
            map_h = map_w / aspect

        pix = pix.scaled(map_w, map_h, Qt.KeepAspectRatio)
        self.setPixmap(pix)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_image_xy(self, px, py):
        """
        translate display frame x, y to image position
        """

        if self.pixmap() is None:
            return

        imageFrame_aspect = self.frame_w / self.frame_h
        displayArea_aspect = self.width() / self.height()

        if imageFrame_aspect > displayArea_aspect:
            # height margin
            disp2frame_scale = self.frame_w / self.width()
            h_margin = (self.height() - (self.width() / imageFrame_aspect))/2
            py -= h_margin
        else:
            # width margin
            disp2frame_scale = self.frame_h / self.height()

        x = np.round(px * disp2frame_scale) - 0.5
        y = np.round(py * disp2frame_scale) - 0.5
        if self.shift_scale_Mtx is not None:
            invMtx = np.linalg.inv(self.shift_scale_Mtx)
            x, y = np.dot(invMtx, [x, y, 1])[:2]

        if self.zoom_factor > 1.0:
            x = self.disp_x0 + x / self.zoom_factor
            y = self.disp_y0 + y / self.zoom_factor

        x = min(max(int(x), 0), self.frame_w-1)
        y = min(max(int(y), 0), self.frame_h-1)

        return x, y

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mouseDoubleClickEvent(self, e):
        if e.button() == Qt.RightButton and len(self.tracking_mark):
            # Add a point at empty frame
            point_name = self.parent.roi_idx_cmbbx.currentText()
            if np.isnan(self.tracking_mark[point_name]['x']):
                px = e.pos().x()
                py = e.pos().y()
                x, y = self.get_image_xy(px, py)
                self.tracking_mark[point_name]['x'] = x
                self.tracking_mark[point_name]['y'] = y
                self.parent.model.edit_point_signal.emit(point_name)
                self.parent.model.select_point_ui_signal.emit(point_name)
                return

        elif e.modifiers() & Qt.ShiftModifier:
            # Add a new point
            px = e.pos().x()
            py = e.pos().y()
            x, y = self.get_image_xy(px, py)

            k = len(self.tracking_mark.keys())+1
            while str(k) in self.tracking_mark:
                k += 1

            # self.tracking_mark is a referense to
            # self.parent.model.tracking_mark, but its entry made in this
            # function could be local. So the new entry is added to
            # self.parent.model.tracking_mark explictly.
            self.parent.model.tracking_mark[str(k)] = {'x': x, 'y': y}
            self.parent.model.edit_point_signal.emit(str(k))
            return

        elif len(self.tracking_mark):
            # Move the point
            point_name = self.parent.roi_idx_cmbbx.currentText()
            if not np.isnan(self.tracking_mark[point_name]['x']):
                px = e.pos().x()
                py = e.pos().y()
                x, y = self.get_image_xy(px, py)
                self.tracking_mark[point_name]['x'] = x
                self.tracking_mark[point_name]['y'] = y
                self.parent.model.edit_point_signal.emit(point_name)
                self.parent.model.select_point_ui_signal.emit(point_name)
                return

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def MouseClickEvent(self, event):
        px = event.pos().x()
        py = event.pos().y()
        im_x, im_y = self.get_image_xy(px, py)
        print(im_x, im_y)

        if event.button() == Qt.LeftButton:
            self.point_mark_xy[0] = im_x
            self.point_mark_xy[1] = im_y

            # Click on a tracking point?
            for k in self.tracking_mark.keys():
                dx = self.tracking_mark[k]['x']
                dy = self.tracking_mark[k]['y']
                rad = self.tracking_mark[k]['rad']
                if np.abs(im_x-dx) <= rad and np.abs(im_y-dy) <= rad:
                    self.parent.model.select_point_ui_signal.emit(k)
                    break

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mousePressEvent(self, event):
        px = event.pos().x()
        py = event.pos().y()
        im_x, im_y = self.get_image_xy(px, py)

        if event.button() == Qt.LeftButton:
            self.point_mark_xy[0] = im_x
            self.point_mark_xy[1] = im_y
            self.parent.model.move_point_signal.emit()

            # Click on point?
            for point_name in self.tracking_mark.keys():
                dx = self.tracking_mark[point_name]['x']
                dy = self.tracking_mark[point_name]['y']
                rad = self.tracking_mark[point_name]['rad']
                if np.abs(im_x-dx) <= rad and np.abs(im_y-dy) <= rad:
                    self.moving = point_name
                    break

        elif event.button() == Qt.RightButton:
            if self.moving is None:
                if self.zoom_factor > 1.0:
                    self.panning = True
                    self.pan_start = event.pos()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mouseMoveEvent(self, event):
        px = event.pos().x()
        py = event.pos().y()
        im_x, im_y = self.get_image_xy(px, py)

        if event.buttons() & Qt.LeftButton:
            self.point_mark_xy[0] = im_x
            self.point_mark_xy[1] = im_y
            if self.moving is not None:
                self.tracking_mark[self.moving]['x'] = im_x
                self.tracking_mark[self.moving]['y'] = im_y
                self.parent.model.select_point_ui_signal.emit(self.moving)
                self.parent.model.move_point_signal.emit()
        elif event.buttons() & Qt.RightButton:
            if self.panning:
                self.current_pan = event.pos() - self.pan_start
                self.zoomc_imx -= self.current_pan.x()
                self.zoomc_imy -= self.current_pan.y()
                self.pan_start = event.pos()
                self.set_pixmap()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def mouseReleaseEvent(self, event):
        if self.moving is not None:
            px = event.pos().x()
            py = event.pos().y()
            im_x, im_y = self.get_image_xy(px, py)

            self.tracking_mark[self.moving]['x'] = im_x
            self.tracking_mark[self.moving]['y'] = im_y
            self.parent.model.edit_point_signal.emit(self.moving)
            self.parent.model.select_point_ui_signal.emit(self.moving)
            self.moving = None
        elif self.panning:
            self.panning = False
            self.current_pan = QPoint(0, 0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def wheelEvent(self, event):
        # Zoom in or out
        zoom_frx = event.position().x()
        zoom_fry = event.position().y()
        self.zoomc_imx, self.zoomc_imy = \
            self.get_image_xy(zoom_frx, zoom_fry)
        num_degrees = event.angleDelta() / 8
        num_steps = num_degrees / 30
        self.zoom_factor *= 1.1 ** num_steps.y()
        self.zoom_factor = max(1.0, self.zoom_factor)
        self.set_pixmap()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def resizeEvent(self, evt):
        self.set_pixmap()
