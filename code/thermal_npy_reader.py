# -*- coding: utf-8 -*-
"""
@author: mamis
"""


# %% import ===================================================================
from pathlib import Path
import numpy as np
import sys
import cv2


# %% THERMAL_NPY_READER class =================================================
class THERMAL_NPY_READER():
    """THERMAL_NPY_READER
    Data reader class for the thermal data in numpy converted from the FLIR
    csq.
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, fname):
        # Load aux info
        fname = Path(fname)
        npzf = np.load(fname)
        for attr in npzf.keys():
            if attr == 'frames':
                self.thermalData = npzf['frames']
            else:
                setattr(self, attr, npzf[attr])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def getFramebyIdx(self, frame_idx):
        try:
            assert frame_idx >= 0 and frame_idx < self.Count
        except Exception:
            sys.stderr.write('{} is out of range [0 - {}).\n'.format(
                    frame_idx, self.Count))
            return None

        frame = self.thermalData[frame_idx, :, :]
        return frame

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def getCircleROIData(self, frame_idx, cx, cy, rad, aggfunc='mean'):
        try:
            assert frame_idx >= 0 and frame_idx < self.Count
        except Exception:
            sys.stderr.write('{} is out of range [0 - {}).\n'.format(
                    frame_idx, self.Count))
            return None

        # Preapare mask
        mask = np.zeros([rad*2, rad*2], dtype=np.uint8)
        mask = cv2.circle(mask, (rad, rad), rad, (1, 1, 1), -1)
        mask[0, :] = 0
        mask[:, 0] = 0

        # Set frame corner
        left = cx - rad
        top = cy - rad
        right = left + rad*2
        bottom = top + rad*2

        # Strip out of image area
        if left < 0:
            mask = mask[:, -left:]
            left = 0

        if top < 0:
            mask = mask[-top:, :]
            top = 0

        if right > self.thermalData.shape[2]:
            mask = mask[:, :-(right-self.thermalData.shape[2])]
            right = self.thermalData.shape[2]

        if bottom > self.thermalData.shape[1]:
            mask = mask[:-(bottom-self.thermalData.shape[1]), :]
            bottom = self.thermalData.shape[1]

        # Read thermal data
        frame_idx = int(frame_idx)
        top = int(top)
        bottom = int(bottom)
        left = int(left)
        right = int(right)
        frame = self.thermalData[frame_idx, top:bottom, left:right]

        # Return masked region
        if aggfunc == 'mean':
            return np.mean(frame[mask > 0])
        elif aggfunc == 'median':
            return np.median(frame[mask > 0])
        elif aggfunc == 'max':
            return np.max(frame[mask > 0])
        elif aggfunc == 'min':
            return np.min(frame[mask > 0])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def getFramebyTime(self, t_sec):
        try:
            assert t_sec >= 0 and t_sec < self.Duration
        except Exception:
            sys.stderr.write('{} is out of range [0 - {}).\n'.format(
                    t_sec, self.Duration))
            return None

        frame_idx = np.round(t_sec * self.FrameRate)
        return self.getFramebyIdx(frame_idx)
