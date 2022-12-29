#!/usr/bin python3
# -*- coding: utf-8 -*-
"""
"""


# %% import
from pathlib import Path
import re
import subprocess
import sys
import platform
import pickle
import time
from datetime import datetime, timedelta

import numpy as np
import cv2
import ffmpeg
from tqdm import tqdm


## %% CSQ_READER ==============================================================
class CSQ_READER():
    """Read csq file

    DEBUG:
    fname = Path.home() / 'Dropbox/ThermalVideoTracking/TVT/Data/id21/id21.csq'
    self = CSQ_READER(fname)
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, fname, temp_dir=(Path.home() / '.TVT'),
                 progressDlg=None, extract_temp_file=False):

        # --- Set parameters --------------------------------------------------
        # Prepare the temp dir
        if not temp_dir.is_dir():
            temp_dir.mkdir()
        self.temp_dir = temp_dir

        # Set exiftool and ffmpeg commands
        if platform.system() == 'Windows':
            self.exiftool = 'exiftool.exe'
        else:
            self.exiftool = 'exiftool'

        self.progressDlg = progressDlg

        fname = Path(fname)

        self.metadata_list = None
        self.rawdata_list = None
        self.thermal_data_frames = None

        self.metadata_pkl_f = fname.parent / (fname.stem + '_meta.pkl')
        self.rawdata_pkl_f = fname.parent / (fname.stem + '_raw.pkl')
        self.extract_temp_file = extract_temp_file
        self.thermaldata_npy_f = fname.parent / (fname.stem + '_temp.npy')

        # -- Read and split binary data ---------------------------------------
        if not self.metadata_pkl_f.is_file() or not self.rawdata_pkl_f.is_file():
            #  Load binary data
            self._logmsg(f"Loading binary data from {fname.name} ...")
            st = time.time()

            with open(fname, 'rb') as fd:
                all_data = bytearray(fd.read())

            self._logmsg(f"Loading data from {fname.name} ... done.",
                         st=st, progress=5)

            # Split and check split
            self._logmsg("Check and correct data split ...")
            st = time.time()

            # Split data
            split_data = self._split(all_data, 'fff')

            # Check split
            split_data = self._check_split_error(split_data)

            self._logmsg("Check and correct data split ... done.",
                         st=st, progress=33)

        # --- Get metadata array ----------------------------------------------
        self._logmsg("Reading meta data ...")
        st = time.time()

        if not self.metadata_pkl_f.is_file():
            fields = ['Emissivity', 'ObjectDistance',
                      'ReflectedApparentTemperature', 'AtmosphericTemperature',
                      'IRWindowTemperature', "IRWindowTransmission",
                      'RelativeHumidity', 'PlanckR1', 'PlanckB', 'PlanckF',
                      'PlanckO', 'PlanckR2', 'RawThermalImageWidth',
                      'RawThermalImageHeight', 'DateTimeOriginal']
            self.metadata_list = self._read_metadata(
                split_data, fields, progress_base_total=[33, 33])
            with open(self.metadata_pkl_f, 'wb') as fd:
                pickle.dump(self.metadata_list, fd)

        else:
            with open(self.metadata_pkl_f, 'rb') as fd:
                self.metadata_list = pickle.load(fd)

        self._logmsg("Reading meta data ... done.", st=st, progress=66)

        # --- Set metadata ----------------------------------------------------
        self.Count = len(self.metadata_list)
        self.Width = self.metadata_list[0]['RawThermalImageWidth']
        self.Height = self.metadata_list[0]['RawThermalImageHeight']
        self.Duration = (self.metadata_list[-1]['DateTimeOriginal'] -
                         self.metadata_list[0]['DateTimeOriginal']
                         ).total_seconds()
        self.FrameRate = self.Count/self.Duration

        # -- Extract raw data to save in movie file ---------------------------
        self._logmsg("Extracting raw data ...")
        st = time.time()

        if not self.rawdata_pkl_f.is_file():
            self.rawdata_list = self._extract_rawdata(split_data)
            with open(self.rawdata_pkl_f, 'wb') as fd:
                pickle.dump(self.rawdata_list, fd)
        else:
            with open(self.rawdata_pkl_f, 'rb') as fd:
                self.rawdata_list = pickle.load(fd)

        self._logmsg("Extracting raw data ... done.", st=st, progress=99)

        self._logmsg("Done", progress=100)
        self.progressDl = None

        # -- Set temperature data --------------------------------------------
        if self.thermaldata_npy_f.is_file() and self.extract_temp_file:
            try:
                self.thermal_data_frames = np.load(self.thermaldata_npy_f,
                    mmap_mode='r+')
            except Exception:
                self.thermal_data_frames = np.ones(
                    [self.Count, self.Height, self.Width], dtype=np.float32) * np.nan
        else:
            self.thermal_data_frames = np.ones(
                [self.Count, self.Height, self.Width], dtype=np.float32) * np.nan

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _logmsg(self, msg, st=None, progress=None):
        if self.progressDlg is None:
            if st is not None:
                etstr = str(timedelta(seconds=time.time()-st)).split('.')[0]
                msg += f" (took {etstr})\n"

            print('\r' + msg, end='')
            sys.stdout.flush()
        else:
            assert not self.progressDlg.wasCanceled()

            self.progressDlg.setLabelText(msg.rstrip())
            if progress is not None:
                self.progressDlg.setValue(progress)
            self.progressDlg.repaint()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _split(self, all_data, sep='fff'):
        """
        Split the FLIR csq binary data

        Parameters
        ----------
        all_data : binrary string
            Binary data string.
        sep : string, optional
            Delimiter to split data. The default is 'fff'.

        Returns
        -------
        split_data : list of binary string
            List of data divided by sep.

        """

        Delims = {'fff': b"\x46\x46\x46\x00",
                  'csq': b"\x46\x46\x46\x00\x52\x54\x50",
                  'jpegls': b"\xff\xd8\xff\xf7"}
        div_pat = Delims[sep]
        split_data = re.split(div_pat, all_data)[1:]
        split_data = [div_pat + fr for fr in split_data]

        return split_data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _check_split_error(self, split_data):
        """ Check and correct errornous split
        """

        while True:

            # Save frame-wise data in .fff files
            for ff in self.temp_dir.glob('frame_*.fff'):
                ff.unlink()

            fffs = []
            for fidx in range(len(split_data)):
                tmp_f = self.temp_dir / f"frame_{fidx:07d}.fff"
                with open(tmp_f, 'wb') as fd:
                    fd.write(split_data[fidx])
                fffs.append(tmp_f)

            # Try reading data
            err_frms = []
            for kk in range(int(np.ceil(len(fffs) / 1000))):
                # Process 10k files
                cmd = f"{self.exiftool} -RawThermalImage"
                fname_pat = f"frame_{kk:04d}*.fff"
                cmd += f" -b {self.temp_dir / fname_pat}"
                pr = subprocess.run(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, shell=True)
                err_lines = pr.stderr.decode().rstrip().split('\n')

                for ll in err_lines:
                    ma = re.search(r'frame_(\d+).fff', ll)
                    if ma:
                        err_frms.append(int(ma.groups()[0]))

                self._logmsg(
                    "Check and correct data split ..." +
                    f" {min((kk+1)*1000, len(fffs))}/{len(fffs)}")

            for ff in self.temp_dir.glob('frame_*.fff'):
                ff.unlink()

            if len(err_frms) == 0:
                return split_data
            else:
                msgstr = f"Found error in frames, {err_frms}. Correct split.\n"
                self._logmsg(msgstr)

            err_frms = sorted(err_frms)
            corr_split_data = []
            stidx = 0
            while len(err_frms):
                err_idx = [err_frms.pop(0)]
                while len(err_frms) and err_frms[0] - err_idx[-1] == 1:
                    err_idx.append(err_frms.pop(0))

                corr_split_data += split_data[stidx:err_idx[0]]
                concat_data = split_data[err_idx[0]]
                for erri in err_idx[1:]:
                    concat_data += split_data[erri]
                corr_split_data.append(concat_data)
                stidx = err_idx[-1]+1

            corr_split_data += split_data[stidx:]

            split_data = corr_split_data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _read_metadata(self, split_data, fields, progress_base_total=[]):
        """
        Read metadata of frames.

        Parameters
        ----------
        split_data : list od bynary data
            List of frame data.
        fields : string list, optional
            List of metadata fileds to read. The default is [].

        Returns
        -------
        metadata : dictionary list
            List of metadata dictionary.
        """

        # Save frame data in fff files
        for ff in self.temp_dir.glob('frame_*.fff'):
            ff.unlink()

        fffs = []
        for ii, data in enumerate(split_data):
            tmp_f = self.temp_dir / f"frame_{ii:07d}.fff"
            with open(tmp_f, 'wb') as fd:
                fd.write(data)
            fffs.append(tmp_f)

        # Read metadata by exiftool
        metadata = []
        ii = 0
        for kk in range(int(np.ceil(len(fffs) / 1000))):
            cmd = f'{self.exiftool} ' + ' '.join([f"-{k}" for k in fields])
            fname_pat = f"frame_{kk:04d}*.fff"
            cmd += f" -q {self.temp_dir / fname_pat}"
            ostr = subprocess.check_output(cmd, shell=True)
            out_lines = ostr.decode().rstrip().split('\n')

            for ln in range(0, len(out_lines), len(fields)):
                metadata_frm = {}
                for ll in out_lines[ln:ln+len(fields)]:
                    key, val = re.search('(.+) : (.+)', ll).groups()
                    key = re.sub(r'[\s|/]', '', key)
                    if key in ('RawThermalImageWidth',
                               'RawThermalImageHeight'):
                        val = int(val)
                    elif key in ('Emissivity', 'ObjectDistance',
                                 'ReflectedApparentTemperature',
                                 'AtmosphericTemperature',
                                 'IRWindowTemperature',
                                 'IRWindowTransmission', 'RelativeHumidity',
                                 'PlanckR1', 'PlanckB', 'PlanckF', 'PlanckO',
                                 'PlanckR2'):
                        val = float(val.split()[0])
                    elif 'Date' in key:
                        if key == 'DateTimeOriginal':
                            val = datetime.strptime(
                                val, '%Y:%m:%d %H:%M:%S.%f%z')
                        else:
                            val = datetime.strptime(
                                val, '%Y:%m:%d %H:%M:%S%z')

                    metadata_frm[key] = val
                metadata.append(metadata_frm)

                if self.progressDlg is not None:
                    if self.progressDlg.wasCanceled():
                        return None

                ii += 1
                msg = f"Reading meta data ... {ii}/{len(fffs)}"
                if len(progress_base_total):
                    start, total = progress_base_total
                    progress = (ii/len(fffs))*total+start
                else:
                    progress = None
                self._logmsg(msg, progress=progress)

        # Cleaning temporal data
        for ff in fffs:
            if ff.is_file():
                ff.unlink()

        return metadata

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _extract_rawdata(self, split_data):
        """
        Parameters
        ----------
        split_data : TYPE
            DESCRIPTION.
        save_f : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        frame_rawdata : TYPE
            DESCRIPTION.

        """

        # Save frame-wise data in .fff files
        for ff in self.temp_dir.glob('frame_*.fff'):
            ff.unlink()

        fffs = []
        for fidx in range(len(split_data)):
            tmp_f = self.temp_dir / f"frame_{fidx:07d}.fff"
            with open(tmp_f, 'wb') as fd:
                fd.write(split_data[fidx])
            fffs.append(tmp_f)

        # Extract raw binary data
        frame_rawdata = []
        for kk in range(int(np.ceil(len(fffs) / 1000))):
            # Process 1k files
            cmd = f"{self.exiftool} -RawThermalImage"
            fname_pat = f"frame_{kk:04d}*.fff"
            cmd += f" -b {self.temp_dir / fname_pat}"
            pr = subprocess.run(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, shell=True)
            raw_data = bytearray(pr.stdout)
            # Split
            frame_rawdata += self._split(raw_data, sep='jpegls')
            self._logmsg("Extracting raw data ..." +
                         f" {min((kk+1)*1000, len(fffs))}/{len(fffs)}")

        # Clean fff files
        for ff in self.temp_dir.glob('frame_*.fff'):
            ff.unlink()

        return frame_rawdata

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _raw2temp(self, raw, E=1, OD=1, RTemp=20, ATemp=20, IRWTemp=20, IRT=1,
                  RH=50, PR1=21106.77, PB=1501, PF=1, PO=-7340,
                  PR2=0.012545258):
        """ Convert raw values from the flir sensor to temperatures in C
            This calculation has been ported to python from
            https://github.com/gtatters/Thermimage/blob/master/R/raw2temp.R
            https://github.com/nationaldronesau/FlirImageExtractor/blob/55cc4e2563e606ea61dd199a8a6d078b352031c2/flir_image_extractor_cli/flir_image_extractor.py#L277
        """

        # constants
        ATA1 = 0.006569
        ATA2 = 0.01262
        ATB1 = -0.002276
        ATB2 = -0.00667
        ATX = 1.9

        # transmission through window (calibrated)
        emiss_wind = 1 - IRT
        refl_wind = 0

        # transmission through the air
        h2o = (RH/100) * np.exp(
                1.5587 + 0.06939 * ATemp - 0.00027816 * ATemp**2
                + 0.00000068455 * ATemp**3)
        tau1 = ATX * np.exp(-np.sqrt(OD/2) * (ATA1+ATB1*np.sqrt(h2o))) + \
            (1-ATX) * np.exp(-np.sqrt(OD/2) * (ATA2+ATB2*np.sqrt(h2o)))
        tau2 = ATX * np.exp(-np.sqrt(OD/2) * (ATA1+ATB1*np.sqrt(h2o))) + \
            (1-ATX) * np.exp(-np.sqrt(OD/2) * (ATA2+ATB2*np.sqrt(h2o)))

        # radiance from the environment
        raw_refl1 = PR1 / (PR2 * (np.exp(PB/(RTemp+273.15)) - PF)) - PO
        raw_refl1_attn = (1 - E) / E * raw_refl1
        raw_atm1 = PR1 / (PR2 * (np.exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm1_attn = (1 - tau1) / E / tau1 * raw_atm1
        raw_wind = PR1 / (PR2 * (np.exp(PB / (IRWTemp + 273.15)) - PF)) - PO
        raw_wind_attn = emiss_wind / E / tau1 / IRT * raw_wind
        raw_refl2 = PR1 / (PR2 * (np.exp(PB / (RTemp + 273.15)) - PF)) - PO
        raw_refl2_attn = refl_wind / E / tau1 / IRT * raw_refl2
        raw_atm2 = PR1 / (PR2 * (np.exp(PB / (ATemp + 273.15)) - PF)) - PO
        raw_atm2_attn = (1 - tau2) / E / tau1 / IRT / tau2 * raw_atm2

        raw_obj = (
            raw / E / tau1 / IRT / tau2
            - raw_atm1_attn
            - raw_atm2_attn
            - raw_wind_attn
            - raw_refl1_attn
            - raw_refl2_attn
        )

        # temperature from radiance
        temp_celcius = PB / np.log(PR1 / (PR2 * (raw_obj + PO)) + PF) - 273.15
        return temp_celcius

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def _get_thermal_data(self, frame_indices, progressDlg=None, update=False,
                          verb=False):

        # --- Check frame index -----------------------------------------------
        frame_indices = np.array(frame_indices)
        try:
            out_range_idx = np.argwhere(
                (frame_indices < 0) | (frame_indices >= self.Count)).ravel()
            assert len(out_range_idx) == 0

        except Exception:
            sys.stderr.write(
                f"Out of range (0 - {self.Count}-1) index:"
                f" {frame_indices[out_range_idx]}\n")
            return None

        # --- Read raw data from videoCap -------------------------------------
        h, w = (self.Height, self.Width)
        thermal_data_array = np.empty([len(frame_indices), h, w])

        if progressDlg is None and len(frame_indices) > 100:
            pbar = tqdm(total=len(frame_indices),
                desc='Reading thermal data')
        else:
            pbar = None

        for ii, fidx in enumerate(frame_indices):
            fridx = int(np.round(fidx)) 
            if progressDlg is not None:
                if progressDlg.wasCanceled():
                    return None
                progressDlg.setValue(ii)
                progressDlg.repaint()

            thermal_data = self.thermal_data_frames[fridx, :, :]
            if not update and not np.any(np.isnan(thermal_data)):
                thermal_data_array[ii, :, :] = thermal_data
            else:
                input_data = self.rawdata_list[fridx]
                process = (
                    ffmpeg
                    .input('pipe:')
                    .output('pipe:', format='rawvideo', pix_fmt='gray16le')
                    .run_async(pipe_stdin=True, pipe_stdout=True, quiet=True)
                    )

                buffer, _ = process.communicate(input=input_data)
                raw = np.frombuffer(buffer, np.uint16).reshape([h, w])
                meta_data = self.metadata_list[fridx]

                thermal_data = self._raw2temp(
                    raw,
                    E=meta_data['Emissivity'],
                    OD=meta_data['ObjectDistance'],
                    RTemp=meta_data['ReflectedApparentTemperature'],
                    ATemp=meta_data['AtmosphericTemperature'],
                    IRWTemp=meta_data['IRWindowTemperature'],
                    IRT=meta_data["IRWindowTransmission"],
                    RH=meta_data['RelativeHumidity'],
                    PR1=meta_data['PlanckR1'],
                    PB=meta_data['PlanckB'],
                    PF=meta_data['PlanckF'],
                    PO=meta_data['PlanckO'],
                    PR2=meta_data['PlanckR2']
                    )
                thermal_data_array[ii, :, :] = thermal_data

                # Save read thermal_data into self.thermal_data_frames
                self.thermal_data_frames[fridx, :, :] = thermal_data

            if pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.close()

        return thermal_data_array

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def getFramebyIdx(self, frame_idx):
        try:
            assert frame_idx >= 0 and frame_idx < self.Count
        except Exception:
            sys.stderr.write('{} is out of range [0 - {}).\n'.format(
                    frame_idx, self.Count-1))
            return None

        thermal_data = np.squeeze(self._get_thermal_data([frame_idx]))

        return thermal_data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def getCircleROIData(self, frame_idx, cx, cy, rad, aggfunc='mean'):
        try:
            assert frame_idx >= 0 and frame_idx < self.Count
        except Exception:
            sys.stderr.write('{} is out of range [0 - {}).\n'.format(
                    frame_idx, self.Count-1))
            return None

        # Preapare mask
        mask = np.zeros([self.Height, self.Width], dtype=np.uint8)
        mask = cv2.circle(mask, (int(cx), int(cy)), rad, (1, 1, 1), -1)

        # Read thermal data
        frame = np.squeeze(self._get_thermal_data([frame_idx]))

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
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def saveTempFrames(self, frame_indices=None, update=False):
        if update:
            if frame_indices is None:
                frame_indices = np.arange(0, self.Count, dtype=int)
            _ = self._get_thermal_data(frame_indices, update=update)
        
        np.save(self.thermaldata_npy_f, self.thermal_data_frames)

## %% main =====================================================================
if __name__ == '__main__':
    fname = '../data/dog_EmotionalContingency/FLIR6551.csq'
    csq_read = CSQ_READER(fname)
    csq_read.saveTempFrames()



