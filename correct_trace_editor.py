#!/usr/bin/env python
# -*- coding: utf-8 -*

# %% import ===================================================================
import sys
from pathlib import Path
from tkinter import filedialog

import csv
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy import fftpack


# %% TrackingCorrection
class TrackingCorrection:
    def __init__(self, rm_irregular_th=0.5,
                 irregular_correction_max_repeat=10,
                 lp_filt_freq=5,
                 flip_th=0.0):

        # Set parameters
        self.rm_irregular_th = rm_irregular_th
        self.irregular_correction_max_repeat = irregular_correction_max_repeat
        self.lp_filt_freq = lp_filt_freq
        self.flip_th = flip_th

        self._body_labals = {
            'human': ['pelvis', 'l_ankle', 'r_ankle'],
            'horse': ['withers', 'l_hoof', 'r_hoof']
        }

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def run(self, filename):
        self._filename = filename

        # Read file
        self._tp_df = self.open(filename)

        # Take trial periods
        trial_label = self.take_trial_periods()

        # Read body position time course
        body_positions = self.read_body_pos()

        # Process
        file_stem = Path(self._filename).stem
        for subj, subj_df in body_positions.items():
            for trial in trial_label.unique():
                if pd.isnull(trial):
                    continue

                tr_df = subj_df[trial_label == trial]
                correct_df = self.correct_flip_at_cross(tr_df)

                for col in correct_df.columns:
                    subj_df.loc[correct_df.index, col] = correct_df.loc[:, col]
                tr_df = subj_df[trial_label == trial]

                fig, ax = plt.subplots()
                fig.set_figwidth(16)
                fig.set_figheight(8)
                tr_df.l_foot_angle_ip.plot(ls=(0, (1, 4)), c='r', label=None)
                tr_df.l_foot_angle.plot(ls='-', c='r', label='Left')
                tr_df.r_foot_angle_ip.plot(ls=(0, (1, 4)), c='b', label=None)
                tr_df.r_foot_angle.plot(ls='-', c='b', label='Right')
                plt.ylabel('foot angle [-: fwd, +: bkwd]')

                seg = tr_df.index[tr_df.point_cross > 0]
                for x in seg:
                    plt.axvline(x, ls=':', c='k', picker=True, pickradius=3)
                plt.legend()
                plt.title(f"{file_stem}\n{subj} {trial}")
                fig.show()

                tr_df_rev = tr_df.copy()
                ep = TrackingCorrection.EditablePlot(
                    self, ax, tr_df_rev)
                ep.connect()
                tr_df_rev = ep.run()

                subj_df[trial_label == trial] = tr_df_rev

        # impute missing points
        imputed_body_positions = \
            self.impute_missing_points(trial_label, body_positions)

        # Save data
        self.save_data(trial_label, body_positions, imputed_body_positions)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def open(self, filename):
        with open(filename) as fd:
            self._header = ''
            for n in range(4):
                self._header += fd.readline()

        tp_df = pd.read_csv(filename, skiprows=0, header=[1, 2], index_col=0)
        self._orig_columns = tp_df.columns
        self._sample_freq = np.mean(1000/np.diff(tp_df.iloc[:, 0].values))

        return tp_df

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def take_trial_periods(self):
        marker_col = np.argwhere(
            [col[1] == 'marker' for col in self._tp_df.columns]).ravel()[0]
        marker = self._tp_df.iloc[:, marker_col]

        st_fr = np.argwhere(
            [pd.notnull(mk) and mk.endswith('start') for mk in marker]).ravel()
        end_fr = np.argwhere(
            [pd.notnull(mk) and mk.endswith('end') for mk in marker]).ravel()

        trial_label = pd.Series(index=self._tp_df.index)
        for st, en in zip(st_fr, end_fr):
            trial = marker[st].split('_')[0]
            trial_label.iloc[st:en] = trial

        return trial_label

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def read_body_pos(self):
        body_positions = {
            k: pd.DataFrame(index=self._tp_df.index, columns=col)
            for k, col in self._body_labals.items()}

        for subj, df in body_positions.items():
            for part in df.columns:
                xy = self._tp_df[[(f'{subj}_{part}', 'x'),
                                  (f'{subj}_{part}', 'y')]]
                df.loc[:, part] = [row.values for idx, row in xy.iterrows()]

        return body_positions

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def eval_foot_angel(self, base_pos, foot_pos):
        foot_angle = pd.Series(index=base_pos.index)
        for idx in base_pos.index:
            foot_angle[idx] = np.arctan2(*(foot_pos[idx] - base_pos[idx]))

        foot_angle0 = foot_angle.copy()
        x = foot_angle.index
        y = foot_angle.values
        mask = pd.notnull(y)
        cs = interpolate.CubicSpline(x[mask], y[mask], extrapolate=False)
        foot_angle_ip = cs(x)

        max_repeat = self.irregular_correction_max_repeat
        if self.rm_irregular_th > 0:
            for rep in range(max_repeat):
                spline_stress = np.abs(cs(x, 1)) + np.abs(cs(x, 2)) + \
                        np.abs(cs(x, 3))
                rm_mask0 = spline_stress > self.rm_irregular_th
                if np.sum(rm_mask0) == 0:
                    break

                rm_mask = rm_mask0.copy()
                on_points = np.argwhere(
                    np.diff(rm_mask0.astype(int)) > 0).ravel()
                off_points = np.argwhere(
                    np.diff(rm_mask0.astype(int)) < 0).ravel()
                for onp in on_points:
                    fw_up_points = np.argwhere(
                        np.diff(spline_stress[onp::-1]) > 0).ravel()
                    if len(fw_up_points):
                        fw_w = fw_up_points[0]
                        rm_mask[onp-fw_w:onp] = True
                    else:
                        rm_mask[:onp] = True

                for offp in off_points:
                    bw_up_points = np.argwhere(
                        np.diff(spline_stress[offp:]) > 0).ravel()
                    if len(bw_up_points):
                        bw_w = bw_up_points[0]+1
                        rm_mask[offp:offp+bw_w] = True
                    else:
                        rm_mask[offp:] = True

                mask = pd.notnull(y) & ~rm_mask
                cs = interpolate.CubicSpline(x[mask], y[mask],
                                             extrapolate=False)
                foot_angle_ip = cs(x)

            foot_angle[foot_angle != foot_angle_ip] = np.nan

        return foot_angle, foot_angle_ip, foot_angle0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_angle_df(self, df):
        df_out = df.copy()
        base_pos = df_out.iloc[:, 0].squeeze()
        l_foot_pos = df_out.iloc[:, 1].squeeze()
        r_foot_pos = df_out.iloc[:, 2].squeeze()

        l_foot_angle, l_foot_angle_ip, l_foot_angle0 = \
            self.eval_foot_angel(base_pos, l_foot_pos)
        r_foot_angle, r_foot_angle_ip, r_foot_angle0 = \
            self.eval_foot_angel(base_pos, r_foot_pos)

        df_out.loc[l_foot_angle.index, 'l_foot_angle'] = l_foot_angle
        df_out.loc[l_foot_angle.index, 'l_foot_angle_ip'] = l_foot_angle_ip
        df_out.loc[r_foot_angle.index, 'r_foot_angle'] = r_foot_angle
        df_out.loc[r_foot_angle.index, 'r_foot_angle_ip'] = r_foot_angle_ip

        # R -> L
        no_flip_diff = np.abs(l_foot_angle_ip - l_foot_angle0)
        flip_diff = np.abs(l_foot_angle_ip - r_foot_angle0)
        flip_frms = np.argwhere(
            ((flip_diff < (np.std(l_foot_angle_ip) / 10)) &
             (flip_diff < no_flip_diff)).values).ravel()

        cross_df = df_out.copy()
        cross_df.iloc[flip_frms, 1] = df_out.iloc[flip_frms, 2]
        cross_df.iloc[flip_frms, 2] = df_out.iloc[flip_frms, 1]
        cross_df.iloc[flip_frms, 3] = df_out.iloc[flip_frms, 5]
        cross_df.iloc[flip_frms, 5] = df_out.iloc[flip_frms, 3]
        cross_df.iloc[flip_frms, 4] = df_out.iloc[flip_frms, 6]
        cross_df.iloc[flip_frms, 6] = df_out.iloc[flip_frms, 4]
        df_out = cross_df

        # L -> R
        no_flip_diff = np.abs(r_foot_angle_ip - r_foot_angle0)
        flip_diff = np.abs(r_foot_angle_ip - l_foot_angle0)
        flip_frms = np.argwhere(
            ((flip_diff < (np.std(r_foot_angle_ip) / 10)) &
             (flip_diff < no_flip_diff)).values).ravel()

        cross_df = df_out.copy()
        cross_df.iloc[flip_frms, 1] = df_out.iloc[flip_frms, 2]
        cross_df.iloc[flip_frms, 2] = df_out.iloc[flip_frms, 1]
        cross_df.iloc[flip_frms, 3] = df_out.iloc[flip_frms, 5]
        cross_df.iloc[flip_frms, 5] = df_out.iloc[flip_frms, 3]
        cross_df.iloc[flip_frms, 4] = df_out.iloc[flip_frms, 6]
        cross_df.iloc[flip_frms, 6] = df_out.iloc[flip_frms, 4]
        df_out = cross_df

        return df_out

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fft_lpfilt(self, sig, filt_freq):
        sig0 = sig.copy()
        sig_mask = pd.notnull(sig)
        sig = sig[sig_mask].values
        sig_fft = fftpack.fft(sig)
        freqs = fftpack.fftfreq(sig.size, d=1.0/self._sample_freq)
        sig_fft[np.abs(freqs) > filt_freq] = 0
        filtered_sig = fftpack.ifft(sig_fft)
        sig0[sig_mask] = filtered_sig.real
        return sig0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def correct_flip_at_cross(self, trial_df):
        angle_df0 = self.get_angle_df(trial_df)
        angle_df = angle_df0.copy()

        angle_df['point_cross'] = 0
        cross_frame = 0
        while True:
            l_f_angle = angle_df.l_foot_angle_ip
            r_f_angle = angle_df.r_foot_angle_ip

            # Low-pass filtering
            l_f_angle_filt = self.fft_lpfilt(l_f_angle, self.lp_filt_freq)
            r_f_angle_filt = self.fft_lpfilt(r_f_angle, self.lp_filt_freq)

            # LR angle distance
            angle_dist = l_f_angle_filt - r_f_angle_filt
            angle_dist *= np.sign(np.mean(angle_dist))

            asum = np.abs(l_f_angle_filt) + np.abs(r_f_angle_filt) + \
                np.abs(angle_dist)
            asum_filt = self.fft_lpfilt(asum, self.lp_filt_freq/2)
            vcross = (np.sign(asum_filt.diff()).diff() > 0)

            # Find crossing frames
            cross_points = np.argwhere(vcross.values).ravel()
            cross_points = cross_points[cross_points > cross_frame]
            if len(cross_points) == 0 or cross_points[0] >= len(angle_df)-2:
                break

            cross_frame = cross_points[0]
            angle_df.iloc[cross_frame, -1] = 1

            st_fr = max(0, cross_frame-5)
            en_fr = min(cross_frame+5, len(l_f_angle))

            l_sp_chg = \
                np.abs(
                    np.nanmean(
                        l_f_angle_filt.diff().iloc[st_fr:cross_frame]) -
                    np.nanmean(
                        l_f_angle_filt.diff().iloc[cross_frame:en_fr]))
            r_sp_chg = \
                np.abs(
                    np.nanmean(
                        r_f_angle_filt.diff().iloc[st_fr:cross_frame]) -
                    np.nanmean(
                        r_f_angle_filt.diff().iloc[cross_frame:en_fr]))

            # Try replacing the points
            cross_df = angle_df.copy()
            cross_df.iloc[cross_frame:, 1] = angle_df.iloc[cross_frame:, 2]
            cross_df.iloc[cross_frame:, 2] = angle_df.iloc[cross_frame:, 1]
            cross_angle_df = self.get_angle_df(cross_df)

            cr_l_f_angle = cross_angle_df.l_foot_angle_ip
            cr_r_f_angle = cross_angle_df.r_foot_angle_ip

            # Low-pass filtering
            cr_l_f_angle_filt = self.fft_lpfilt(cr_l_f_angle,
                                                self.lp_filt_freq)
            cr_r_f_angle_filt = self.fft_lpfilt(cr_r_f_angle,
                                                self.lp_filt_freq)

            cr_l_sp_chg = \
                np.abs(
                    np.nanmean(
                        cr_l_f_angle_filt.diff().iloc[st_fr:cross_frame]) -
                    np.nanmean(
                        cr_l_f_angle_filt.diff().iloc[cross_frame:en_fr]))
            cr_r_sp_chg = \
                np.abs(
                    np.nanmean(
                        cr_r_f_angle_filt.diff().iloc[st_fr:cross_frame]) -
                    np.nanmean(
                        cr_r_f_angle_filt.diff().iloc[cross_frame:en_fr]))

            if (l_sp_chg - cr_l_sp_chg) > self.flip_th and \
                    (r_sp_chg - cr_r_sp_chg) > self.flip_th:
                # Velocity change becomes smooth with replacing
                angle_df = cross_angle_df

        return angle_df

    # /////////////////////////////////////////////////////////////////////////
    class EditablePlot():
        def __init__(self, parent, ax, data_df):
            self.ax = ax
            self.fig = ax.get_figure()
            self.data_df = data_df
            self.seg_x = np.array([ln.get_xdata()[0] for ln in ax.lines[2:]])
            self.press = False
            self.move_seg = None
            self.parent = parent

        def connect(self):
            self.cidpress = self.fig.canvas.mpl_connect(
                'button_press_event', self.on_press)
            self.cidrelease = self.fig.canvas.mpl_connect(
                'button_release_event', self.on_release)
            self.cidmotion = self.fig.canvas.mpl_connect(
                'motion_notify_event', self.on_motion)
            self.cidpick = self.fig.canvas.mpl_connect(
                'pick_event', self.on_pick)

        def on_press(self, event):
            if event.dblclick:
                xpos = event.xdata
                seg_x = np.array([ln.get_xdata()[0]
                                  for ln in self.ax.lines[2:]])
                xdata = self.data_df.index
                if xpos > seg_x[0]:
                    lx = seg_x[seg_x < xpos].max()
                else:
                    lx = xdata[0]

                if xpos < seg_x[-1]:
                    rx = seg_x[seg_x > xpos].min()
                else:
                    rx = xdata[-1]
                xmask = (xdata >= lx) & (xdata < rx)

                # Flip points
                rep_idx = np.argwhere(xmask).ravel()
                l_pos = self.data_df.iloc[rep_idx, 1]
                self.data_df.iloc[rep_idx, 1] = self.data_df.iloc[rep_idx, 2]
                self.data_df.iloc[rep_idx, 2] = l_pos

                # Refit line
                self.data_df = self.parent.get_angle_df(self.data_df)
                self.ax.lines[0].set_ydata(self.data_df.l_foot_angle_ip.values)
                self.ax.lines[1].set_ydata(self.data_df.l_foot_angle.values)
                self.ax.lines[2].set_ydata(self.data_df.r_foot_angle_ip.values)
                self.ax.lines[3].set_ydata(self.data_df.r_foot_angle.values)
                self.fig.canvas.draw()

        def on_motion(self, event):
            if self.press and self.move_seg is not None:
                xdata = event.xdata
                self.move_seg.set_xdata([xdata, xdata])
                self.fig.canvas.draw()

        def on_release(self, event):
            self.press = False
            self.move_seg = None

        def on_pick(self, event):
            print('on_pick')
            print(event)
            self.press = True

            self.move_seg = event.artist

        def run(self):
            fig = self.ax.figure

            # Wait for close
            while plt.fignum_exists(fig.number):
                plt.pause(0.1)
                continue

            return self.data_df

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def impute_missing_points(self, trial_label, body_positions):
        imputed_body_positions = {}
        for subj, subj_df in body_positions.items():
            imp_subj_df = subj_df.copy()
            imputed_body_positions[subj] = imp_subj_df

            for trial in trial_label.unique():
                if pd.isnull(trial):
                    continue
                tr_df = imp_subj_df[trial_label == trial]
                for col_i, lr in enumerate('lr'):
                    correcting_frames = np.argwhere(
                        pd.isnull(tr_df[f"{lr}_foot_angle"]) &
                        pd.notnull(tr_df.iloc[:, 0]) &
                        pd.notnull(tr_df[f"{lr}_foot_angle_ip"])).ravel()

                    for fr in correcting_frames:
                        if np.all(pd.isna(tr_df.iloc[fr:, col_i+1])):
                            break

                        prev_base_point = tr_df.iloc[fr-1, 0]
                        prev_foot_point = tr_df.iloc[fr-1, col_i+1]
                        if np.all(pd.isna(prev_base_point)) or \
                                np.all(pd.isna(prev_foot_point)):
                            continue
                        current_base_point = tr_df.iloc[fr, 0]
                        prev_angle, eval_angle = \
                            tr_df[f"{lr}_foot_angle_ip"].values[[fr-1, fr]]
                        rot = eval_angle - prev_angle
                        prev_vect = prev_foot_point - prev_base_point
                        RotMtx = np.array([[np.cos(-rot), -np.sin(-rot)],
                                           [np.sin(-rot), np.cos(-rot)]])
                        eval_vect = np.dot(RotMtx, prev_vect[:, None]).ravel()
                        eval_point = current_base_point + eval_vect
                        tr_df.iloc[fr, col_i+1][:] = eval_point

        return imputed_body_positions

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_data(self, trial_label, body_positions, imputed_body_positions):
        # LR corrected
        for subj, subj_df in body_positions.items():
            for bpart in subj_df.columns[:3]:
                xy = np.concatenate([
                    row[None, :] for row in subj_df[bpart].values])
                self._tp_df[(f"{subj}_{bpart}", 'x')] = xy[:, 0]
                self._tp_df[(bpart, 'y')] = xy[:, 1]

        corr_tp_df = self._tp_df[self._orig_columns]
        filename = Path(self._filename)
        cor_filename = filename.parent / \
            (filename.stem + '_corrected' + '.csv')
        with open(cor_filename, 'w') as fd:
            fd.write(self._header)
            corr_tp_df.to_csv(fd, header=False, quoting=csv.QUOTE_NONNUMERIC,
                              encoding='cp932')

        # LR corrected and impute missing
        tp_df_imp = self._tp_df.copy()
        for subj, subj_df in imputed_body_positions.items():
            for bpart in subj_df.columns[:3]:
                xy = np.concatenate([
                    row[None, :] for row in subj_df[bpart].values])
                tp_df_imp[(f"{subj}_{bpart}", 'x')] = xy[:, 0]
                tp_df_imp[(bpart, 'y')] = xy[:, 1]

        corr_imp_tp_df = tp_df_imp[self._orig_columns]
        filename = Path(self._filename)
        cor_filename = filename.parent / \
            (filename.stem + '_corrected_imputed' + '.csv')
        with open(cor_filename, 'w') as fd:
            fd.write(self._header)
            corr_imp_tp_df.to_csv(
                fd, header=False, quoting=csv.QUOTE_NONNUMERIC,
                encoding='cp932')

        # Save foot angles
        angle_df = pd.DataFrame(trial_label)
        angle_df.columns = ('trial',)
        for subj, subj_df in body_positions.items():
            cols = [f"{subj}_{col}" for col in subj_df.columns]
            subj_df.columns = cols
            angle_df = pd.concat([angle_df, subj_df.copy()], axis=1)

        angle_filename = filename.parent / \
            (filename.stem + '_f_angle' + '.csv')
        angle_df.to_csv(angle_filename)


# __main__ ====================================================================
if __name__ == '__main__':
    tracking_correction = TrackingCorrection()
    
    # Select tracking point files
    filetypes = (
        ('tracking csv files', '*tracking_points.csv'),
        ('csv files', '*.csv'),
        ('All files', '*.*')
        )

    filenames = filedialog.askopenfilenames(
        title='Open tracking points file',
        initialdir='./',
        filetypes=filetypes)
    if len(filenames) == 0:
        sys.exit()

    for filename in filenames:
        tracking_correction.run(filename)
