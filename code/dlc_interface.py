# -*- coding: utf-8 -*-
"""
DeepLabCut interface class, DLCinter
Most of the comments for functions were copied from Nath et al., (2019).

Use 'pythonw' to boot this on anaconda environment.

Nath, T., Mathis, A., Chen, A.C., Patel, A., Bethge, M., Mathis, M.W., 2019.
Using DeepLabCut for 3D markerless pose estimation across species and
behaviors. Nat Protoc.

@author: mmisaki
"""


# %% import
import os
import sys
import shutil
from pathlib import Path
import io
import subprocess
import shlex
import platform
import socket
import re
import pickle
from contextlib import redirect_stdout
import traceback
import unicodedata
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import deeplabcut as dlc
from deeplabcut.utils import auxiliaryfunctions
import yaml


# %%
def slugify(value, allow_unicode=True):
    """
    Taken from
    https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value)
        value = value.encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


# %% DLCinter
class DLCinter():
    """ Model class. Interface to deeplabcut
    """

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, DATA_ROOT, config_path=None, ui_parent=None):
        """Initialize the DLCinter class

        Parametes
        ---------
        config_path: Path or string (optional)
            Path to a config file (.yaml).
        """
        self.OS = platform.system()
        self.HOSTNAME = slugify(socket.gethostname())

        self.DATA_ROOT = DATA_ROOT
        self._config_path = None  # config file portable across hosts
        self._config_work_path = None  # config file used in the current host
        if config_path is not None:
            self.config_conv(config_path)

        self.ui_parent = ui_parent

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_msg(self, msg):
        if self.ui_parent is not None:
            self.ui_parent.msg_dlg(msg)
        else:
            sys.stdout.write(msg+'\n')
            sys.stdout.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def show_err_msg(self, msg):
        if self.ui_parent is not None:
            self.ui_parent.error_MessageBox(msg)
        else:
            sys.stderr.write(msg+'\n')
            sys.stderr.flush()

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def check_config_file(self, popup_err=True):
        if self._config_work_path is None or \
                not self._config_work_path.is_file():
            if popup_err:
                self.show_err_msg('No DLC config file is set.')
            return False

        else:
            return True

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, config_path0):
        if not Path(config_path0).is_file():
            self.show_err_msg(f"Not found {config_path0}.")
            return

        self.set_config(config_path0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def boot_dlc_gui(self):
        cmd = 'python -m deeplabcut'
        subprocess.Popen(shlex.split(cmd))

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def set_config(self, config_path0):
        """Set config file (_config_path, _config_work_path) with converting
        paths. Convert paths in config to a relative one from DATA_ROOT or vice
        versa.

        If the config includes ${DATA_ROOT}, a config file with full-path
        is created and set to self.config_work_path.
        Otherwise, paths in the config are converted to relative to DATA_ROOT,
        and a converted config is saved in a file to set in self.config_path.
        """

        # Read config file
        with open(config_path0, 'r') as stream:
            config_data = yaml.safe_load(stream)

        # Convert paths
        if '${DATA_ROOT}' in config_data['project_path']:
            # File with relative path is read.
            # convert to absolute path
            project_path = config_data['project_path']
            project_path = project_path.replace('${DATA_ROOT}/', '')
            project_path = str((self.DATA_ROOT / project_path).resolve())

            video_sets = {}
            for vf0 in config_data['video_sets'].keys():
                vf = vf0.replace('${DATA_ROOT}/', '')
                vf = str((self.DATA_ROOT / vf).resolve())
                video_sets[vf] = config_data['video_sets'][vf0]

            config_data['project_path'] = project_path
            config_data['video_sets'] = video_sets

            out_f = Path(config_path0).parent / f'config_{self.HOSTNAME}.yaml'
            self._config_path = config_path0
            self._config_work_path = out_f

        else:
            # File with absolute path is read.
            if not Path(config_data['project_path']).is_dir():
                errmsg = "Not found project_path,"
                errmsg += f" {config_data['project_path']},"
                errmsg += f" in {config_path0}.\n"
                sys.stderr.write(errmsg)
                return

            # convert to relative path
            project_path = Path(config_data['project_path']).resolve()
            DATA_ROOT = self.DATA_ROOT.resolve()
            try:
                project_path = os.path.relpath(project_path, DATA_ROOT)
                project_path = str(project_path).replace(os.sep, '/')
                project_path = '${DATA_ROOT}/' + project_path
            except Exception:
                project_path = Path(config_data['project_path'])
                project_path = str(project_path).replace(os.sep, '/')
                print(project_path)

            video_sets = {}
            for vf0 in config_data['video_sets'].keys():
                try:
                    vf = os.path.relpath(Path(vf0).resolve(), DATA_ROOT)
                    vf = str(vf).replace(os.sep, '/')
                    vf = '${DATA_ROOT}/' + vf
                except Exception:
                    vf = str(vf0)
                video_sets[vf] = config_data['video_sets'][vf0]

            config_data['project_path'] = project_path
            config_data['video_sets'] = video_sets
            config_data['skeleton'] = []

            out_f = Path(config_path0).parent / 'config_rel.yaml'
            self._config_path = out_f

            # Copy config_path0
            cp_f = Path(config_path0).parent / f'config_{self.HOSTNAME}.yaml'
            if cp_f != Path(config_path0):
                shutil.copy(config_path0, cp_f)
            self._config_work_path = cp_f

        # Write config file
        with io.open(str(out_f), 'w', encoding='utf8') as outfile:
            yaml.dump(config_data, outfile, default_flow_style=False,
                      allow_unicode=True)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def new_project(self, proj_name='', experimenter_name='', video_files=[],
                    work_dir='./', copy_videos=False):
        """ Create a new project directory structure and the project
        configuration file. Each project is identified by the name of the
        project (e.g., Reaching), the name of the experimenter
        (e.g., YourName), as well as the date of creation.

        The name of project directory is;
        ‘projName+experimenterName+'date of creation of the project’

        Parameters
        ----------
        proj_name: string
            The name of the project
        experimenter_name: string
            The name of the experimenter
        video_files: array like
            list of the full path of the videos. These are (initially) used to
            create the training dataset.
        work_dir: Path or string
            The working directory, where the project directory will be created.
            If workDir is unspecified, the project directory is created in the
            current working directory.
        copy_videos: bool
            Flag to copy the videos to the project directory.
            If copy_videos is unspecified, symbolic links for the videos are
            created in the videos directory.

        Return
        ------
        Config file (.yaml) is created and the path to it is set in
        self.config_path.
        No variable is returned.
        """

        # Create new DLC project
        if self.OS == 'Windows':
            copy_videos = True

        iof = io.StringIO()
        with redirect_stdout(iof):
            config_path0 = dlc.create_new_project(
                    proj_name, experimenter_name, video_files, work_dir,
                    copy_videos=copy_videos)
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

        if config_path0 is not None:
            self.set_config(config_path0)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def add_video(self, video_files, copy_videos=False):
        """Add new videos

        Parameters
        ----------
        video_files: array like
            list of the full path of the videos. These are (initially) used to
            create the training dataset.
        copy_videos: bool
            Flag to copy the videos to the project directory.
            If copy_videos is unspecified, symbolic links for the videos are
            created in the videos directory.

        Return
        ------
        Video files are added in the project.
        No variable is returned.
        """

        if not self.check_config_file():
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.add_new_videos(self._config_work_path, video_files,
                               copy_videos)
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

        self.set_config(self._config_work_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_config(self):
        if not self.check_config_file():
            return

        # Read config file
        with open(self._config_work_path, 'r') as stream:
            config_data = yaml.safe_load(stream)

        return config_data

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def edit_config(self, edit_gui_fn=None, edit_keys=None,
                    default_values=None, gui_title='Edit DLC configuration'):
        if not self.check_config_file():
            return

        # Set editing keys
        if edit_keys is None:
            edit_keys = ['Task', 'bodyparts', 'corner2move2', 'date',
                         'default_net_type', 'iteration', 'move2corner',
                         'start', 'stop', 'x1', 'x2', 'y1', 'y2']

        # Read config file
        config_data = self.get_config()

        editing_config_data = {}
        for k in edit_keys:
            if k in config_data:
                editing_config_data[k] = config_data[k]

        # Set default values
        if default_values is not None:
            for k, v in default_values.items():
                if k in editing_config_data:
                    editing_config_data[k] = v

        # Edit config
        if edit_gui_fn is not None:
            try:
                editing_config_data = edit_gui_fn(editing_config_data,
                                                  gui_title)
            except Exception:
                print(traceback.format_exc())
                editing_config_data = None

        if editing_config_data is None:
            return -1

        config_data.update(editing_config_data)

        bodyparts = config_data['bodyparts']
        rm_skel = []
        for skel in config_data['skeleton']:
            if skel[0] not in bodyparts or skel[1] not in bodyparts:
                rm_skel.append(skel)

        for rm in rm_skel:
            rmidx = config_data['skeleton'].index(rm)
            config_data['skeleton'].pop(rmidx)

        # Write config file
        with io.open(self._config_work_path, 'w', encoding='utf8') as outfile:
            yaml.dump(config_data, outfile, default_flow_style=False,
                      allow_unicode=True)

        self.set_config(self._config_work_path)

        return 0

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def extract_frames(self, edit_gui_fn, mode='automatic', algo='uniform',
                       crop=False, user_feedback=False, cluster_color=False):
        """ Extract video frames for training the network.
        Frames reflecting the diversity of the behavior with respect to
        postures, luminance conditions, background conditions, animal
        identities, and other variable aspects of the data will be extracted.

        The extracted frames from all the videos are stored in a separate
        subdirectory named after the video file’s name under the ‘labeled-data’
        directory.

        the user can change the number of frames to extract from each video by
        setting the numframes2pick variable in the config.yaml file.

        Parameters
        ----------
        mode: 'automatic'/'manual', optional, default is 'automatic'

        algo: 'uniform'/'kmeans', optional, default is 'uniform'
            Only required for 'automatic' mode.

        crop: bool, optional, default is False
            If True, a user interface pops up with a frame to
            select the cropping parameters. Use the left click to draw a
            cropping area and hit the button set cropping parameters to save
            the cropping parameters for a video.

        user_feedback: bool
            If you have already labeled some folders and want to extract data
            for new videos, you can skip the labeled folders by set
            user_feedback True, then a dialog, where the user is asked for each
            video if (additional/any) frames from this video should be
            extracted will pop up.
            If False during automatic mode, frames for all
            videos are extracted.

        cluster_color: bool, default is False
            If true, the color channels are considered. This increases the
            computational complexity.
            If False, each downsampled image is treated as a grayscale vector
            (discarding color information).
        """

        if not self.check_config_file():
            return

        # Edit bodyparts configurations
        edit_keys = ['numframes2pick']
        if self.edit_config(edit_gui_fn, edit_keys=edit_keys,
                            gui_title='Set number of labeling frames') < 0:
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.extract_frames(self._config_work_path, mode, algo, crop=crop,
                               userfeedback=user_feedback,
                               cluster_color=cluster_color)
        ostr = iof.getvalue()
        if len(ostr):
            msgstr = iof.getvalue()
            msgstr += "Select 'menu -> DLC -> Lebel frames' for the labeling."
            self.show_msg(msgstr)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def label_frames(self, edit_gui_fn):
        """Labeling of all the extracted frames using an interactive GUI.
        The body parts to label (points of interest) should already have been
        named in the project’s configuration file (config.yaml).

        When the labeling toolbox is invoked, use the ‘Load Frames’ button to
        select the directory that stores the extracted frames from one of the
        videos.
        A right click places the first body part, and, subsequently, you can
            either select one of the radio buttons (top right) to select a
            body part to label, or use the built-in mechanism that
            automatically advances to the next body part.
        If a body part is not visible, simply do not label the part and
            select the next body part you want to label.
        Each label will be plotted as a dot in a unique color. You can also
            move the label around by left-clicking and dragging.
        Once the position is satisfactory, you can select another radio button
            (in the top right) to switch to another label (it also
            auto-advances, but you can manually skip labels if needed).
        Once all the visible body parts are labeled, then you can click ‘Next’
            to load the following frame, or ‘Previous’ to look at and/or adjust
            the labels on previous frames.
        You need to save the labels after all the frames from one of the
            videos are labeled by clicking the ‘Save’ button.
        You can save at intermediate points, and then relaunch the GUI to
            continue labeling (or refine your already-applied labels).
        Saving the labels will create a labeled dataset in a hierarchical data
            format (HDF) file and comma-separated (CSV) file in the
            subdirectory corresponding to the particular video in
            ‘labeled-data’.
        """

        if not self.check_config_file():
            return

        # Edit bodyparts configurations
        edit_keys = ['bodyparts']
        if self.edit_config(edit_gui_fn, edit_keys=edit_keys,
                            gui_title='Set body parts') < 0:
            return

        if self.OS == 'Darwin':
            pycmd = "import deeplabcut as dlc; "
            pycmd += f"config_path = '{self._config_work_path}'; "
            pycmd += "dlc.label_frames(config_path)"
            cmd = f'pythonw -c "{pycmd}"'
            subprocess.check_call(shlex.split(cmd))
        elif self.OS == 'Windows':
            cmd = "python -m deeplabcut"
            subprocess.check_call(shlex.split(cmd))

            iof = io.StringIO()
            with redirect_stdout(iof):
                dlc.label_frames(self._config_work_path)
            ostr = iof.getvalue()
            if len(ostr):
                self.show_msg(iof.getvalue())
        else:
            iof = io.StringIO()
            with redirect_stdout(iof):
                dlc.label_frames(self._config_work_path)
            ostr = iof.getvalue()
            if len(ostr):
                self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def check_labels(self):
        if not self.check_config_file():
            return

        with open(self._config_work_path, 'r') as stream:
            config_data = yaml.safe_load(stream)

        labeledImg_dir = Path(config_data['project_path']) / 'labeled-data'
        for vf in config_data['video_sets'].keys():
            dd = labeledImg_dir / (Path(vf).stem + '_labeled')
            if dd.is_dir():
                shutil.rmtree(str(dd))

        iof = io.StringIO()
        with redirect_stdout(iof):
            try:
                dlc.check_labels(self._config_work_path)
            except Exception:
                print(traceback.format_exc())

        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def create_training_dataset(self, num_shuffles=1):
        if not self.check_config_file():
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.create_training_dataset(self._config_work_path,
                                        num_shuffles=num_shuffles)
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

        self.set_config(self._config_work_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def train_network(self, proc_type='run_here', analyze_videos=[]):
        if not self.check_config_file():
            return

        # --- Check running process -------------------------------------------
        if self.OS == 'Windows':
            cmd = 'tasklist | FIND "run_dlc_train.py"'
            try:
                out = subprocess.check_output(cmd, shell=True)
                if len(out.decode().rstrip().split('\n')) > 1:
                    errmsg = "Training process is running.\n"
                    sys.stderr.write(errmsg)
                    return
            except Exception:
                pass

        else:
            try:
                out = subprocess.check_output('pgrep -f run_dlc_train.py',
                                              shell=True)
                if len(out.decode().rstrip().split('\n')) > 1:
                    pid = out.decode().rstrip().split('\n')[0]
                    msg = f"Training process is running as process {pid}.\n"
                    msg += "You can kill it by 'pkill -f run_dlc_train' "
                    msg += " in a console."
                    self.show_msg(msg)
                    return
            except Exception:
                pass

        # --- Make training script --------------------------------------------
        work_dir = Path(self._config_work_path).resolve().parent

        # Update config_rel.yaml file
        self.set_config(self._config_work_path)
        conf_path = Path(self._config_path).name

        cmd_path = Path.home() / 'TVT' / 'code' / 'run_dlc_train.py'
        log_f = work_dir / 'DLC_train.out'

        if not cmd_path.is_file():
            self.show_err_msg(f'Not found {cmd_path}.')
        cmd_path = os.path.relpath(cmd_path, work_dir)
        script_f = work_dir / 'DLC_training.sh'
        cmd = f'python {cmd_path} --config {conf_path}'
        cmd += " --evaluate_network"
        if len(analyze_videos):
            video_path = [str(Path(os.path.relpath(pp.resolve(), work_dir)))
                          for pp in analyze_videos]
            cmd += f" --analyze_videos {' '.join(video_path)}"
            cmd += " --filterpredictions"

        with open(script_f, 'w') as fd:
            fd.write(cmd)

        if self.OS == 'Windows':
            run_cmd = f"bash.exe {script_f.name}"
        else:
            run_cmd = f"/bin/bash {script_f.name}"

        if proc_type == 'run_subprocess':
            run_cmd = f"cd {work_dir} && " + run_cmd
            subprocess.Popen(run_cmd, stdout=open(log_f, 'a'),
                             stderr=open(log_f, 'a'), shell=True)

            msg = 'Training has been started.\n'
            msg += f"Progress is being written in {log_f}\n"
            msg += 'It may take a long time.'
            self.show_msg(msg)

        elif proc_type == 'prepare_script':
            msg = f"The process script is made as\n {script_f}\n\n"
            msg += "Run the script in a console"
            msg += " by copy and paste the following lines;\n"
            msg += " (modify the path if necessary)\n\n"
            msg += "conda activate TVT\n"
            msg += f"cd {work_dir}\n"
            msg += f"nohup {run_cmd} > {log_f.relative_to(work_dir)} &"
            self.show_msg(msg)

        elif proc_type == 'run_here':
            dlc.train_network(self._config_work_path)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def find_analysis_results(self, video_path, shuffle=1):
        """ Get video analysis result filnenames """

        videoname = Path(video_path).stem
        config_data = self.get_config()
        Task = config_data['Task']
        date = config_data['date']
        snapshotindex = config_data['snapshotindex']
        if snapshotindex == -1:
            trainFraction = config_data['TrainingFraction'][0]
            modelfolder = Path(config_data["project_path"])
            GetModelFolder = auxiliaryfunctions.get_model_folder
            modelfolder /= GetModelFolder(trainFraction, shuffle,
                                          config_data)
            Snapshots = [int(re.search(r'\d+', fn.stem).group())
                         for fn in (modelfolder / 'train').glob('*.index')]
            if len(Snapshots):
                snapshotindex = np.max(Snapshots)
                self.edit_config(
                    edit_keys=['snapshotindex'],
                    default_values={'snapshotindex': int(snapshotindex)})
            else:
                return []

        pred_f_temp = f"{videoname}DeepCut_resnet50_{Task}{date}"
        pred_f_temp += f"shuffle{shuffle}_{snapshotindex}*"

        res_fs = [ff for ff in Path(video_path).parent.glob(pred_f_temp)]
        return res_fs

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def evaluate_network(self, plotting=True):
        if not self.check_config_file():
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.evaluate_network(self._config_work_path, plotting=plotting)
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def analyze_videos(self, video_path, shuffle=1):
        if not self.check_config_file():
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.analyze_videos(self._config_work_path, [str(video_path)],
                               shuffle=shuffle, save_as_csv=True,
                               videotype='.mp4')
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def filterpredictions(self, video_path):
        if not self.check_config_file():
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.filterpredictions(self._config_work_path, [str(video_path)],
                                  save_as_csv=True)
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def plot_trajectories(self, video_path, filtered=False):
        if not self.check_config_file():
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.plot_trajectories(self._config_work_path, [str(video_path)],
                                  videotype='.mp4', filtered=filtered)
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def create_labeled_video(self, video_path, filtered=False):
        if not self.check_config_file():
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.create_labeled_video(self._config_work_path, [str(video_path)],
                                     filtered=filtered)
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def extract_outlier_frames(self, video_path):
        if not self.check_config_file():
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.extract_outlier_frames(
                    self._config_work_path, [str(video_path)],
                    videotype='.mp4', automatic=True)
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def refine_labels(self):
        if not self.check_config_file():
            return

        if self.OS == 'Darwin':
            pycmd = 'import deeplabcut as dlc; '
            pycmd += f'config_path = "{self._config_work_path}"; '
            pycmd += 'dlc.refine_labels(config_path)'
            cmd = f"pythonw -c '{pycmd}'"
            subprocess.run(cmd, shell=True)
        else:
            iof = io.StringIO()
            with redirect_stdout(iof):
                dlc.refine_labels(self._config_work_path)
            ostr = iof.getvalue()
            if len(ostr):
                self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def merge_datasets(self):
        if not self.check_config_file(popup_err=False):
            return

        iof = io.StringIO()
        with redirect_stdout(iof):
            dlc.merge_datasets(self._config_work_path)
        ostr = iof.getvalue()
        if len(ostr):
            self.show_msg(iof.getvalue())

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_save_params(self):

        if not self.check_config_file(popup_err=False):
            return None

        saving_params = ['_config_path']
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
