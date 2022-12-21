#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: mamis
"""


# %% import ===================================================================
import os
os.environ['MPLBACKEND'] = 'TKAgg'

import deeplabcut as dlc
import argparse
from pathlib import Path
import yaml
import io
import platform


# %% ==========================================================================
def set_config(config_path0, APP_ROOT, HOSTNAME):
    """Set config file (_config_path, _config_work_path) with converting
    paths. Convert paths in config to a relative one from APP_ROOT or vice
    versa.

    If the config includes ${APP_ROOT}, a config file with full-path
    is created and set to self.config_work_path.
    Otherwise, paths in the config are converted to relative to APP_ROOT,
    and a converted config is saved in a file to set in self.config_path.
    """

    # Read config file
    with open(config_path0, 'r') as stream:
        config_data = yaml.safe_load(stream)

    # Convert paths
    if '${APP_ROOT}' in config_data['project_path']:
        # File with relative path is read.
        # convert to absolute path
        project_path = config_data['project_path']
        project_path = project_path.replace('${APP_ROOT}/', '')
        project_path = str((APP_ROOT / project_path).resolve())

        video_sets = {}
        for vf0 in config_data['video_sets'].keys():
            vf = vf0.replace('${APP_ROOT}/', '')
            vf = str((APP_ROOT / vf).resolve())
            video_sets[vf] = config_data['video_sets'][vf0]

        config_data['project_path'] = project_path
        config_data['video_sets'] = video_sets

        out_f = Path(config_path0).parent / f'config_{HOSTNAME}.yaml'

        # Write config file
        with io.open(str(out_f), 'w', encoding='utf8') as outfile:
            yaml.dump(config_data, outfile, default_flow_style=False,
                      allow_unicode=True)
    else:
        out_f = config_path0

    return out_f


# %% ==========================================================================
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run DLC train_network.')
    parser.add_argument('--config', help='DLC configuration file',
                        required=True)
    parser.add_argument('-c', '--create_training_dset', action='store_true',
                        default=False,
                        help='Run create_training_dataset')
    parser.add_argument('--shuffle', type=int, default=1)
    parser.add_argument('--trainingsetindex', type=int, default=0,
                        help='Integer specifying which training set ' +
                        'fraction to use')
    parser.add_argument('--gputouse', type=float, default=None)
    parser.add_argument('--max_snapshots_to_keep', type=int, default=5)
    parser.add_argument('--displayiters', type=int, default=1000)
    parser.add_argument('--saveiters', type=int, default=20000)
    parser.add_argument('--maxiters', type=int, default=200000)
    parser.add_argument('--evaluate_network', action='store_true',
                        default=False)
    parser.add_argument('--analyze_videos', nargs='+')
    parser.add_argument('--filterpredictions', action='store_true',
                        default=False)

    args = parser.parse_args()
    config_f = Path(args.config)
    create_training_dset = args.create_training_dset
    shuffle = args.shuffle
    trainingsetindex = args.trainingsetindex
    gputouse = args.gputouse
    if gputouse is not None:
        gputouse = int(gputouse)
    max_snapshots_to_keep = args.max_snapshots_to_keep
    displayiters = args.displayiters
    saveiters = args.saveiters
    maxiters = args.maxiters
    evaluate_network = args.evaluate_network
    analyze_videos = args.analyze_videos
    filterpredictions = args.filterpredictions

    # --- Set config file -----------------------------------------------------
    APP_ROOT = Path(__file__).absolute().parent.parent
    if platform.system() == 'Windows':
        HOSTNAME = platform.node()
    else:
        HOSTNAME = os.uname()[1]
    if not config_f.stem.endswith(HOSTNAME):
        config_f = set_config(config_f, APP_ROOT, HOSTNAME)

    config_f = Path(config_f).absolute()

    # --- Run -----------------------------------------------------------------
    print('+' * 70)
    print(f"Run dlc.train_network for config {config_f}\n")

    if create_training_dset:
        dlc.create_training_dataset(config_f)

    dlc.train_network(config_f, shuffle=shuffle,
                      trainingsetindex=trainingsetindex, gputouse=gputouse,
                      max_snapshots_to_keep=max_snapshots_to_keep,
                      displayiters=displayiters, saveiters=saveiters,
                      maxiters=maxiters)

    if evaluate_network:
        dlc.evaluate_network(config_f, plotting=False)

    if len(analyze_videos):
        dlc.analyze_videos(config_f, analyze_videos, shuffle=1,
                           save_as_csv=True, videotype='.mp4')
        if filterpredictions:
            dlc.filterpredictions(config_f, analyze_videos, save_as_csv=True)
