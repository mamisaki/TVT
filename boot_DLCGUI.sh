#!/bin/bash

# anaconda environment setup
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate DEEPLABCUT

cd $HOME/TVT/code
python DLC_GUI.py
