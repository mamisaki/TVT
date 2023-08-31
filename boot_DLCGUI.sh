#!/bin/bash

# anaconda environment setup
. $HOME/anaconda3/etc/profile.d/conda.sh
conda activate TVT

cd $HOME/TVT/code
python DLC_GUI.py
