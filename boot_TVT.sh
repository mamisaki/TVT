#!/bin/bash

if [ -d $HOME/anaconda3 ]; then
  CONDA_DIR=$HOME/anaconda3
fi

if [ -d $HOME/miniconda3 ]; then
  CONDA_DIR=$HOME/miniconda3
fi

# anaconda environment setup
. $CONDA_DIR/etc/profile.d/conda.sh
conda activate TVT

cd $HOME/TVT/code
python TVT.py
