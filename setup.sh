#!/bin/bash

if [ -d $HOME/anaconda3 ]; then
  CONDA_DIR=$HOME/anaconda3
fi

if [ -d $HOME/miniconda3 ]; then
  CONDA_DIR=$HOME/miniconda3
fi

cd $CONDA_DIR/envs/TVT/lib/python3.10/site-packages/tensorrt_libs
if [ ! -f libnvinfer.so.7 ]; then
  ln -sf libnvinfer.so.* libnvinfer.so.7
fi

if [ ! -f libnvinfer_plugin.so.7 ]; then
  ln -sf libnvinfer_plugin.so.* libnvinfer_plugin.so.7
fi

cd $CONDA_DIR/envs/TVT
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/deactivate.d/env_vars.s

echo '#!/bin/sh' > ./etc/conda/activate.d/env_vars.sh
echo 'export CONDA_DIR='$CONDA_DIR >> ./etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH0=$LD_LIBRARY_PATH' >> ./etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CONDA_DIR/envs/TVT/lib:$CONDA_DIR/envs/TVT/lib/python3.10/site-packages/tensorrt_libs:$CONDA_DIR/envs/TVT/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH' >> ./etc/conda/activate.d/env_vars.sh
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' >> ./etc/conda/activate.d/env_vars.sh

echo '#!/bin/sh' > ./etc/conda/deactivate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH0' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset LD_LIBRARY_PATH0' >> ./etc/conda/deactivate.d/env_vars.sh

