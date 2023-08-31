#!/bin/bash

cd $HOME/anaconda3/envs/TVT/lib/python3.10/site-packages/tensorrt_libs
ln -sf libnvinfer.so.8 libnvinfer.so.7
ln -sf libnvinfer_plugin.so.8 libnvinfer_plugin.so.7

cd $HOME/anaconda3/envs/TVT
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/deactivate.d/env_vars.s

echo '#!/bin/sh' > ./etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH0=$LD_LIBRARY_PATH' >> ./etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$HOME/anaconda3/envs/TVT/lib:$HOME/anaconda3/envs/TVT/lib/python3.10/site-packages/tensorrt_libs:$HOME/anaconda3/envs/TVT/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH' >> ./etc/conda/activate.d/env_vars.sh
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' >> ./etc/conda/activate.d/env_vars.sh

echo '#!/bin/sh' > ./etc/conda/deactivate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH0' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset LD_LIBRARY_PATH0' >> ./etc/conda/deactivate.d/env_vars.sh

