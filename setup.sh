#!/bin/bash

cd $HOME/anaconda3/envs/DEEPLABCUT
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/deactivate.d/env_vars.sh

echo '#!/bin/sh' > ./etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH0=$LD_LIBRARY_PATH' >> ./etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$HOME/anaconda3/envs/DEEPLABCUT/lib:$LD_LIBRARY_PATH' >> ./etc/conda/activate.d/env_vars.sh

echo '#!/bin/sh' > ./etc/conda/deactivate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH0' >> ./etc/conda/deactivate.d/env_vars.sh
echo 'unset LD_LIBRARY_PATH0' >> ./etc/conda/deactivate.d/env_vars.sh

