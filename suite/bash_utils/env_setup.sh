#!/bin/bash
ENV_PATH=$SLURM_TMPDIR/env
if [ -f $ENV_PATH ]; then
   echo "(ENV_CHECK) File $ENV_PATH exists."
  #  rm -rf $ENV_PATH
else
   echo "(ENV_CHECK) File $ENV_PATH does not exist."
fi

module load StdEnv/2020  gcc/9.3.0  cuda/11.4 arrow/8.0.0 python/3.10

virtualenv --no-download $ENV_PATH
#virtualenv $ENV_PATH
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
#pip install --upgrade pip

pip install --no-index numpy==1.23.5
pip install --no-index -r requirements.txt
#pip install -r requirements.txt
pip install --no-deps /home/amirresm/files/research/summarization/adapters-0.1.2-py3-none-any.whl