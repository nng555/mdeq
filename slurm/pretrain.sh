#!/bin/bash

# set up environment
. /h/nng/venv/mdeq/bin/activate

# symlink checkpoint directory to run directory
ln -s /checkpoint/$USER/$SLURM_JOB_ID /h/nng/slurm/2021-02-16/contrastive_mdeq_large/$SLURM_JOB_ID
cd /h/nng/programs/mdeq
python3 tools/contrastive_train.py --cfg experiments/cifar/contrastive_mdeq_LARGE.yaml

