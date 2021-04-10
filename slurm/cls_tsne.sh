#!/bin/bash

# set up environment
. /h/nng/venv/mdeq/bin/activate

# symlink checkpoint directory to run directory
ln -s /checkpoint/$USER/$SLURM_JOB_ID /h/nng/slurm/2021-02-16/mdeq_eval/$SLURM_JOB_ID
cd /h/nng/programs/mdeq
python3 tools/cls_tsne.py --cfg experiments/cifar/contrastive_mdeq_LARGE.yaml --pretrained output/cifar10/contrastive_mdeq_LARGE/model_best.pth.tar

