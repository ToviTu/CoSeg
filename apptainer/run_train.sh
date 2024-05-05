#! /bin/bash

. /home/research/jianhong.t/Instruction-tuned-Flamingo-MLLM/src/set_environ_var.sh
singularity run --nv --bind /scratch,/storage1 /scratch/t.tovi/lang-modeling_latest.sif python /home/research/jianhong.t/OpenVocab_Seg_with_AutoRegres/train_model_v0.4.py
