#! /bin/bash

singularity run --nv --bind /scratch/ ./instruct-flamingo_latest.sif python ./OpenVocab_Seg_with_AutoRegres/train.py