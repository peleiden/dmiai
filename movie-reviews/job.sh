#!/bin/sh
#BSUB -q gpua100
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J "movie-review"
#BSUB -R "rusage[mem=40GB]"
#BSUB -n 1
#BSUB -W 24:00
#BSUB -o $HOME/joblogs/%J.out
#BSUB -e $HOME/joblogs/%J.err
#BSUB -u s183912@dtu.dk
#BSUB -N

LOC=/work3/$USER/dmiai/movie-review
alias python=/appl/python/3.8.4/bin/python3
python train.py $LOC --batch-size 64 --model-name roberta-base --lr 2e-6 --epochs 10
python plot.py $LOC
