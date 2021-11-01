#!/bin/sh
#BSUB -q gpua100
#BSUB -R "select[gpu40gb]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -J "movie-review"
#BSUB -R "rusage[mem=40GB]"
#BSUB -n 1
#BSUB -W 10:00
#BSUB -o $HOME/joblogs/%J.out
#BSUB -e $HOME/joblogs/%J.err
#BSUB -u s183912@dtu.dk
#BSUB -N

LOC=/work3/$USER/dmiai/movie-review
python train.py $LOC --batch-size 16 --model-name roberta-large
python plot.py $LOC
