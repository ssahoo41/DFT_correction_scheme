#!/bin/bash
#PBS -l nodes=4:ppn=12
#PBS -l walltime=48:00:00
#PBS -N overall_subsample_trend
#PBS -o stdout
#PBS -e stderr
#PBS -m abe
#PBS -M ssahoo41@gatech.edu
#PBS -A GT-amedford6-joe

cd $PBS_O_WORKDIR

#module load intel/19.0.5
source ~/.bashrc

pyenv activate miniconda3-latest

python overall_subsample_trend.py
