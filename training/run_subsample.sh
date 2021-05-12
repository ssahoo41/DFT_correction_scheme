#!/bin/bash
#PBS -l nodes=1:ppn=24
#PBS -l walltime=36:00:00
#PBS -N model_training
#PBS -o stdout_fitting
#PBS -e stderr_fitting
#PBS -A GT-amedford6-joe

cd $PBS_O_WORKDIR

source ~/.bashrc

pyenv activate miniconda3-latest

python model_fitting.py 

