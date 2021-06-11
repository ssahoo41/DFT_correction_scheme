#!/bin/bash
#PBS -l nodes=2:ppn=12
#PBS -l walltime=6:00:00
#PBS -N model_fitting_3.5_5fold
#PBS -o stdout_fitting_3.5_5fold
#PBS -e stderr_fitting_3.5_5fold
#PBS -A GT-amedford6-joe

cd $PBS_O_WORKDIR

source ~/.bashrc

pyenv activate miniconda3-latest

python model_fitting.py

