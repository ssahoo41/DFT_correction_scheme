#!/bin/bash
#PBS -l nodes=2:ppn=12
#PBS -l walltime=6:00:00
#PBS -N model_fitting_3.2_CV
#PBS -o stdout_fitting_3.2_CV
#PBS -e stderr_fitting_3.2_CV
#PBS -A GT-amedford6-joe

cd $PBS_O_WORKDIR

source ~/.bashrc

pyenv activate miniconda3-latest

python model_fitting.py

