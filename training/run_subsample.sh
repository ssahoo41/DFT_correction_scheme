#!/bin/bash
#PBS -l nodes=1:ppn=12
#PBS -l walltime=5:00:00
#PBS -N model_training
#PBS -o stdout_system_subsampling
#PBS -e stderr_system_subsampling
#PBS -A GT-amedford6-joe

cd $PBS_O_WORKDIR

source ~/.bashrc

pyenv activate miniconda3-latest

python system_subsample.py ${SYSTEM} GGA_PBE 

