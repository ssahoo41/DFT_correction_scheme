#!/bin/bash
#PBS -l nodes=2:ppn=12
#PBS -l walltime=12:00:00
#PBS -N overall_subsample
#PBS -o stdout
#PBS -e stderr
#PBS -m abe
#PBS -M ssahoo41@gatech.edu
#PBS -A GT-amedford6-joe

cd $PBS_O_WORKDIR

#module load intel/19.0.5
source ~/.bashrc

pyenv activate miniconda3-latest

python overall_subsample.py
