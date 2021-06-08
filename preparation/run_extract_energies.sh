#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=10:00:00
#PBS -N extract_energies
#PBS -o stdout
#PBS -e stderr
#PBS -m abe
#PBS -M ssahoo41@gatech.edu
#PBS -A GT-amedford6-joe

cd $PBS_O_WORKDIR

#module load intel/19.0.5
source ~/.bashrc

pyenv activate miniconda3-latest

python extract_reference_energies.py
