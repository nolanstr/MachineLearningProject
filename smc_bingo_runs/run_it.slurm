#!/bin/bash
#SBATCH --time=96:00:00 # walltime, abbreviated by -t
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=u1079968@umail.utah.edu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=20GB
# additional information for allocated clusters
#SBATCH --account=hochhalter-np     # account - abbreviated by -A
#SBATCH --partition=hochhalter-shared-np # partition, abbreviated by -p

##SBATCH --pty 
##SBATCH --qos=hochhalter-high-np

#pwd

python run_bingo.py
