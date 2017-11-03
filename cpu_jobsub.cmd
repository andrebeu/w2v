#!/bin/bash

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=abeukers@princeton.edu

#SBATCH -t 40:00:00			# runs for 48 hours (max)  
#SBATCH --ntasks-per-node=1
#SBATCH -N 1				# node count 
#SBATCH -c 16				# number of cores 

echo "AAA"

wd_dir="/tigress/abeukers/wd/w2v"

module load anaconda3/4.4.0
module load cudnn/cuda-8.0/6.0

corpus_fpath="${1}"
results_dir="${2}"

printf "\n --corp_fpath is ${corpus_fpath}"
printf "\n --results in ${results_dir} \n"

srun python ${wd_dir}/w2v_train.py "${corpus_fpath}" "${results_dir}"
# srun python -u ${wd_dir}/w2v_corpRSM.py 

sacct --format="CPUTime,MaxRSS"