#!/bin/bash

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=abeukers@princeton.edu

#SBATCH --gpu-accounting

#SBATCH -t 40:00:00		# runs for 48 hours (max)  
#SBATCH -c 16				# number of cores 4
#SBATCH -N 1				# node count 
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:2		# number of gpus 4


wd_dir="/tigress/abeukers/wd/w2v"

module load anaconda3/4.4.0
module load cudnn/cuda-8.0/6.0

corpus_fpath="${1}"
results_dir="${2}"

printf "\n --corp_fpath is ${corpus_fpath}"
printf "\n --results_dir is ${results_dir}"

srun python ${wd_dir}/w2v3_gpu.py "${corpus_fpath}" "${results_dir}"


printf "\n\nGPU profiling \n\n"
nvidia-smi --query-accounted-apps=gpu_serial,gpu_utilization,mem_utilization,max_memory_usage,time\
			--format=csv
