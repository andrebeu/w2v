#!/bin/bash

# loops through directory of clusters submitting a job to each
printf "\n\n -- cluster_loop\n"
printf " --  submitting several jobs in a loop \n"

wd_dir="/tigress/abeukers/wd/w2v"

slurm_dir="${wd_dir}/slurm_out/$(ls -l ./slurm_out | grep -c ^d)"
results_dir="${wd_dir}/results/$(ls -l ./results | grep -c ^d)"
mkdir -p ${slurm_dir}
mkdir -p ${results_dir}
cd ${slurm_dir}

echo "slurm dir is ${slurm_dir}"
echo "results dir is ${results_dir}"

corpus_dir="${wd_dir}/data/corpus_partitions/4_k10"

# loop submitting jobs on each corpus found in corpus_dir
for corpus_fpath in ${corpus_dir}/[FOX,MSN]*.txt;do
	echo $corpus_fpath
	sbatch ${wd_dir}/gpu_jobsub.cmd ${corpus_fpath} ${results_dir}
done


