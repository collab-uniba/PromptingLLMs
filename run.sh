#!/bin/bash
#SBATCH -A IscrC_LLM4SE
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_lprod
#SBATCH --time 0-01:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=246000
#SBATCH --job-name=llm4se_inference
#SBATCH --out=llm4se_inference.out
#SBATCH --err=llm4se_inference.out

srun ./predict_exe
