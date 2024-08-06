#!/bin/bash
#SBATCH -A IscrC_LLM4SE
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_lprod
#SBATCH --time 0-20:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=492000
#SBATCH --job-name=llm4se_inference
#SBATCH --out=logs/nlbse24-logs-fix/Mixtral-8x7B-Instruct-v0.1.out
#SBATCH --err=logs/nlbse24-logs-fix/Mixtral-8x7B-Instruct-v0.1.err


srun ./predict_exe
