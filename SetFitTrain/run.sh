#!/bin/bash
#SBATCH -A IscrC_LLM4SE
#SBATCH -p boost_usr_prod
#SBATCH --qos boost_qos_lprod
#SBATCH --time 0-30:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=123000
#SBATCH --job-name=llm4se_inference
#SBATCH --out=logs/setfit24_curve_3.out
#SBATCH --err=logs/setfit24_curve_3.err


srun ./predict_exe
