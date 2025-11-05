#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH -o 'output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct/th/20250110-004711/1/th.out'
#SBATCH -e 'output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct/th/20250110-004711/1/th.err'
#SBATCH --time=0-10:30:00
#SBATCH -p lrz-hgx-h100-92x4 
source activate multi-direction
cd ~/storage/code/refusal-multilingual
source venv/bin/activate
BNB_CUDA_VERSION=122 TRANSFORMERS_CACHE=/dss/dsshome1/lxc02/di75taw/storage/runs_models/huggingface python3 -m pipeline.run_pipeline --config_path output/model_ablation_source_lang/Qwen/Qwen2.5-7B-Instruct/th/20250110-004711/1/th.yaml 