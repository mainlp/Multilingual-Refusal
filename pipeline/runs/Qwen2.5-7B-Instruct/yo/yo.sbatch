#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH -o 'output/yo_vector_sweep/Qwen/Qwen2.5-7B-Instruct/yo/20250102-155445/1/yo.out'
#SBATCH -e 'output/yo_vector_sweep/Qwen/Qwen2.5-7B-Instruct/yo/20250102-155445/1/yo.err'
#SBATCH --time=0-05:30:00
#SBATCH -q mcml
#SBATCH -p mcml-hgx-a100-80x4
source activate multi-direction
cd ~/storage/code/refusal-multilingual
source venv/bin/activate
BNB_CUDA_VERSION=122 TRANSFORMERS_CACHE=/dss/dsshome1/lxc02/di75taw/storage/runs_models/huggingface python3 -m pipeline.run_pipeline --config_path output/yo_vector_sweep/Qwen/Qwen2.5-7B-Instruct/yo/20250102-155445/1/yo.yaml 