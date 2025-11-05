import os
from datetime import datetime
from itertools import permutations, product, combinations
import time
import mmengine

import subprocess




def get_job_count():
    """
    Get the current number of jobs running for the user.
    """
    try:
        # Run the `squeue` command and count lines excluding the header
        result = subprocess.run(
            ['squeue', '--me', '-h'], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        # Count the number of lines in the output
        if result.stdout.strip():  # Ensure the output isn't empty
            job_count = len(result.stdout.strip().split('\n'))
        else:
            job_count = 0
        return job_count
    except Exception as e:
        print(f"Error occurred while fetching job count: {e}")
        return -1 

def wait_for_available_slot(max_jobs, check_interval=60):
    """
    Wait until the number of running jobs is less than the max_jobs threshold.
    """
    while True:
        job_count = get_job_count()
        if job_count == -1:
            print("Error fetching job count. Retrying in 60 seconds...")
        elif job_count < max_jobs:
            print(f"Job count is {job_count}. Proceeding...")
            break
        else:
            print(f"Job count is {job_count}. Waiting for available slots...")
        time.sleep(check_interval)


# from scripts.experiment import generate_prompts, PERSONA_DICT
now = datetime.now()
current_time = time.strftime("%Y%m%d-%H%M%S")            
model_list =  ['Qwen/Qwen2.5-7B-Instruct', "meta-llama/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it"] 
            #    -8B-Instruct",'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct','Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct' ]
max_jobs = 6
# ["Qwen/Qwen-7B-Chat", "meta-llama/Llama-2-70b-chat-hf"]
# ["meta-llama/Llama-2-7b-chat-hf"  , "meta-llama/Llama-2-13b-chat-hf", "google/gemma-7b-it"] #,  , "google/gemma-2b-it", 'meta-llama/Meta-Llama-3-8B-Instruct']#, "Qwen/Qwen-7B-Chat", "google/gemma-7b-it", ] #, ] # "meta-llama/Meta-Llama-3.1-8B-Instruct" ,, ,"google/gemma-7b-it", ,
# model_list = ["meta-llama/Llama-2-7b-chat-hf","meta-llama/Meta-Llama-3-8B-Instruct"]
num_gpu = 1
current_time = current_time
# current_time= '20240916-013820'
base_config_path = 'configs/cfg.yaml'
cfg = mmengine.Config.fromfile(base_config_path)





# experiment_name = 'partial_ortho_0.2_OR_add_coeff'
var_1_list = ['de', 'es', 'fr', 'it', 'nl', 'ja', 'pl', 'ru', 'zh', 'ko', 'ar']
            #   , 'es', 'fr', 'it', 'nl', 'ja', 'pl', 'ru', 'zh', 'ko', 'ar']
# [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# 
# 
# [0]
# [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
# 
#  #



# steer_kl_threshold_list = [1]
# var_2 = 'lambda'
# lambda_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# [0.8]
# 
# []
# [ 0.6 , 0.8, ]


# top_n_list = [1, 2, 3, 4, 5]
# var_2 = 'top_n'



experiment_name = 'model_ablation_source_lang'  
'zh_vector_sweep' 
'second_ultra_lang_sweep_fix' 
'zh_vector_sweep' 
'baseline' #''
# [ 0.2, 0.4, 0.6, 0.8, 1.0]


# ablaation_kl_threshold_list = [0.1, 0.5, 1.0, 2.0, 3.0]
cfg.system = None
seeds = [1]
for seed in seeds:
    cfg.random_seed = seed
    for model in model_list: #, 'safety_falcon', 'mistral','falcon',       
        # for variable_1 in var_1_list:

            # for variable_2 in top_n_list:
                
                cfg.model_path = model
                cfg.source_lang = 'de'
                # cfg.lang = variable_1
                
                cfg.batch_size = 64

                # cfg.ortho_lambda = 0.8
                
                # cfg.top_n = variable_2
                
                
                # run_name = variable_1
                run_name = 'de'
                
                file_name = f"{run_name}.sbatch"
                
                output_dir = f"output/{experiment_name}/{model}/{run_name}/{current_time}/{seed}"
                
                cfg.artifact_path = output_dir
                
                # save the config file to yaml  
                cfg_path = f"{output_dir}/{run_name}.yaml"
                cfg.dump(cfg_path)
                
                # also snapshot the current python file
                os.makedirs(f"{output_dir}/scripts", exist_ok=True)
                os.system(f"cp scripts/experiment_slurm.py {output_dir}/scripts")


                if not os.path.exists(output_dir):
                    # create if not exists, nested dir will also be created
                    os.makedirs(output_dir, exist_ok=True)    
                with open (f"{output_dir}/{run_name}.sbatch", 'w') as rsh:
                    rsh.write(
                    f'''\
#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:{num_gpu}
#SBATCH --ntasks=1
#SBATCH -o '{output_dir}/{run_name}.out'
#SBATCH -e '{output_dir}/{run_name}.err'
#SBATCH --time=0-03:30:00
#SBATCH -q mcml
#SBATCH -p mcml-hgx-h100-92x4
source activate multi-direction
cd ~/storage/code/refusal-multilingual
source venv/bin/activate
BNB_CUDA_VERSION=122 \
TRANSFORMERS_CACHE=/dss/dsshome1/lxc02/di75taw/storage/runs_models/huggingface \
python3 -m pipeline.run_pipeline --config_path {cfg_path} \
'''
        )
                cmd = f"sbatch '{output_dir}/{run_name}.sbatch'"
                print(f"Waiting until job count is less than {max_jobs}...")
                wait_for_available_slot(max_jobs, 60)

                # Your script logic goes here
                print("Job count is within limit. Running the script...")
                os.system(cmd)
    # wait for 1 hour
        # time.sleep(3600)
            # #SBATCH -q mcml
#SBATCH -q mcml
#SBATCH -p mcml-dgx-a100-40x8

# SBATCH -p mcml-hgx-a100-80x4
    
# lrz-dgx-a100-80x8
# 
# python3 -m scripts.multi_test --config {cfg_path} \
# 
\

# 
#BNB_CUDA_VERSION=122 \
# HF_HOME=/dss/dsshome1/lxc02/di75taw/storage/runs_models/huggingface \
# TRANSFORMERS_CACHE=/dss/dsshome1/lxc02/di75taw/storage/runs_models/huggingface \