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



to_collect = [
    
    {   'model': '01-ai/Yi-6B-Chat',
        'experiment': 'second_ultra_lang_sweep',
        'time': '20241209-163214',
        },
    
    {'model':   'google/gemma-2b-it',
        'experiment': 'second_ultra_lang_sweep',
        'time': '20241209-163214',
        },
     { 'model': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'experiment': 'second_ultra_lang_sweep',
        'time': '20241209-163214',
    },
     { 'model': 'meta-llama/Meta-Llama-3-70B-Instruct',
        'experiment': 'second_ultra_lang_sweep_fix',
        'time': '20250102-155445',
    },
     
     { 'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'experiment': 'second_ultra_lang_sweep_fix',
        # 'time': '20250102-200441',
        'time': '20250104-171226'
    },
    #  { 'model': 'meta-llama/Llama-2-13b-chat-hf',
    #     'experiment': 'second_ultra_lang_sweep_fix',
    #     'time': '20241213-200420',
    # }, problem with the model
     { 'model': 'google/gemma-2-9b-it',
        'experiment': 'second_ultra_lang_sweep_fix',
        'time': '20241212-141650',
    },
    
    { 'model': 'Qwen/Qwen2.5-3B-Instruct',
        'experiment': 'second_ultra_lang_sweep_fix',
        'time': '20250104-171226',
    },
    
    { 'model': 'Qwen/Qwen2.5-7B-Instruct',
      'experiment': 'second_ultra_lang_sweep_fix',
      'time': '20250104-171226',  
    },
    { 'model': 'Qwen/Qwen2.5-14B-Instruct',
      'experiment': 'second_ultra_lang_sweep_fix',
      'time':'20250107-111851'
        # '20241224-220423',  
    },
    { 'model': 'Qwen/Qwen2.5-32B-Instruct',
      'experiment': 'second_ultra_lang_sweep_fix',
      'time': '20250107-163624'
        # '20241224-220423',  
    },
    
     
]

to_collect = [
                 {'model':   'google/gemma-2-9b-it',
        'experiment': 'gemma2_9b_zh_test',
        'time': '20241217-002452',
    },
    
    {'model':   'meta-llama/Meta-Llama-3.1-8B-Instruct',
        'experiment': 'zh_vector_sweep',
        'time': '20250105-231003'
            # '20250103-230929',
    },
    {'model':   'Qwen/Qwen2.5-7B-Instruct',
        'experiment': 'zh_vector_sweep',
        'time': '20250105-231003'
            # '20250103-230929',
    },



    # {'model':   'google/gemma-2-9b-it',
    #     'experiment': 'de_vector_sweep',
    #     'time': '20250106-014713',

    # },
    
    # {'model':   'meta-llama/Meta-Llama-3.1-8B-Instruct',
    #     'experiment': 'de_vector_sweep',
    #     'time': '20250106-014713'
    #         # '20250103-230929',
    # },
    # {'model':   'Qwen/Qwen2.5-7B-Instruct',
    #     'experiment': 'de_vector_sweep',
    #     'time': '20250106-014713'
    #         # '20250103-230929',
    # },

    # {'model':   'google/gemma-2-9b-it',
    #     'experiment': 'th_vector_sweep',
    #     'time': '20250111-160023',

    # },
    
    # {'model':   'meta-llama/Meta-Llama-3.1-8B-Instruct',
    #     'experiment': 'th_vector_sweep',
    #     'time': '20250108-154935'
    #         # '20250103-230929',
    # },
    # {'model':   'Qwen/Qwen2.5-7B-Instruct',
    #     'experiment': 'th_vector_sweep',
    #     'time': '20250110-005533'
    #         # '20250103-230929',
    # },

    ]

# transfer the to_collect to a dictionary, which model as key

to_collect = {model['model']: model for model in to_collect}

# from scripts.experiment import generate_prompts, PERSONA_DICT
now = datetime.now()
# current_time = time.strftime("%Y%m%d-%H%M%S")          
current_time = '20250102-155445'  

            #    , , "google/gemma-2-9b-it"] "meta-llama/Meta-Llama-3.1-8B-Instruct"
            #    -8B-Instruct",'Qwen/Qwen2.5-3B-Instruct', ,'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct' ]
max_jobs = 10
# ["Qwen/Qwen-7B-Chat", "meta-llama/Llama-2-70b-chat-hf"]
# ["meta-llama/Llama-2-7b-chat-hf"  , "meta-llama/Llama-2-13b-chat-hf", "google/gemma-7b-it"] #,  , "google/gemma-2b-it", 'meta-llama/Meta-Llama-3-8B-Instruct']#, "Qwen/Qwen-7B-Chat", "google/gemma-7b-it", ] #, ] # "meta-llama/Meta-Llama-3.1-8B-Instruct" ,, ,"google/gemma-7b-it", ,

num_gpu = 2
base_config_path = 'configs/cfg.yaml'
cfg = mmengine.Config.fromfile(base_config_path)


model_list = to_collect.keys()

model_list = ['google/gemma-2-9b-it',  "meta-llama/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct"]
# model_list = [ "Qwen/Qwen-7B-Chat", "meta-llama/Llama-2-70b-chat-hf", "meta-llama/Meta-Llama-3-70B-Instruct", "google/gemma-2-9b-it",  "google/gemma-2b-it", "meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"]
# [ 'Qwen/Qwen2.5-7B-Instruct' , "meta-llama/Meta-Llama-3.1-8B-Instruct", "google/gemma-2-9b-it",]
experiment_name = 'yo_vector_sweep'
'second_ultra_lang_sweep_fix' 
'multi_inference'
'second_ultra_lang_sweep_fix' 
# 'model_ablation_source_lang' 
'multi_inference'

'th_vector_sweep' 

'zh_vector_sweep' 
'baseline' #''

# var_1_list = ['en', 'de', 'es', 'fr', 'it', 'nl', 'ja', 'pl', 'ru', 'zh', 'ko', 'ar', 'th', 'yo']
var_1_list = ['yo']
            #   , 'es', 'fr', 'it', 'nl', 'ja', 'pl', 'ru', 'zh', 'ko', 'ar']
# var_1_list = ['en', 'th', 'yo']         
# var_1_list = ['en']
# var_1_list = 
# ['th']
# , 'yi']

# var_1_list = ['harmful', 'harmless']


cfg.system = None
seeds = [1]
for seed in seeds:
    cfg.random_seed = seed
    for model in model_list: #, 'safety_falcon', 'mistral','falcon',  
        # experiment_name = to_collect[model]['experiment']
        # current_time = to_collect[model]['time']    
        for variable_1 in var_1_list:

            # for variable_2 in top_n_list:
                
                cfg.model_path = model

                if 'zh' in experiment_name:
                    cfg.source_lang = 'zh'
                elif 'th' in experiment_name:
                    cfg.source_lang = 'th'
                elif 'de' in experiment_name:
                    cfg.source_lang = 'de'
                cfg.source_lang = 'yo'
                # cfg.lang = 'yo'
                # variable_1
                
                
                cfg.lang = variable_1
                # cfg.lang = variable_1
                # cfg.lang = 'yo'
                
                
                if cfg.source_lang in ['yo', 'yi']:
                    cfg.filter_val = False
                
                if '32b' in model.lower():
                    cfg.batch_size = 16
                elif '70b' in model.lower():
                    cfg.batch_size = 8
                else:
                    cfg.batch_size = 32

                # cfg.ortho_lambda = 0.8
                
                # cfg.top_n = variable_2
                
                
                run_name = variable_1
                # run_name = 'yi'
                
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
#SBATCH --time=0-05:30:00
#SBATCH -q mcml
#SBATCH -p mcml-hgx-a100-80x4
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
#SBATCH -p mcml-dgx-a100-40x8

    #SBATCH -q mcml
#SBATCH -p mcml-hgx-a100-80x4
# lrz-dgx-a100-80x8

# 
#SBATCH -q mcml
# 
#SBATCH -p mcml-hgx-a100-80x4
# 
#SBATCH -p lrz-hgx-h100-94x4
# 
# 
#SBATCH -p mcml-hgx-a100-80x4
# python3 -m scripts.multi_inference --config {cfg_path} --type {variable_1}
#  
# python3 -m scripts.multi_inference --config configs/cfg.yaml --type harmful
\

# #SBATCH -p mcml-hgx-h100-94x4
#SBATCH -q mcml
# python3 -m scripts.multi_test --config {cfg_path} \
# 
# SBATCH -p lrz-dgx-a100-80x8
#BNB_CUDA_VERSION=122 \
# HF_HOME=/dss/dsshome1/lxc02/di75taw/storage/runs_models/huggingface \
# TRANSFORMERS_CACHE=/dss/dsshome1/lxc02/di75taw/storage/runs_models/huggingface \
# python3 -m pipeline.run_pipeline --config_path {cfg_path} \
# 
# python3 -m scripts.multi_test --config {cfg_path} \
# 
    