import random 
from itertools import permutations, combinations

import subprocess
import time
from collections import deque
from utils.utils import get_available_gpus, check_and_allocate_gpus
# from utils.utils_sys import read_training_variables, create_combination

import time
from itertools import product
import os
from dotenv import load_dotenv
load_dotenv(override=True)
import argparse
import os
from datetime import datetime
from itertools import permutations, product, combinations
import time
import mmengine




key = os.environ.get('DEEPL_KEY')






def generate_job_queue(path):
    job_queue = deque()
    # configs = read_training_variables(path='configs/eval.yaml')


    
    
    # current_time = '20240306-232014'
    # current_time = configs['current_time']

        
    current_time = time.strftime("%Y%m%d-%H%M%S")   
    # current_time = '20241209-163214'         
    model_list = ['Qwen/Qwen2.5-7B-Instruct',"meta-llama/Meta-Llama-3.1-8B-Instruct","google/gemma-2-9b-it"]
    # ["meta-llama/Meta-Llama-3-70B-Instruct" ]
    #  
    # ["meta-llama/Meta-Llama-3.1-8B-Instruct" ]
    # ['01-ai/Yi-6B-Chat']
    # []
                #   , 'Qwen/Qwen2.5-7B-Instruct', , 'Qwen/Qwen2.5-32B-Instruct']
    # ["google/gemma-2-9b-it"]
    # 
    # 
    # ["meta-llama/Llama-2-13b-chat-hf"]
    # ["meta-llama/Llama-2-13b-chat-hf", "google/gemma-2-2b-it",  "google/gemma-2-9b-it",  "google/gemma-2-27b-it" ] 
    
    # ['01-ai/Yi-1.5-34B-Chat', '01-ai/Yi-1.5-9B-Chat',, 'Qwen/Qwen2.5-7B-Instruct', 'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct']
                #   'Qwen/Qwen2-7B-Instruct', 'Qwen/Qwen2-1.5B-Instruct', 'Qwen/Qwen2-72B-Instruct' ]
    # 
    
    # ["google/gemma-2b-it"]
    # [,  "google/gemma-2b-it"]
    # 
    # ["meta-llama/Llama-2-13b-chat-hf"]
    # ["google/gemma-7b-it", "meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf" ,"meta-llama/Meta-Llama-3-8B-Instruct"]#, "meta-llama/Llama-2-7b-chat-hf", ] #s, , ]#, "google/gemma-7b-it",  ,  "google/gemma-2b-it", ]#, "Qwen/Qwen-7B-Chat", "google/gemma-7b-it", ] #, ] # "meta-llama/Meta-Llama-3.1-8B-Instruct" ,, ,"google/gemma-7b-it",,"meta-llama/Llama-2-7b-chat-hf",, "Qwen/Qwen-7B-Chat",
    # 
    # model_list = ["meta-llama/Llama-2-7b-chat-hf","meta-llama/Meta-Llama-3-8B-Instruct"]"meta-llama/Llama-2-70b-chat-hf"
    # current_time= '20241224-220423'
    base_config_path = 'configs/cfg.yaml'
    cfg = mmengine.Config.fromfile(base_config_path)





    # experiment_name = 'partial_ortho_0.2_OR_add_coeff'
    var_1 = 'lang'
    var_1_list = ['ko','ru','ja']#, 'th']
    # ['de', 'es', 'fr', 'it', 'nl', 'ja', 'pl', 'ru', 'zh', 'ko', 'ar']
    # ['ru']
    # 
    # ['zh']
    # 
    # ['zh']
    # 
    # 
    # 
    # [ 'it', 'ko', 'ar']
    # 
    # ['ko', 'ar', 'en', 'de', 'fr', '']
    # [0.6, 0.8, 1.0] #0, 0.2, 0.4, 
    # [0]
    # [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    # 
    #  #



    # steer_kl_threshold_list = [1]
    # var_2 = 'model'
    # var_2_list =  [0.8]
    # [0,0.2, 0.4, 0.6, 0.8, 1.0]
                    # 

  
    # [0.8]
    # 
    # []
    # [ 0.6 , 0.8, ]


    # top_n_list = [1, 2, 3, 4, 5]
    # var_2 = 'top_n'



    
    experiment_name =    'model_ablation_source_lang'   
    # 'second_ultra_lang_sweep_fix'  
    #  'th_vector_sweep'  
    # 'model_ablation_source_lang'   
    # 'zh_vector_sweep' 
    # 'second_ultra_lang_sweep_fix' 
 
   
 
    # 'model_ablation' 
    # 

    # 
    # 
    
    'gemma2_9b_zh_test'
    # 'yi_zh_test'

 
   
    # 
    # 'first_ultra_lang_sweep' 
    # 'xstest_contrast_harm_filter_or_nofilter_local'
    #    
    'add_1+0_xstest_constrast_harm_filter_or_nofilter'
    'baseline/system_prompt'
    # 'real_no_ablation_add_coeff_-1'  

    'real_no_ablation_xstest_constrast_harm_filter_or_nofilter'

    
    'system_prompt'
     #''
    # [ 0.2, 0.4, 0.6, 0.8, 1.0]

    seeds = [1]
    # ablaation_kl_threshold_list = [0.1, 0.5, 1.0, 2.0, 3.0]
    cfg.system = None
    for seed in seeds:
        cfg.random_seed = seed  
        for model in model_list: #, 'safety_falcon', 'mistral','falcon',       
            for variable_1 in var_1_list:
                # for variable_2 in var_2_list:
                # for variable_2 in top_n_list:
                    
                    cfg.model_path = model
                    cfg.source_lang = variable_1
                    cfg.lang = variable_1
                    
                    if '32b' in model.lower():
                        cfg.batch_size = 16
                    elif '70b' in model.lower():
                        cfg.batch_size = 8
                    else:
                        cfg.batch_size = 64
                    # 4
                    # cfg.n_train = 256
                    # cfg.ortho_lambda = variable_2
                    # cfg.ortho_lambda = 0.8
                    
                    # cfg.top_n = variable_2
                    
                    
                    run_name = variable_1
                    # run_name = 'th'
                    # run_name = 'harmfulablation'
                    
                    file_name = f"{run_name}.sbatch"
                    
                    output_dir = f"output/{experiment_name}/{model}/{run_name}/{current_time}/{seed}"
                    
                    cfg.artifact_path = output_dir
                    
                    # save the config file to yaml  
                    cfg_path = f"{output_dir}/{run_name}.yaml"
                    cfg.dump(cfg_path)
                    
                    # also snapshot the current python file
                    os.makedirs(f"{output_dir}/scripts", exist_ok=True)
                    os.system(f"cp scripts/experiment.py {output_dir}/scripts")

                    if not os.path.exists(output_dir):
                        # create if not exists, nested dir will also be created
                        os.makedirs(output_dir, exist_ok=True)    
                    script = f'''\
BNB_CUDA_VERSION=122 \
DEEPL_KEY={key} \
python3 -m pipeline.run_pipeline --config {cfg_path} \
'''
                    script = f'''TRANSFORMERS_CACHE=/mounts/Users/student/xinpeng/data/runs_models/huggingface \
''' + script
                    gpu_needed = 1  
                    if '32B' in model or '70B' in model:
                        gpu_type = ["80GB"]
                    else:
                        gpu_type = ["80GB", "40GB"]
                    # ['80GB'] #, 
                    job_queue.appendleft({"script":script, "gpus_needed": gpu_needed, "gpu_type": gpu_type})
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open (output_dir + f'script.sh', 'w') as rsh:
                        rsh.write(script)
        
    return job_queue


def main(path):
    job_queue = generate_job_queue(path)
    while True:
        job_queue = check_and_allocate_gpus(job_queue)
        # end if the job queue is empty
        if not job_queue:
            print('job queue is empty')
        time.sleep(60)  # Check for GPU availability every minute
        print('checking job queue')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the experiment')
    parser.add_argument('--path', type=str, help='path to the config file')

    args = parser.parse_args()
    main(args.path)   
    # python3 -m pipeline.run_pipeline --config_path {cfg_path} \
        
        
        
#         # BNB_CUDA_VERSION=122 \
# DEEPL_KEY={key} \

# 
# '''
# 
# python3 -m pipeline.run_pipeline --config {cfg_path} \
    # 
# 
# 
# 
        # 
# python3 -m scripts.multi_test --config {cfg_path} \
# 
# 
# 
# 
            # 