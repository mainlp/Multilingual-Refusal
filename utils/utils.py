import subprocess
import re
import pandas as pd
import GPUtil
import torch.nn.functional as F
import yaml
from itertools import product
import numpy as np
import json
from transformers import AutoTokenizer
def get_available_mig_gpus():
    """
    Get the list of available MIG-mode GPUs.
    """
    # run the command "mig list" and get the output
    result = subprocess.run(['mig', 'list'], text=True, capture_output=True)


    # # Sample input string
    # output_str = """
    # ☑️  4/0 -> MIG-9dfbe979-06a1-5cab-9421-4d953ccc212b (24MiB / 19968MiB)
    # ☑️  4/1 -> MIG-addf3595-2623-5ed6-81f3-5c0436ac1be2 (25MiB / 19968MiB)
    # ☑️  4/2 -> MIG-2f4f1b86-4eb6-5049-8b0a-54af025f0035 (25MiB / 19968MiB)
    # ☑️  4/3 -> MIG-f860893a-b684-545c-a03b-1d86722e740a (12MiB / 9728MiB)
    # ☑️  5/0 -> MIG-6a7aa340-75e4-52fb-b906-3d7ef118662c (37MiB / 40192MiB)
    # ☑️  5/1 -> MIG-bc5c3190-8f61-5675-87e8-47322ffe2138 (37MiB / 40192MiB)
    # 🔥 6/0 -> MIG-65b91f1a-e4e8-5ce9-b732-3913baf5a6d1 (254MiB / 40192MiB)
#         xinpeng / 1778756 / 'python'
    # ☑️  6/1 -> MIG-2c4cee2c-ec97-5007-9a07-f73212d353ca (37MiB / 40192MiB)
    # ☑️  7/0 -> MIG-e6c9b2f7-e93e-5f29-9ab1-864c18ba8261 (37MiB / 40192MiB)
    # ☑️  7/1 -> MIG-f3649993-0669-5e8a-8220-10ccda377d74 (37MiB / 40192MiB)
    # """

    # Using regex to find all UUIDs and associated status symbols
    # Define the function to round memory sizes
    def round_memory_size(memory_in_mib):
        """Convert memory size in MiB to the nearest 20GB or 40GB."""
        gigabytes = int(memory_in_mib) / 1024  # Convert MiB to GB
        
        if gigabytes <= 10:
            return "10GB"
        if 10 < gigabytes <= 20:
            return "20GB"
        elif 20 < gigabytes <= 40:
            return "40GB"
        elif 40 < gigabytes :
            return "80GB"

    
    results = re.findall(r'(🔥|☑️)\s+.*?->\s+(MIG-[a-f0-9\-]+)\s+\((\d+)MiB / (\d+)MiB\)', result.stdout)
    # Separating into in-use (with fire) and not in-use (without fire)
    in_use = [(uuid, round_memory_size(total_size)) for status, uuid, used_size, total_size in results if "🔥" in status]
    not_in_use = [(uuid, round_memory_size(total_size)) for status, uuid, used_size, total_size in results if "☑️" in status]
    
    return in_use, not_in_use



def get_available_gpus(leave_empty=2):

    available_cuda_gpus = GPUtil.getAvailable(order='memory', limit=8, maxLoad=0.9, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
    # If available_cuda_gpus is not empty, then add 80GB to each element, and return the list of turples
    if available_cuda_gpus and len(available_cuda_gpus) > leave_empty:
        num_to_use = len(available_cuda_gpus) - leave_empty
    else:
        num_to_use = 0
    if available_cuda_gpus:
        available_cuda_gpus = available_cuda_gpus[:num_to_use]
        available_cuda_gpus = [(gpu, '80GB') for gpu in available_cuda_gpus]
    else:
        available_cuda_gpus = []
    inuse_mig_gpus, available_mig_gpus =  get_available_mig_gpus()
    
    return available_cuda_gpus+available_mig_gpus
    
# Replace this with your GPU's UUID
# gpu_uuid = "Your-GPU-UUID-Here"
# status = get_gpu_status_by_uuid(gpu_uuid)
# print(status)



def convert_xlsx_to_csv(xlsx_file_path, csv_file_path):
    # Read the XLSX file
    data = pd.read_excel(xlsx_file_path)

    # Save the data to a CSV file
    data.to_csv(csv_file_path, index=False)
    
    
    
def get_tokenizer_from_path(path):

    if 'mixtral' in path:
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mixtral-8x7B-Instruct-v0.1')
    elif 'mistral' in path:
        tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
    elif 'llama2-7b' in path:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    elif 'llama2-13b' in path:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-13b-chat-hf')
    elif 'llama2-70b' in path:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat-hf')
    elif 'Yi-34b' in path:
        tokenizer = AutoTokenizer.from_pretrained('01-ai/Yi-34B-Chat')
    elif 'Yi-6b' in path:
        tokenizer = AutoTokenizer.from_pretrained('01-ai/Yi-6B-Chat')
    elif 'gemma-7b-it' in path:
        tokenizer = AutoTokenizer.from_pretrained('google/gemma-7b-it')
        
    return tokenizer


def get_references_probability(references_list, tokenizer, score, option_id):
 
    references_probabilities = []
    alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    for idx, reference in enumerate(references_list):
        if option_id:
            reference = alphabet[idx] + '. ' + reference
        reference_tokens = tokenizer(reference, return_tensors="pt", padding=False, truncation=True)
        reference_tokens = {k: v.to('cuda') for k, v in reference_tokens.items()}
        reference_ids = reference_tokens['input_ids'][0, 1:]

        # get the sequence probability

        
        
        # Get the probability of the reference\
        score_to_look = score[:len(reference_ids)]
        
        # Convert logits to probabilities
        probabilities = [F.softmax(score, dim=1) for score in score_to_look]

        # Indexing the probabilities with reference_ids
        indexed_probabilities = [prob[0, idx].item() for prob, idx in zip(probabilities, reference_ids)]

        # transform the prob to log base ,for the sum of log probabilities
        indexed_probabilities = [np.log(prob) for prob in indexed_probabilities]
        
        # Calculate the product of probabilities, on the log base
        sequence_probability =  np.sum(indexed_probabilities)
        references_probabilities.append(sequence_probability.item())
        
        
    return references_probabilities




def prepare_prompt(model_family, question, references_list, steer_text, args):
    general_prefix = args.general_instruct
    
    
    if args.pride_instruct:
        # general_prefix = 'Please read the multiple-choice question below carefully and select ONE of the listed options and repeat the answer. \n'
        subcategories = args.subcategories
        general_prefix = f'The following are multiple choice questions about {subcategories}. You should directly answer the question by choosing the correct option. \n'
    
    # if not args.option_id:
    #     general_prefix = 'Please read the multiple-choice question below carefully and select ONE of the listed options and repeat the answer. \n'
    general_example = 'Here is an example of the format: \n\nQuestion: Question 1 \n\nA. Option 1 \nB. Option 2 \nC. Option 3 \nAnswer: A/B/C \n'
    example_1 = 'Question: What type of vacation do you prefer? \nA. Beach holiday \nB. Adventure travel \nC. Cultural exploration \nD. Staycation \nE. Refused \nAnswer: E \n\n'
    example_2 = 'Question: What is your favorite color? \nA. Red \nB. Blue \nC. Green \nD. Yellow \nE. Refused \nAnswer: C \n\n'
    example_3 = 'Question: What is your favorite food? \nA. Pizza \nB. Pasta \nC. Steak \nD. Salad \nE. Refused \nAnswer: B \n\n'
    example_4 = 'Question: Do you prefer cooking at home or dining out? \nA. Cooking at home \nB. Dining out \nC. No preference \nD. Refused \nAnswer: D \n\n'
    example_5 = '\nQuestion: In politics today,  do you consider yourself a \nA. Republican\nB. Democrat\nC. Independent\nD. Something else\nE. Refused\nAnswer: B\n\n' 
    prefix = general_prefix
    if args.general_example:
        prefix = prefix + general_example + '\n'
    if not args.instruction_tuned:
        prefix = general_prefix + example_1 + example_2 + example_3 + example_4
    if steer_text != 'None':
        prefix = steer_text + ' ' + prefix
    question = prefix + 'Question: ' + question + ' \nOptions: \n'
    alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    if not args.option_id:
         for idx, option in enumerate(references_list):
            question = question + option + ' \n' if model_family != 'falcon' else question + option + ' '
    else:
        for idx, option in enumerate(references_list):
            question = question + alphabet[idx] + '. ' + option + ' \n' if model_family != 'falcon' else question + alphabet[idx] + '. ' + option + ' '
    
    question =  question + 'Answer:'
    if model_family in ['llama2-arxiv']:
        # TODO: Set the gneration config, such as temperature, max_new_tokens, to return the hidden states
        question = '<s>[INST] ' + question + ' [/INST]'
    elif model_family in ['mistral', 'mixtral', 'llama2', 'falcon', 'Yi', 'finetuned', 'gemma']: 
        # 'Please read the multiple-choice question below carefully and select ONE of the listed options and only give a single letter. \n'
        # general_prefix = 'Please read the multiple-choice question below carefully and select ONE of the listed options and explain the reason.\n'
        # prefix =  'Please read the multiple-choice question below carefully and select ONE of the listed options.\n'
        # prefix = general_prefix + example_5
        # prefix = example_5
        # create a list from 'A' to 'Z'
        # add the references with A, B, C, D ... prefix
        if args.instruction_tuned and model_family == 'falcon':
            question = steer_text + ' ' + general_prefix + "User:" + question.replace( prefix, '') + ' \nAssistant:'  
        # elif args.instruction_tuned and model_family != 'gemma':
        #     question =  question + 'Answer:'
    elif model_family in ['safety_llama2', 'safety_mistral', 'safety_llama2-full', 'safety_falcon'] :
        prefix = "A chat between a user and an AI assistant. The assistant answers the user's questions.\n\n ### User: " + prefix            
        question =  question + '\n' + 'Answer:' + '\n' +  '### Assistant:'
    return question



def check_and_allocate_gpus(job_queue, leave_empty=4):
    available_gpus = get_available_gpus(leave_empty=leave_empty)

    # If there are jobs in the queue, allocate them
    while job_queue:
        job = job_queue.pop()
        # extract the gpus with GPU- prefix
        available_CUDA_gpus = [gpu[0] for gpu in available_gpus if isinstance(gpu[0], int)]
        # available_MIG_gpus = [gpu for gpu in available_gpus if gpu[0].startswith('MIG-')]
        if job["gpus_needed"] >1:
            # if the job needs more than one gpu, then only allocate the GPU- ones
            avaible_required_gpus = available_CUDA_gpus
        
        else:
            # Allocate GPUs to the job
            # consider the available gpus and the job requirement
            avaible_required_gpus = [gpu[0] for gpu in available_gpus if gpu[1] in job["gpu_type"]]

        if len(avaible_required_gpus) >= job["gpus_needed"]:
            # Allocate GPU(s) to this job, select the gpu that has the same type
            selected_gpus = avaible_required_gpus[:job["gpus_needed"]]
            selected_gpus_sizes = [gpu[1] for gpu in available_gpus if gpu[0] in selected_gpus]
            
            available_gpus = [gpu for gpu in available_gpus if gpu[0] not in selected_gpus]
            # Set environment variable for CUDA to use only the selected GPUs
            # env = {"CUDA_VISIBLE_DEVICES": ','.join(map(str, selected_gpus))}
            # format the selected gpus to a string starting with CUDA_VISIBLE_DEVICES, so that it can be added to the command line
            selected_gpus_prefix = 'CUDA_VISIBLE_DEVICES=' + ','.join(map(str, selected_gpus)) + ' '
            
            
            script_to_run = selected_gpus_prefix + job["script"]
            
            # if '20GB' in selected_gpus_sizes:
            #     script_to_run += ' --batch_size 8'
            
            # elif '40GB' in selected_gpus_sizes:
            #     script_to_run += ' --batch_size 16'
            
            # Run the job script with the allocated GPU(s)
            print(f"Running {selected_gpus_prefix + job['script']} on GPU(s): {selected_gpus}")
            
            
            
            
            # subprocess.Popen( 'CUDA_VISIBLE_DEVICES=MIG-f3649993-0669-5e8a-8220-10ccda377d74 ' + job["script"], shell=True)#, env=env)
            subprocess.Popen(script_to_run, shell=True)
        else:
            # Not enough GPUs available for this job, put it back in the queue
            job_queue.append(job)
            break
    
    return job_queue



def read_training_variables(path):
    #read yaml configuration file
    with open(path, 'r') as stream:
        try:
            variables = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    # create variballe with the same name as the key in the yaml file
    # for key, value in variables.items():
    #     exec(key + " = value")
    
    return variables


def create_combination(config_variables):
    for key, value in config_variables.items():
        if not isinstance(value, list):
            config_variables[key] = [value]
    # Use itertools.product to generate all combinations
    combinations = list(product(*config_variables.values()))
    # Convert combinations back to dictionaries
    combination_dicts = [{key: value for key, value in zip(config_variables.keys(), combo)} for combo in combinations]
    return combination_dicts


def entropy(labels, base=None):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0
    # Compute entropy
    base = np.e if base is None else base
    for i in probs:
        ent -= i * np.log(i) / np.log(base)
    return ent

def read_json_to_df(output_file):
    # read json file
    with open(output_file) as f:
        # load json file where the lines are separated by \
            # n
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
    df = pd.DataFrame(data)
    return df




class LoggerWriter:
    def __init__(self, log_func):
        self.log_func = log_func
        self.line = ""

    def write(self, message):
        if message != "\n":  # Skip empty lines
            self.line += message
            if message.endswith("\n"):  # Log complete lines
                self.log_func(self.line.strip())
                self.line = ""

    def flush(self):
        if self.line:  # Flush remaining content
            self.log_func(self.line.strip())
            self.line = ""