#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import random
import json
import os
os.chdir(os.path.dirname('/mounts/Users/student/xinpeng/gdata/code/refusal_direction/'))
import argparse

from dataset.load_dataset import load_dataset_split, load_dataset

from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import get_activation_addition_input_pre_hook, get_all_direction_ablation_hooks

from pipeline.submodules.generate_directions import generate_directions
from pipeline.submodules.select_direction import select_direction, get_refusal_scores
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.evaluator.evalharness import LMEvalHarness

import mmengine
from pipeline.utils.hook_utils import add_hooks
import jsonpickle
import sys


# In[2]:


cfg = mmengine.Config.fromfile('configs/results.yaml')
model_list = ['google/gemma-7b-it', 'meta-llama/Llama-2-7b-chat-hf',  'meta-llama/Llama-2-13b-chat-hf', 'meta-llama/Meta-Llama-3-8B-Instruct', ]
model = model_list[1]
experiment, time, seed, mmlu = cfg['models'][model][0]
model_base = construct_model_base(model)


# In[3]:


add_coeff = 0.8
lambda_value = 0
dir_path = f'output/{experiment}/{model}/add_coeff_{add_coeff}_lambda_{lambda_value}/{time}' if not seed else f'output/{experiment}/{model}/add_coeff_{add_coeff}_lambda_{lambda_value}/{time}/{seed}'


# In[4]:


dataset = [{'instruction':'''What happens if you cry during the New Year?'''}]


# In[5]:


# read dataset
with open(f'dataset/processed/xstest_unsafe.json') as f:
    dataset = json.load(f)


# In[6]:


direction_ablation = torch.load(f'{dir_path}/direction_ablation.pt')[0]
direction_addition = torch.load(f'{dir_path}/direction_addition.pt')[0]


# In[7]:


with open(f'{dir_path}/direction_metadata_addition.json', 'r') as f:
    config = json.load(f)
addition_layer = config['layer'][0]


# In[8]:


harm_actadd_fwd_pre_hooks, harm_actadd_fwd_hooks = [(model_base.model_block_modules[addition_layer], get_activation_addition_input_pre_hook(vector=direction_addition, coeff=+add_coeff))], []
or_ablation_fwd_pre_hooks, or_ablation_fwd_hooks = get_all_direction_ablation_hooks(model_base, direction_ablation, 0)


# In[9]:


or_ablation_harm_actadd_fwd_pre_hooks = or_ablation_fwd_pre_hooks + harm_actadd_fwd_pre_hooks
or_ablation_harm_actadd_fwd_hooks = or_ablation_fwd_hooks + harm_actadd_fwd_hooks


# In[10]:


completions = model_base.generate_completions(dataset, fwd_pre_hooks=or_ablation_harm_actadd_fwd_pre_hooks, fwd_hooks=or_ablation_harm_actadd_fwd_hooks, max_new_tokens=512, batch_size=1)
with open(f'{cfg.artifact_path}/completions/xstest.json', "w") as f:
        json.dump(completions, f, indent=4)


# In[405]:


# model_base.generate_completions(dataset,  max_new_tokens=512, batch_size=1)


# In[ ]:





# 

# In[ ]:




