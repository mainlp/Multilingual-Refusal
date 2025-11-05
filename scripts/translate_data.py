from dotenv import load_dotenv
load_dotenv(override=True)
import os
os.chdir(os.path.dirname('/mounts/Users/student/xinpeng/gdata/code/refusal-multilingual/'))

from dataset.load_dataset import load_dataset_split
import deepl
import json
from tqdm import tqdm
from deep_translator import GoogleTranslator


data_type = 'harmful'
split = 'train'


# translate the harmful_test set to the target language
def translate_harmful_to_target_language(target_lang, split):
    translator = GoogleTranslator(source='en', target=target_lang)
    if data_type == 'harmless':
        if split == 'test':
            split_ = split + '_500_sampled'
        else :
            split_ = split + '_200_sampled'
    else:
        split_ = split
    harmful_test_set = load_dataset_split(data_type, split=split_)
    harmful_test_set_translated = []
    
    if split == 'test':
    
        for item in tqdm(harmful_test_set):
            try:
                translated_item = translator.translate(item['instruction'])
                to_add = {
                    'instruction': item['instruction'],
                    'instruction_translated': translated_item,
                    'category': item['category']
                }
                harmful_test_set_translated.append(to_add)
            except Exception as e:
                print(f"Could not translate: {item['instruction']}")
                print(f"Error: {e}")
                continue
    else:
        for item in tqdm(harmful_test_set):
            try:
                translated_item = translator.translate(item['instruction'])
                to_add = {
                    'instruction': translated_item,
                    'category': item['category']
                }
                harmful_test_set_translated.append(to_add)
            except:
                continue
        # print(f"Translating: {item['instruction']} -> {translated_item.text}")
        # break
    return harmful_test_set_translated

# 'en', 'de','es', 'fr','it', 'nl', 'pl',  'ar','th',  'yo', 'ru', 
target_langs =  [ 'zh-CN', 'ja','ko',]
#  'th',  'yo', 'ru', 'zh', 'ja','ko',
# ['de', 'es', 'fr', 'it', 'nl', 'ja', 'pl', 'ru', 'zh', 'ko', 'ar']
# target_langs = []


for target_lang in target_langs:
    for data_type in [ "jailbreakbench"]:
        for split in ['test']:
            print(f"Translating {data_type}_{split} to {target_lang}")
            harmful_test_set_translated = translate_harmful_to_target_language(target_lang, split)
            with open(f'dataset/splits_multi/{data_type}_{split}_translated_{target_lang}.json', 'w') as f:
                json.dump(harmful_test_set_translated, f, indent=4)