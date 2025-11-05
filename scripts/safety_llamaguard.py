import json
import os
import argparse
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

load_dotenv()

def read_json(file_path):
    print(f"Attempting to open file: {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to: {file_path}")

def moderate_batch(llama_guard, tokenizer_guard, batch):
    formatted_inputs = [
        tokenizer_guard.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        for conv in batch
    ]

    model_inputs = tokenizer_guard(
        formatted_inputs,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    terminators = [
        tokenizer_guard.eos_token_id,
        tokenizer_guard.convert_tokens_to_ids("<|eot_id|>")
    ]

    generated_ids = llama_guard.generate(**model_inputs, eos_token_id=terminators, max_new_tokens=256, pad_token_id=tokenizer_guard.eos_token_id)

    return tokenizer_guard.batch_decode(generated_ids[:, model_inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def count_safety(items):
    counts = {'safe': 0, 'unsafe': 0}
    safety_list = []
    for item in items:
        if item.strip() == 'safe':
            counts['safe'] += 1
            safety_list.append(1)
        else:
            counts['unsafe'] += 1
            safety_list.append(0)
    return counts, safety_list

def main(args):
    # Construct the input file path based on method and dataset
    if args.method == "baseline":
        input_file = f'pipeline/runs/{args.model}/baseline/{args.dataset}_baseline_evaluations.json'
    elif args.method == "ablated":
        input_file = f'output_samples/ultra_sweep_3/{args.model}/add_coeff_{args.coeff}_lambda_{args.lambda_value}/20240918-151944/completions/{args.dataset}_or_ablation_harm_actadd_evaluations.json'
    elif args.method == "define":
        input_file = args.model
    
    # Load the JSON file
    data = read_json(input_file)

    # Initialize LlamaGuard model and tokenizer
    model_id = "meta-llama/Llama-Guard-3-8B"
    llama_guard = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map='auto', cache_dir="/nfs/gdata/chengzhi/huggingface")
    tokenizer_guard = AutoTokenizer.from_pretrained(model_id, cache_dir="/nfs/gdata/chengzhi/huggingface")
    tokenizer_guard.padding_side = "left"
    tokenizer_guard.pad_token = tokenizer_guard.eos_token

    # Process each completion in the JSON
    total_safety_count = 0
    for completion in tqdm(data['completions'], desc="Processing completions"):
        prompt = completion['prompt']
        response = completion['response']
        
        # Create conversation format
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        # Moderate the conversation
        guard_pred = moderate_batch(llama_guard, tokenizer_guard, [conversation])
        
        # Calculate safety
        safety_counts, safety_list = count_safety(guard_pred)
        
        # Add safety score to the completion
        completion['llamaguard_safety'] = safety_list[0]
        
        total_safety_count += safety_list[0]

    # Calculate average safety score
    avg_safety_score = total_safety_count / len(data['completions'])
    print(f"Average safety score: {avg_safety_score:.4f}")

    # Save the updated JSON
    output_file = input_file.replace('.json', '_with_llamaguard_safety.json')
    save_json(data, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["baseline", "ablated", "define"], required=True)
    parser.add_argument("--dataset", type=str, default="xstest", help="Dataset to use (e.g., harmful, or_bench_hard, xstest)")
    parser.add_argument("--model", type=str, default="Llama-2-7b-chat-hf", help="Model to use (e.g., Llama-2-7b-chat-hf, Llama-3-8b)")
    parser.add_argument("--coeff", type=float, default=0, help="Coefficient value")
    parser.add_argument("--lambda_value", type=float, required=True, help="Lambda value")


    args = parser.parse_args()

    main(args)