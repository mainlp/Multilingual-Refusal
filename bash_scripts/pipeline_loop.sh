#  "google/gemma-7b-it" "meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Meta-Llama-3-8B-Instruct")
# models=("google/gemma-2b-it")
models=("meta-llama/Meta-Llama-3-8B-Instruct")
for model in "${models[@]}"
do
    CUDA_VISIBLE_DEVICES=0 \
    TRANSFORMERS_CACHE=/mounts/Users/student/xinpeng/data/runs_models/huggingface \
    HF_HOME=/mounts/Users/student/xinpeng/data/runs_models/huggingface \
    BNB_CUDA_VERSION=122 \
    python3 -m pipeline.run_pipeline --model_path "$model"
done