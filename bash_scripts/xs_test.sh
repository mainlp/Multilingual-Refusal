CUDA_VISIBLE_DEVICES=0 \
TRANSFORMERS_CACHE=/mounts/Users/student/xinpeng/data/runs_models/huggingface \
HF_HOME=/mounts/Users/student/xinpeng/data/runs_models/huggingface \
BNB_CUDA_VERSION=122 \
python scripts/xstest_eval.py --model_name meta-llama/Llama-2-7b-chat-hf