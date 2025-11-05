CUDA_VISIBLE_DEVICES=3 \
TRANSFORMERS_CACHE=/mounts/Users/student/xinpeng/data/runs_models/huggingface \
HF_HOME=/mounts/Users/student/xinpeng/data/runs_models/huggingface \
BNB_CUDA_VERSION=122 \
python3 -m pipeline.run_pipeline --config_path configs/cfg.yaml