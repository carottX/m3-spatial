CUDA_VISIBLE_DEVICES=0 python render_metrics.py \
--source_path /scratch1/train \
--model_path /scratch1/train_m3/train_bsz8_gpu1_embTrue_clipTrue_sigTrue_dinoFalse_seemFalse_llaTrue_llvFalse_dim160_temp0.05_test/run_0004 \
--preload_dataset_to_gpu_threshold 0 \
--local_sampling \
--render \
--skip_train \
--use_embed \
--use_clip \
--use_siglip \
# --use_seem \
# --use_llama3 \
# --use_llamav