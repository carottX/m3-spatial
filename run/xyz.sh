python compute_mem_xyz.py \
--model_path /scratch1/train_m3/train_bsz8_gpu1_embTrue_clipTrue_sigTrue_dinoFalse_seemFalse_llaTrue_llvFalse_dim160_temp0.05_test/run_0004 \
--temperature 1 \
--out_dir ./output \
--local_sampling \
--preload_dataset_to_gpu_threshold 0 \
--source_path /scratch1/train \
--render \