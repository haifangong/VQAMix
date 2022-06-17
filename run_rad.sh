# CUDA_VISIBLE_DEVICES=1 python3 main.py --model BAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/BAN_MEVF-RAD-VQAMix-0 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 0
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/BAN_MEVF-RAD-VQAMix-0 --output="results-RAD"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model BAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/BAN_MEVF-RAD-VQAMix-1 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 1
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/BAN_MEVF-RAD-VQAMix-1 --output="results-RAD"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model BAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/BAN_MEVF-RAD-VQAMix-2 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 2
# CUDA_VISIBLE_DEVICES=3 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/BAN_MEVF-RAD-VQAMix-2 --output="results-RAD"
CUDA_VISIBLE_DEVICES=3 python3 main.py --model BAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/BAN_MEVF-RAD-VQAMix-3 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 3
CUDA_VISIBLE_DEVICES=3 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/BAN_MEVF-RAD-VQAMix-3 --output="results-RAD"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model BAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/BAN_MEVF-RAD-VQAMix-4 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 4
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/BAN_MEVF-RAD-VQAMix-4 --output="results-RAD"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model SAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/SAN_MEVF-RAD-VQAMix-0 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 0
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model SAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/SAN_MEVF-RAD-VQAMix-0 --output="results-RAD"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model SAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/SAN_MEVF-RAD-VQAMix-1 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 1
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model SAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/SAN_MEVF-RAD-VQAMix-1 --output="results-RAD"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model SAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/SAN_MEVF-RAD-VQAMix-2 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 2
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model SAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/SAN_MEVF-RAD-VQAMix-2 --output="results-RAD"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model SAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/SAN_MEVF-RAD-VQAMix-3 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 3
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model SAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/SAN_MEVF-RAD-VQAMix-3 --output="results-RAD"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --model SAN --use_RAD --RAD_dir data_RAD --output saved_model_rad/SAN_MEVF-RAD-VQAMix-4 --epochs 80 --use_mix --alpha=1 --use_mix_cond --use_mix_cond_q --seed 4
# CUDA_VISIBLE_DEVICES=1 python3 test.py --model SAN --use_RAD --RAD_dir data_RAD --input saved_model_rad/SAN_MEVF-RAD-VQAMix-4 --output="results-RAD"
