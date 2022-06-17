# run ablation on lnl
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LNL-0 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LNL-1 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LNL-2 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LNL-3 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LNL-4 --output="results_abla" --epoch 79
# # run ablation on lcl v
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-V-0 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-V-1 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-V-2 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-V-3 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-V-4 --output="results_abla" --epoch 79
# # run ablation on lcl q
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-Q-0 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-Q-1 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-Q-2 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-Q-3 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-Q-4 --output="results_abla" --epoch 79
# # run ablation on lcl interation vq
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-IN-0 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-IN-1 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-IN-2 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-IN-3 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-IN-4 --output="results_abla" --epoch 79
# # run ablation on lcl union vq
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-UNION-0 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-UNION-1 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-UNION-2 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-UNION-3 --output="results_abla" --epoch 79
# CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_RAD --input saved_models/VQAMix-LCL-UNION-4 --output="results_abla" --epoch 79
# run on path vqa set
CUDA_VISIBLE_DEVICES=2 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_models/SAN_MEVF-PATH-0 --output="results_abla" --epoch 79
CUDA_VISIBLE_DEVICES=2 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_models/SAN_MEVF-PATH-1 --output="results_abla" --epoch 79
CUDA_VISIBLE_DEVICES=2 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_models/SAN_MEVF-PATH-2 --output="results_abla" --epoch 79
CUDA_VISIBLE_DEVICES=2 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_models/SAN_MEVF-PATH-3 --output="results_abla" --epoch 79
CUDA_VISIBLE_DEVICES=2 python3 test.py --model SAN --use_RAD --RAD_dir data_PathVQA --input saved_models/SAN_MEVF-PATH-4 --output="results_abla" --epoch 79
# run on path vqa set
CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_models/BAN_MEVF-PATH-0 --output="results_abla" --epoch 79
CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_models/BAN_MEVF-PATH-1 --output="results_abla" --epoch 79
CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_models/BAN_MEVF-PATH-2 --output="results_abla" --epoch 79
CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_models/BAN_MEVF-PATH-3 --output="results_abla" --epoch 79
CUDA_VISIBLE_DEVICES=2 python3 test.py --model BAN --use_RAD --RAD_dir data_PathVQA --input saved_models/BAN_MEVF-PATH-4 --output="results_abla" --epoch 79
