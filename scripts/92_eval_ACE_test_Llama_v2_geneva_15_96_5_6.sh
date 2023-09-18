export CUDA_VISIBLE_DEVICES=4

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-15-96-5-6-epoch10/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-15-96-5-6/epoch_10/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8


# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-15-96-5-6-epoch20/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-15-96-5-6/epoch_20/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8


# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-15-96-5-6-epoch29/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-15-96-5-6/epoch_29/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8
