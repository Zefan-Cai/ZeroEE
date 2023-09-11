export CUDA_VISIBLE_DEVICES=7

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/test-Llama-2-7b-geneva-20-40-10-epoch18/ \
    --model /home/caizf/projects/ZeroEE/output/Llama-2-7b-geneva-20-40-10/epoch_18/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

