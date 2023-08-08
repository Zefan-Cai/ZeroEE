export CUDA_VISIBLE_DEVICES=0

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /home/caizf/projects/ZeroEE/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /home/caizf/projects/ZeroEE/results/ace_ace_test/ \
    --model /home/caizf/projects/ZeroEE/output/gpt_1.3b/ \
    --tokenizer /home/caizf/models/opt-1.3b/ \
    --eval_batch_size 256

