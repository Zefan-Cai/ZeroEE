export CUDA_VISIBLE_DEVICES=7

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace_v1 \
    --valid_file ACE_valid-clean.json \
    --test_file ACE_valid.json \
    --save_dir /local1/zefan/results/Llama-2-7b-GenData-5definitions/ \
    --model /local1/zefan/output/Llama-2-7b-GenData-5definitions/ \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

