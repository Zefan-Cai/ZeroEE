export CUDA_VISIBLE_DEVICES=7

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace_v1 \
    --valid_file ACE_valid_GenerationStyle.json \
    --test_file ACE_test_GenerationStyle.json \
    --save_dir /local1/zefan/results/Llama-2-7b-GenData-1definitions_ace_v1/ \
    --model /local1/zefan/output/Llama-2-7b-GenData-1definitions/ \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

