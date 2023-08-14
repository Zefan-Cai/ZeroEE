export CUDA_VISIBLE_DEVICES=7

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace \
    --valid_file ACE_valid-clean.json \
    --test_file ACE_test.json \
    --save_dir /local1/zefan/results/Llama_v2_geneva_30_test/ \
    --model /local1/zefan/output/Llama-2-7b-geneva-30/epoch_22 \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

