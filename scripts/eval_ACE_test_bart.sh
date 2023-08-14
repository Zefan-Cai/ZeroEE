export CUDA_VISIBLE_DEVICES=0

# cot
python -m open_instruct.eval.ace.run_eval_bart \
    --data_dir /local1/zefan/data/ace \
    --valid_file ACE_valid-clean.json \
    --test_file ACE_test.json \
    --save_dir /local1/zefan/results/test_ace_bart_large_epoch44/ \
    --model /local1/zefan/output/bart-large-ACE-15/epoch_44/ \
    --tokenizer /local1/zefan/models/bart-large/ \
    --eval_batch_size 4

