export CUDA_VISIBLE_DEVICES=4

# cot
python -m open_instruct.eval.ace.run_eval_degree \
    --data_dir /local1/zefan/data/degree_ed_ace05e_defi \
    --valid_file dev.json \
    --test_file test.json \
    --save_dir /local1/zefan/results/test_ace_bart_large_degree_epoch7/ \
    --model /local1/zefan/output/bart-large-degree/epoch_7/ \
    --tokenizer /local1/zefan/models/bart-large/ \
    --eval_batch_size 128

