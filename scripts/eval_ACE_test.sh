export CUDA_VISIBLE_DEVICES=0

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /local1/zefan/results/ace_ace_bart-large/ \
    --model /local1/zefan/output/ace_13_bart-large/ \
    --tokenizer /local1/zefan/models/bart-large/ \
    --eval_batch_size 256

