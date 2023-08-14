export CUDA_VISIBLE_DEVICES=0

# cot
python -m open_instruct.eval.ace.run_eval_degree \
    --data_dir /local1/zefan/data/degree_ed_ace05e_defi \
    --valid_file dev.json \
    --test_file test.json \
    --save_dir /local1/zefan/results/test_Llama/ \
    --model /local1/zefan/output/Llama-2-7b-ace-15/epoch_41/ \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 4

