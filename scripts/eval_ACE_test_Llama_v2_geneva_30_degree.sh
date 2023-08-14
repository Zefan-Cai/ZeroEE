export CUDA_VISIBLE_DEVICES=7

# cot
python -m open_instruct.eval.ace.run_eval_degree \
    --data_dir /local1/zefan/data/degree_ed_ace05e_defi \
    --valid_file dev.json \
    --test_file test.json \
    --save_dir /local1/zefan/results//Llama-2-7b-geneva-30-ace-degree/ \
    --model /local1/zefan/output/Llama-2-7b-geneva-30/epoch_25 \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

