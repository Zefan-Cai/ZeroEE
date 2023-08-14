export CUDA_VISIBLE_DEVICES=0

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace \
    --valid_file ACE_valid-clean.json \
    --test_file ACE_test.json \
    --save_dir /local1/zefan/results/Llama_v2_ace_8_test/ \
    --model /local1/zefan/output/Llama-2-7b-ace-8/epoch_8 \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

