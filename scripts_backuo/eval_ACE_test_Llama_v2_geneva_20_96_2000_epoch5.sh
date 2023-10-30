export CUDA_VISIBLE_DEVICES=2

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace \
    --valid_file ACE_valid.json \
    --test_file ACE_test.json \
    --save_dir /local1/zefan/results/test_Llama-2-7b-ace-20-96-2000-epoch5/ \
    --model /local1/zefan/output/Llama-2-7b-geneva-20-96-2000/epoch_5/ \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 8

