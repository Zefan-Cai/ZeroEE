export CUDA_VISIBLE_DEVICES=6

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace_v2 \
    --valid_file ACE_valid_GenerationStyle.json \
    --test_file ACE_test_GenerationStyle.json \
    --save_dir /local1/zefan/results/Llama2_Geneva_20_96_2000_GenData200_epoch6_ACE_v2/ \
    --model /local1/zefan/output/Llama2_Geneva_20_96_2000_GenData200/epoch_6/torch_model_save \
    --tokenizer /local1/zefan/models/Llama-2-7b-hf/ \
    --eval_batch_size 128

zw