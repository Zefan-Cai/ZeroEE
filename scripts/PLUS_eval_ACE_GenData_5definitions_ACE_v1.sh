export CUDA_VISIBLE_DEVICES=7

MODEL_NAME="$1"

# cot
python -m open_instruct.eval.ace.run_eval \
    --data_dir /local1/zefan/data/ace_v2 \
    --valid_file ACE_valid_v2_inference.json \
    --test_file ACE_test_v2_inference.json \
    --save_dir /local1/zefan/results/${MODEL_NAME} \
    --model /local1/zefan/output/${MODEL_NAME} \
    --tokenizer /local1/zefan/output/${MODEL_NAME} \
    --eval_batch_size 8

