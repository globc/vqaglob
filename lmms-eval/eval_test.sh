export CUDA_VISIBLE_DEVICES=7
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model instructblip \
#     --model_args setup="ANSWER" \
#     --tasks docvqa_valglob,chartqaglob,infovqa_valglob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4


accelerate launch --num_processes=1 -m lmms_eval \
    --model instructblip \
    --model_args setup="COT_CHAT_BASIC" \
    --tasks glob \
    --batch_size 1 \
    --output_path ./logs/ \
    --limit 1000 \
    #--num_fewshot 4 >> output2.txt

accelerate launch --num_processes=1 -m lmms_eval \
    --model instructblip \
    --model_args setup="COT_CHAT_BASIC" \
    --tasks glob \
    --batch_size 1 \
    --output_path ./logs/ \
    --limit 1000 \
    --num_fewshot 4 >> output2.txt