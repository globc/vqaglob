(
export CUDA_VISIBLE_DEVICES=7
accelerate launch --num_processes=1 -m lmms_eval \
    --model qwen_vl \
    --model_args fewshot_images="True,setup=ANSWER" \
    --tasks glob \
    --batch_size 1 \
    --output_path ./logs/ \
    --limit 1000 \
    --num_fewshot 4 >> output4.txt

export CUDA_VISIBLE_DEVICES=7
accelerate launch --num_processes=1 -m lmms_eval \
    --model llava \
    --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b,use_flash_attention_2=False,device_map=cuda:0,fewshot_images=False,setup=CHAT_NULL" \
    --tasks glob \
    --batch_size 1 \
    --output_path ./logs/ \
    --limit 1000 \
    --num_fewshot 4 >> output5.txt

) &

(
export CUDA_VISIBLE_DEVICES=9
accelerate launch --num_processes=1 -m lmms_eval \
    --model idefics \
    --model_args fewshot_images="True,setup=ANSWER" \
    --tasks glob \
    --batch_size 1 \
    --output_path ./logs/ \
    --limit 1000 \
    --num_fewshot 4 >> output6.txt

export CUDA_VISIBLE_DEVICES=9
accelerate launch --num_processes=1 -m lmms_eval \
    --model instructblip \
    --model_args setup="CHAT_NULL" \
    --tasks glob \
    --batch_size 1 \
    --output_path ./logs/ \
    --limit 1000 \
    --num_fewshot 4 >> output7.txt
) &

wait



# (
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model qwen_vl \
#     --model_args fewshot_images="True,setup=COT_2" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4
# ) &

# (
# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b,use_flash_attention_2=False,device_map=cuda:0,fewshot_images=False,setup=COT_2" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4
# ) &

# (
# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model idefics \
#     --model_args fewshot_images="True,setup=COT_2" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4
# ) &

# wait

# (
# export CUDA_VISIBLE_DEVICES=1
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model qwen_vl \
#     --model_args fewshot_images="True,setup=COT_3" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4
# ) &

# (
# export CUDA_VISIBLE_DEVICES=2
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b,use_flash_attention_2=False,device_map=cuda:0,fewshot_images=False,setup=COT_3" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4
# ) &

# (
# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model idefics \
#     --model_args fewshot_images="True,setup=COT_3" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4
# ) &

# wait

# (
# export CUDA_VISIBLE_DEVICES=3
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model instructblip \
#     --model_args setup="COT_CHAT_BASIC" \
#     --tasks vqav2_valglob,textvqa_valglob,docvqa_valglob,ok_vqa_val2014glob,aok_vqaglob,ai2dglob,vizwiz_vqa_valglob,gqa,chartqaglob,infovqa_valglob,contextual_val \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 0

# ) &

# (
# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model multimodal_cot \
#     --model_args answer_model="idefics" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 0
# ) &

# wait

###############################################################

# (
# export CUDA_VISIBLE_DEVICES=5
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model instructblip \
#     --model_args setup="CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 2 >> output2.txt
# ) &

# (
# export CUDA_VISIBLE_DEVICES=6
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model instructblip \
#     --model_args setup="CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4 >> output3.txt
# ) &

# wait

# (
# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model qwen_vl \
#     --model_args fewshot_images="False,setup=CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 2 >> output1.txt
# ) &

# (
# export CUDA_VISIBLE_DEVICES=5
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model qwen_vl \
#     --model_args fewshot_images="False,setup=CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4 >> output2.txt
# ) &

# (
# export CUDA_VISIBLE_DEVICES=6
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model qwen_vl \
#     --model_args fewshot_images="True,setup=CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 2 >> output3.txt
# ) &

# wait

# (
# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model multimodal_cot \
#     --model_args answer_model="idefics" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 0 >> output1.txt
# ) &

# (
# export CUDA_VISIBLE_DEVICES=5
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model qwen_vl \
#     --model_args fewshot_images="True,setup=CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4 >> output2.txt
# ) &

wait

# (
# export CUDA_VISIBLE_DEVICES=5
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b,use_flash_attention_2=False,device_map=cuda:0,fewshot_images=False,setup=CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 2 > output2.txt
# ) &

# (
# export CUDA_VISIBLE_DEVICES=6
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b,use_flash_attention_2=False,device_map=cuda:0,fewshot_images=False,setup=CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4 > output3.txt
# ) &

# wait

# (
# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b,use_flash_attention_2=False,device_map=cuda:0,fewshot_images=True,setup=CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 2 >> output1.txt
# ) &

# (
# export CUDA_VISIBLE_DEVICES=5
# accelerate launch --num_processes=1 -m lmms_eval \
#     --model llava \
#     --model_args pretrained="liuhaotian/llava-v1.6-vicuna-7b,use_flash_attention_2=False,device_map=cuda:0,fewshot_images=True,setup=CHAT_NULL" \
#     --tasks glob \
#     --batch_size 1 \
#     --output_path ./logs/ \
#     --limit 1000 \
#     --num_fewshot 4 >> output2.txt
# ) &

# (
# export CUDA_VISIBLE_DEVICES=6
# accelerate launch --num_processes=1 -m lmms_eval \
#   --model_args: fewshot_images="False,setup=FORMAT" \
#   --tasks glob \
#   --batch_size 16 \
#   --output_path: ./logs/ \
#   --limit 1000 \
#   --num_fewshot 0 >> output3.txt
# ) &


# wait

# (
# export CUDA_VISIBLE_DEVICES=4
# accelerate launch --num_processes=1 -m lmms_eval \
#   --model_args: fewshot_images="False,setup=EXPERT" \
#   --tasks glob \
#   --batch_size 16 \
#   --output_path: ./logs/ \
#   --limit 1000 \
#   --num_fewshot 0 >> output1.txt
# ) &

# (
# export CUDA_VISIBLE_DEVICES=5
# accelerate launch --num_processes=1 -m lmms_eval \
#   --model_args: fewshot_images="False,setup=CHAT" \
#   --tasks glob \
#   --batch_size 16 \
#   --output_path: ./logs/ \
#   --limit 1000 \
#   --num_fewshot 0 >> output2.txt

# ) &

# (
# export CUDA_VISIBLE_DEVICES=6
# accelerate launch --num_processes=1 -m lmms_eval \
#   --model_args: fewshot_images="True,setup=ANSWER" \
#   --tasks glob \
#   --batch_size 16 \
#   --output_path: ./logs/ \
#   --limit 1000 \
#   --num_fewshot 2 >> output3.txt

# ) &

# wait