# hyper-parameters
model_name_or_path=/medical-data/zsxm/codes/LLaVA-Med/weights/llava_med_in_text_60k
output_dir=/medical-data/zsxm/codes/LLaVA-Med/checkpoints/quilt1m_clip
data_path=/medical-data/zsxm/codes/LLaVA-Med/data/alignment/quilt1m_train.json
image_folder=/medical-data/zsxm/public_dataset/image-caption/Quilt-1M/images
vision_tower=openai/clip-vit-large-patch14 # microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
GPUS=2
per_device_batch_size=2
accumulation_steps=16
train_epochs=1
model_max_length=2048
# 设置tune_mm_mlp_adapter为True则LM不调整，否则LM权重也会调整
tune_mm_mlp_adapter=True

############### Option 1 to run code with one single GPU ###############
# export CUDA_DEVICE_ORDER=PCI_BUS_ID
# export CUDA_VISIBLE_DEVICES=0
# python llava/train/train.py --model_name_or_path ${model_name_or_path} \
#     --data_path ${data_path} \
#     --image_folder ${image_folder} \
#     --tune_mm_mlp_adapter ${tune_mm_mlp_adapter} \
#     --output_dir ${output_dir} \
#     --vision_tower ${vision_tower} \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end True \
#     --bf16 True \
#     --num_train_epochs ${train_epochs} \
#     --per_device_train_batch_size ${per_device_batch_size} \
#     --per_device_eval_batch_size ${per_device_batch_size} \
#     --gradient_accumulation_steps ${accumulation_steps} \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 3 \
#     --learning_rate 2e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length ${model_max_length} \
#     --lazy_preprocess True \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 8 \
#     --report_to tensorboard

############### Option 2 to run code with multi-GPU ###############
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,2
torchrun --nnodes=1 --nproc_per_node=${GPUS} --master_port=25001 llava/train/train_mem.py \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --image_folder ${image_folder} \
    --tune_mm_mlp_adapter True \
    --output_dir ${output_dir} \
    --vision_tower ${vision_tower} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --num_train_epochs ${train_epochs} \
    --per_device_train_batch_size ${per_device_batch_size} \
    --per_device_eval_batch_size ${per_device_batch_size} \
    --gradient_accumulation_steps ${accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${model_max_length} \
    --lazy_preprocess True \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'


# Note: to support FSDP when pre-training the projection layer only, a special torch version is need [pip install --pre torch==2.1.0.dev20230424+cu117 torchaudio==2.1.0.dev20230424+cu117 torchvision==0.16.0.dev20230424+cu117 --index-url https://download.pytorch.org/whl/nightly/cu117].
# --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'




