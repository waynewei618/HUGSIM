#!/bin/zsh
cuda=$1
out=$2
model_path=${INVERSEFORM_MODEL_PATH:-/workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth}
batch_size=${INVERSEFORM_BATCH_SIZE:-4}

export CUDA_VISIBLE_DEVICES=$cuda

if [ ! -f "${model_path}" ]; then
    echo "InverseForm model checkpoint not found: ${model_path}" >&2
    exit 1
fi

arr=("front" "front_left" "front_right" "left" "right" "back")
for cam in ${arr[@]}
do
    echo ${cam}
    torchrun --nproc_per_node=1 validation.py \
    --input_dir ${out}/images/${cam}_camera \
    --output_dir ${out}/semantics/${cam}_camera \
    --model_path ${model_path} \
    --arch "ocrnet.HRNet_Mscale" --hrnet_base "48" --has_edge True \
    --batch_size ${batch_size}
    echo Done
done
