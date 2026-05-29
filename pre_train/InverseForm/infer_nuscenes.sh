#!/bin/zsh
cuda=$1
out=$2
model_path=${INVERSEFORM_MODEL_PATH:-/workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth}

export CUDA_VISIBLE_DEVICES=$cuda

if [ ! -f "${model_path}" ]; then
    echo "InverseForm model checkpoint not found: ${model_path}" >&2
    exit 1
fi

arr=("FRONT" "FRONT_LEFT" "FRONT_RIGHT" "BACK_LEFT" "BACK_RIGHT" "BACK")
for cam in ${arr[@]}
do
    echo CAM_${cam}
    torchrun --nproc_per_node=1 validation.py \
    --input_dir ${out}/images/CAM_${cam} \
    --output_dir ${out}/semantics/CAM_${cam} \
    --model_path ${model_path} \
    --arch "ocrnet.HRNet_Mscale" --hrnet_base "48" --has_edge True
    echo Done
done
