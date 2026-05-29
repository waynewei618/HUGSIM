cuda=$1
out=$2
cams=$3
model_path=${INVERSEFORM_MODEL_PATH:-/workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth}

export CUDA_VISIBLE_DEVICES=$cuda

if [ ! -f "${model_path}" ]; then
    echo "InverseForm model checkpoint not found: ${model_path}" >&2
    exit 1
fi

for cam in $cams
do
    echo cam_${cam}
    torchrun --nproc_per_node=1 validation.py \
    --input_dir ${out}/images/cam_${cam} \
    --output_dir ${out}/semantics/cam_${cam} \
    --model_path ${model_path} \
    --arch "ocrnet.HRNet_Mscale" --hrnet_base "48" --has_edge True
    echo Done
done
