#!/bin/bash
set -euo pipefail

cuda=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES=$cuda

base_dir=${WAYMO_BASE_DIR:-"/workspace/data/waymo"}
segment=${WAYMO_SEGMENT:-"segment-16801666784196221098_2480_000_2500_000_with_camera_labels.tfrecord"}

seg_prefix=$(echo $segment| cut -c 9-15)
seq_name=${seg_prefix}
out=${WAYMO_OUT:-"/workspace/HUGSIM/outputs/waymo/$seq_name"}
cameras="1 2 3"

if [ ! -f "${base_dir}/${segment}" ]; then
    echo "Waymo segment not found: ${base_dir}/${segment}" >&2
    exit 1
fi

mkdir -p $out

# load images, camera pose, etc
python waymo/load.py -b ${base_dir} -c ${cameras} -o ${out} -s ${segment}

# generate semantic mask
cd InverseForm
./infer_waymo.sh ${cuda} ${out}
cd -

python utils/create_dynamic_mask.py --data_path ${out} --data_type waymo
python utils/estimate_depth.py --out ${out}
python utils/merge_depth_wo_ground.py --out ${out} --total 200000
python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype waymo
