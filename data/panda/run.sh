#!/bin/zsh

resource_dir=${HUGSIM_RESOURCE_DIR:-"/workspace/data"}
cuda=${CUDA_VISIBLE_DEVICES:-0}
seq=${PANDASET_SEQ:-001}
data=${PANDASET_BASE_DIR:-"${resource_dir}/pandaset"}
out_root=${HUGSIM_OUTPUT_DIR:-"${resource_dir}/HUGSIM"}
out=${PANDASET_OUT:-"${out_root}/pandaset/${seq}"}
downsample=${PANDASET_DOWNSAMPLE:-2}

export CUDA_VISIBLE_DEVICES=$cuda

if [ ! -d "${data}/${seq}" ]; then
    echo "PandaSet sequence not found: ${data}/${seq}" >&2
    exit 1
fi

mkdir -p "${out}"

python panda/load.py --datapath "${data}" --seq "${seq}" --out "${out}" --downsample "${downsample}" --video

# generate semantic mask
cd InverseForm
./infer_pandaset.sh "${cuda}" "${out}"
cd -

python utils/create_dynamic_mask.py --data_path "${out}" --data_type pandaset
python utils/estimate_depth.py --out "${out}"
python utils/merge_depth_wo_ground.py --out "${out}" --total 200000
python utils/merge_depth_ground.py --out "${out}" --total 200000 --datatype pandaset
