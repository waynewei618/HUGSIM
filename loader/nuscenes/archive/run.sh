#!/bin/zsh

export PYTHONPATH="${PWD}:$PYTHONPATH"

resource_dir=${HUGSIM_RESOURCE_DIR:-"/workspace/data"}
cuda=${CUDA_VISIBLE_DEVICES:-0}
data=${NUSC_BASE_DIR:-"${resource_dir}/NuScenes"}
version=${NUSC_VERSION:-"interp_12Hz_trainval"}
start=${NUSC_START:-0}
end=${NUSC_END:-180}
out_root=${HUGSIM_OUTPUT_DIR:-"${resource_dir}/HUGSIM"}
seq_list=${NUSC_SEQ_LIST:-${NUSC_SEQ:-"scene-0038"}}

if [ ! -d "${data}/${version}" ]; then
        echo "NuScenes version not found: ${data}/${version}" >&2
        exit 1
fi

for seq in ${(z)seq_list}; do
        echo $seq
        out=${NUSC_OUT:-"${out_root}/nusc/${seq}"}

        export CUDA_VISIBLE_DEVICES=$cuda

        mkdir -p ${out}
        python nusc/load.py --datapath "${data}" --version "${version}" --seq "${seq}" --out "${out}" \
                --start "${start}" --end "${end}" --downsample 2 --video

        python utils/vis_bbox_2d.py --out "${out}"
        
        # # generate semantic mask
        # cd InverseForm
        # ./infer_nuscenes.sh ${cuda} ${out}
        # cd -

        # python utils/create_dynamic_mask.py --data_path ${out} --data_type nuscenes

        # # COLMAP sparse model
        # rm -rf ${out}/colmap_sparse*
        # rm ${out}/database.db*
        # rm -rf ${out}/prior
        # python nusc/prepare_colmap.py -i ${out}

        # echo "convert model into ply format"
        # colmap model_converter \
        #         --input_path ${out}/colmap_sparse_tri \
        #         --output_path ${out}/sparse_ba.ply \
        #         --output_type PLY

        # python colmap/update_campose.py --datapath ${out}
        # python utils/vis_bbox_2d.py --out ${out}

        # python utils/estimate_depth.py --out ${out}
        # python utils/merge_depth_wo_ground.py --out ${out} --total 200000
        # python utils/merge_depth_ground.py --out ${out} --total 200000 --datatype nuscenes
done
