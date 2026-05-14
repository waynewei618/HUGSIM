# PandaSet 数据预处理

本文记录 PandaSet 数据在训练前的预处理流程。命令默认在 Docker 容器内的 `/workspace/HUGSIM` 执行。

## 输入

- 原始数据集路径：`/workspace/data/pandaset`
- 本次跑通的场景：`001`
- 输出路径：`/workspace/HUGSIM/outputs/pandaset/001`
- InverseForm 语义分割权重：`/workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth`
- UniDepth 权重目录：`/workspace/HUGSIM/checkpoints/unidepth-v2-vitl14`

运行前确认 PandaSet devkit 可导入：

```bash
pixi run python - <<'PY'
from pandaset import DataSet
print("pandaset import ok")
PY
```

如果不可导入，按 `docs/pixi_environment_setup.md` 中的 PandaSet devkit 说明先补依赖。

## 运行步骤

### 1. 导出图像和元数据

```bash
pixi run python data/panda/load.py \
  --datapath /workspace/data/pandaset \
  --seq 001 \
  --out /workspace/HUGSIM/outputs/pandaset/001 \
  --downsample 2 \
  --video
```

该步骤会写出：

- `images/<camera>/*.jpg`
- `meta_data.json`
- `view.mp4`

### 2. 生成语义分割

使用一张空闲 GPU 时：

```bash
cd /workspace/HUGSIM/data/InverseForm
INVERSEFORM_MODEL_PATH=/workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth \
INVERSEFORM_BATCH_SIZE=4 \
pixi run ./infer_pandaset.sh 0 /workspace/HUGSIM/outputs/pandaset/001
```

两张 GPU 都空闲时，可以把相机拆成两组并行跑。两个 `torchrun` 需要使用不同端口：

```bash
cd /workspace/HUGSIM/data/InverseForm

for cam in front front_left front_right; do
  CUDA_VISIBLE_DEVICES=0 pixi run torchrun --master_port 29500 --nproc_per_node=1 validation.py \
    --input_dir /workspace/HUGSIM/outputs/pandaset/001/images/${cam}_camera \
    --output_dir /workspace/HUGSIM/outputs/pandaset/001/semantics/${cam}_camera \
    --model_path /workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth \
    --arch ocrnet.HRNet_Mscale --hrnet_base 48 --has_edge True --batch_size 6
done
```

```bash
cd /workspace/HUGSIM/data/InverseForm

for cam in left right back; do
  CUDA_VISIBLE_DEVICES=1 pixi run torchrun --master_port 29501 --nproc_per_node=1 validation.py \
    --input_dir /workspace/HUGSIM/outputs/pandaset/001/images/${cam}_camera \
    --output_dir /workspace/HUGSIM/outputs/pandaset/001/semantics/${cam}_camera \
    --model_path /workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth \
    --arch ocrnet.HRNet_Mscale --hrnet_base 48 --has_edge True --batch_size 6
done
```

输出为：

- `semantics/<camera>/*.npy`

默认只保存 `.npy` 语义结果。若需要额外保存可视化图和叠加图，设置：

```bash
export INVERSEFORM_SAVE_DEBUG=1
```

### 3. 生成动态 mask

```bash
cd /workspace/HUGSIM
pixi run python data/utils/create_dynamic_mask.py \
  --data_path /workspace/HUGSIM/outputs/pandaset/001 \
  --data_type pandaset
```

输出为：

- `masks/<camera>/*.npy`
- `masks/<camera>/*.png`

### 4. 估计单目深度

运行前检查空闲 GPU，并用 `CUDA_VISIBLE_DEVICES` 显式指定。示例使用 GPU 0：

```bash
cd /workspace/HUGSIM
CUDA_VISIBLE_DEVICES=0 HUGSIM_DISABLE_XFORMERS=1 pixi run python data/utils/estimate_depth.py \
  --out /workspace/HUGSIM/outputs/pandaset/001
```

输出为：

- `depth/<camera>/*.pt`

### 5. 合并点云

```bash
cd /workspace/HUGSIM
pixi run python data/utils/merge_depth_wo_ground.py \
  --out /workspace/HUGSIM/outputs/pandaset/001 \
  --total 200000
```

```bash
cd /workspace/HUGSIM
pixi run python data/utils/merge_depth_ground.py \
  --out /workspace/HUGSIM/outputs/pandaset/001 \
  --total 200000 \
  --datatype pandaset
```

输出为：

- `points3d.ply`
- `ground_points3d.ply`

## 输出检查

场景 `001` 预期每个相机各 80 帧：

```bash
out=/workspace/HUGSIM/outputs/pandaset/001

for root in images semantics masks depth; do
  echo "${root}"
  find "${out}/${root}" -mindepth 1 -maxdepth 1 -type d | sort | while read cam_dir; do
    cam=$(basename "${cam_dir}")
    case "${root}" in
      images) pattern="*.jpg" ;;
      semantics|masks) pattern="*.npy" ;;
      depth) pattern="*.pt" ;;
    esac
    printf "  %s " "${cam}"
    find "${cam_dir}" -maxdepth 1 -name "${pattern}" | wc -l
  done
done

ls -lh "${out}"/meta_data.json "${out}"/view.mp4 \
  "${out}"/points3d.ply "${out}"/ground_points3d.ply
```

本次 `001` 场景跑通后，输出检查结果为：

- `images`、`semantics`、`masks`、`depth` 下 6 个相机目录均为 80 帧。
- `meta_data.json`、`view.mp4`、`points3d.ply`、`ground_points3d.ply` 均已生成。
