# NuScenes 前处理流程

本文记录 HUGSIM 中 NuScenes 场景在 train 之前的数据准备流程。12Hz 标注生成不在本文展开，见 `docs/nuscenes_12Hz_processing.md`。

> **注意**：本文所有命令默认在 Docker 容器 `ubuntu_dev` 内执行，工作目录为 `/workspace/HUGSIM`。

## 输入数据

本次跑通的 NuScenes 根目录为：

```bash
/workspace/data/NuScenes
```

其中已包含 12Hz 标注版本：

```bash
/workspace/data/NuScenes/interp_12Hz_trainval
```

本次处理场景为：

```bash
scene-0038
```

输出目录为：

```bash
/workspace/data/HUGSIM/nusc/scene-0038
```

## 模型权重

本流程会读取以下模型权重：

| 用途 | 默认路径或本次使用路径 |
|---|---|
| InverseForm 语义分割 | `/workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth` |
| UniDepth 深度估计 | `/workspace/HUGSIM/checkpoints/unidepth-v2-vitl14` |

权重目录管理和临时路径覆盖规则见 `docs/pixi_environment_setup.md`。

## 运行流程

### 1. 提取 HUGSIM 场景数据

```bash
cd /workspace/HUGSIM/data
mkdir -p /workspace/data/HUGSIM/nusc/scene-0038

PYTHONPATH=/workspace/HUGSIM/data:$PYTHONPATH \
pixi run --manifest-path /workspace/HUGSIM/pixi.toml \
python nusc/load.py \
    --datapath /workspace/data/NuScenes \
    --version interp_12Hz_trainval \
    --seq scene-0038 \
    --out /workspace/data/HUGSIM/nusc/scene-0038 \
    --start 0 \
    --end 180 \
    --downsample 2 \
    --video
```

这一步会导出 6 个相机的图像、`meta_data.json`、相机刚体配置、前视相机高度信息、地面 lidar 点云和预览视频。

### 2. 生成 2D bbox 可视化

```bash
cd /workspace/HUGSIM/data

PYTHONPATH=/workspace/HUGSIM/data:$PYTHONPATH \
pixi run --manifest-path /workspace/HUGSIM/pixi.toml \
python utils/vis_bbox_2d.py \
    --out /workspace/data/HUGSIM/nusc/scene-0038
```

输出写入：

```bash
/workspace/data/HUGSIM/nusc/scene-0038/vis_bbox
```

### 3. 生成语义分割

`data/InverseForm/infer_nuscenes.sh` 默认读取 `/workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth`，也可通过 `INVERSEFORM_MODEL_PATH` 临时覆盖。显式逐相机运行示例：

```bash
cd /workspace/HUGSIM/data/InverseForm
out=/workspace/data/HUGSIM/nusc/scene-0038

for cam in FRONT FRONT_LEFT FRONT_RIGHT BACK_LEFT BACK_RIGHT BACK; do
  CUDA_VISIBLE_DEVICES=0 \
  pixi run --manifest-path /workspace/HUGSIM/pixi.toml \
  torchrun --nproc_per_node=1 validation.py \
      --input_dir ${out}/images/CAM_${cam} \
      --output_dir ${out}/semantics/CAM_${cam} \
      --model_path /workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth \
      --arch "ocrnet.HRNet_Mscale" \
      --hrnet_base "48" \
      --has_edge True
done
```

### 4. 生成动态 mask

```bash
cd /workspace/HUGSIM/data

PYTHONPATH=/workspace/HUGSIM/data:$PYTHONPATH \
pixi run --manifest-path /workspace/HUGSIM/pixi.toml \
python utils/create_dynamic_mask.py \
    --data_path /workspace/data/HUGSIM/nusc/scene-0038 \
    --data_type nuscenes
```

### 5. 估计深度

```bash
cd /workspace/HUGSIM/data

CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/workspace/HUGSIM/data:$PYTHONPATH \
pixi run --manifest-path /workspace/HUGSIM/pixi.toml \
python utils/estimate_depth.py \
    --out /workspace/data/HUGSIM/nusc/scene-0038 \
    --model_path /workspace/HUGSIM/checkpoints/unidepth-v2-vitl14
```

xFormers 配置、验证和临时回退方式见 `docs/pixi_environment_setup.md`。

### 6. 融合非地面点云

```bash
cd /workspace/HUGSIM/data

PYTHONPATH=/workspace/HUGSIM/data:$PYTHONPATH \
pixi run --manifest-path /workspace/HUGSIM/pixi.toml \
python utils/merge_depth_wo_ground.py \
    --out /workspace/data/HUGSIM/nusc/scene-0038 \
    --total 200000
```

输出：

```bash
/workspace/data/HUGSIM/nusc/scene-0038/points3d.ply
```

### 7. 融合地面点云

```bash
cd /workspace/HUGSIM/data

PYTHONPATH=/workspace/HUGSIM/data:$PYTHONPATH \
pixi run --manifest-path /workspace/HUGSIM/pixi.toml \
python utils/merge_depth_ground.py \
    --out /workspace/data/HUGSIM/nusc/scene-0038 \
    --total 200000 \
    --datatype nuscenes
```

输出：

```bash
/workspace/data/HUGSIM/nusc/scene-0038/ground_points3d.ply
/workspace/data/HUGSIM/nusc/scene-0038/ground_param.pkl
```

## 输出结构

```text
/workspace/data/HUGSIM/nusc/scene-0038/
├── images/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   └── CAM_BACK_RIGHT/
├── semantics/
├── masks/
├── depth/
├── vis_bbox/
├── cam_rigid_config.json
├── front_info.json
├── ground_lidar.ply
├── ground_param.pkl
├── ground_points3d.ply
├── meta_data.json
├── points3d.ply
└── view.mp4
```

本次输出检查结果：

| 项目 | 数量或结果 |
|---|---|
| `images/*.jpg` | 1080 |
| `semantics/*.npy` | 1080 |
| `masks/*.npy` | 1080 |
| `depth/*.pt` | 1080 |
| `vis_bbox/*.jpg` | 1080 |
| `meta_data.json` frames | 1080 |
| train cameras | 864 |
| test cameras | 216 |
| `points3d.ply` | 199800 points |
| `ground_points3d.ply` | 195915 points |
| `ground_param.pkl` | 180 front camera poses |

## 训练前检查

可用以下命令确认训练数据读取路径可以加载图像、语义、mask、深度、点云和地面参数：

```bash
cd /workspace/HUGSIM

CUDA_VISIBLE_DEVICES=0 pixi run python - <<'PY'
import os, pickle
from omegaconf import OmegaConf
from scene import load_cameras
from scene.dataset_readers import fetchPly

source = "/workspace/data/HUGSIM/nusc/scene-0038"
cfg = OmegaConf.merge(
    OmegaConf.load("configs/gs_base.yaml"),
    OmegaConf.load("configs/nusc.yaml"),
)
cfg.source_path = source
cfg.model.data_device = "cpu"

train_cams, test_cams, _ = load_cameras(cfg, cfg.data_type, True)
print("train_cams", len(train_cams), "test_cams", len(test_cams))
print("first_image", tuple(train_cams[0].original_image.shape), train_cams[0].image_name)
print("first_semantic", tuple(train_cams[0].semantic2d.shape))
print("first_depth", tuple(train_cams[0].depth.shape))
print("first_mask", tuple(train_cams[0].mask.shape))
print("ground_points", fetchPly(os.path.join(source, "ground_points3d.ply")).points.shape)
print("points3d", fetchPly(os.path.join(source, "points3d.ply")).points.shape)
with open(os.path.join(source, "ground_param.pkl"), "rb") as f:
    poses, height, cmds = pickle.load(f)
print("ground_param", poses.shape, float(height), len(cmds), sorted(set(cmds)))
PY
```

预期关键输出：

```text
train_cams 864 test_cams 216
first_image (3, 450, 800) CAM_FRONT_00000
first_semantic (1, 450, 800)
first_depth (450, 800)
first_mask (450, 800)
ground_points (195915, 3)
points3d (199800, 3)
ground_param (180, 4, 4) 1.4972307012023558 180 [0, 1, 2]
```

## 后续训练

前处理完成后，ground/full 重建训练命令、暂停恢复方式、checkpoint 和本次 `scene-0038` 指标见 `docs/reconstruction_training.md`。
