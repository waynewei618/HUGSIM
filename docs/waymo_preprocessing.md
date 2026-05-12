# Waymo 前处理流程

本文记录 HUGSIM 中 Waymo segment 的前处理流程，包含数据输入、模型权重位置、运行命令和输出结构。

> **注意**：本文所有命令默认在 Docker 容器 `ubuntu_dev` 内执行，工作目录为 `/workspace/HUGSIM`。

## 输入数据

本次跑通的输入文件为：

```bash
/workspace/data/waymo/segment-16801666784196221098_2480_000_2500_000_with_camera_labels.tfrecord
```

`data/waymo/run.sh` 默认读取：

```bash
WAYMO_BASE_DIR=/workspace/data/waymo
WAYMO_SEGMENT=segment-16801666784196221098_2480_000_2500_000_with_camera_labels.tfrecord
```

脚本会根据 segment 名称截取 `1680166` 作为默认序列名，默认输出到：

```bash
/workspace/HUGSIM/outputs/waymo/1680166
```

如需处理其他 segment，可通过环境变量覆盖路径：

```bash
export WAYMO_BASE_DIR=/workspace/data/waymo
export WAYMO_SEGMENT=<your_segment>.tfrecord
export WAYMO_OUT=/workspace/HUGSIM/outputs/waymo/<seq_name>
```

## 模型权重

本流程会读取以下模型权重：

| 用途 | 默认路径 | 可覆盖环境变量 |
|---|---|---|
| InverseForm 语义分割 | `/workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth` | `INVERSEFORM_MODEL_PATH` |
| InverseForm distance regressor | `/workspace/HUGSIM/checkpoints/distance_measures_regressor.pth` | `INVERSEFORM_DISTANCE_MODEL_PATH` |
| UniDepth 深度估计 | `/workspace/HUGSIM/checkpoints/unidepth-v2-vitl14` | `UNIDEPTH_MODEL_PATH` |

## 运行流程

进入项目根目录后执行：

```bash
cd /workspace/HUGSIM/data
bash waymo/run.sh
```

等价的显式命令：

```bash
cd /workspace/HUGSIM/data
CUDA_VISIBLE_DEVICES=0 \
WAYMO_BASE_DIR=/workspace/data/waymo \
WAYMO_SEGMENT=segment-16801666784196221098_2480_000_2500_000_with_camera_labels.tfrecord \
WAYMO_OUT=/workspace/HUGSIM/outputs/waymo/1680166 \
bash waymo/run.sh
```

脚本内部步骤如下：

1. `waymo/load.py` 读取 `.tfrecord`，导出图像、相机参数、位姿、点云和 `meta_data.json`
2. `InverseForm/infer_waymo.sh` 对 `cam_1`、`cam_2`、`cam_3` 生成语义分割
3. `utils/create_dynamic_mask.py` 根据语义结果生成动态物体 mask
4. `utils/estimate_depth.py` 使用本地 UniDepth 权重估计每帧深度
5. `utils/merge_depth_wo_ground.py` 融合非地面深度点云
6. `utils/merge_depth_ground.py` 融合地面点云，生成最终前处理结果

## 输出结构

本次输出目录为：

```bash
/workspace/HUGSIM/outputs/waymo/1680166
```

主要内容：

```text
outputs/waymo/1680166/
├── images/
│   ├── cam_1/
│   ├── cam_2/
│   └── cam_3/
├── semantics/
│   ├── cam_1/
│   ├── cam_2/
│   └── cam_3/
├── masks/
│   ├── cam_1/
│   ├── cam_2/
│   └── cam_3/
├── depth/
│   ├── cam_1/
│   ├── cam_2/
│   └── cam_3/
├── front_info.json
├── ground_lidar.ply
├── ground_param.pkl
├── ground_points3d.ply
├── meta_data.json
└── points3d.ply
```

`outputs/` 与 `checkpoints/` 已在 `.gitignore` 中忽略，不随代码提交。

## 路径覆盖

```bash
export WAYMO_BASE_DIR=<waymo_data_dir>
export WAYMO_SEGMENT=<segment_filename>
export WAYMO_OUT=<output_dir>
```

这些变量分别覆盖输入目录、segment 文件名和输出目录。
