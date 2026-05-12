# NuScenes 2Hz → 12Hz 标注帧率提升流程

本文记录使用 [ASAP](https://github.com/JeffWang987/ASAP) 将 NuScenes 原始 2Hz 关键帧标注扩展为 12Hz 高频标注的流程。

> **注意**：本文所有命令默认在 Docker 容器 `ubuntu_dev` 内执行，工作目录为 `/workspace/HUGSIM`。

## 背景

NuScenes 相机硬件本身以 12Hz 拍摄，但官方只对其中 2Hz 的**关键帧**（key frame）提供了标注。数据集中 `samples/` 存放关键帧（约 3,400 张/相机），`sweeps/` 存放全部 12Hz 图像（约 16,000 张/相机）——图像本身无需提升帧率，非关键帧的图像文件已经存在。

问题在于非关键帧缺少两样东西：
1. **索引**：NuScenes 的 `sample_data` 表中没有非关键帧的条目
2. **标注**：非关键帧没有 3D bounding box 等标注

ASAP 解决这两个问题：
- 通过 `sample_data` 的 `next` 指针链，遍历关键帧之间的非关键帧（每两个关键帧之间约 5 帧），为它们生成 `sample_data` 条目，指向 `sweeps/` 中已有的图像
- 通过物体轨迹插值（interpolation），为非关键帧生成 3D 标注

最终输出 `interp_12Hz_trainval` 版本，图像和标注均为 12Hz。

## 环境配置

### 1. 创建 Conda 环境

ASAP 依赖 Python 3.7、PyTorch 1.9.0、CUDA 11.1 及 mmcv-full 1.4.0。创建独立环境：

```bash
conda create -n ASAP python=3.7 -y
```

### 2. 安装依赖

```bash
# 激活环境
conda activate ASAP

# nuscenes-devkit
pip install nuscenes-devkit

# PyTorch 1.9.0 + CUDA 11.1
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y

# mmcv-full 1.4.0
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

### 3. 验证环境

```bash
python -c '
import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.cuda.is_available())
import nuscenes
import mmcv; print("mmcv:", mmcv.__version__)
'
```

预期输出：

```
PyTorch: 1.9.0 CUDA: True
mmcv: 1.4.0
```

## 数据准备

### 数据路径

NuScenes 原始数据位于 `/workspace/data/NuScenes`。ASAP 的脚本默认把数据路径写在脚本内的 `data_path` 变量中，例如 `external/ASAP/scripts/ann_generator.sh` 默认值为：

```bash
data_path="./data/nuscenes"
```

运行前将脚本中的 `data_path` 改为实际数据目录：

```bash
cd /workspace/HUGSIM/external/ASAP
sed -i 's#data_path="./data/nuscenes"#data_path="/workspace/data/NuScenes"#' scripts/ann_generator.sh
```

### 数据版本

- 下载 NuScenes V1.0 trainval 完整数据集（Blobs + Meta）
- 需要 `samples`、`sweeps`、`maps` 及 `v1.0-trainval` 目录

## 生成 12Hz 标注

### 使用插值策略（推荐）

ASAP 提供两种标注生成策略：
- **interp**：基于物体轨迹线性插值（简单快速，HUGSIM 使用的默认方式）
- **advanced**：需先训练 CenterPoint 生成 20Hz LiDAR 检测结果再构建时序数据库

HUGSIM 使用 `interp` 策略即可。执行前需创建一个占位文件，因为脚本无条件加载 LiDAR 推理结果（插值策略实际不使用）：

```bash
cd /workspace/HUGSIM/external/ASAP
mkdir -p out/lidar_20Hz
echo '{}' > out/lidar_20Hz/results_nusc.json
```

运行标注生成：

```bash
conda activate ASAP
bash scripts/ann_generator.sh 12 --ann_strategy 'interp'
```

### 输出

生成结果位于数据目录下的 `interp_12Hz_trainval/`：

```
/workspace/data/NuScenes/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
└── interp_12Hz_trainval/    ← 新生成
    ├── attribute.json
    ├── calibrated_sensor.json
    ├── category.json
    ├── ego_pose.json
    ├── instance.json
    ├── log.json
    ├── map.json
    ├── sample.json
    ├── sample_annotation.json
    ├── sample_data.json
    ├── scene.json
    ├── sensor.json
    └── visibility.json
```

生成规模（以 NuScenes trainval 全集为例）：
- 约 35,000 个新 sample
- 约 1,100,000 条新 sample_annotation
- 约 420,000 条新 sample_data

## 后续处理

生成 `interp_12Hz_trainval` 后，使用 HUGSIM 主环境执行场景提取、语义分割、动态 mask、深度估计和点云融合。具体流程见 `docs/nuscenes_preprocessing.md`。
