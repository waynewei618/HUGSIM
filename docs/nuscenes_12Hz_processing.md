# NuScenes 2Hz → 12Hz 标注帧率提升流程

本文记录使用 [ASAP](https://github.com/JeffWang987/ASAP) 将 NuScenes 原始 2Hz 关键帧标注扩展为 12Hz 高频标注的完整流程，以及后续 HUGSIM 场景数据提取的环境配置与使用方法。

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

ASAP 依赖 Python 3.7、PyTorch 1.9.0、CUDA 11.1 及 mmcv-full 1.4.0。在 Docker 容器 `ubuntu_dev` 中创建独立环境：

```bash
docker exec ubuntu_dev bash -c "conda create -n ASAP python=3.7 -y"
```

### 2. 安装依赖

```bash
# nuscenes-devkit（使用清华 PyPI 镜像）
docker exec ubuntu_dev bash -c "source activate ASAP && \
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nuscenes-devkit"

# PyTorch 1.9.0 + CUDA 11.1
docker exec -e http_proxy="http://127.0.0.1:7890" \
  -e https_proxy="http://127.0.0.1:7890" \
  ubuntu_dev bash -c "source activate ASAP && \
  conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 \
  -c pytorch -c conda-forge -y"

# mmcv-full 1.4.0
docker exec -e http_proxy="http://127.0.0.1:7890" \
  -e https_proxy="http://127.0.0.1:7890" \
  ubuntu_dev bash -c "source activate ASAP && \
  pip install mmcv-full==1.4.0 \
  -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html"
```

### 3. 验证环境

```bash
docker exec ubuntu_dev bash -c "source activate ASAP && python -c '
import torch; print(\"PyTorch:\", torch.__version__, \"CUDA:\", torch.cuda.is_available())
import nuscenes
import mmcv; print(\"mmcv:\", mmcv.__version__)
'"
```

预期输出：

```
PyTorch: 1.9.0 CUDA: True
mmcv: 1.4.0
```

## 数据准备

### 目录结构

NuScenes 原始数据位于 `/workspace/data/NuScenes`（Docker 内部路径），ASAP 脚本期望在仓库根目录下的 `data/nuscenes/` 访问数据。通过软链接连接：

```bash
docker exec ubuntu_dev bash -c "
  mkdir -p /workspace/HUGSIM/external/ASAP/data
  ln -s /workspace/data/NuScenes /workspace/HUGSIM/external/ASAP/data/nuscenes
"
```

链接后的目录结构：

```
external/ASAP/
├── data/
│   └── nuscenes/          → /workspace/data/NuScenes
│       ├── maps/
│       ├── samples/
│       ├── sweeps/
│       └── v1.0-trainval/
├── sAP3D/
├── scripts/
└── ...
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
docker exec ubuntu_dev bash -c "
  cd /workspace/HUGSIM/external/ASAP
  mkdir -p out/lidar_20Hz
  echo '{}' > out/lidar_20Hz/results_nusc.json
"
```

运行标注生成：

```bash
docker exec ubuntu_dev bash -c "
  cd /workspace/HUGSIM/external/ASAP
  source activate ASAP
  bash scripts/ann_generator.sh 12 --ann_strategy 'interp'
"
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

## 提取 HUGSIM 场景数据

标注生成后，使用 HUGSIM 的 `data/nusc/load.py` 提取单个场景。

### 依赖

`load.py` 使用 `nuscenes-devkit`，可以直接复用 ASAP 环境：

```bash
docker exec ubuntu_dev bash -c "
  cd /workspace/HUGSIM/data
  source activate ASAP
  export PYTHONPATH=\"\${PWD}:\$PYTHONPATH\"
  python nusc/load.py \
    --datapath /workspace/data/NuScenes \
    --version interp_12Hz_trainval \
    --seq scene-0038 \
    --out /path/to/output/scene-0038 \
    --start 0 \
    --end 180 \
    --downsample 2 \
    --video
"
```

参数说明：
| 参数 | 说明 |
|---|---|
| `--datapath` | NuScenes 数据根目录 |
| `--version` | 数据版本，使用 `interp_12Hz_trainval` |
| `--seq` | 场景名，如 `scene-0038` |
| `--out` | 输出目录 |
| `--start` | 起始帧（默认 0） |
| `--end` | 结束帧（-1 表示全部） |
| `--downsample` | 降采样倍数（默认 2） |
| `--video` | 同时生成预览视频 |

### 输出结构

```
<out>/
├── images/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   └── CAM_BACK_RIGHT/
├── meta_data.json
├── cam_rigid_config.json
├── front_info.json
├── ground_lidar.ply
└── view.mp4              # 若指定 --video
```

## 环境对照表

| 用途 | 环境 | Python | PyTorch | CUDA |
|---|---|---|---|---|
| ASAP 标注生成 | Conda `ASAP` | 3.7 | 1.9.0 | 11.1 |
| HUGSIM 主环境 | Pixi `hugsim-env` | 3.10+ | 2.4.1 | 11.8 |

`data/nusc/load.py` 等数据预处理脚本仅依赖 `nuscenes-devkit`，两个环境均可运行。推荐使用 `ASAP` 环境直接运行，避免环境切换。
