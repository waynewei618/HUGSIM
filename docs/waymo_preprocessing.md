# Waymo 前处理流程

本文记录 HUGSIM 中 Waymo segment 的前处理流程，包含数据输入、模型权重位置、环境注意事项、运行命令和输出结构。

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

如需处理其他 segment，优先通过环境变量覆盖路径，不需要创建数据软链接：

```bash
export WAYMO_BASE_DIR=/workspace/data/waymo
export WAYMO_SEGMENT=<your_segment>.tfrecord
export WAYMO_OUT=/workspace/HUGSIM/outputs/waymo/<seq_name>
```

## 模型权重

流程中使用到的模型权重统一放在项目的 `checkpoints/` 目录下，避免重复下载：

```text
checkpoints/
├── distance_measures_regressor.pth
├── hrnet48_OCR_HMS_IF_checkpoint.pth
└── unidepth-v2-vitl14/
    ├── config.json
    └── model.safetensors
```

对应代码默认路径：

| 用途 | 默认路径 | 可覆盖环境变量 |
|---|---|---|
| InverseForm 语义分割 | `/workspace/HUGSIM/checkpoints/hrnet48_OCR_HMS_IF_checkpoint.pth` | `INVERSEFORM_MODEL_PATH` |
| InverseForm distance regressor | `/workspace/HUGSIM/checkpoints/distance_measures_regressor.pth` | `INVERSEFORM_DISTANCE_MODEL_PATH` |
| UniDepth 深度估计 | `/workspace/HUGSIM/checkpoints/unidepth-v2-vitl14` | `UNIDEPTH_MODEL_PATH` |

下载 GitHub、HuggingFace 或 pip 包时，可优先设置国内访问配置：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export ftp_proxy="http://127.0.0.1:7890"
export HF_ENDPOINT="https://hf-mirror.com"
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple <package>
```

## xFormers

UniDepth 会使用 xFormers 的 memory efficient attention。当前环境为：

```text
torch 2.4.1+cu118
CUDA 11.8
GPU compute capability 8.6
```

PyPI 上默认解析到的 xFormers wheel 可能与 CUDA 版本不匹配。本项目已将 xFormers 作为本地子模块记录在 `external/xformers`，使用 `v0.0.28.post1` 对应提交 `d3948b5c`，并在本地针对 CUDA 11.8 编译。

安装或重建 pixi 环境前，先确认 xFormers 位于适配 tag：

```bash
git -C external/xformers describe --tags --always --dirty
git -C external/xformers rev-parse HEAD
```

预期输出包含：

```text
v0.0.28.post1
d3948b5cb9a3711032a0ef0e036e809c7b08c1e0
```

如果是手动 clone 或本地分支状态不对，安装前切到适配 tag：

```bash
git -C external/xformers fetch --tags origin
git -C external/xformers checkout v0.0.28.post1
git -C external/xformers submodule update --init --recursive
```

验证命令：

```bash
pixi run python -m xformers.info | egrep "xFormers|build.cuda_version|build.torch_version|memory_efficient_attention.cutlassF|gpu.compute|TORCH_CUDA_ARCH_LIST"
```

预期关键输出：

```text
xFormers 0.0.29+d3948b5c.d20260512
memory_efficient_attention.cutlassF-pt: available
gpu.compute_capability: 8.6
build.cuda_version: 1108
build.torch_version: 2.4.1+cu118
build.env.TORCH_CUDA_ARCH_LIST: 8.6
```

如果后续 xFormers CUDA 算子再次不可用，可以临时设置：

```bash
export HUGSIM_DISABLE_XFORMERS=1
```

这会让 `data/utils/estimate_depth.py` 禁用 UniDepth 的 xFormers 路径，回退到 PyTorch attention。

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

## 常见问题

### Waymo 数据路径不一致

不要创建软链接。优先修改或覆盖脚本路径：

```bash
export WAYMO_BASE_DIR=<waymo_data_dir>
export WAYMO_SEGMENT=<segment_filename>
export WAYMO_OUT=<output_dir>
```

### 权重重复下载

检查三个默认权重路径是否存在于 `checkpoints/`。如果要临时使用其他位置，通过对应环境变量覆盖，不建议改回在线下载。

### xFormers CUDA 算子不可用

先验证当前安装是否匹配：

```bash
pixi run python - <<'PY'
import torch, xformers
from xformers.ops import memory_efficient_attention
print(torch.__version__, torch.version.cuda, xformers.__version__)
q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.float16)
y = memory_efficient_attention(q, q, q)
torch.cuda.synchronize()
print(tuple(y.shape), y.dtype, y.device)
PY
```

如需回退：

```bash
export HUGSIM_DISABLE_XFORMERS=1
```
