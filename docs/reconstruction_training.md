# HUGSIM 重建训练记录

本文记录本次在 Docker 容器 `ubuntu_dev` 内跑通的 HUGSIM 场景重建训练流程、排错记录和注意事项。所有命令默认在容器内执行，工作目录为 `/workspace/HUGSIM`。

## 基本流程

重建训练分两步：

1. `train_ground.py` 先训练地面模型，生成 `ckpts/ground_chkpnt30000.pth`
2. `train.py` 再训练完整场景，读取上一步的 ground checkpoint

标准命令格式：

```bash
cd /workspace/HUGSIM

CUDA_VISIBLE_DEVICES=0 \
pixi run python -u train_ground.py \
    --data_cfg ./configs/<dataset>.yaml \
    --source_path <preprocessed_scene_path> \
    --model_path <reconstruction_output_path>

CUDA_VISIBLE_DEVICES=0 \
pixi run python -u train.py \
    --data_cfg ./configs/<dataset>.yaml \
    --source_path <preprocessed_scene_path> \
    --model_path <reconstruction_output_path>
```

`train.py` 默认要求：

```text
<model_path>/ckpts/ground_chkpnt30000.pth
```

如果该文件不存在，完整场景训练会在加载 ground checkpoint 时失败。

## 训练前检查

建议先确认容器、GPU、pixi 环境和相机加载正常：

```bash
cd /workspace/HUGSIM

nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

pixi run python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

检查场景相机是否能加载：

```bash
cd /workspace/HUGSIM

pixi run python - <<'PY'
from omegaconf import OmegaConf
from scene import load_cameras

cfg = OmegaConf.merge(
    OmegaConf.load("configs/gs_base.yaml"),
    OmegaConf.load("configs/nusc.yaml"),
)
cfg.source_path = "/workspace/data/HUGSIM/nusc/scene-0038"
cfg.model_path = "/workspace/HUGSIM/outputs/nusc/scene-0038"
train_cams, test_cams, _ = load_cameras(cfg, cfg.data_type, True)
print("train_cams", len(train_cams), "test_cams", len(test_cams), "first", train_cams[0].image_name)
PY
```

本次 NuScenes `scene-0038` 检查结果：

```text
train_cams 864 test_cams 216 first CAM_FRONT_00000
```

Waymo `1680166` 检查结果：

```text
train_cams 477 test_cams 117 first cam_1_000000
```

## 本次 NuScenes 训练

输入场景：

```bash
/workspace/data/HUGSIM/nusc/scene-0038
```

输出目录：

```bash
/workspace/HUGSIM/outputs/nusc/scene-0038
```

命令：

```bash
cd /workspace/HUGSIM

CUDA_VISIBLE_DEVICES=0 \
pixi run python -u train_ground.py \
    --data_cfg ./configs/nusc.yaml \
    --source_path /workspace/data/HUGSIM/nusc/scene-0038 \
    --model_path /workspace/HUGSIM/outputs/nusc/scene-0038

CUDA_VISIBLE_DEVICES=0 \
pixi run python -u train.py \
    --data_cfg ./configs/nusc.yaml \
    --source_path /workspace/data/HUGSIM/nusc/scene-0038 \
    --model_path /workspace/HUGSIM/outputs/nusc/scene-0038
```

完成情况：

| 阶段 | 输出 |
|---|---|
| ground 7000 | `ckpts/ground_chkpnt7000.pth` |
| ground 15000 | `ckpts/ground_chkpnt15000.pth` |
| ground 30000 | `ckpts/ground_chkpnt30000.pth` |
| full 7000 | `ckpts/chkpnt7000.pth` 和 dynamic checkpoint |
| full 15000 | `ckpts/chkpnt15000.pth` 和 dynamic checkpoint |
| full 30000 | `ckpts/chkpnt30000.pth`、dynamic checkpoint、unicycle checkpoint |

最终 full 训练指标：

| split | PSNR | SSIM | LPIPS | L1 |
|---|---:|---:|---:|---:|
| test | 25.3245 | 0.7635 | 0.2217 | 0.0357 |
| train | 26.5774 | 0.8005 | 0.2114 | 0.0301 |

## 本次 Waymo 训练

输入和输出目录相同：

```bash
/workspace/HUGSIM/outputs/waymo/1680166
```

这是前处理输出目录，同时也作为重建训练目录。

ground 训练命令：

```bash
cd /workspace/HUGSIM

CUDA_VISIBLE_DEVICES=0 \
pixi run python -u train_ground.py \
    --data_cfg ./configs/waymo.yaml \
    --source_path /workspace/HUGSIM/outputs/waymo/1680166 \
    --model_path /workspace/HUGSIM/outputs/waymo/1680166
```

full 训练命令：

```bash
cd /workspace/HUGSIM

CUDA_VISIBLE_DEVICES=0 \
pixi run python -u train.py \
    --data_cfg ./configs/waymo.yaml \
    --source_path /workspace/HUGSIM/outputs/waymo/1680166 \
    --model_path /workspace/HUGSIM/outputs/waymo/1680166
```

本次 Waymo full 训练经历了暂停和恢复，最终从 `7000` checkpoint 恢复后完成到 `30000`。

最终 full 训练指标：

| split | PSNR | SSIM | LPIPS | L1 |
|---|---:|---:|---:|---:|
| test | 24.7736 | 0.7630 | 0.2107 | 0.0305 |
| train | 26.6486 | 0.8136 | 0.1952 | 0.0248 |

完成后关键文件：

```text
/workspace/HUGSIM/outputs/waymo/1680166/ckpts/ground_chkpnt30000.pth
/workspace/HUGSIM/outputs/waymo/1680166/ckpts/chkpnt30000.pth
/workspace/HUGSIM/outputs/waymo/1680166/ckpts/dynamic_0_chkpnt30000.pth
/workspace/HUGSIM/outputs/waymo/1680166/ckpts/dynamic_1_chkpnt30000.pth
/workspace/HUGSIM/outputs/waymo/1680166/ckpts/dynamic_2_chkpnt30000.pth
/workspace/HUGSIM/outputs/waymo/1680166/results.json
```

## 暂停和恢复训练

### 查看进度

日志里 `tqdm` 会包含大量 carriage return，直接 `tail` 有时不易读。可用：

```bash
log=/workspace/HUGSIM/outputs/waymo_1680166_full_resume_from7000_20260513_051456.log

grep -ao "Training progress:[^\r]*" "$log" | tail -5
grep -nE "Traceback|Error|Exception|RuntimeError|Saving Checkpoint|Evaluating|Training complete" "$log" | tail -40
```

查看训练进程和 GPU：

```bash
pgrep -af "train.py .*waymo|pixi run python -u train.py" || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

### 暂停训练

先查进程：

```bash
pgrep -af "train.py .*waymo/1680166|pixi run python -u train.py"
```

再停止对应的 Python、pixi 和外层 shell 进程：

```bash
kill <python_pid> <pixi_pid> <shell_pid>
```

如外层 `pixi` 残留且确认主 Python 已退出，可再强制清理：

```bash
kill -9 <pixi_pid>
```

暂停不会自动保存当前 iteration。能恢复的点只包括已经落盘的 checkpoint，例如本次中途停止前日志到约 `12280/30000`，但最近可恢复 checkpoint 仍是 `7000`。

### 从 checkpoint 恢复

本次为 `train.py` 增加了 `--resume_iteration`，可从 full checkpoint 继续：

```bash
cd /workspace/HUGSIM

CUDA_VISIBLE_DEVICES=0 \
pixi run python -u train.py \
    --data_cfg ./configs/waymo.yaml \
    --source_path /workspace/HUGSIM/outputs/waymo/1680166 \
    --model_path /workspace/HUGSIM/outputs/waymo/1680166 \
    --resume_iteration 7000
```

恢复会读取：

```text
<model_path>/ckpts/chkpnt7000.pth
<model_path>/ckpts/dynamic_<id>_chkpnt7000.pth
```

注意：当前 checkpoint 只保存模型参数和 iteration，不保存 Adam optimizer 状态。恢复后模型参数从 checkpoint 继续，学习率按恢复 iteration 继续调度，但 Adam 动量会重新初始化。

## 已发现问题和处理

### 1. 系统 Python 没有项目依赖

容器内直接运行 `python` 会缺少 `torch` 等包：

```text
ModuleNotFoundError: No module named 'torch'
```

本项目训练应使用：

```bash
pixi run python ...
```

### 2. LPIPS 的 AlexNet 权重放在 `checkpoints/`

项目已将 torch hub 缓存固定到：

```text
/workspace/HUGSIM/checkpoints/torch/hub
```

AlexNet 权重应放在：

```text
/workspace/HUGSIM/checkpoints/torch/hub/checkpoints/alexnet-owt-7be5be79.pth
```

如果该文件存在，训练不会再下载到 home cache。若重新部署环境，优先把该文件放回 `checkpoints/`，不要依赖默认在线下载。

### 3. `source_path == model_path` 会触发 SameFileError

Waymo 前处理输出目录同时作为训练输出目录时，`Scene` 初始化会复制：

```text
meta_data.json
ground_param.pkl
input.ply
```

如果源和目标相同，原始代码会报：

```text
shutil.SameFileError: '<path>/meta_data.json' and '<path>/meta_data.json' are the same file
```

已处理方式：复制前判断 `src` 和 `dst` 的绝对路径，相同则跳过。

### 4. Waymo 配置缺少 `uc_opt_pos`

`configs/waymo.yaml`、`configs/pandaset.yaml`、`configs/kitti360.yaml` 默认没有 `uc_opt_pos`，原始 `train.py` 无条件读取会报：

```text
omegaconf.errors.ConfigAttributeError: Missing key uc_opt_pos
```

已处理方式：使用 `cfg.get("uc_opt_pos", False)`。NuScenes 中该字段仍按配置生效。

### 5. 恢复 checkpoint 需要初始化 densification 统计张量

从 full checkpoint 恢复模型参数后，`max_radii2D` 等 densification 相关张量需要重新初始化，否则后续 densification 逻辑可能缺少状态。

已处理方式：

- `GaussianModel.restore()` 恢复后初始化 `max_radii2D`
- `ObjModel.restore()` 恢复后初始化 `max_radii2D`
- 恢复后的 ground model 使用 `finetune=True`，保持 full 训练阶段只微调 ground feature 的行为

## 训练中的现象

### checkpoint 和验证阶段会暂停进度条

在 `7000`、`15000`、`30000` checkpoint 点，进度条会长时间停住，但进程仍在：

- 写大 checkpoint，Waymo full `chkpnt15000.pth` 和 `chkpnt30000.pth` 约 1.4GB
- 保存 dynamic checkpoint
- 跑 test/train 验证
- 保存 `point_cloud_vis`

判断是否卡死应结合：

```bash
ps -o pid,stat,etime,pcpu,pmem,cmd -p <python_pid>
nvidia-smi
grep -nE "Saving Checkpoint|Evaluating|Training complete" <log>
```

### `dist3d` 有时波动很大

训练中 `dist3d` 偶尔出现较大值，例如十几、数百甚至更高。只要 loss 没有持续 NaN、进程没有异常退出、验证指标正常更新，可以继续观察。

### 30000 结束后还会验证和保存

进度条到 `100%` 后不代表立即结束。还需要：

1. 写 `chkpnt30000.pth`
2. 写 dynamic checkpoint
3. 跑 test/train 验证
4. 保存 point cloud 可视化
5. 写 `results.json`
6. 打印 `Training complete.`

本次 Waymo 30000 结束后，日志出现：

```text
[ITER 30000] Saving Checkpoint
[ITER 30000] Evaluating test: L1 0.0305 PSNR 24.7736 SSIM 0.7630 Lpips 0.2107
[ITER 30000] Evaluating train: L1 0.0248 PSNR 26.6486 SSIM 0.8136 Lpips 0.1952
[ITER 30000] Saving Gaussians
Training complete.
```

## 后台运行模板

长时间训练建议后台运行并保存日志：

```bash
cd /workspace/HUGSIM

log=/workspace/HUGSIM/outputs/waymo_1680166_full_resume_from7000_$(date +%Y%m%d_%H%M%S).log

nohup env CUDA_VISIBLE_DEVICES=0 \
pixi run python -u train.py \
    --data_cfg ./configs/waymo.yaml \
    --source_path /workspace/HUGSIM/outputs/waymo/1680166 \
    --model_path /workspace/HUGSIM/outputs/waymo/1680166 \
    --resume_iteration 7000 \
    > "$log" 2>&1 &

echo $! > /workspace/HUGSIM/outputs/waymo_1680166_full_resume_from7000.pid
echo "$log"
```

持续监控模板：

```bash
log=/workspace/HUGSIM/outputs/waymo_1680166_full_resume_from7000_20260513_051456.log

while true; do
  if ! pgrep -f "/workspace/HUGSIM/.pixi/envs/default/bin/python -u train.py .*waymo/1680166" >/dev/null; then
    echo PROCESS_DONE
    grep -nE "Traceback|Error|Exception|RuntimeError|Saving Checkpoint|Evaluating|Training complete" "$log" | tail -100
    break
  fi
  date --iso-8601=seconds
  grep -ao "Training progress:[^\r]*" "$log" | tail -1
  grep -nE "Traceback|Error|Exception|RuntimeError|Saving Checkpoint|Evaluating|Training complete" "$log" | tail -20
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | sed -n "1p"
  sleep 60
done
```
