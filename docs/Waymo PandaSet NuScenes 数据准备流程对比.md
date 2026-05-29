# Waymo PandaSet NuScenes 数据准备流程对比

本文记录 2026-05-28 在 `ubuntu_dev` 容器内对 Waymo、PandaSet、NuScenes 三套数据准备流程的实跑结果、数据流动和差异。

结论先写：

- 三套流程最终都能产出 HUGSIM 训练前需要的 `meta_data.json`、`images`、`semantics`、`masks`、`depth`、`points3d.ply`、`ground_points3d.ply`、`ground_param.pkl`。
- 三套流程的下游基本一致，主要差异集中在 loader：原始数据读取方式、坐标系归一化、动态物体来源、地面高度来源、相机命名和裁剪策略不同。
- PandaSet 没有在 loader 阶段写 `front_info.json`，地面合并阶段使用硬编码前视相机高度 `2.2`。
- NuScenes 的 `data/nusc/run.sh` 当前只执行 load 和 bbox 可视化，语义、mask、depth、merge 后半段仍是注释；本次是按完整链路手动补跑。
- 本次容器中 `pixi run` 会长时间无输出，因此实跑使用 `/workspace/HUGSIM/.pixi/envs/default/bin/python` 和同目录 `torchrun` 直接调用。

## 本次实跑数据

输出根目录：

```bash
/workspace/HUGSIM/outputs/dataflow_probe_20260528_083656
```

| 数据集 | 输入 | 输出目录 | 图像 | semantics | masks | depth | 点云 |
|---|---|---|---:|---:|---:|---:|---|
| Waymo | `segment-16801666784196221098_2480_000_2500_000_with_camera_labels.tfrecord` | `waymo_1680166` | 594 | 594 | 594 npy + 594 png | 594 | 完整 |
| PandaSet | `001` | `pandaset_001` | 480 | 480 | 480 npy + 480 png | 480 | 完整 |
| NuScenes | `scene-0038`, `interp_12Hz_trainval`, `0:180` | `nusc_scene-0038` | 1080 | 1080 | 1080 npy + 1080 png | 1080 | 完整 |

目录体积：

| 数据集 | 体积 |
|---|---:|
| Waymo | 5.0G |
| PandaSet | 3.1G |
| NuScenes | 4.8G |

元数据检查：

| 数据集 | `meta_data.frames` | `meta_data.verts` | 前视相机位姿数 | 前视高度 | high level command |
|---|---:|---:|---:|---:|---|
| Waymo | 594 | 3 | 198 | 2.0952592859360584 | `[2]` |
| PandaSet | 480 | 70 | 80 | 2.2 | `[2]` |
| NuScenes | 1080 | 2 | 180 | 1.4961337709869373 | `[0, 1, 2]` |

相机和帧数：

| 数据集 | 相机 | 每相机帧数 |
|---|---|---:|
| Waymo | `cam_1`, `cam_2`, `cam_3` | 198 |
| PandaSet | `front_camera`, `front_left_camera`, `front_right_camera`, `back_camera`, `left_camera`, `right_camera` | 80 |
| NuScenes | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` | 180 |

## 总体数据流

三套数据集进入 HUGSIM 后，下游数据流一致：

```text
原始数据
  -> loader
      -> images/<camera>/*
      -> meta_data.json
      -> 可选 front_info.json / ground_lidar.ply / cam_rigid_config.json / view.mp4
  -> InverseForm semantic inference
      -> semantics/<camera>/*.npy
  -> create_dynamic_mask.py
      -> masks/<camera>/*.npy
      -> masks/<camera>/*.png
  -> estimate_depth.py
      -> depth/<camera>/*.pt
  -> merge_depth_wo_ground.py
      -> points3d.ply
  -> merge_depth_ground.py
      -> ground_points3d.ply
      -> ground_param.pkl
```

核心约定是 `meta_data.json`。每个 frame 至少包含：

- `rgb_path`：相对输出目录的图像路径。
- `camtoworld`：归一化后的相机到世界矩阵。
- `intrinsics`：4x4 内参矩阵，前 3x3 用于投影和反投影。
- `width` / `height`：下采样或裁剪后的图像尺寸。
- `timestamp`：相对起始时间。
- `dynamics`：当前帧可见或有效的动态物体位姿。

`meta_data["verts"]` 保存动态物体局部 bbox 顶点。`create_dynamic_mask.py` 会读取 `verts` 和每帧 `dynamics`，把动态 bbox 投影到图像上，再和语义类别中的车、人等类别求交集，最终保存 `~mask`。这里的 mask 含义是有效静态区域，动态物体区域为 `False`。

## Waymo 数据流

入口：

```bash
cd /workspace/HUGSIM/data
PYTHONPATH=/workspace/HUGSIM/data:/workspace/HUGSIM \
/workspace/HUGSIM/.pixi/envs/default/bin/python waymo/load.py \
  -b /workspace/data/waymo \
  -c 1 2 3 \
  -o /workspace/HUGSIM/outputs/dataflow_probe_20260528_083656/waymo_1680166 \
  -s segment-16801666784196221098_2480_000_2500_000_with_camera_labels.tfrecord
```

loader 行为：

- 读取单个 Waymo `.tfrecord`。
- 从首帧 lidar 点云中截取近车区域，RANSAC 拟合地面，写 `ground_lidar.ply` 和 `front_info.json`。
- 导出 3 个相机图像：`cam_1`、`cam_2`、`cam_3`，格式为 PNG。
- 使用 `cam_1` 第 0 帧的 `camtoworld` 逆矩阵作为全局归一化原点。
- 从 `frame.laser_labels` 读取 3D bbox，保留 `VEHICLE`、`PEDESTRIAN`、`CYCLIST`，轨迹运动量大于 1 的实例进入动态物体集合。
- 写 `meta_data.json`，其中包含 `frames`、`verts` 和 `inv_pose`。

关键源码：

- `data/waymo/load.py`：首帧 lidar 地面、图像导出、动态物体过滤、`meta_data.json` 写出。
- `data/waymo/run.sh`：完整串联 load、semantic、mask、depth、merge。

本次结果：

```text
198 frames x 3 cameras = 594 images
front_info.height = 2.0952592859360584
meta_data.verts = 3
ground_param = (198, 4, 4), height 2.0952592859360584, commands [2]
```

## PandaSet 数据流

入口：

```bash
cd /workspace/HUGSIM/data
PYTHONPATH=/workspace/HUGSIM/data \
/workspace/HUGSIM/.pixi/envs/default/bin/python panda/load.py \
  --datapath /workspace/data/pandaset \
  --seq 001 \
  --out /workspace/HUGSIM/outputs/dataflow_probe_20260528_083656/pandaset_001 \
  --downsample 2 \
  --video
```

注意：PandaSet devkit 的 import 名也是 `pandaset`，项目根目录下也存在 `pandaset/` 目录。运行 `data/panda/load.py` 时如果把 `/workspace/HUGSIM` 放进 `PYTHONPATH`，会遮蔽官方 devkit，导致 `from pandaset import DataSet` 失败。本次 loader 只使用：

```bash
PYTHONPATH=/workspace/HUGSIM/data
```

loader 行为：

- 通过官方 PandaSet devkit 读取 `/workspace/data/pandaset/001`。
- 固定处理 `PANDASET_SEQ_LEN = 80` 帧。
- 导出 6 个相机图像：`front_camera`、`front_left_camera`、`front_right_camera`、`back_camera`、`left_camera`、`right_camera`，格式为 JPG。
- 使用 `front_camera` 第 0 帧 pose 的逆矩阵作为全局归一化原点。
- `back_camera` 会裁掉底部 250 像素，再按 `downsample=2` 缩放。
- 从 `sequence.cuboids` 读取动态物体，条件是 `stationary == False` 且类别属于动态类别集合。
- 每个有效 cuboid 的 pose 会同步写入同一时刻的 6 个相机 frame。
- 写 `meta_data.json` 和 `view.mp4`。

PandaSet 与 Waymo/NuScenes 最大差异是地面高度：loader 不生成 `front_info.json`，`merge_depth_ground.py --datatype pandaset` 内部直接使用：

```python
front_cam_height = 2.2
```

关键源码：

- `data/panda/load.py`：PandaSet devkit 读取、6 相机图像和 metadata 写出、动态 cuboid 过滤。
- `data/panda/utils.py`：`back_camera` 底部裁剪。
- `data/panda/run.sh`：完整串联 load、semantic、mask、depth、merge。

本次结果：

```text
80 frames x 6 cameras = 480 images
front_cam_height = 2.2
meta_data.verts = 70
ground_param = (80, 4, 4), height 2.2, commands [2]
```

## NuScenes 数据流

入口：

```bash
cd /workspace/HUGSIM/data
mkdir -p /workspace/HUGSIM/outputs/dataflow_probe_20260528_083656/nusc_scene-0038

PYTHONPATH=/workspace/HUGSIM/data \
/workspace/HUGSIM/.pixi/envs/default/bin/python nusc/load.py \
  --datapath /workspace/data/NuScenes \
  --version interp_12Hz_trainval \
  --seq scene-0038 \
  --out /workspace/HUGSIM/outputs/dataflow_probe_20260528_083656/nusc_scene-0038 \
  --start 0 \
  --end 180 \
  --downsample 2 \
  --video
```

注意：`data/nusc/load.py` 在写 `front_info.json` 和 `ground_lidar.ply` 前会使用 `--out`，但创建输出目录在后面。因此当前代码需要先手动 `mkdir -p --out`。

loader 行为：

- 读取 NuScenes devkit 数据，版本为 `interp_12Hz_trainval`。
- 从 `scene-0038` 中截取 sample `[0:180]`。
- 从首帧 lidar 点云拟合地面，写 `ground_lidar.ply` 和 `front_info.json`。
- 额外写 `cam_rigid_config.json`，供后续刚体相机 BA / COLMAP 流程使用。
- 导出 6 个相机图像：`CAM_FRONT`、`CAM_FRONT_LEFT`、`CAM_FRONT_RIGHT`、`CAM_BACK`、`CAM_BACK_LEFT`、`CAM_BACK_RIGHT`，格式为 JPG。
- 使用 `CAM_FRONT` 第 0 帧相机 pose 的逆矩阵作为全局归一化原点。
- `CAM_BACK` 会裁掉底部 80 像素，再按 `downsample=2` 缩放。
- 动态物体来自 sample annotations，同一 instance 的首尾位移阈值大于 2 时视为动态。
- 写 `meta_data.json`、`front_info.json`、`cam_rigid_config.json`、`ground_lidar.ply`、`view.mp4`。

关键源码：

- `data/nusc/load.py`：首帧 lidar 地面、rig config、图像导出、动态 instance 过滤。
- `data/nusc/utils.py`：`CAM_BACK` 底部裁剪。
- `data/nusc/run.sh`：当前只实际执行 load 和 `vis_bbox_2d.py`；semantic、mask、COLMAP、depth、merge 代码块仍为注释。

本次结果：

```text
180 frames x 6 cameras = 1080 images
front_info.height = 1.4961337709869373
front_info.rect_mat = non-null 3x3 matrix
meta_data.verts = 2
ground_param = (180, 4, 4), height 1.4961337709869373, commands [0, 1, 2]
```

## 下游模块对比

### 语义分割

三套流程都调用 `data/InverseForm/validation.py`，输出 `semantics/<camera>/*.npy`。

相机名由不同脚本或手动命令决定：

| 数据集 | semantic 输入目录 |
|---|---|
| Waymo | `images/cam_1`、`images/cam_2`、`images/cam_3` |
| PandaSet | `images/front_camera` 等 6 个 `_camera` 目录 |
| NuScenes | `images/CAM_FRONT` 等 6 个 `CAM_*` 目录 |

并发运行 `torchrun` 时必须使用不同 `--master_port`。默认端口 `29500` 会冲突。

### 动态 mask

入口统一：

```bash
python utils/create_dynamic_mask.py --data_path <out> --data_type <waymo|pandaset|nuscenes>
```

差异只在相机名集合：

| `data_type` | 相机集合 |
|---|---|
| `waymo` | `cam_1`, `cam_2`, `cam_3` |
| `pandaset` | `front_camera`, `front_left_camera`, `front_right_camera`, `back_camera`, `left_camera`, `right_camera` |
| `nuscenes` | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT` |

生成逻辑一致：

1. 读取当前 frame 的 `camtoworld`、`intrinsics`、`dynamics`。
2. 读取对应 `semantics/*.npy`。
3. 将动态 bbox 顶点从物体局部坐标变换到世界坐标，再投影到图像。
4. bbox 投影区域与语义动态类别求交。
5. 保存 `~mask`，即有效静态区域。

### 深度估计

入口统一：

```bash
python utils/estimate_depth.py --out <out>
```

数据流：

1. 遍历 `meta_data["frames"]`。
2. 按 `rgb_path` 读取图像。
3. 从 `intrinsics` 取 3x3 相机内参。
4. 调用本地 UniDepth 模型。
5. 写 `depth/<camera>/<frame>.pt`。

本次使用模型：

```bash
/workspace/HUGSIM/checkpoints/unidepth-v2-vitl14
```

运行时使用：

```bash
HUGSIM_DISABLE_XFORMERS=1
```

UniDepth 会提示 KNN / extract_patches 自定义 op 未编译，实际影响是速度，不影响本次完成。

### 非地面点云合并

入口统一：

```bash
python utils/merge_depth_wo_ground.py --out <out> --total 200000
```

数据流：

1. 读取每帧图像、深度、内参、位姿。
2. 根据深度反投影为相机坐标点云。
3. 读取动态 mask，保留静态有效区。
4. 读取语义，保留 `semantics > 1` 的非地面区域。
5. 每帧随机采样，变换到世界坐标。
6. 写 `points3d.ply`。

### 地面点云合并

入口统一：

```bash
python utils/merge_depth_ground.py --out <out> --total 200000 --datatype <waymo|pandaset|nuscenes>
```

数据流：

1. 读取每帧图像、深度、语义、内参、位姿。
2. 保留 `semantics <= 1` 的地面区域。
3. 反投影并变换到世界坐标。
4. 收集前视相机轨迹：
   - Waymo：`/cam_1/`
   - PandaSet：`/front_camera/`
   - NuScenes：`/CAM_FRONT/`
5. 确定前视相机高度：
   - Waymo：读取 `front_info.json`
   - PandaSet：硬编码 `2.2`
   - NuScenes：读取 `front_info.json`
6. 将地面点投到最近前视相机局部坐标，把局部竖直方向设为前视高度，再变回世界坐标。
7. 写 `ground_points3d.ply`。
8. 根据未来 20 帧前视相机横向位移生成 high level command，并写 `ground_param.pkl`。

high level command 编码：

| 值 | 含义 |
|---:|---|
| 0 | right |
| 1 | left |
| 2 | forward |

## 横向差异总结

| 维度 | Waymo | PandaSet | NuScenes |
|---|---|---|---|
| loader 输入 | 单 `.tfrecord` | PandaSet devkit sequence | NuScenes devkit scene |
| 本次长度 | 198 帧 | 80 帧 | 180 帧 |
| 相机数 | 3 | 6 | 6 |
| 图像格式 | PNG | JPG | JPG |
| 输出视频 | 无 | `view.mp4` | `view.mp4` |
| 地面 lidar | 有 | 无 | 有 |
| `front_info.json` | 有 | 无 | 有 |
| `cam_rigid_config.json` | 无 | 无 | 有 |
| 坐标原点 | `cam_1` 第 0 帧 | `front_camera` 第 0 帧 | `CAM_FRONT` 第 0 帧 |
| 动态物体来源 | `laser_labels` | `cuboids` | `sample annotations` |
| 动态过滤 | 类别 + 运动量 > 1 | 非 stationary + 类别 | instance 位移 > 2 |
| 后视相机裁剪 | 无 | `back_camera` 裁底 250 px | `CAM_BACK` 裁底 80 px |
| 地面高度 | lidar 平面估计 | 固定 2.2 | lidar 平面估计 |
| run.sh 完整度 | 完整链路 | 完整链路 | 后半段注释 |

## 本次踩坑和处理

1. 宿主机和容器里都没有 `rg`，调查时使用 `find`、`grep`、`sed`。
2. 项目程序按约定在 `ubuntu_dev` 容器中运行，宿主机不直接执行项目 Python。
3. 容器系统 Python 缺项目依赖，`pixi run` 在当前环境会长时间无输出，因此实跑改用 `.pixi/envs/default/bin/python`。
4. PandaSet devkit 需要安装官方包。仅验证 `import pandaset` 不够，因为项目根目录的空 `pandaset/` 目录会遮蔽官方 devkit。
5. PandaSet loader 需要避免把 `/workspace/HUGSIM` 放入 `PYTHONPATH`，否则 `from pandaset import DataSet` 会失败。
6. NuScenes loader 当前需要提前创建 `--out` 目录。
7. 并行跑 InverseForm 时，多个 `torchrun` 不能共享默认 `29500` 端口。
8. NuScenes `run.sh` 不是完整数据准备链路，本次完整结果来自显式补跑 semantic、mask、depth、merge。

## 产物检查命令

```bash
root=/workspace/HUGSIM/outputs/dataflow_probe_20260528_083656

for d in waymo_1680166 pandaset_001 nusc_scene-0038; do
  echo "=== ${d}"
  echo -n "images "; find "$root/$d/images" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l
  echo -n "semantics_npy "; find "$root/$d/semantics" -type f -name "*.npy" | wc -l
  echo -n "masks_npy "; find "$root/$d/masks" -type f -name "*.npy" | wc -l
  echo -n "masks_png "; find "$root/$d/masks" -type f -name "*.png" | wc -l
  echo -n "depth_pt "; find "$root/$d/depth" -type f -name "*.pt" | wc -l
  ls -lh "$root/$d"/meta_data.json \
         "$root/$d"/points3d.ply \
         "$root/$d"/ground_points3d.ply \
         "$root/$d"/ground_param.pkl
done
```

## 后续建议

- 把 NuScenes `run.sh` 的后半段恢复为可选完整链路，或者明确拆成 `load_only` 与 `full_prepare` 两个脚本。
- 修正 `data/nusc/load.py` 的输出目录创建顺序，在写 `front_info.json` 前执行 `os.makedirs(outdir, exist_ok=True)`。
- PandaSet 文档和脚本中明确 `PYTHONPATH` 约束，避免项目根目录遮蔽官方 devkit。
- 如果后续要同时批量处理多场景，InverseForm 的 `torchrun --master_port` 应按任务自动分配。
