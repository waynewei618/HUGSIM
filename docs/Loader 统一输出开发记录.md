# Loader 统一输出开发记录

本文记录 2026-05-29 对 Waymo、PandaSet、NuScenes 三套 loader 的统一输出开发过程，并合并原数据准备流程对比、三套 loader 处理记录和 NuScenes 12Hz 标注记录。当前已新增 ME 数据集 loader；ME 需要先按 ego 轨迹重划分 sub-scene，再进入统一 loader 输出。

本文只覆盖原始数据进入 `loader/` 并生成统一中间格式的阶段；后续训练前处理记录见 `docs/数据到训练开发记录.md`。旧 `data/<dataset>/load.py` 和 `data/utils/*.py` 中已有新实现替代的入口已删除；没有新实现替代的旧辅助脚本只保留在对应 `archive/` 目录作参考。

## 目标和约束

本次 loader 输出统一为自车中心逻辑：

- 图像目录统一使用 `CAM_*` 相机名。
- ego 坐标系统一为右手系：`x` 向前、`y` 向左、`z` 向上。
- 新增 `camera_paras.json`，单独保存相机相对于自车的安装平移、旋转和相机内参。
- `meta_data.json` 每帧保存 `ego_to_world`、时间戳和动态物体，不再保存每帧 `camtoworld`、`intrinsics`、`width` 和 `height`。
- 动态物体在每帧中保存为相对当前 ego 的 `object_to_ego`。
- 新增 `geo_reference.json`，记录局部米制世界和真实地理坐标之间是否存在可用锚点。
- loader 阶段不再输出 `cam_rigid_config.json`；相机间相对位姿后续由 `camera_paras.json` 运行时缓存推导。
- 新实现只放在 `loader/<dataset>/`，不直接改旧 `data/<dataset>/load.py`。

统一后的 loader 阶段输出结构：

```text
<out>/
├── images/
│   ├── CAM_FRONT/
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   └── CAM_BACK_RIGHT/
├── camera_paras.json
├── geo_reference.json
├── front_info.json
├── ground_lidar.ply
├── meta_data.json
└── view.mp4
```

Waymo 当前只导出前三个相机，因此只会生成：

```text
images/CAM_FRONT
images/CAM_FRONT_LEFT
images/CAM_FRONT_RIGHT
```

## 当前 loader 运行流程

默认已进入 Docker 容器内的 `/workspace/HUGSIM`：

```text
原始数据
  -> loader/<dataset>/load.py
      -> images/CAM_*
      -> meta_data.json
      -> camera_paras.json
      -> geo_reference.json
      -> front_info.json
      -> ground_lidar.ply
      -> view.mp4
```

按数据集选择一个 loader 入口运行。

Waymo：

```bash
.pixi/envs/default/bin/python loader/waymo/load.py \
  -b <waymo_root> \
  -s <segment.tfrecord> \
  -c 1 2 3 \
  -o <loader_out> \
  --downsample 2
```

PandaSet：

```bash
.pixi/envs/default/bin/python loader/pandaset/load.py \
  --datapath <pandaset_root> \
  --seq <sequence_id> \
  --out <loader_out> \
  --downsample 2
```

NuScenes：

```bash
.pixi/envs/default/bin/python loader/nuscenes/load.py \
  --datapath <nuscenes_root> \
  --version <version> \
  --seq <scene_name> \
  --out <loader_out> \
  --downsample 2 \
  --start 0 \
  --end <end_frame>
```

ME 需要先重划分，再运行 loader。控制机原始数据默认路径是 `/mnt/compute-data/e2e/me`，算力机原始数据默认路径是 `/data/e2e/me`。

重划分：

```bash
.pixi/envs/default/bin/python loader/me/resplit_subscenes.py \
  -s <scene_id> \
  --data-root /mnt/compute-data/e2e/me \
  --output-root outputs/me/resplit \
  --max-frames 150 \
  --max-distance 200 \
  --overlap-frames 0 \
  --min-distance 0.05 \
  --overwrite
```

重划分输出结构：

```text
outputs/me/resplit/<scene_id>_<max_distance>m/<new_sub_scene_id>/
```

例如 `--max-distance 200` 会输出到 `outputs/me/resplit/20250317_161633_1_200m/1/`。目录名中的距离只保留米制字符串，避免路径里出现小数点。

loader：

```bash
.pixi/envs/default/bin/python loader/me/load.py \
  --datapath outputs/me/resplit \
  --seq <scene_id>_200m \
  --sub-scene <sub_scene_id> \
  --downsample 2
```

未显式传 `--out` 时，ME loader 会从输入路径中删除第一个 `resplit` 段作为默认输出路径。例如输入 `outputs/me/resplit/20250317_161633_1_200m/1` 时，默认输出为 `outputs/me/20250317_161633_1_200m/1`。

## 代码结构

新增和维护的 loader 文件：

```text
loader/
├── common.py
├── me/
│   ├── load.py
│   └── resplit_subscenes.py
├── nuscenes/
│   └── load.py
├── pandaset/
│   └── load.py
└── waymo/
    └── load.py
```

`loader/common.py` 负责共享约定：

- 统一相机名常量。
- Waymo/PandaSet/ME 到 `CAM_*` 的映射。
- 输出目录初始化，并清理旧的 `cam_rigid_config.json`。
- `front_info.json` 写出。
- `camera_paras.json` 写出。
- `geo_reference.json` 写出。
- crop/downsample 后最终图像坐标系内参计算。
- `view.mp4` 拼接布局。
- 简单的 4x4 位姿求逆工具。

`view.mp4` 使用统一布局：

```text
CAM_FRONT_LEFT  CAM_FRONT  CAM_FRONT_RIGHT
CAM_BACK_RIGHT  CAM_BACK   CAM_BACK_LEFT
```

Waymo 只有前排相机时，只拼接前排。

## 相机命名映射

Waymo 映射：

| 原始相机 | 统一相机 |
|---:|---|
| `1` | `CAM_FRONT` |
| `2` | `CAM_FRONT_LEFT` |
| `3` | `CAM_FRONT_RIGHT` |
| `4` | `CAM_BACK_LEFT` |
| `5` | `CAM_BACK_RIGHT` |

本次实跑仍只使用 Waymo `1 2 3`。

PandaSet 映射：

| 原始相机 | 统一相机 |
|---|---|
| `front_camera` | `CAM_FRONT` |
| `front_left_camera` | `CAM_FRONT_LEFT` |
| `front_right_camera` | `CAM_FRONT_RIGHT` |
| `back_camera` | `CAM_BACK` |
| `left_camera` | `CAM_BACK_LEFT` |
| `right_camera` | `CAM_BACK_RIGHT` |

NuScenes 原始相机名已经是 `CAM_*`，保持不变。

ME 映射：

| 原始相机 | 统一相机 |
|---|---|
| `CAM_FRONT_120` | `CAM_FRONT` |
| `CAM_FRONT_LEFT` | `CAM_FRONT_LEFT` |
| `CAM_FRONT_RIGHT` | `CAM_FRONT_RIGHT` |
| `CAM_BACK` | `CAM_BACK` |
| `CAM_BACK_LEFT` | `CAM_BACK_LEFT` |
| `CAM_BACK_RIGHT` | `CAM_BACK_RIGHT` |

`camera_paras.json` 的相机 key 使用统一相机名；`source_camera_name` 保留原始相机名。例如 `CAM_FRONT` 的 `source_camera_name` 是 `CAM_FRONT_120`。

## 数据集对比

| 数据集 | 当前入口 | 原始输入 | 当前相机输出 | 动态物体来源 | 地理锚点 |
|---|---|---|---|---|---|
| Waymo | `loader/waymo/load.py` | 单个 `.tfrecord` | 当前实跑 `CAM_FRONT`、`CAM_FRONT_LEFT`、`CAM_FRONT_RIGHT` | `laser_labels` | `geo_reference.available=false` |
| PandaSet | `loader/pandaset/load.py` | PandaSet sequence | 6 个 `CAM_*` | `sequence.cuboids` | 从 `meta/gps.json` 写首帧和末帧 GPS |
| NuScenes | `loader/nuscenes/load.py` | NuScenes scene/version | 6 个 `CAM_*` | sample annotations | `geo_reference.available=false` |
| ME | `loader/me/resplit_subscenes.py` + `loader/me/load.py` | resplit 后的 ME sub-scene | 6 个 `CAM_*` | `annotations_info/*.json` | `geo_reference.available=false` |

| 数据集 | 原始相机命名 | 统一命名 | 特殊图像处理 | 地面信息 |
|---|---|---|---|---|
| Waymo | `1/2/3/4/5` | `CAM_FRONT` 等 | 当前无额外 crop，只 downsample | 首帧 lidar 拟合，写 `ground_lidar.ply` 和 `front_info.json` |
| PandaSet | `front_camera` 等 | `CAM_FRONT` 等 | `back_camera` 裁底 250 像素 | 首帧 lidar 拟合，写 `ground_lidar.ply` 和 `front_info.json` |
| NuScenes | 已是 `CAM_*` | 保持不变 | `CAM_BACK` 裁底 80 像素 | 首帧 lidar 拟合，写 `ground_lidar.ply` 和 `front_info.json` |
| ME | `CAM_FRONT_120` 等 | `CAM_FRONT` 等 | `CAM_FRONT_120` 裁上下和左右边，`CAM_BACK` 裁底 56 像素 | 首帧 `AT128` lidar 拟合，写 `ground_lidar.ply` 和 `front_info.json` |

## 坐标和文件约定

### ego 坐标

统一 ego 坐标：

```text
x: 前
y: 左
z: 上
```

这是右手系。Waymo 和 NuScenes 原始 ego 坐标已经符合该约定。PandaSet 的 native lidar 局部坐标表现为 `x` 向右、`y` 向前、`z` 向上，因此 loader 在写出前显式乘固定轴变换，把 PandaSet native 坐标转成 HUGSIM ego：

$$
C_{\text{pandaset native}\rightarrow\text{hugsim ego}}
=
\begin{bmatrix}
0 & 1 & 0 & 0\\
-1 & 0 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}
$$

每个序列的归一化 world 以第 0 帧 ego 为原点。`meta_data.frames[*].ego_to_world` 表示当前帧 ego 到该归一化 world 的 4x4 位姿。

`meta_data.origin_ego_to_global` 表示第 0 帧 ego 到数据集原生 global 坐标的 4x4 位姿，可以作为高斯局部 world 和数据集 global 坐标之间的桥梁：

$$
T_{\text{global}\leftarrow\text{world}} = T_{\text{origin ego}\rightarrow\text{global}}
$$

它本身不是经纬度。真实地理锚点单独写在 `geo_reference.json`。

### camera_paras.json

`camera_paras.json` 只保存相机安装参数和内参，不保存逆矩阵、投影矩阵或相机间相对位姿。

相机坐标采用 OpenCV 约定：

```text
x: 右
y: 下
z: 前
```

每个相机条目包含：

```json
{
  "source_camera_name": "front_camera",
  "translation": [0.0, 0.0, 0.0],
  "rotation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
  "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
  "intrinsics": {
    "matrix": [[...], [...], [...], [...]],
    "fx": 0.0,
    "fy": 0.0,
    "cx": 0.0,
    "cy": 0.0,
    "width": 0,
    "height": 0
  }
}
```

其中 `translation + rotation` 的语义是 `camera_to_ego`，即相机坐标系在 ego 坐标系下的安装位姿。后续需要从 ego 坐标变到相机坐标时，在代码中缓存：

$$
T_{\text{ego}\rightarrow\text{camera}} = T_{\text{camera}\rightarrow\text{ego}}^{-1}
$$

`intrinsics` 必须对应最终写入 `images/<camera>/` 的图片坐标系。统一处理顺序是先 crop，再更新主点，再 downsample，再缩放内参：

$$
f_x' = \frac{f_x}{s},\quad
f_y' = \frac{f_y}{s},\quad
c_x' = \frac{c_x - l}{s},\quad
c_y' = \frac{c_y - t}{s}
$$

其中 $l$ 是左侧 crop 像素数，$t$ 是顶部 crop 像素数，$s$ 是 downsample。`camera_paras.json` 中的 `width/height` 也是最终图片尺寸。

### meta_data.json

统一后，每个 frame 至少包含以下字段：

```json
{
  "rgb_path": "./images/CAM_FRONT/000000.jpg",
  "camera_name": "CAM_FRONT",
  "source_camera_name": "front_camera",
  "ego_to_world": [[...], [...], [...], [...]],
  "timestamp": 0.0,
  "dynamics": {
    "track_id": {
      "object_to_ego": [[...], [...], [...], [...]]
    }
  }
}
```

`meta_data.json` 不再保存每帧 `width/height`。下游需要宽高或内参时，通过 `camera_name` 到 `camera_paras.json.cameras` 中读取最终图片尺寸和最终内参。

当前 crop 配置：

- PandaSet `back_camera`：沿用旧逻辑裁掉底部 250 像素后再 resize。
- NuScenes `CAM_BACK`：沿用旧逻辑裁掉底部 80 像素后再 resize。
- Waymo 当前没有额外 crop，只做 downsample。
- ME `CAM_FRONT_120`：裁上 360 像素、裁下 520 像素，并在 downsample 后等效裁左右各 480 像素；代码中先把左右 crop 换算回原图像素，再统一调用 `build_final_intrinsic` 和 `crop_and_downsample_image`。
- ME `CAM_BACK`：裁掉底部 56 像素。

`object_to_ego` 表示动态物体局部坐标到当前帧 ego 坐标的 4x4 位姿。要把动态物体变到相机坐标，可使用：

$$
T_{\text{object}\rightarrow\text{camera}}
= T_{\text{camera}\rightarrow\text{ego}}^{-1}
  T_{\text{object}\rightarrow\text{ego}}
$$

ME 的动态物体候选类别为 `MotorVehicle`、`Pedestrian`、`TwoWheels` 和 `Tricycle`。判定是否动态时，先把标注中心从当前 ego 转到数据集 global 坐标，再按 XY 平面最大位移是否超过 1m 过滤；不能直接使用 ego 坐标下的中心位移，否则自车运动会把静态路边目标误判为动态。ME 标注中的 `PC_3D[9:12]` 速度字段在当前测试段全为 0，不作为动态判定依据。

`meta_data.json` 不再保存每帧 `intrinsics`、`camtoworld`、`width` 和 `height`。需要相机 world 位姿时，由下式推导：

$$
T_{\text{camera}\rightarrow\text{world}}
= T_{\text{ego}\rightarrow\text{world}}
  T_{\text{camera}\rightarrow\text{ego}}
$$

### geo_reference.json

`geo_reference.json` 只负责记录真实地理锚点是否可用，不参与相机投影和训练坐标计算。

统一字段：

```json
{
  "local_world": "first_ego_pose",
  "global_frame": "dataset_native_global",
  "dataset": "pandaset",
  "available": true
}
```

PandaSet 本地原始数据有 `meta/gps.json`，因此写出第 0 帧 GPS，并额外写出当前导出范围最后一帧 GPS，便于后续场景拼接：

```json
{
  "available": true,
  "first": {
    "frame_index": 0,
    "timestamp": 1557539924.49981,
    "latitude": 37.77466814570412,
    "longitude": -122.40106707219165,
    "height": 2.938702280237465,
    "xvel": 0.017196079484729808,
    "yvel": 7.733904794845615
  },
  "last": {
    "frame_index": 79,
    "timestamp": 1557539932.400371,
    "latitude": 37.77522169392809,
    "longitude": -122.40041067111173,
    "height": 2.695573289464273,
    "xvel": 0.014876800469027484,
    "yvel": 12.918896373525008
  }
}
```

Waymo、NuScenes 和 ME 只提供数据集原生米制位姿或粗粒度地点信息，不提供可直接使用的 first ego 经纬度，因此写：

```json
{
  "available": false,
  "location": "location_sf"
}
```

## loader 实现

### Waymo

入口：

```bash
loader/waymo/load.py
```

主要处理：

- `.tfrecord` 中的相机 `1/2/3` 输出为 `CAM_FRONT/CAM_FRONT_LEFT/CAM_FRONT_RIGHT`。
- Waymo 相机外参乘 `OPENGL_TO_WAYMO` 后写入 `camera_paras.json`，得到 OpenCV camera 到 ego 的 `camera_to_ego`。
- 每帧使用图像包里的 ego pose 生成 `ego_to_world`。
- 动态物体由 Waymo `laser_labels` 得到，保存为当前 ego 下的 `object_to_ego`。
- 写 `geo_reference.json`，`available=false`，保留 `context.stats.location` 和 segment name。
- 首帧 lidar 地面拟合后写 `ground_lidar.ply` 和 `front_info.json`。
- 输出 `view.mp4`，不输出 `cam_rigid_config.json`。

运行示例：

```bash
cd /workspace/HUGSIM
.pixi/envs/default/bin/python loader/waymo/load.py \
  -b /workspace/data/waymo \
  -s segment-16801666784196221098_2480_000_2500_000_with_camera_labels.tfrecord \
  -c 1 2 3 \
  -o /workspace/HUGSIM/outputs/loader_ego_probe_20260529_015116/waymo_1680166
```

### PandaSet

入口：

```bash
loader/pandaset/load.py
```

主要处理：

- 6 个 PandaSet 原始相机全部映射为 `CAM_*`。
- 以 `sequence.lidar.poses[i]` 作为每帧 native ego pose，再经 PandaSet native 到 HUGSIM ego 的固定轴变换生成 `ego_to_world`。
- 每个相机使用首帧 `camera.pose` 相对首帧 lidar ego 的 native 位姿，经固定轴变换后写入 `camera_paras.json`。
- 保留 `back_camera` 裁掉底部 250 像素的旧行为。
- 写 `geo_reference.json`，从 `meta/gps.json` 读取第 0 帧和导出末帧的 `lat/long/height/xvel/yvel`。
- 动态物体由 `sequence.cuboids` 得到，并转成当前 HUGSIM ego 下的 `object_to_ego`。
- 通过 PandaSet devkit 读取首帧 lidar，拟合地面并写 `ground_lidar.ply` 和 `front_info.json`。

PandaSet 有一个特殊问题：仓库根目录存在空的 `pandaset/` 目录，会遮蔽官方 PandaSet devkit。新 loader 在导入官方 `pandaset.DataSet` 时会临时移除项目根路径，避免误导入本仓库的空目录。

运行示例：

```bash
cd /workspace/HUGSIM
.pixi/envs/default/bin/python loader/pandaset/load.py \
  --datapath /workspace/data/pandaset \
  --seq 001 \
  --out /workspace/HUGSIM/outputs/loader_ego_probe_20260529_015116/pandaset_001
```

### ME

ME 当前分两步处理：先在 `loader/me/resplit_subscenes.py` 中根据 ego pose 重划分 sub-scene，再由 `loader/me/load.py` 把某一个重划分后的 sub-scene 转成统一 loader 输出。

入口：

```bash
loader/me/resplit_subscenes.py
loader/me/load.py
```

重划分主要处理：

- 默认原始数据根目录为 `/mnt/compute-data/e2e/me`；算力机对应路径为 `/data/e2e/me`。
- 默认输出根目录为 `outputs/me/resplit`。
- 新 scene 目录命名为 `<scene_id>_<max_distance>m`，例如 `20250317_161633_1_200m`。
- 子场景目录结构为 `outputs/me/resplit/<scene_id>_<max_distance>m/<new_sub_scene_id>/`。
- 默认 `--max-frames 150`、`--max-distance 200`、`--overlap-frames 0`、`--min-distance 0.05`。
- 只复制源 `meta.json`、`annotations_info/*.json`、相机图像和 `AT128` lidar 文件，不修改标注、图像和点云内容。
- 额外生成 `ego_trajectory.png` 用于检查轨迹；不生成 `ego_poses.json` 或 `odometer.json`。

loader 主要处理：

- `--datapath` 是 resplit 根目录，默认 `outputs/me/resplit`。
- `--seq` 是 resplit scene 目录名，例如 `20250317_161633_1_200m`。
- `--sub-scene` 是数字 sub-scene id，例如 `1`。
- `--out` 可选；未传时默认删除输入路径中的 `resplit` 段，例如 `outputs/me/resplit/20250317_161633_1_200m/1` 输出为 `outputs/me/20250317_161633_1_200m/1`。
- `CAM_FRONT_120` 输出为统一相机 `CAM_FRONT`，但 `camera_paras.json` 和 `meta_data.json` 中保留 `source_camera_name=CAM_FRONT_120`。
- 相机内参复用 `loader/common.py` 的 `build_final_intrinsic`，图像 crop/downsample 复用 `crop_and_downsample_image`。
- `meta_data.frames[*].ego_to_world` 使用首帧 ego pose 作为局部 world 原点，由 `meta.json` 中的 `pose[0].matrix4` 推导。
- 动态物体输出为当前 ego 下的 `object_to_ego`，box 顶点采用统一约定：`x=length`、`y=width`、`z=height`，以 box 中心为原点。
- 动态 track 使用 global XY 位移超过 1m 判定，避免把静态路边目标因 ego 相对运动误判为动态。
- 写 `geo_reference.json`，`available=false`，说明 ME 只提供米制 ego pose，没有精确经纬度锚点。
- 首帧 `AT128` lidar 地面拟合后写 `ground_lidar.ply` 和 `front_info.json`。
- 始终生成 `view.mp4`，不提供 `--no_video`。

当前默认测试场景：

```bash
scene_id=20250317_161633_1
seq=20250317_161633_1_200m
```

重划分示例：

```bash
cd /workspace/HUGSIM
.pixi/envs/default/bin/python loader/me/resplit_subscenes.py \
  -s 20250317_161633_1 \
  --data-root /mnt/compute-data/e2e/me \
  --output-root outputs/me/resplit \
  --max-frames 150 \
  --max-distance 200 \
  --overlap-frames 0 \
  --min-distance 0.05 \
  --overwrite
```

loader 示例：

```bash
cd /workspace/HUGSIM
.pixi/envs/default/bin/python loader/me/load.py \
  --datapath outputs/me/resplit \
  --seq 20250317_161633_1_200m \
  --sub-scene 1
```

### NuScenes

NuScenes 相机硬件为 12Hz，但官方关键帧标注为 2Hz。若要使用 12Hz 标注版本，先按本文后面的 ASAP 小节生成 `interp_12Hz_trainval`。

入口：

```bash
loader/nuscenes/load.py
```

主要处理：

- NuScenes 原始相机名已经是 `CAM_*`，目录名保持不变。
- 优先使用 `nusc.utils` 工具函数；本地没有该包时回退到仓库内 NuScenes 归档工具函数。
- 每个相机的 calibrated sensor pose 直接作为 `camera_to_ego` 写入 `camera_paras.json`。
- 每个相机 sample_data 的 ego pose 转为 `ego_to_world`。
- 动态物体从 annotation box world pose 转为当前 ego 下的 `object_to_ego`。
- 保留 `CAM_BACK` 裁掉底部 80 像素的旧行为。
- 写 `geo_reference.json`，`available=false`，保留 `log.json` 中的 `location/logfile/vehicle/date_captured`。
- 继续写 `front_info.json`、`ground_lidar.ply`、`view.mp4`，不输出 `cam_rigid_config.json`。

运行示例：

```bash
cd /workspace/HUGSIM
.pixi/envs/default/bin/python loader/nuscenes/load.py \
  --datapath /workspace/data/NuScenes \
  --version interp_12Hz_trainval \
  --seq scene-0038 \
  --out /workspace/HUGSIM/outputs/loader_ego_probe_20260529_015116/nusc_scene-0038 \
  --start 0 \
  --end 180
```

## NuScenes 12Hz 标注

ASAP 只用于在 NuScenes loader 之前生成高频标注版本。它不属于 HUGSIM 主 pixi 环境。

背景：

- NuScenes `samples/` 是 2Hz 关键帧，`sweeps/` 已包含 12Hz 图像。
- 缺的是非关键帧的 `sample_data` 索引和 3D 标注。
- ASAP 通过 `sample_data.next` 链和物体轨迹插值生成 `interp_12Hz_trainval`。

ASAP 环境使用独立 Conda：

```bash
conda create -n ASAP python=3.7 -y
conda activate ASAP
pip install nuscenes-devkit
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

生成前把 ASAP 脚本中的 NuScenes 路径改为本机路径：

```bash
cd /workspace/HUGSIM/external/ASAP
sed -i 's#data_path="./data/nuscenes"#data_path="/workspace/data/NuScenes"#' scripts/ann_generator.sh
```

HUGSIM 当前使用插值策略：

```bash
cd /workspace/HUGSIM/external/ASAP
mkdir -p out/lidar_20Hz
echo '{}' > out/lidar_20Hz/results_nusc.json

conda activate ASAP
bash scripts/ann_generator.sh 12 --ann_strategy 'interp'
```

输出目录：

```text
/workspace/data/NuScenes/interp_12Hz_trainval/
├── sample.json
├── sample_annotation.json
├── sample_data.json
└── ...
```

生成 `interp_12Hz_trainval` 后，回到 HUGSIM 主环境运行 `loader/nuscenes/load.py --version interp_12Hz_trainval`。

## 验证记录

静态检查：

```bash
cd /workspace/HUGSIM
.pixi/envs/default/bin/python -m py_compile \
  loader/common.py \
  loader/me/resplit_subscenes.py \
  loader/me/load.py \
  loader/waymo/load.py \
  loader/pandaset/load.py \
  loader/nuscenes/load.py
```

2026-05-29 增加 `geo_reference.json`、将 frame `width/height` 移到 `camera_paras.json` 并统一最终图像内参后，上述静态检查再次通过。内参 helper 也做了最小数值检查：crop 左 10、上 20、右 30、下 40，再 downsample 2 后，`cx/cy/fx/fy/width/height` 均符合公式。

2026-06-05 新增 ME resplit 和 loader 后，`loader/me/load.py` 已通过 `py_compile`，并用 `20250317_161633_1` 场景完成 sub-scene 1/2 的 loader 实跑。

schema 调整后的探针输出根目录：

```bash
/workspace/HUGSIM/outputs/loader_intrinsic_probe_20260529
```

探针检查结果：

| 数据集 | 输出帧数 | 相机数 | `meta_data.frames[*]` 宽高字段 | `camera_paras` 尺寸和图片一致 | `geo_reference.json` |
|---|---:|---:|---|---|---|
| Waymo | 594 | 3 | 无 | 是 | 有 |
| PandaSet | 480 | 6 | 无 | 是 | 有 |
| NuScenes | 6 | 6 | 无 | 是 | 有 |

其中 NuScenes 探针只跑 `--start 0 --end 1`，用于快速覆盖 6 相机和 `CAM_BACK` 裁剪逻辑。

实跑输出根目录：

```bash
/workspace/HUGSIM/outputs/loader_ego_probe_20260529_015116
```

产物统计：

| 数据集 | 输出目录 | 图像数 | 相机目录 | camera_paras 相机数 |
|---|---|---:|---|---:|
| Waymo | `waymo_1680166` | 594 | `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT` | 3 |
| PandaSet | `pandaset_001` | 480 | 6 个 `CAM_*` | 6 |
| NuScenes | `nusc_scene-0038` | 1080 | 6 个 `CAM_*` | 6 |

字段检查结果：

| 数据集 | `meta_data.frames` | `meta_data.verts` | 缺字段 | 禁止字段 | 路径错误 | 动态字段错误 |
|---|---:|---:|---:|---:|---:|---:|
| Waymo | 594 | 3 | 0 | 0 | 0 | 0 |
| PandaSet | 480 | 70 | 0 | 0 | 0 | 0 |
| NuScenes | 1080 | 2 | 0 | 0 | 0 | 0 |

当前统一 loader 输出均包含：

```text
meta_data.json
camera_paras.json
geo_reference.json
front_info.json
ground_lidar.ply
view.mp4
```

当前统一 loader 输出均不包含：

```text
cam_rigid_config.json
```

ME 当前测试记录：

| 阶段 | 路径 | 文件数 | 大小 | 备注 |
|---|---|---:|---:|---|
| resplit | `outputs/me/resplit/20250317_161633_1_200m/1` | 769 | 955M | 只复制源标注、图像、lidar，额外生成 `ego_trajectory.png` |
| resplit | `outputs/me/resplit/20250317_161633_1_200m/2` | 897 | 1.1G | 只复制源标注、图像、lidar，额外生成 `ego_trajectory.png` |
| loader | `outputs/me/20250317_161633_1_200m/1` | 642 | 590M | 已同步到算力机同名路径 |
| loader | `outputs/me/20250317_161633_1_200m/2` | 750 | 699M | 已同步到算力机同名路径 |

ME loader 输出字段检查：

| sub-scene | `meta_data.frames` | 图像帧 | 相机目录 | `meta_data.verts` | `CAM_FRONT.source_camera_name` | `CAM_FRONT` 尺寸 | `view.mp4` |
|---:|---:|---:|---|---:|---|---|---|
| 1 | 576 | 96 x 6 | 6 个 `CAM_*` | 28 | `CAM_FRONT_120` | 960 x 640 | 有 |
| 2 | 672 | 112 x 6 | 6 个 `CAM_*` | 125 | `CAM_FRONT_120` | 960 x 640 | 有 |

ME 动态物体一致性检查：

- 所有 `frames[*].dynamics` 中的 object id 都能在 `meta_data.verts` 中找到对应顶点。
- 所有 `object_to_ego` 都是有限的 4x4 矩阵。
- 所有 `verts` 都是有限的 8x3 顶点。
- sub-scene 1 中，使用 ego 坐标直接判动会得到 36 个 track；改为 global XY 位移后得到 28 个 track，去掉了自车运动导致的静态目标误判。

动态物体投影抽查：

| 数据集 | 投影动态框数 | 至少一个顶点可见 |
|---|---:|---:|
| Waymo | 696 | 255 |
| PandaSet | 22818 | 5026 |
| NuScenes | 1482 | 281 |

投影检查使用：

$$
T_{\text{object}\rightarrow\text{camera}}
= T_{\text{camera}\rightarrow\text{ego}}^{-1}
  T_{\text{object}\rightarrow\text{ego}}
$$

并使用 `camera_paras.json` 中的内参投影到图像平面。

检查 loader 输出：

```bash
out=<loader_out>

find "$out/images" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l
ls -lh "$out"/meta_data.json "$out"/camera_paras.json "$out"/geo_reference.json
ls -lh "$out"/front_info.json "$out"/ground_lidar.ply "$out"/view.mp4
test ! -e "$out/cam_rigid_config.json"
```

## 当前边界

本文边界停在 loader 输出。下游训练前处理、训练和离线仿真准备见 `docs/数据到训练开发记录.md` 和 `docs/HUGSIM 程序运行 Pipeline.md`。旧 `data/` 代码按替代关系清理：

- `data/<dataset>/load.py` 和 `data/utils/*.py` 中已有新实现替代的入口已删除。
- 未被新实现替代的 COLMAP、fisheye、统计和运行脚本移动到对应 `archive/` 目录作为参考。

下游以 `camera_paras.json` 和 `meta_data.frames[*].camera_name` 为准，不再从每帧读取 `intrinsics`、`camtoworld`、`width` 或 `height`。

旧输出格式不再作为新 loader 的兼容目标。如果已有旧产物需要使用统一链路，应重新运行新 loader。

以下旧文档内容已经合并到本文，并以统一 loader schema 修正：

- `docs/Waymo 前处理流程.md`
- `docs/PandaSet 数据预处理.md`
- `docs/NuScenes 前处理流程.md`
- `docs/NuScenes 2Hz → 12Hz 标注帧率提升流程.md`
- `docs/Waymo PandaSet NuScenes 数据准备流程对比.md`
- `docs/数据准备流程对比.md`

旧重建训练记录已删除。重建和离线仿真准备入口见 `docs/数据到训练开发记录.md` 和 `docs/HUGSIM 程序运行 Pipeline.md`。
