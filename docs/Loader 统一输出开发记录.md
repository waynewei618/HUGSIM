# Loader 统一输出开发记录

本文记录 2026-05-29 对 Waymo、PandaSet、NuScenes 三套 loader 的统一输出开发过程，并合并原数据准备流程对比、三套 loader 处理记录和 NuScenes 12Hz 标注记录。

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

## 代码结构

新增和维护的 loader 文件：

```text
loader/
├── common.py
├── nuscenes/
│   └── load.py
├── pandaset/
│   └── load.py
└── waymo/
    └── load.py
```

`loader/common.py` 负责共享约定：

- 统一相机名常量。
- Waymo/PandaSet 到 `CAM_*` 的映射。
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

## 数据集对比

| 数据集 | 当前入口 | 原始输入 | 当前相机输出 | 动态物体来源 | 地理锚点 |
|---|---|---|---|---|---|
| Waymo | `loader/waymo/load.py` | 单个 `.tfrecord` | 当前实跑 `CAM_FRONT`、`CAM_FRONT_LEFT`、`CAM_FRONT_RIGHT` | `laser_labels` | `geo_reference.available=false` |
| PandaSet | `loader/pandaset/load.py` | PandaSet sequence | 6 个 `CAM_*` | `sequence.cuboids` | 从 `meta/gps.json` 写首帧和末帧 GPS |
| NuScenes | `loader/nuscenes/load.py` | NuScenes scene/version | 6 个 `CAM_*` | sample annotations | `geo_reference.available=false` |

| 数据集 | 原始相机命名 | 统一命名 | 特殊图像处理 | 地面信息 |
|---|---|---|---|---|
| Waymo | `1/2/3/4/5` | `CAM_FRONT` 等 | 当前无额外 crop，只 downsample | 首帧 lidar 拟合，写 `ground_lidar.ply` 和 `front_info.json` |
| PandaSet | `front_camera` 等 | `CAM_FRONT` 等 | `back_camera` 裁底 250 像素 | 首帧 lidar 拟合，写 `ground_lidar.ply` 和 `front_info.json` |
| NuScenes | 已是 `CAM_*` | 保持不变 | `CAM_BACK` 裁底 80 像素 | 首帧 lidar 拟合，写 `ground_lidar.ply` 和 `front_info.json` |

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

`object_to_ego` 表示动态物体局部坐标到当前帧 ego 坐标的 4x4 位姿。要把动态物体变到相机坐标，可使用：

$$
T_{\text{object}\rightarrow\text{camera}}
= T_{\text{camera}\rightarrow\text{ego}}^{-1}
  T_{\text{object}\rightarrow\text{ego}}
$$

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

Waymo 和 NuScenes 公开数据只提供数据集原生米制位姿和粗粒度地点字符串，不提供可直接使用的 first ego 经纬度，因此写：

```json
{
  "available": false,
  "location": "location_sf"
}
```

## 三套 loader 实现

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
  loader/waymo/load.py \
  loader/pandaset/load.py \
  loader/nuscenes/load.py
```

2026-05-29 增加 `geo_reference.json`、将 frame `width/height` 移到 `camera_paras.json` 并统一最终图像内参后，上述静态检查再次通过。内参 helper 也做了最小数值检查：crop 左 10、上 20、右 30、下 40，再 downsample 2 后，`cx/cy/fx/fy/width/height` 均符合公式。

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

三套输出均包含：

```text
meta_data.json
camera_paras.json
geo_reference.json
front_info.json
ground_lidar.ply
view.mp4
```

三套输出均不包含：

```text
cam_rigid_config.json
```

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
