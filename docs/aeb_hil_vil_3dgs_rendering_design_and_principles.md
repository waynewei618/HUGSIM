# AEB HIL/VIL 3DGS 渲染改造与原理说明

本文档面向只熟悉基础 pinhole 成像的读者，解释本次 AEB HIL/VIL 渲染代码为什么这样改、数据怎样流动、3DGS 场景和相机模型到底是什么关系，以及实车 `front_120/cam1` 超广角渲染异常的原因和处理思路。

相关代码：

```text
aeb_hil_vil_render/gaussian_scene_renderer.py
aeb_hil_vil_render/compose_compare_video.py
aeb_hil_vil_render/extract_scene_inputs.py
aeb_hil_vil_render/reconstruction_compare.sh
aeb_hil_vil_render/camera_intrinsics.json
aeb_hil_vil_render/vtd_lookup_table.py
gaussian_renderer/__init__.py
scene/cameras.py
```

## 1. 最核心的一句话

3DGS 场景本身不是某一种相机模型。

3DGS 场景主要保存的是世界坐标系里的很多 3D Gaussian：

- 位置 `xyz`
- 尺度 `scale`
- 旋转 `rotation`
- 不透明度 `opacity`
- 颜色/球谐系数 `features`
- 可选动态物体模型 `dynamic_*.pth`

相机只在“把这些 3D Gaussian 投影成一张 2D 图片”的时候参与。渲染时需要给 renderer 一个虚拟相机：

```text
相机内参 K
相机外参 camera_to_world
图片宽高 width, height
```

底层 gsplat rasterization 做的事情可以理解为：

```text
world point -> camera point -> pixel point -> splat/rasterize -> RGB image
```

所以，“3DGS 场景和相机模型不一致”这个说法不精确。更准确的说法是：

```text
3DGS 场景不绑定相机模型；
但渲染器必须用某个投影模型把场景投到图像上。
如果投影模型、外参、畸变处理、训练视角覆盖范围不合理，渲染结果会异常。
```

## 2. 从 pinhole 成像开始

你已经熟悉 pinhole，可以直接对应到当前代码。

一个世界坐标点 `X_world = [x, y, z, 1]^T`，先用外参变成相机坐标：

```text
X_camera = world_to_camera @ X_world
world_to_camera = inverse(camera_to_world)
```

再用内参矩阵 `K` 投到像素坐标：

```text
[u, v, 1]^T ~ K @ [Xc / Zc, Yc / Zc, 1]^T
```

常见 OpenCV 风格内参：

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

其中：

- `fx, fy`：焦距的像素单位表达。
- `cx, cy`：主点坐标，通常以图像左上角为原点。
- `width, height`：渲染图像分辨率。

本项目中，`scene/cameras.py` 的 `Camera` 类只保存这些关键数据：

```python
self.K = torch.from_numpy(K).float().cuda()
self.c2w = torch.from_numpy(c2w).float().cuda()
self.width = width
self.height = height
```

`gaussian_renderer/__init__.py` 里真正传给 gsplat 的是：

```python
viewmats=torch.linalg.inv(viewpoint.c2w)[None, ...]
Ks=viewpoint.K[None, :3, :3]
width=viewpoint.width
height=viewpoint.height
```

这就是 pinhole 模型在当前 3DGS 渲染代码里的落点。

## 3. 本次代码结构改造

旧设计里 `render_front_view.py` 名字和职责都绑定“前视”。但新的需求是：

```text
给什么相机内参和外参，就渲染什么相机视角。
```

所以代码改成了更通用的核心类：

```text
aeb_hil_vil_render/gaussian_scene_renderer.py
```

核心类是 `GaussianSceneRenderer`：

```python
renderer = GaussianSceneRenderer(scene_path)

image = renderer.render_image(
    intrinsics=K,
    camera_to_world=camera_to_world,
    width=3840,
    height=2160,
    timestamp=t,
)
```

这个类做了两件事：

1. `__init__` 阶段只加载一次 3DGS 权重。
2. 每一帧只接收相机参数并返回一张 `uint8 RGB` 图片。

`__init__` 加载内容包括：

- `cfg.yaml`
- `scene.pth`
- `dynamic_*.pth`
- 背景色
- HUGSIM/gsplat 的 `render` 函数

这样设计的原因是，3DGS 权重很大，不应该每帧重新加载。视频渲染时，应该复用同一个 renderer 实例，然后逐帧传入新的相机位姿。

## 4. 逐帧视频渲染的数据流

`compose_compare_video.py` 负责把轨迹、相机标定和 renderer 串起来。

输入文件有两个关键 JSON：

```text
aeb_trajectory.json
aeb_camera.json
```

`aeb_trajectory.json` 每一帧提供自车世界位姿：

```json
{
  "ego_position": [0.0, 0.0, 0.0],
  "ego_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
  "timestamp": 0.0
}
```

代码会把它转成：

```text
ego_to_world
```

`aeb_camera.json` 提供相机相对自车的安装外参：

```text
camera_to_ego
```

每一帧真正传给渲染器的相机外参是：

```text
camera_to_world = ego_to_world @ camera_to_ego
```

这条链路非常重要：

```text
相机坐标 -> 自车坐标 -> 世界坐标
camera_to_world = ego_to_world @ camera_to_ego
```

如果你手上的标定是 `ego_to_camera`，不能直接填到 `camera_to_ego`，必须先求逆：

```text
camera_to_ego = inverse(ego_to_camera)
```

外参方向错，是渲染画面严重异常的最常见原因之一。

## 5. `extract_scene_inputs.py` 的注意事项

`extract_scene_inputs.py` 是为了从已经训练好的 HUGSIM 场景里快速提取一套“能跑通”的示例输入。

但是训练场景通常只有采集相机本身的世界位姿，没有真实的：

```text
自车坐标系 ego
相机安装外参 camera_to_ego
```

所以示例提取时采用了一个近似：

```text
ego_to_world = 训练前视相机 camera_to_world
camera_to_ego = identity
```

这只能用于快速查看重建效果。实际 AEB HIL/VIL 场景必须替换为真实自车位姿和真实相机安装外参。

如果继续使用这个示例 `camera_to_ego = identity` 去渲染真实车载相机，很容易出现视角高度、朝向、平移都不对的问题。

## 6. 实车 `front_120/cam1` 渲染链路

实车 AEB 前视相机参数写在：

```text
aeb_hil_vil_render/camera_intrinsics.json
```

当前默认使用：

```text
camera_id = front_120/cam1
```

该相机的关键参数：

```text
width  = 3840
height = 2160
fov_x  = 133.25666 deg
fov_y  = 104.94207 deg
fx     ~= 829.73
fy     ~= 829.58
cx     ~= 1917.57
cy     ~= 1081.59
```

这是一个 4K 超广角相机。它不是普通窄 FOV 前视相机。

`compose_compare_video.py` 会做这些事：

1. 从 `camera_intrinsics.json` 找到 `front_120/cam1`。
2. 用它的 `camera_matrix` 替换普通示例相机内参。
3. 用它的 `width/height` 设置输出分辨率。
4. 读取 `near/far`。
5. 如果存在 VTD lookup table，就使用 lookup table 处理超广角映射。
6. 逐帧调用 `GaussianSceneRenderer` 生成实车相机视频。
7. 写出每帧耗时 CSV。

输出默认是：

```text
aeb_real_front_120_rendered.mp4
aeb_real_front_120_rendered.timing.csv
```

## 7. 为什么超广角直接渲染会异常

你看到的异常图像表现为边缘和上方严重拉伸、模糊、拖影。这个现象和 3DGS 场景“有没有相机模型”不是一回事。

核心原因是：用单个 pinhole 相机直接覆盖 `133 x 105` 度的超大 FOV，会让图像边缘的投影非常极端。

在 pinhole 模型里：

```text
u = fx * Xc / Zc + cx
v = fy * Yc / Zc + cy
```

当视线接近超广角边缘时，很多点的 `Zc` 变小，`Xc / Zc` 或 `Yc / Zc` 会快速变大。对 3DGS 来说，一个 3D Gaussian 投到 2D 后会变成一个屏幕空间椭圆 splat。超广角边缘会把这些 splat 拉得很大，产生明显的涂抹和模糊。

此外还有两个现实问题：

- 训练数据里的相机视角可能没有覆盖真实 120 度前视相机的全部视野。
- VTD 实车相机图像可能不是单纯 pinhole 图，而是经过 lookup table 畸变/校正后的图。

所以，异常不是因为“3DGS 场景绑定了某个相机模型”，而是因为：

```text
用一个普通 pinhole rasterization 直接渲染超广角 4K 输出，投影边缘数值条件很差；
同时真实相机输出还可能需要畸变查表映射。
```

## 8. 当前对超广角的处理

当前代码里有两层处理。

第一层是普通分块渲染：

```text
tile_width
tile_height
tile_overlap
max_splat_radius
```

它把大图拆成多个小块，每个小块单独渲染，再用 overlap 权重融合。这样可以降低单次 rasterization 覆盖范围太大的问题。

第二层是 ray projection lookup table 渲染，这是 `front_120/cam1` 更关键的处理。

代码位置：

```text
GaussianSceneRenderer.render_ray_map_request()
```

基本思想是：

1. VTD lookup table 给出最终图像每个像素对应的源图像采样坐标。
2. 代码把这些采样坐标反算成相机射线方向。
3. 把整张超广角图的射线场拆成多个局部 tile。
4. 每个 tile 构造一个局部 pinhole 相机。
5. 分别渲染这些局部 pinhole 图。
6. 再把 tile 渲染结果按 lookup/ray map 采样回最终 4K 图像。

这样做的直觉是：

```text
不要强迫一个 pinhole 相机一次性吃下 133 度 FOV；
而是把超广角相机拆成多个较小视场的局部 pinhole 相机。
```

这能减少超广角边缘的 2D splat 过度拉伸。

## 9. lookup table 在这里的意义

`aeb_hil_vil_render/vtd_lookup_table.py` 负责读取 VTD `.dat` 查表文件。

它会生成两个矩阵：

```text
map_x
map_y
```

OpenCV 的 `cv2.remap` 语义是：

```text
output[y, x] = input[map_y[y, x], map_x[y, x]]
```

也就是说，lookup table 描述的是“输出图像的每个像素应该从输入图像哪个位置采样”。

普通后处理模式是：

```text
先渲染一张 pinhole 图 -> 再用 lookup table remap 成 VTD 风格图
```

ray projection 模式是：

```text
先把 lookup table 对应的每个像素转成射线 -> 分块渲染这些射线 -> 拼成最终图
```

对超广角相机，后者更合理。

## 10. 动态物体和时间戳

`CameraRenderRequest` 里有两个容易忽略的字段：

```text
timestamp
dynamics
```

`timestamp` 用于当前帧时间。`dynamics` 用于每个动态物体在当前帧的世界变换。

底层 `gaussian_renderer.render()` 会把静态背景 Gaussian 和动态物体 Gaussian 合起来：

```text
all_gaussians = static_scene + transformed_dynamic_objects
```

所以如果轨迹 JSON 里有动态物体变换，每帧渲染时也会应用。

## 11. `previous_camera` 是什么

`GaussianSceneRenderer` 里有：

```python
self.previous_camera
```

它主要用于时间相关的信息，例如 optical flow 或动态物体上一帧状态。虽然当前 RGB 渲染不一定强依赖它，但视频渲染时保持上一帧相机是合理的。

切换到另一段视频或另一个相机时，需要调用：

```python
renderer.reset_temporal_state()
```

当前 `compose_compare_video.py` 在开始渲染实车相机视频前会重置它，避免普通对比视频的上一帧状态影响实车视频。

## 12. appearance affine 和 tile 渲染

HUGSIM 的 `GaussianModel` 可能启用 `affine` 外观模型。底层代码会根据相机位置和朝向预测一个颜色仿射修正：

```python
cam_xyz = affine_c2w[:3, 3]
cam_dir = affine_c2w[:3, 2]
```

普通渲染时，外观模型使用当前相机的 `c2w`。

ray tile 渲染时，每个 tile 都是人为构造出来的局部小相机。如果直接把局部 tile 相机拿去驱动 appearance affine，可能导致同一帧不同 tile 颜色不一致。

所以当前代码给 tile render 传入：

```text
appearance_camera_to_world = 原始整帧相机 camera_to_world
```

这样 tile 之间共享同一个外观条件，减少拼接时的颜色不连续。

## 13. 每帧耗时统计

`GaussianSceneRenderer` 支持精确统计实车 4K 渲染每一帧耗时：

```python
renderer.enable_timing(clear=True)
...
renderer.save_timing_csv(path)
renderer.print_timing_summary()
```

CSV 字段包括：

```text
frame_index
image_name
width
height
timestamp
camera_ms
render_ms
postprocess_ms
total_ms
```

含义：

- `camera_ms`：构造 HUGSIM `Camera` 对象的 CPU/GPU 数据准备耗时。
- `render_ms`：CUDA rasterization 渲染耗时，使用 CUDA event 统计。
- `postprocess_ms`：tensor 转 `uint8`、lookup table remap、tile stitch 等后处理耗时。
- `total_ms`：整帧端到端耗时。

如果问“4K 实车相机一张图片需要渲染多久”，应以生成的：

```text
aeb_real_front_120_rendered.timing.csv
```

为准。不同场景 Gaussian 数量、动态物体数量、GPU 型号、tile 数量、lookup table 是否命中缓存，都会影响最终耗时。

## 14. 排查渲染异常的顺序

如果画面异常，建议按这个顺序排查。

1. 先用训练相机复现对比视频。

   如果训练相机视角都渲染不对，优先查 3DGS 场景、权重加载、坐标读取。

2. 检查外参方向。

   确认传入的是 `camera_to_ego`，不是 `ego_to_camera`。最终应满足：

   ```text
   camera_to_world = ego_to_world @ camera_to_ego
   ```

3. 检查坐标轴定义。

   自车坐标系和 HUGSIM 世界坐标系必须一致。如果实车数据是右前上、前左上、东北天等，需要先转换到场景使用的坐标约定。

4. 检查旋转顺序和单位。

   当前支持 `ego_quaternion_wxyz`、`ego_quaternion_xyzw`、`ego_rpy`。RPY 默认是弧度，组合顺序为：

   ```text
   R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
   ```

5. 检查内参分辨率是否匹配。

   `K` 必须对应当前 `width/height`。如果把 1920x1080 的 `K` 用到 3840x2160，`fx/fy/cx/cy` 通常要按比例缩放。

6. 检查主点原点。

   当前使用 OpenCV 风格，像素原点在左上角。若外部系统使用左下角原点，需要转换 `cy`。

7. 检查 lookup table 文件是否存在且尺寸匹配。

   `front_120/cam1` 的 `.dat` lookup table 必须能解析成 `3840x2160` 的 `map_x/map_y`。

8. 检查训练视角覆盖。

   3DGS 只能重建训练数据看见过的内容。真实 120 度相机看到训练相机没覆盖的区域时，画面边缘或远处可能缺失、模糊或漂浮。

9. 检查 near/far。

   当前实车相机读取 VTD 的 `near=1.0` 和 `far=2500.0`。如果近处物体被裁掉或远处异常，需要结合场景尺度检查裁剪面。

## 15. 一键入口

当前一键脚本是：

```bash
CUDA_VISIBLE_DEVICES=0 bash aeb_hil_vil_render/reconstruction_compare.sh \
  /home/sil/workspace/HUGSIM/outputs/nusc/scene-0038 \
  /home/sil/workspace/HUGSIM/data-or-original-scene-path
```

脚本会先提取轨迹和相机 JSON，再调用 `compose_compare_video.py` 生成：

```text
aeb_front_compare.mp4
aeb_real_front_120_rendered.mp4
aeb_real_front_120_rendered.timing.csv
```

其中：

- `aeb_front_compare.mp4`：原始采集视频和训练相机渲染结果左右对比。
- `aeb_real_front_120_rendered.mp4`：用 `front_120/cam1` 实车 AEB 前视相机内参渲染的视频。
- `aeb_real_front_120_rendered.timing.csv`：实车相机每帧耗时。

## 16. 推荐理解方式

可以把整个系统拆成三层：

```text
第一层：3DGS 场景
世界坐标里的 Gaussian 集合，本身不关心最终用哪个相机看。

第二层：相机几何
K、camera_to_world、width、height 决定从哪里看、怎么看、图像多大。

第三层：工程适配
轨迹逐帧循环、实车相机标定、VTD lookup table、超广角分块、视频编码、耗时统计。
```

本次改造的目标就是把这三层拆清楚：

```text
gaussian_scene_renderer.py 负责第一层和第二层的渲染接口。
compose_compare_video.py 负责第三层的视频流程。
reconstruction_compare.sh 负责一键跑通当前业务入口。
```

这样后续即使相机不再是前视、输出不再是对比视频、轨迹来源换成 HIL/VIL 实车记录，也不需要重写底层 3DGS 渲染类，只需要换输入的 `K`、`camera_to_world` 和视频组织逻辑。
