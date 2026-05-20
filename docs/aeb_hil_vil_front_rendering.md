# AEB HIL/VIL 前视视频渲染

本文档记录当前阶段的实际项目入口：基于自车世界位姿轨迹和实际前视相机标定，从导出的 HUGSIM 3DGS 场景渲染前视视频，并可与原采集视频合成对比。

相关代码集中在：

```text
/workspace/HUGSIM/aeb_hil_vil_render/
```

## 程序拆分

当前目录保留以下单一职责程序和模块：

- `reconstruction_compare.sh`：一键生成重建效果观察用的原图/渲染左右对比视频。
- `extract_scene_inputs.py`：从训练场景中提取示例渲染输入。
- `gaussian_scene_renderer.py`：核心 3DGS 场景渲染类，加载场景权重和渲染默认参数后，根据传入的相机内参、`camera_to_world` 外参、输出分辨率和时间戳返回图像。
- `tiled_camera_renderer.py`：超广角分块渲染策略，负责局部相机切分、GPU 重投影和 guard band 羽化融合。
- `compose_compare_video.py`：根据自车轨迹和相机标定逐帧调用核心渲染类，并将原采集视频与渲染图像左右合成。

## 一键观察重建效果

输入为包含 `meta_data.json`、`cfg.yaml`、`scene.pth` 的场景路径和原始图片所在路径：

```bash
CUDA_VISIBLE_DEVICES=0 bash aeb_hil_vil_render/reconstruction_compare.sh \
  /workspace/HUGSIM/outputs/nusc/scene-0038 \
  /workspace/data/HUGSIM/nusc/scene-0038
```

第二个参数可以是原始数据场景根目录、`images` 目录，或前视相机图片目录。例如：

```text
/workspace/data/HUGSIM/nusc/scene-0038
/workspace/data/HUGSIM/nusc/scene-0038/images
/workspace/data/HUGSIM/nusc/scene-0038/images/CAM_FRONT
```

脚本会依次生成：

```text
outputs/aeb_hil_vil_render/<dataset>/<scene>/aeb_trajectory.json
outputs/aeb_hil_vil_render/<dataset>/<scene>/aeb_camera.json
outputs/aeb_hil_vil_render/<dataset>/<scene>/aeb_front_original.mp4
outputs/aeb_hil_vil_render/<dataset>/<scene>/aeb_front_compare.mp4
outputs/aeb_hil_vil_render/<dataset>/<scene>/aeb_real_front_120_rendered.mp4
outputs/aeb_hil_vil_render/<dataset>/<scene>/aeb_real_front_120_rendered.timing.csv
outputs/aeb_hil_vil_render/<dataset>/<scene>/aeb_trajectory_plots.png
```

其中 `aeb_front_compare.mp4` 左侧为原采集图片合成视频，右侧为 3DGS 渲染视频，用于快速观察重建效果。
`aeb_real_front_120_rendered.mp4` 使用 `camera_intrinsics.json` 中 `camera_id` 为 `front_120/cam1` 的实车 AEB 前视相机内参渲染，外参沿用 `aeb_camera.json` 中的 `camera_to_ego`。
对默认实车相机 `front_120/cam1`，程序先按 VTD Display XML 的畸变前 pinhole 内参渲染，再使用 `aeb_hil_vil_render/vtd_front_120/front_120_parameters.json` 指向的 `front.dat` 做查表畸变，输出与 VTD 最终 `front_120` 更一致的图像。
`aeb_real_front_120_rendered.timing.csv` 由调用脚本记录实车相机每帧耗时，包括 `render_camera()` 调用耗时、可选后处理耗时和二者合计；核心渲染类内部不维护 timing 状态。
`aeb_trajectory_plots.png` 将里程-高度和水平面 x-y 轨迹绘制在同一张图中；当前 HUGSIM 场景坐标里高度使用 scene `y`，水平面 x-y 使用 `(scene z, -scene x)`。

## 1. 从训练场景提取输入

命令只传训练完成后导出的场景目录：

```bash
pixi run python aeb_hil_vil_render/extract_scene_inputs.py \
  /workspace/HUGSIM/outputs/waymo/1680166 \
  /workspace/HUGSIM/outputs/waymo/1680166 \
  /workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166
```

输出写入第三个参数指定的目录：

```text
/workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_trajectory.json
/workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_camera.json
/workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_front_original.mp4
/workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_trajectory_plots.png
```

提取脚本默认使用 `meta_data.json` 中的 `cam_1` 作为 Waymo 示例前视相机。如果导出场景目录没有原始图片，脚本会尝试读取 `cfg.yaml` 中的 `source_path`，再从源数据目录生成 `aeb_front_original.mp4`。

注意：训练场景通常只有前视相机的世界位姿，没有单独保存真实自车坐标系和相机安装外参。示例提取出的 `aeb_trajectory.json` 使用前视相机位姿作为自车位姿，`aeb_camera.json` 中 `camera_to_ego` 为单位矩阵。实际 AEB HIL/VIL 项目必须替换为真实自车位姿和真实相机标定。

## 2. 核心 3DGS 图像渲染类

`gaussian_scene_renderer.py` 不绑定前视相机，也不绑定 AEB 轨迹。它的核心接口是：

```python
from aeb_hil_vil_render.gaussian_scene_renderer import GaussianSceneRenderer

renderer = GaussianSceneRenderer("/workspace/HUGSIM/outputs/waymo/1680166")
image = renderer.render_camera(
    intrinsics=intrinsics,
    camera_to_world=camera_to_world,
    width=960,
    height=640,
    timestamp=0.0,
)
```

其中 `image` 是 `uint8` RGB 图像。`GaussianSceneRenderer.__init__` 只加载一次 `cfg.yaml`、`scene.pth`、`dynamic_*.pth`、训练时的背景设置和渲染默认参数；`near/far` 默认沿用上游训练渲染函数的 `0.01/500.0`，调用者需要时可显式覆盖。`render_camera()` 直接接收构造底层 `scene.cameras.Camera` 所需的相机输入，外参输入约定固定为 `camera_to_world`。核心渲染类只负责渲染一张图，不维护耗时统计、不保存图片/视频、不做调用侧后处理，也不负责超广角分块合成。

## 3. AEB 轨迹和相机标定

`trajectory.json` 最小结构：

```json
{
  "frames": [
    {
      "ego_position": [0.0, 0.0, 0.0],
      "ego_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
      "mileage": 0.0,
      "timestamp": 0.0
    },
    {
      "ego_position": [0.0, 0.0, 0.5],
      "ego_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
      "mileage": 0.5,
      "timestamp": 0.1
    }
  ]
}
```

`mileage` 从首帧 `0.0` 开始，按相邻帧 `ego_position` 的 3D 欧氏距离累计，单位与 `ego_position` 一致。当前 HUGSIM 场景坐标里高度使用 `ego_position` 的第二维 `y`，水平面 x-y 使用 `(scene z, -scene x)`。

也可以用实测常见的 `roll/pitch/yaw` 表达，默认单位为弧度：

```json
{
  "angle_unit": "rad",
  "frames": [
    {
      "ego_position": [0.0, 0.0, 0.0],
      "ego_rpy": [0.0, 0.0, 0.0],
      "timestamp": 0.0
    }
  ]
}
```

自车旋转表示的是 `ego_to_world` 旋转。四元数推荐使用 `ego_quaternion_wxyz`，顺序为 `[w, x, y, z]`；如果实测数据是 `[x, y, z, w]`，使用字段 `ego_quaternion_xyzw`。`ego_rpy` 按 `roll, pitch, yaw` 读取，旋转组合为 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。

`camera.json` 最小结构：

```json
{
  "intrinsics": [[1030.0, 0.0, 480.0], [0.0, 1030.0, 320.0], [0.0, 0.0, 1.0]],
  "camera_to_ego": [[1.0, 0.0, 0.0, 1.8], [0.0, 1.0, 0.0, 1.4], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
  "width": 960,
  "height": 640,
  "fps": 10.0
}
```

`camera_to_ego` 必须是实际前视相机相对于自车坐标系的 4x4 外参矩阵；不能用默认参数、虚构安装位置或只有 3D 平移替代。

`compose_compare_video.py` 按帧计算：

```text
camera_to_world = ego_to_world @ camera_to_ego
```

然后把每帧的 `intrinsics`、`camera_to_world`、`width`、`height`、`timestamp` 传给渲染入口。`compose_compare_video.py` 默认用 `TiledCameraRenderer` 包装 `GaussianSceneRenderer`，`--render-tile-rows 1 --render-tile-cols 1` 时会转调核心整图渲染路径。

## 4. 合成对比视频

将原采集视频放左侧，逐帧渲染图像放右侧：

```bash
pixi run python aeb_hil_vil_render/compose_compare_video.py \
  /workspace/HUGSIM/outputs/waymo/1680166 \
  /workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_front_original.mp4 \
  /workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_trajectory.json \
  /workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_camera.json \
  /workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_front_compare.mp4 \
  --real-camera-output /workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_real_front_120_rendered.mp4
```

默认不分块渲染，即 `--render-tile-rows 1 --render-tile-cols 1`。当前超广角前视渲染基线使用 `4x4` 分块和 guard band 羽化融合：

```bash
--render-tile-rows 4 \
--render-tile-cols 4
```

实车相机渲染会默认写出同名 timing CSV：

```text
/workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_real_front_120_rendered.timing.csv
```

默认实车相机 ID 为 `front_120/cam1`，内参文件为 `aeb_hil_vil_render/camera_intrinsics.json`。如需切换，可传：

```bash
--real-camera-id front_120/cam1 \
--real-camera-intrinsics /workspace/HUGSIM/aeb_hil_vil_render/camera_intrinsics.json \
--real-camera-distortion-parameters /workspace/HUGSIM/aeb_hil_vil_render/vtd_front_120/front_120_parameters.json \
--real-camera-timing-output /workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/custom_timing.csv
```

`--real-camera-distortion-parameters` 可以传本项目保存的参数 JSON，也可以直接传 VTD `.dat` 查表文件。需要观察畸变前 pinhole 中间结果时，传 `--disable-real-camera-distortion`。

输出：

```text
/workspace/HUGSIM/outputs/aeb_hil_vil_render/waymo/1680166/aeb_front_compare.mp4
```
