# AEB 场景中插入静止非原生车辆

本文说明如何通过 `gui/app.py` 在 HUGSIM 3DGS 场景中配置一辆非原生 3DRealCar 车辆，并说明车辆高度如何贴到地面。

这里的“非原生车辆”指不来自当前重建场景 `dynamic_*.pth` 的车辆，而是从 3DRealCar 车辆库中选一辆插入到 3DGS 场景里。车辆库路径固定使用：

```bash
export PATH_3DRealCar=/data/realcar3d
```

车辆模型已经转换完成，GUI 直接使用：

```text
/data/realcar3d/converted/*.ply
/data/realcar3d/converted/*.splat
```

不需要再运行 `eval_render/convert_vehicles.py`。

## 适用范围

当前 `gui/app.py` 生成的是闭环仿真使用的 scenario yaml，后续由 `closed_loop.py` / `HUGSimEnv` 读取。它不是 `render_3dgs/reconstruction_compare/compose_compare_video.py` 的输入。

当前 AEB 前视离线渲染已经支持直接插入一辆静态非原生车辆。`GaussianSceneRenderer` 初始化时会把 `aeb_trajectory.json` 加载为 `self.ego_trajectory`，根据指定里程 `s` 从轨迹中插值得到车辆插入位置和朝向，再交给 `render_3dgs/core/static_vehicle_insertion.py` 加载 `/data/realcar3d/<vehicle_id>/gs.pth`，并把它作为一个固定 `body_to_world` 动态对象合入每帧渲染。

`compose_compare_video.py` 仍然读取：

- `${scene_path}/scene.pth`
- `${scene_path}/dynamic_*.pth`
- `trajectory.json` 每帧里的 `dynamics`

它不读取 GUI 导出的 `config.yaml`。如果要在 AEB 前视离线渲染中直接插入静止非原生车，使用：

```bash
pixi run python render_3dgs/reconstruction_compare/compose_compare_video.py \
  "${scene_path}" \
  "${output_path}/aeb_front_original.mp4" \
  "${output_path}/aeb_trajectory.json" \
  "${output_path}/aeb_camera.json" \
  "${output_path}/aeb_front_compare.mp4" \
  --real-camera-output "${output_path}/aeb_real_front_120_rendered.mp4" \
  --insert-static-vehicle-id "2024_07_02_14_38_58" \
  --insert-static-vehicle-s 35.0
```

`--insert-static-vehicle-s` 是 `trajectory.json` 中从 0 开始累计的 `mileage`。不传时默认使用轨迹最大里程，也就是把车放在轨迹终点。`--insert-static-vehicle-height` 默认为 `-0.3`，含义和 GUI 导出的 `plan_list` 第三项一致，用于在估计地面高度上微调车身贴地效果。`--realcar-path` 不传时使用 `$PATH_3DRealCar`，再退回 `/data/realcar3d`。

代码职责划分如下：

- `GaussianSceneRenderer` 保存 `ego_trajectory`，负责按 `s` 从轨迹中取插入位置、按 `ground_param.pkl` 计算贴地高度，并生成 `body_to_world`。
- `create_static_vehicle_insertion(vehicle_path, body_to_world, sh_degree)` 只负责加载非原生车辆模型，并把给定的插入位姿包装成渲染需要的静态 `dynamics`。
- `compose_compare_video.py` 只负责从命令行接收车辆 ID、`s` 和高度偏置，并把 `trajectory_path` 传给渲染器。

用 GUI 插入非原生车辆的标准闭环链路仍然是：

```text
gui/app.py 放车 -> 导出 scenario yaml -> closed_loop.py 闭环仿真渲染
```

## 启动 GUI

进入 Docker 容器后执行：

```bash
cd gui
python app.py \
  --scene ${export_path} \
  --car_folder ${PATH_3DRealCar}/converted
```

`--scene ${export_path}` 需要包含：

```text
${export_path}/vis/scene.splat
${export_path}/vis/semantic.ply
${export_path}/ground_param.pkl
```

`--car_folder` 指向转换后的 `.splat` 车辆目录。这里的 `.splat` 只用于浏览器预览和下拉框枚举；闭环仿真真正加载的是：

```text
${PATH_3DRealCar}/${vehicle_id}/gs.pth
${PATH_3DRealCar}/${vehicle_id}/wlh.json
```

所以闭环仿真的 `configs/sim/<dataset>_base.yaml` 里的 `realcar_path` 也应指向 `/data/realcar3d`。

## GUI 实际做了什么

`gui/app.py` 的后端逻辑很薄：

- 从 `${export_path}/vis/scene.splat` 提供右侧 3DGS 预览场景。
- 从 `${export_path}/vis/semantic.ply` 提供左侧语义点云。
- 枚举 `${PATH_3DRealCar}/converted/*.splat`，作为车辆外观下拉框。
- 提供 `/car/<vehicle>.splat`，让前端加载选中的车辆预览模型。
- 提供 `/get_height`，按 `ground_param.pkl` 估算给定 `(x, z)` 的地面高度。

前端的放车流程在 `gui/static/js/point_cloud.js` 和 `gui/static/js/main.js`：

1. 在左侧视图点击第一个点，确定车辆的水平位置 `(x, z)`。
2. 鼠标移动或再次点击第二个点，确定车辆朝向。
3. 点击 `Insert the Vehicle`，浏览器读取选中的车辆 `.splat`，按朝向旋转后平移到 `(x, y, z)` 预览。
4. 点击 `Save Configurations`，导出 `config.yaml`。

注意：点击位置不是直接取语义点云上的点。`find_intersection()` 使用鼠标射线和水平平面 `y = 0` 求交，只用这个交点的 `x`、`z` 作为车辆水平位置。车辆实际高度由地面模型另算。

## 导出的 scenario yaml

GUI 导出的每辆车写入 `plan_list`，格式是：

```yaml
plan_list:
- - x
  - z
  - height
  - yaw
  - speed
  - "vehicle_id"
  - ConstantPlanner
  - {}
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `x` | 车辆在 scene 世界坐标中的 `x` |
| `z` | 车辆在 scene 世界坐标中的 `z` |
| `height` | 相对地面高度偏置，不是绝对世界 `y` |
| `yaw` | 写入 planner 的朝向角 |
| `speed` | 初始速度；静止车应设为 `0` |
| `vehicle_id` | 3DRealCar 车辆目录名 |
| `ConstantPlanner` | 行为控制器 |
| `{}` | 控制器参数 |

静止前车建议手动确认 `speed: 0`。GUI 速度滑条默认值是 `2`，`ConstantPlanner` 会按速度继续更新车辆位置；不改成 `0` 就不是静止车。

示例：

```yaml
mode: easy_00
plan_list:
- - 18.50
  - 42.00
  - -0.30
  - 0.0
  - 0.0
  - "2024_04_22_10_35_34"
  - ConstantPlanner
  - {}
load_HD_map: false
start_euler:
- 0.0
- 0.0
- 0.0
start_ab:
- 0.0
- 0.0
start_velo: 1
start_steer: 0
scene_name: <scene_name>
iteration: 30000
```

## 车辆如何贴到地面

不要把点击点的 `y`、`ego_position[1]` 或相机高度当作车辆高度。HUGSIM 插入车辆时使用的是：

```text
vehicle_world_y = ground_height(x, z) + height
```

其中 `height` 就是 scenario yaml 中 `plan_list` 的第三项。GUI 当前保存时固定写入：

```text
height = -0.3
```

闭环仿真中，`sim/utils/plan.py` 的 `planner.ground_height(u, v)` 会读取 `ground_param.pkl`：

1. 在前视相机轨迹中找到离 `(u, v)` 最近的相机位姿。
2. 把世界点 `[u, 0, v]` 变到最近相机坐标系。
3. 将局部坐标的 `y` 置为 `0`，表示落在该相机局部地面平面上。
4. 再变回世界坐标，得到该位置的地面世界高度。
5. 加上 `ground_param.pkl` 中保存的 `cam_height`，作为 planner 使用的地面高度基准。

然后 planner 生成车辆 `b2w`：

```python
b2w[:3, 3] = np.array([x, ground_height(x, z) + height, z])
```

AEB 前视离线渲染中的直接插入逻辑也使用同一类 `b2w`：先按 `--insert-static-vehicle-s` 在自车轨迹里插值得到 `(x, z)`，用轨迹水平切线确定车辆朝向，再按 `ground_param.pkl` 估计地面高度并加上 `--insert-static-vehicle-height`。默认 `s` 是轨迹最大里程，默认 `height` 是 `-0.3`。

所以贴地调节只需要改 `height`：

- 车辆悬空：把 `height` 调小，例如从 `-0.3` 改成 `-0.4`。
- 车辆陷入地面：把 `height` 调大，例如从 `-0.3` 改成 `-0.2`。
- 建议按 `0.05` 到 `0.10` 米一步微调。

默认 `-0.3` 是当前 GUI 和示例 scenario 使用的起点，通常比直接填 `0` 更接近 3DRealCar 模型的落地效果。

## GUI 预览高度和仿真高度的区别

GUI 预览里，`main.js` 插入车辆 `.splat` 时会调用：

```text
car_y = /get_height(x, z).y + 1.4
```

这里的 `+1.4` 只是浏览器中把车辆 `.splat` 大致抬到地面上的预览偏置。它不会写入导出的 `config.yaml`。

导出的 `config.yaml` 只保存 `height = -0.3`。闭环仿真真正使用的是 `sim/utils/plan.py` 里的 `ground_height(x, z) + height`。因此：

- GUI 预览高度只用于确认车辆大概在路面附近。
- 最终贴地效果以闭环仿真渲染为准。
- 不要把 GUI 预览里的 `+1.4` 改写到 scenario yaml 的 `height` 字段。

另外，GUI 的 `update` 按钮主要用于更新 yaw 预览；当前代码中更新预览时没有重新使用完整的 `ground_height(x, z)` 偏置。判断是否贴地时，不要只看 GUI 预览，应该看闭环仿真输出。

## 朝向如何保存

前端用两次点击的方向向量计算预览 yaw：

```text
yaw_preview = -atan2(dir.z, dir.x)
```

导出 yaml 时会转换为：

```text
yaw_saved = -yaw_preview - pi / 2
```

planner 渲染车辆时再使用：

```text
R = Rotation.from_euler("y", [-yaw_saved - pi / 2 - rectify_angle])
```

在没有 HD map `rectify_angle` 修正时，最终车辆朝向会和 GUI 预览方向一致。通常不需要手算这个角度，直接用 GUI 第二次点击或 yaw 滑条调方向即可。

## 放置建议

- 车辆尽量放在自车轨迹附近的可行驶路面上；`ground_height` 使用最近前视相机位姿估算地面，离轨迹太远时误差会变大。
- 静止 AEB 前车使用 `ConstantPlanner`，并把 `speed` 改为 `0`。
- 如果只想换车辆外观，改 `vehicle_id` 即可；ID 必须对应 `/data/realcar3d/<vehicle_id>/gs.pth`。
- 如果车辆高度不对，优先改 `plan_list` 第三项 `height`，不要改 `x`、`z` 或创建软链接。
- `ground_param.pkl` 必须和当前 export scene 对应；换场景或换 export 目录后要重新确认。

## 检查点

运行前确认：

```text
${export_path}/vis/scene.splat
${export_path}/vis/semantic.ply
${export_path}/ground_param.pkl
/data/realcar3d/converted/<vehicle_id>.splat
/data/realcar3d/<vehicle_id>/gs.pth
/data/realcar3d/<vehicle_id>/wlh.json
```

闭环仿真配置中确认：

```text
configs/sim/<dataset>_base.yaml: realcar_path: /data/realcar3d
scenario yaml: plan_list 第三项 height 从 -0.3 开始微调
scenario yaml: 静止车 speed 为 0
```
