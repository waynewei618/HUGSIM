# TiledCameraRenderer 分块实现模块

本文档专门解析 `render_3dgs/core/tiled_camera_renderer.py`。该模块负责把一个目标 pinhole 相机拆成多个局部小 FOV tile 相机渲染，再把这些局部渲染结果重投影回目标图像坐标并融合。

它解决的是 3DGS 在超广角 pinhole 边缘处容易出现的屏幕空间 splat 过度拉伸问题。最终输出仍按调用方传入的目标相机内参、`camera_to_world` 和分辨率定义；分块只改变中间 rasterization 的局部相机，不改变最终图像像素对应的目标相机光线。

## 模块定位

`TiledCameraRenderer` 是 `GaussianSceneRenderer` 内部使用的分块实现类。业务调用侧只使用统一入口：

```python
image = renderer.render_camera(
    intrinsics=intrinsics,
    camera_to_world=camera_to_world,
    width=3840,
    height=2160,
    timestamp=timestamp,
    dynamics=dynamics,
    tile_rows=2,
    tile_cols=2,
)
```

当 `tile_rows == 1` 且 `tile_cols == 1` 时，`GaussianSceneRenderer` 直接调用 `camera_view_render.py` 渲染整图：

```python
CameraViewRender.render_camera(...)
```

当 `tile_rows` 或 `tile_cols` 大于 `1` 时，`GaussianSceneRenderer` 复用同一个 `CameraViewRender` 实例创建 `TiledCameraRenderer`，再进入 `render_tiled_camera()` 分块路径。`camera_view_render.py` 的初始化只发生一次。

当前默认常量：

| 常量 | 默认值 | 含义 |
| --- | ---: | --- |
| `DEFAULT_TILE_GUARD_PIXELS` | `64` | 每个 tile 在目标图像坐标中向外扩展的 guard band 像素数 |
| `DEFAULT_TILE_RENDER_BATCH_SIZE` | `4` | 单次 gsplat batched rasterization 最多渲染的 tile 相机数量 |

## 总体流程

分块路径的主流程在 `render_tiled_camera()` 和 `_render_camera_tiled()`：

1. 校验目标相机内参、外参、宽高、near/far 和分块数量。
2. 创建一个完整目标相机 `reference_viewpoint`，用于帧结束后恢复 `camera_view_render.previous_camera`。
3. 根据目标相机和分块参数构造所有 tile render task。
4. 按 tile 渲染图尺寸分组；同尺寸 tile 才能放进同一个 batch。
5. 每个 batch 调用 `CameraViewRender.render_viewpoint_batch()` 一次性渲染多个局部 tile 相机。
6. 对每个 tile 结果用 `torch.nn.functional.grid_sample()` 重采样回目标图像的 guard 区域。
7. 用 feather 权重累加到 `output_accum`，同时累加 `weight_accum`。
8. 最终输出 `output_accum / weight_accum`，并用 `render_to_uint8()` 转成 `uint8` RGB 图像。

输出尺寸永远是调用方传入的 `height x width`，不是 tile 渲染图尺寸。

## Tile 区域划分

`tile_ranges(length, parts)` 用 `np.linspace(0, length, parts + 1)` 把目标图像按行列切成非空区间：

```text
[x0, x1) x [y0, y1)
```

每个 tile 有两个区域：

| 区域 | 作用 |
| --- | --- |
| core | tile 在最终目标图像中负责的主区域 |
| guard | core 向四周扩展 `guard_pixels` 后的区域，用于覆盖 splat 半径、重投影插值和融合过渡 |

代码中对应：

```python
guard_x0 = max(0, tile_x0 - self.guard_pixels)
guard_y0 = max(0, tile_y0 - self.guard_pixels)
guard_x1 = min(width, tile_x1 + self.guard_pixels)
guard_y1 = min(height, tile_y1 + self.guard_pixels)
```

tile 实际渲染图尺寸为：

```text
tile_width  = guard_x1 - guard_x0
tile_height = guard_y1 - guard_y0
```

也就是说，每个局部 tile 相机渲染的是 guard 区域对应的视场，不只是 core 区域。

## 局部 Tile 相机旋转

目标相机内参记为 $K_{\mathrm{real}}$。对每个 tile，先取 core 区域中心像素：

$$
p_c =
\begin{bmatrix}
(x_0 + x_1) / 2 \\
(y_0 + y_1) / 2 \\
1
\end{bmatrix}
$$

再用目标相机内参反投影出中心光线：

$$
d_c = K_{\mathrm{real}}^{-1} p_c
$$

代码入口是：

```python
center_ray = ray_from_pixel(k_real_inv, center_u, center_v)
tile_to_real_rotation = make_local_tile_rotation(center_ray)
```

`make_local_tile_rotation()` 构造一个从局部 tile 相机坐标系到目标相机坐标系的旋转矩阵：

- 局部 tile 相机的 `z` 轴对准中心光线 `center_ray`。
- 优先用目标相机的 `x` 轴投影到垂直于 `z` 轴的平面上，作为局部 `x` 轴。
- 如果中心光线接近目标相机 `x` 轴导致退化，则改用目标相机 `y` 轴叉乘构造。
- 局部 `y` 轴由 `np.cross(z_axis, x_axis)` 得到。

返回矩阵按列堆叠：

```python
np.stack([x_axis, y_axis, z_axis], axis=1)
```

因此 `tile_to_real_rotation` 满足：

```text
d_real = tile_to_real_rotation @ d_tile
```

局部 tile 相机与目标相机共享相机中心，只改变朝向。对应外参为：

```python
tile_camera_to_world = camera_to_world @ make_transform(tile_to_real_rotation)
```

## Tile 内参构造

局部 tile 相机必须完整覆盖 guard 区域对应的目标相机光线。`make_tile_intrinsics()` 使用 guard 区域的 8 个采样点估计局部 FOV：

- 左上、右上、左下、右下 4 个角点。
- 上边中心、下边中心、左边中心、右边中心 4 个边中心点。

采样点先由目标相机反投影为 `real_rays`，再旋转到局部 tile 相机坐标：

$$
d_{\mathrm{tile}}
= R_{\mathrm{tile}\rightarrow\mathrm{real}}^{\mathsf{T}}
K_{\mathrm{real}}^{-1}p
$$

然后计算归一化像平面坐标：

$$
x = d_{\mathrm{tile},x} / d_{\mathrm{tile},z}, \quad
y = d_{\mathrm{tile},y} / d_{\mathrm{tile},z}
$$

取这些采样点的 `min_x/max_x/min_y/max_y`，构造让该范围映射到 tile 渲染图边界的 pinhole 内参：

$$
f_x = \frac{W_{\mathrm{tile}} - 1}{x_{\max} - x_{\min}}, \quad
f_y = \frac{H_{\mathrm{tile}} - 1}{y_{\max} - y_{\min}}
$$

$$
c_x = -f_x x_{\min}, \quad
c_y = -f_y y_{\min}
$$

如果采样点中有局部 tile 相机后方的光线，或 FOV 退化到接近 0，函数会报错。

## 重投影合成

局部 tile 图不能直接贴回目标图像，因为它的光轴已经变成 tile 中心光线。合成时以最终目标图像像素为准，把目标像素反查到 tile 图像：

$$
p_{\mathrm{tile}}
\sim
K_{\mathrm{tile}}
R_{\mathrm{tile}\rightarrow\mathrm{real}}^{\mathsf{T}}
K_{\mathrm{real}}^{-1}
p_{\mathrm{real}}
$$

代码中对应：

```python
real_to_tile_homography = tile_intrinsics[:3, :3] @ tile_to_real_rotation.T @ k_real_inv
tile_pixels = real_to_tile_homography @ pixel_homogeneous
tile_pixels[:2] /= tile_pixels[2:3]
```

因为 tile 相机和目标相机共享相机中心，二者只差一个旋转，所以这里是一个纯旋转相机之间的 2D homography，不需要深度。

`_tile_composite_tensors()` 会把 guard 区域内每个目标像素对应的 tile 像素坐标转换成 `grid_sample()` 需要的 `[-1, 1]` 归一化坐标：

```python
grid_x = tile_u * (2.0 / (tile_width - 1)) - 1.0
grid_y = tile_v * (2.0 / (tile_height - 1)) - 1.0
```

合成时使用：

```python
sampled_batch = F.grid_sample(
    render_batch,
    sample_grids,
    mode="bilinear",
    padding_mode="border",
    align_corners=True,
)
```

`padding_mode="border"` 表示如果少量采样点落在 tile 渲染图外，会取边界像素。正常情况下，tile 内参和 guard 覆盖应让 core 区域采样落在有效范围内。

## Feather 融合

`tile_feather_weights()` 生成每个 tile guard 区域的融合权重。core 区域权重为 `1`；进入 guard band 后，权重按 smoothstep 平滑降到 `0`。

单轴权重由 `feather_axis_weights()` 计算：

```python
smoothstep(t) = t * t * (3.0 - 2.0 * t)
```

二维权重为横纵两个方向相乘：

```python
weight = wx * wy
```

合成阶段不会直接覆盖像素，而是做加权累加：

```python
output_accum[..., guard_y0:guard_y1, guard_x0:guard_x1] += sampled_tile * weights
weight_accum[..., guard_y0:guard_y1, guard_x0:guard_x1] += weights
```

最后统一归一化：

```python
output = output_accum / torch.clamp(weight_accum, min=1e-6)
```

这样 tile 重叠区域会自然加权平均，避免硬接缝。

## Batched Rasterization

`TiledCameraRenderer` 不直接使用底层 `gaussian_renderer.call_rasterization()`。分块后每个局部 tile 相机交给 `camera_view_render.py` 中的 `CameraViewRender.render_viewpoint_batch()` 渲染，保证整图渲染和 tile 渲染共用同一套场景张量、动态物体和 appearance affine 逻辑。

`CameraViewRender.render_viewpoint_batch()` 要求同一个 batch 内所有 tile 相机宽高一致：

```python
if any(viewpoint.width != width or viewpoint.height != height for viewpoint in viewpoints):
    raise ValueError("batched viewpoints must have identical image dimensions")
```

因此 `_render_camera_tiled()` 会先按 `(viewpoint.width, viewpoint.height)` 分组，再按 `render_batch_size` 切 batch：

```python
tasks_by_size[(viewpoint.width, viewpoint.height)].append(task)
```

每个 batch 会一次性传入多个相机：

```python
viewmats = torch.stack([torch.linalg.inv(viewpoint.c2w) for viewpoint in viewpoints], dim=0)
intrinsics = torch.stack([viewpoint.K[:3, :3] for viewpoint in viewpoints], dim=0)
```

rasterization 只渲染 RGB：

```python
render_mode="RGB"
```

合成网格和 feather 权重也都提前转成 GPU tensor，合成阶段在 GPU 上完成。整帧结束时才通过 `render_to_uint8()` 把结果转成 CPU 可写视频的 `uint8` 图像。

## 动态物体和 Appearance Affine

`CameraViewRender._scene_tensors_for_viewpoint()` 负责准备当前帧使用的背景和动态前景 Gaussian：

- 背景来自 `camera_view_render.gaussians`。
- 动态物体来自 `viewpoint.dynamics`。
- 每个动态物体用当前帧的 `body_to_world` 变换位置和旋转。
- 背景和前景通过 `cat_bgfg()` 拼成一次 rasterization 输入。

分块 batch 内的所有 tile 属于同一目标帧，因此使用同一份动态物体状态。

如果训练出的 Gaussian 模型启用了 `gaussians.affine`，`CameraViewRender._apply_affine_batch()` 会按每个 tile 相机的相机位置和朝向单独计算 appearance affine，并对该 tile 渲染结果做颜色修正。这里没有共用完整目标相机的 appearance，因为每个 tile 的局部视角不同。

## 缓存内容

`_tile_static_tasks()` 会缓存与帧无关的 tile 几何和合成张量。缓存 key 包含：

```python
(width, height, tile_rows, tile_cols, guard_pixels, k_real.tobytes())
```

缓存内容包括：

- tile 的 core/guard 区域。
- `tile_to_real_rotation`。
- `tile_intrinsics`。
- GPU 上的 `sample_grid`。
- GPU 上的 `feather_weights`。

每帧变化的 `camera_to_world`、`timestamp`、`dynamics` 和 `image_name` 不进入缓存。它们会在 `_build_tile_render_tasks()` 中与缓存的静态 task 组合，生成当前帧的 tile viewpoint。

## Temporal State 处理

底层 `CameraViewRender` 有 `previous_camera` 状态。分块渲染一帧时会产生多个局部 tile 相机，如果把状态留在最后一个 tile，相当于把下一帧的上一相机误设为局部相机。

当前实现先创建完整目标相机：

```python
reference_viewpoint = self.camera_view_render._make_camera(...)
```

所有 tile 合成完成后再设置：

```python
self.camera_view_render.previous_camera = reference_viewpoint
```

这样对外表现仍然像每帧只渲染了一个完整目标相机。

## 命令入口

`compose_compare_video.py` 通过命令行参数把分块数量传给 `GaussianSceneRenderer.render_camera()`：

```bash
pixi run python render_3dgs/reconstruction_compare/compose_compare_video.py \
  <scene_export> \
  <aeb_front_original.mp4> \
  <aeb_trajectory.json> \
  <aeb_camera.json> \
  <aeb_front_compare.mp4> \
  --real-camera-output <aeb_real_front_120_rendered.mp4> \
  --render-tile-rows 2 \
  --render-tile-cols 2
```

默认参数为：

```text
--render-tile-rows 1
--render-tile-cols 1
```

默认值表示不启用分块路径。当前超广角调试常用 `2 x 2` 或 `4 x 4`，具体取值需要在画质、显存和耗时之间权衡。

## 性能口径

当前实现的主要性能优化点：

- 同尺寸 tile 使用 batched cameras rasterization。
- 每个 batch 默认最多 `4` 个 tile 相机。
- tile 渲染只取 RGB，不额外渲染 depth 或 3D feature。
- 目标像素到 tile 图像的采样网格缓存到 GPU。
- feather 权重缓存到 GPU。
- 重投影和融合都在 GPU 上执行。

已记录的 VTD 等效 `front_120/cam1` 输出统计如下，分辨率均为 `3840 x 2160`，`total_ms` 包含分块 3DGS 渲染和 `front.dat` LUT 后处理：

| 场景 | 帧数 | `total_ms` 平均 | 最小/最大 | `render_camera_ms` 平均 | `postprocess_ms` 平均 | 对应帧率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `me/1` | `97` | `300.547ms` | `249.804/604.931ms` | `261.930ms` | `38.617ms` | `3.33FPS` |
| `pandaset/003` | `80` | `239.828ms` | `219.367/497.182ms` | `200.744ms` | `39.085ms` | `4.17FPS` |
| `pandaset/004` | `80` | `277.669ms` | `228.873/546.196ms` | `237.452ms` | `40.216ms` | `3.60FPS` |

首帧通常包含 CUDA kernel 首次编译、tile 合成网格构建和 feather 权重缓存构建，因此最大耗时经常出现在首帧。

## 限制

- 该模块只处理 pinhole 目标相机的分块渲染。VTD `front_120` 的最终 LUT 畸变不在这里做，而是在后处理阶段由 `lut_distortion.py` 完成。
- 分块渲染不能补足训练数据没有覆盖到的视角，也不能修复 3DGS 本身的几何错误、漂浮物或动态轨迹错误。
- tile 数量增加会减小局部 FOV，但也会增加 rasterization 次数、重投影面积和融合开销。
- `guard_pixels` 太小容易出现边界接缝，太大会增加每个 tile 的渲染面积。
- `render_batch_size` 太大可能增加显存峰值；显存不足时应降低该值。
