# LUT 查表畸变模块

本文档专门记录 `render_3dgs/lut_distortion.py`。该模块只负责读取 VTD distortion lookup table，并把已经渲染出的 pinhole 图像按 LUT 重映射为最终畸变图像。

在当前 AEB HIL/VIL 前视渲染流程中，它对应 VTD `front_120` 链路里的第二步：

```text
3DGS 场景
  -> 按 VTD 畸变前 pinhole 相机渲染图像
  -> 使用 front.dat 做 lookup-table remap
  -> 得到 VTD 等效的 front_120 最终图像
```

`lut_distortion.py` 不负责生成 LUT、不解析 VTD XML、不推导相机内参，也不决定使用哪一路相机。调用侧需要先提供尺寸一致的 pinhole 渲染图和对应 `.dat` 查表文件。

## 文件职责

模块内只有三个公开入口：

| 入口 | 职责 |
| --- | --- |
| `read_vtd_lookup_table(path)` | 读取 VTD `.dat` 查表文件，返回 OpenCV `remap` 需要的 `map_x`、`map_y` |
| `apply_lookup_table_distortion(image, map_x, map_y)` | 对单张图像执行查表畸变 |
| `LookupTableDistorter(lookup_table_path)` | 缓存 LUT，提供可复用的逐帧 callable 后处理器 |

常规视频流程应使用 `LookupTableDistorter`，因为 `front.dat` 很大，逐帧重新读取 LUT 会明显拖慢处理。

## LUT 文件格式

`read_vtd_lookup_table()` 当前支持 VTD lookup table version `1`。文件按二进制方式打开，但内容按行解析：

```text
1
<width> <height>
<x0>,<y0>,<flag0>|<x1>,<y1>,<flag1>|...
...
```

解析规则：

- 第一行必须是 `1`，否则报 `ValueError`。
- 第二行是 LUT 输出尺寸：`width height`。
- 后续必须有 `height` 行。
- 每行必须包含 `width * 3` 个浮点值。
- 每个像素三元组为 `source_x, source_y, flag`。
- 当前实现只使用前两个值，`flag` 会被读取但不参与计算。

返回值：

```python
map_x.shape == (height, width)
map_y.shape == (height, width)
map_x.dtype == np.float32
map_y.dtype == np.float32
```

语义是：最终输出图像中像素 `(u, v)` 的颜色，从输入 pinhole 图像的浮点坐标 `(map_x[v, u], map_y[v, u])` 采样得到。采样由 OpenCV 做双线性插值。

## 图像重映射行为

`apply_lookup_table_distortion()` 会调用：

```python
cv2.remap(
    image,
    map_x,
    map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0,
)
```

因此实际行为为：

- 输入图像可以是 `HxW` 灰度图，也可以是 `HxWxC` 多通道图。
- 输入图像的 `height, width` 必须与 LUT 的 `height, width` 完全一致。
- LUT 坐标落在输入图像外时，输出填 `0`，也就是黑色。
- 图像通道顺序不会被转换；输入是 RGB，输出仍按 RGB 数组处理。
- 输出 dtype 通常跟随输入图像 dtype。

该函数不会自动 resize。如果渲染出的 pinhole 图像尺寸不是 LUT 尺寸，应先修正相机内参和渲染分辨率，而不是在畸变前后临时缩放图像。

## 在当前流程中的接入位置

`compose_compare_video.py` 在生成实车相机渲染视频时接入该模块：

```python
lookup_table_path = lookup_table_path_from_distortion_parameters(distortion_parameters)
postprocess = LookupTableDistorter(lookup_table_path)
```

随后 `write_rendered_camera_video()` 每帧先调用 3DGS 渲染：

```python
image = renderer.render_camera(...)
```

再执行后处理：

```python
image = postprocess(image)
```

默认实车相机 `front_120/cam1` 使用：

```text
/workspace/HUGSIM/render_3dgs/vtd_front_120/front_120_parameters.json
```

该 JSON 中的 `local_files.lookup_table` 指向同目录的 `front.dat`。`lut_distortion.py` 本身不解析这个 JSON；从参数 JSON 找到 `.dat` 路径的逻辑在 `compose_compare_video.py` 的 `lookup_table_path_from_distortion_parameters()` 中。

## 直接使用示例

在已有 pinhole 图像数组时，可以直接调用：

```python
import imageio.v2 as imageio

from render_3dgs.lut_distortion import LookupTableDistorter

distorter = LookupTableDistorter(
    "/workspace/HUGSIM/render_3dgs/vtd_front_120/front.dat"
)

image = imageio.imread("/workspace/tmp/front_120_pinhole.png")
distorted = distorter(image)
imageio.imwrite("/workspace/tmp/front_120_distorted.png", distorted)
```

在完整视频渲染流程中，通常不直接写 Python，而是给 `compose_compare_video.py` 传参数：

```bash
pixi run python render_3dgs/compose_compare_video.py \
  /workspace/HUGSIM/outputs/pandaset/003 \
  /workspace/HUGSIM/outputs/render_3dgs/pandaset/003/aeb_front_original.mp4 \
  /workspace/HUGSIM/outputs/render_3dgs/pandaset/003/aeb_trajectory.json \
  /workspace/HUGSIM/outputs/render_3dgs/pandaset/003/aeb_camera.json \
  /workspace/HUGSIM/outputs/render_3dgs/pandaset/003/aeb_front_compare.mp4 \
  --real-camera-output /workspace/HUGSIM/outputs/render_3dgs/pandaset/003/aeb_real_front_120_rendered.mp4 \
  --real-camera-distortion-parameters /workspace/HUGSIM/render_3dgs/vtd_front_120/front_120_parameters.json
```

如果只想观察畸变前的 pinhole 中间图，可传：

```bash
--disable-real-camera-distortion
```

## 常见错误

| 报错 | 直接原因 | 处理方式 |
| --- | --- | --- |
| `VTD lookup table not found` | `.dat` 路径不存在 | 检查 `front.dat` 符号链接或 `--real-camera-distortion-parameters` 路径 |
| `unsupported lookup table version` | 第一行不是 `1` | 当前读取器只支持 version `1` |
| `invalid lookup table size line` | 第二行不能解析成两个整数 | 检查 LUT 文件是否损坏或传错文件 |
| `ended before row ...` | 实际行数少于声明的 `height` | 检查 LUT 文件完整性 |
| `row ... contains ... values` | 某行不是 `width * 3` 个值 | 检查 LUT 文件格式 |
| `image must be HxW or HxWxC` | 输入不是 2D 或 3D 图像数组 | 调用前整理图像数组维度 |
| `image size ... does not match lookup table size ...` | 渲染分辨率和 LUT 尺寸不同 | 使用 LUT 对应的 pinhole 相机分辨率重新渲染 |
| `lookup-table distortion requires OpenCV` | 环境里没有 `cv2` | 使用项目 pixi 环境运行 |

## 性能和内存

以 `3840 x 2160` 的 `front_120` LUT 为例：

- `front.dat` 文件约 `218MB`。
- `map_x` 和 `map_y` 各占约 `31.6MiB`，合计约 `63.3MiB`。
- 当前实现使用 CPU 版 OpenCV `remap`。
- 已记录的 `front_120` 视频流程中，LUT 后处理约为 `39-40ms/frame`。

因此长视频处理时应复用同一个 `LookupTableDistorter` 实例。除非 LUT 文件变化，否则不要在每帧里调用 `read_vtd_lookup_table()`。

## 边界和约定

- LUT 畸变是最终相机输出的一部分，不是训练相机标定的替代品。
- `map_x/map_y` 描述的是从输出像素到输入 pinhole 图像的反向采样关系。
- 该模块不保存图片或视频；保存逻辑由调用侧负责。
- 该模块不维护 timing 状态；逐帧耗时由 `compose_compare_video.py` 调用侧记录。
- 当前实现没有检查 LUT 文件末尾是否还有多余内容；它按声明的 `height` 读取对应行数。
