# VTD front_120 参数

本目录集中保存 `/data/vtd/Data/Setups/es39_pilot_net_image_QZ` 中 `front_120` 相关配置，用于后续让 3DGS 输出与 VTD 最终 `front_120` 图像一致。

## 文件

```text
front_120_parameters.json      # 解析后的关键参数和 LUT 抽样结果
front_120_display.xml          # VTD 畸变前 pinhole RenderSurface 相机配置
front_120_IGbase.xml           # VTD 后处理 pipeline，包含 Distortion 步骤
front.dat -> /data/vtd/...     # VTD distortion lookup table，当前为符号链接
shaders/PinholeDistortVert.glsl
shaders/PinholeDistortFrag.glsl
```

`front.dat` 约 `218MB`，当前不复制实体文件，只在本目录放符号链接：

```text
/data/vtd/Data/Setups/es39_pilot_net_image_QZ/ImageGenerator/dat/front.dat
```

如果后续需要离线打包，再把该符号链接替换为实体文件。

## VTD 渲染链路

当前 VTD `simServer.xml` 实际启动：

```text
Config/ImageGenerator/front_120/front_120_Cfg.xml
```

该配置的输出链路是：

```text
3D 场景
  -> 3840x2160、水平 133.25666°、垂直 104.94207° 的 pinhole OriginalScene
  -> 使用 front.dat + PinholeDistortFrag.glsl 做查表畸变
  -> 最终 front_120 图像
```

因此 3DGS 要生成与 VTD 一致的前视图像，也应按同样流程：

```text
3DGS 场景
  -> 用 front_120_parameters.json 中 pinhole_camera.intrinsics 渲染 3840x2160 图像
  -> 用 front.dat 对该图像 remap
  -> 输出最终 front_120 图像
```

不能直接把目标相机改成普通 `120°` pinhole。VTD 最终图像是 LUT 畸变图，不是 pinhole 图。
