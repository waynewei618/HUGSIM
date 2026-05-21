# 超广角 3DGS 渲染

## 1. 当前问题描述

真实车辆前视相机 `front_120` 的 VTD 配置不是单一 pinhole 相机。VTD 先用一个水平 FOV 约 $133.26^\circ$、垂直 FOV 约 $104.94^\circ$、分辨率为 `3840 x 2160` 的内部 pinhole RenderSurface 渲染，再通过 `front.dat` 查表畸变输出最终图像。`front_120` 名称对应的是查表畸变后的最终输出，其中心水平线有效 FOV 约为 $120.13^\circ$。

### 1.1 图片质量

当前画面的主要问题是边缘区域和上部中间区域仍有涂抹、拉伸和局部黑块。图像中心区域还能保留一部分道路、车辆和远处结构，但越接近左右边缘、上边缘和角点，Gaussian splat 越容易被拉成长条状，导致建筑立面、树枝、道路边缘和车辆轮廓变得模糊、变宽、拖尾。

从视觉现象上看，这不是简单的颜色偏差或视频编码问题，而是屏幕空间 splat 本身在超广角边缘被投影成过大的扁长椭圆。多个过大的椭圆相互叠加后，就表现为低频背景被抹开、物体边界被拉长、画面边缘结构不清晰。

早期未加 `front.dat` LUT 畸变时，pandaset/003 场景第 40 帧 pinhole 中间图示例：

![pandaset/003 front_120 pinhole frame 040](assets/pandaset003_real_front120_pinhole_frame040.png)

## 2. 原因分析：边缘 2D 协方差被拉成大而扁的椭圆

3DGS 中每个 Gaussian 不是无尺寸的点，而是一个带 3D 协方差的椭球。渲染时，3D 协方差会通过相机投影变成屏幕空间的 2D 协方差，最终 rasterizer 根据这个 2D 协方差绘制一个椭圆 splat。

设某个 Gaussian 在相机坐标系下的协方差为：

$$
\Sigma_c
$$

屏幕空间 2D 协方差近似为：

$$
\Sigma_{2d} = J \Sigma_c J^{\mathsf{T}}
$$

其中 $J$ 是 pinhole 投影在 Gaussian 中心点处的一阶雅可比矩阵。

pinhole 投影为：

$$
\begin{aligned}
u &= f_x \frac{X_c}{Z_c} + c_x \\
v &= f_y \frac{Y_c}{Z_c} + c_y
\end{aligned}
$$

对应雅可比为：

$$
J =
\begin{bmatrix}
\frac{f_x}{Z_c} & 0 & -\frac{f_x X_c}{Z_c^2} \\
0 & \frac{f_y}{Z_c} & -\frac{f_y Y_c}{Z_c^2}
\end{bmatrix}
$$

令：

$$
\begin{aligned}
x &= \frac{X_c}{Z_c} \\
y &= \frac{Y_c}{Z_c}
\end{aligned}
$$

则：

$$
J =
\begin{bmatrix}
\frac{f_x}{Z_c} & 0 & -\frac{f_x x}{Z_c} \\
0 & \frac{f_y}{Z_c} & -\frac{f_y y}{Z_c}
\end{bmatrix}
$$

这个式子里，超广角边缘最关键的量就是 $x$ 和 $y$。它们不是像素坐标，而是归一化像平面坐标。FOV 越大，图像边缘的 $|x|$、$|y|$ 越大：

$$
\begin{aligned}
|x_{\mathrm{edge}}| &\approx \tan\left(\frac{\mathrm{FOV}_x}{2}\right) \\
|y_{\mathrm{edge}}| &\approx \tan\left(\frac{\mathrm{FOV}_y}{2}\right)
\end{aligned}
$$

对当前直接使用的 VTD 内部 pinhole RenderSurface：

$$
\begin{aligned}
\frac{\mathrm{FOV}_x}{2} &\approx 66.63^\circ,\quad |x_{\mathrm{edge}}| \approx 2.31 \\
\frac{\mathrm{FOV}_y}{2} &\approx 52.47^\circ,\quad |y_{\mathrm{edge}}| \approx 1.30
\end{aligned}
$$

因此在图像边缘，雅可比矩阵第三列中的 $-\frac{f_x x}{Z_c}$、$-\frac{f_y y}{Z_c}$ 会变大。这意味着相机坐标系中的深度方向扰动，会被更强地映射成屏幕上的横向或纵向位移。换句话说，Gaussian 在 3D 里的一个普通椭球，投影到屏幕边缘后会变成更大的 2D 椭圆。

用一个简化例子可以直接看到这个结果。假设：

$$
\begin{aligned}
f_x &= f_y = f \\
\Sigma_c &= \sigma^2 I
\end{aligned}
$$

则：

$$
\Sigma_{2d}
= \left(\frac{f\sigma}{Z_c}\right)^2
\begin{bmatrix}
1 + x^2 & xy \\
xy & 1 + y^2
\end{bmatrix}
$$

这个 2D 协方差矩阵的切向标准差量级约为：

$$
\frac{f\sigma}{Z_c}
$$

径向标准差量级约为：

$$
\frac{f\sigma}{Z_c}\sqrt{1 + x^2 + y^2}
$$

所以边缘位置相对中心位置会出现额外的径向拉伸：

- 水平边缘：$\sqrt{1 + 2.31^2} \approx 2.52$
- 垂直边缘：$\sqrt{1 + 1.30^2} \approx 1.64$
- 角点附近：$\sqrt{1 + 2.31^2 + 1.30^2} \approx 2.83$

这就解释了当前画面质量问题：同样大小的 3D Gaussian，在图像中心附近可能只是一个正常的小 splat；到了超广角边缘，由于 $x = X_c/Z_c$、$y = Y_c/Z_c$ 变大，$\Sigma_{2d}$ 会被拉成更大、更扁、方向更接近径向的椭圆。

这种大而扁的边缘椭圆会覆盖过多像素，并把本来局部的颜色和透明度贡献扩散到更宽区域。多个 Gaussian 同时发生这种投影放大后，边缘图像就会出现涂抹、拖影、模糊和结构拉长。

## 3. 根源：远离光轴后一阶雅可比近似误差变大

上一节的协方差投影还有一个更本质的前提：$\Sigma_{2d} = J \Sigma_c J^{\mathsf{T}}$ 使用的是 pinhole 投影在 Gaussian 中心点处的一阶局部线性近似。

也就是说，它默认 Gaussian 覆盖的那一小块 3D 空间经过投影后，仍然可以近似看成一个线性变换。但 pinhole 投影本身不是线性的：

$$
\begin{aligned}
u &= f_x \frac{X_c}{Z_c} + c_x \\
v &= f_y \frac{Y_c}{Z_c} + c_y
\end{aligned}
$$

从泰勒展开看：

$$
f(p + \delta)
= f(p) + J\delta + \frac{1}{2}\delta^{\mathsf{T}}H\delta + \cdots
$$

3DGS 的协方差投影只保留了一阶项 $J\delta$，忽略了 Hessian 对应的二阶项和更高阶项。这个近似是否有效，取决于投影函数在该 Gaussian 覆盖范围内是否足够接近线性。

靠近光轴时，$x = X_c/Z_c$、$y = Y_c/Z_c$ 较小，pinhole 投影在局部更接近线性，一阶雅可比通常还能较好描述这个 Gaussian 的屏幕投影。

远离光轴时，尤其是在超广角图像边缘，$x$、$y$ 会变大。雅可比中的深度耦合项随之变大：

$$
\begin{aligned}
\frac{\partial u}{\partial Z_c}
&= -\frac{f_x X_c}{Z_c^2}
= -\frac{f_x x}{Z_c} \\
\frac{\partial v}{\partial Z_c}
&= -\frac{f_y Y_c}{Z_c^2}
= -\frac{f_y y}{Z_c}
\end{aligned}
$$

二阶项也会随 $x$、$y$ 增大而变得更明显，例如：

$$
\begin{aligned}
\frac{\partial^2 u}{\partial Z_c^2}
&= \frac{2f_x X_c}{Z_c^3}
= \frac{2f_x x}{Z_c^2} \\
\frac{\partial^2 v}{\partial Z_c^2}
&= \frac{2f_y Y_c}{Z_c^3}
= \frac{2f_y y}{Z_c^2}
\end{aligned}
$$

因此，同一个 3D Gaussian 在图像中心附近可能仍满足局部线性假设；但到了超广角边缘，投影函数在 Gaussian 覆盖范围内变化更剧烈，一阶雅可比已经难以代表整个 Gaussian 的真实投影形状。

从角度上也可以理解这一点。pinhole 投影近似为：

$$
u = f_x \tan(\theta)
$$

当 $\theta$ 接近光轴方向时，$\tan(\theta)$ 的局部变化较平缓；当 $\theta$ 增大到超广角边缘时，$\tan(\theta)$ 和它的高阶变化都会快速增大，线性近似误差也随之放大。

## 4. VTD `front_120` 投影链路校核

`/data/vtd/Data/Setups/es39_pilot_net_image_QZ` 里存在两套名字相近的 `Front_120` 配置：

```text
Config/ImageGenerator/front_120/
Config/Img2/
```

当前 `Config/SimServer/simServer.xml` 实际启动的是：

```text
cmdline="$VI_CURRENT_SETUP/Config/ImageGenerator/front_120/front_120_Cfg.xml"
```

因此应以 `Config/ImageGenerator/front_120/` 为准，不是 `Config/Img2/Front_120_Display.xml` 里的旧配置。

### 4.1 Display XML 是畸变前的内部 pinhole 视场

`Config/ImageGenerator/front_120/front_120_display.xml` 中的相机配置为：

```xml
<AsymmetricPerspectiveAngles
  left="66.60195"
  right="66.65471"
  bottom="52.43017"
  top="52.51190"
  near="1"
  far="2500"
  offsetHPR="0.00000 0.00000 0.00000"
  offsetXYZ="-0.00000 1.84000 1.34500"/>
```

按该 Display XML 直接计算：

```text
horizontal FOV = 66.60195 + 66.65471 = 133.25666°
vertical FOV   = 52.43017 + 52.51190 = 104.94207°
```

对应 pinhole 内参约为：

```text
fx = 829.73
fy = 829.58
cx = 1917.57
cy = 1081.59
```

这组数值是 VTD 内部 RenderSurface 的畸变前相机，不是最终输出图像的完整模型。

### 4.2 `front.dat` 查表畸变后，最终水平 FOV 约为 120°

`Config/ImageGenerator/front_120/front_120_IGbase.xml` 中，`OriginalScene` 之后还有 `Distortion` 步骤：

```xml
<Step type="PPSTextureRect" name="Distortion" >
  <Inputs>
    <TextureInput inputNo="0" sourceStep="HaloBloomGlare" />
    <TextureInput inputNo="1" sourceLookupTable="./dat/front.dat" />
    <TextureInput inputNo="2" textureType="TextureRectangle" sourceImage="lensImage.rgb" />
  </Inputs>
  <Program vertexShader="./Shaders/PinholeDistortVert.glsl"
           fragmentShader="./Shaders/PinholeDistortFrag.glsl" >
    <Uniform type="sampler2D" name="u_originalScene" value="0" />
    <Uniform type="sampler2D" name="u_lookupTex" value="1"/>
    <Uniform type="sampler2D" name="u_edgeBlendTex" value="2"/>
  </Program>
</Step>
```

`PinholeDistortFrag.glsl` 的核心逻辑是：

```glsl
vec2 texCoord = texture(u_lookupTex, gl_TexCoord[0].st ).xy;
vec3 color = texture(u_originalScene, texCoord).rgb;
```

也就是最终输出像素先查 `front.dat`，得到畸变前 133° pinhole 图上的采样坐标，再从 `OriginalScene` 取色。

对 `front.dat` 抽样可见，最终输出上边中心像素并不采样畸变前图像最上边，而是采样约 `y=544` 的位置：

```text
output pixel (1919, 0)    -> source pixel (1918.5, 543.8)
output pixel (1919, 2159) -> source pixel (1918.5, 1614.8)
output pixel (0, 1080)    -> source pixel (481.5, 1080.2)
output pixel (3839,1080)  -> source pixel (3363.5,1080.2)
```

将这些源图坐标代回 133° pinhole 内参，最终输出图像中心线有效视场约为：

```text
horizontal FOV ≈ 120.13°
vertical FOV   ≈ 65.68°
```

因此 `front_120` 名称是合理的：它对应 `front.dat` 查表畸变后的最终水平视场，而不是 Display XML 中畸变前的 133° 内部渲染视场。

### 4.3 3DGS 要对齐 VTD，必须补同样的畸变步骤

要输出和 VTD `front_120` 一致的图片，流程应对齐为：

```text
VTD:
3D 场景 -> 133° pinhole OriginalScene -> front.dat 畸变/LUT -> 最终 front_120 图像

3DGS:
3DGS 场景 -> 133° pinhole 渲染图 -> front.dat 畸变/LUT -> 最终 front_120 图像
```

早期 AEB 3DGS 渲染路径只做了第一步：

```text
3DGS 场景 -> 133° pinhole 渲染图
```

也就是说，当时的 `aeb_real_front_120_rendered.mp4` 更接近 VTD 的 `OriginalScene` 中间结果，而不是 VTD 最终输出的 `front_120` 图像。直接把 133° pinhole 图当最终结果，会把 VTD 本来通过 `front.dat` 压缩、重采样的区域直接显示出来，容易放大上部和边缘区域的拖影、黑块和结构错误。

不能简单把相机内参改成 120° pinhole 来替代这一步，因为 VTD 最终图不是普通 120° pinhole，而是查表畸变后的图像。正确方向是复用同一个：

```text
/data/vtd/Data/Setups/es39_pilot_net_image_QZ/ImageGenerator/dat/front.dat
```

当前 `compose_compare_video.py` 的默认实车 `front_120/cam1` 输出已经在渲染后调用通用 LUT 畸变模块，对 3DGS 渲染出的 133° pinhole 图做同样的 remap。畸变资源由调用方参数提供，默认使用：

```text
render_3dgs/vtd_front_120/front_120_parameters.json
```

如果要观察畸变前中间结果，可在调用时传 `--disable-real-camera-distortion`。

## 5. 当前渲染效果讨论

### 5.1 问题缓解了，但没有完全解决

补上 `front.dat` LUT 后，输出像素已经与 VTD 最终 `front_120` 的采样关系对齐。但视频中仍能看到上部中间区域存在明显黑块、拉伸和不稳定纹理。这说明当前残留问题不只是输出投影链路问题，还和训练数据覆盖范围、场景几何质量有关。

### 5.2 pandaset/003 原始前视不是大广角

pandaset/003 原始 `front_camera` 内参为：

```text
resolution = 1920 x 1080
fx = 1970.0131
fy = 1970.0091
cx = 970.0002
cy = 483.2988
```

按 pinhole 模型计算：

```text
front_camera:
  horizontal FOV ≈ 51.96°
  vertical FOV   ≈ 30.64°
  diagonal FOV   ≈ 58.40°
```

而当前直接用于 3DGS 的 VTD 内部 pinhole `front_120/cam1` 为：

```text
resolution = 3840 x 2160
horizontal FOV ≈ 133.26°
vertical FOV   ≈ 104.94°
```

因此，现在是在用一个远大于原始中心前视相机视场的内部 pinhole 相机做渲染。`front_120` 畸变前图中有大量像素对应的光线，原始中心前视相机从未观测过。PandaSet 还有 `front_left_camera`、`front_right_camera`、`left_camera`、`right_camera` 等相机，它们对水平侧向区域有补充，但它们不是同一个光心的中心大广角前视相机。

### 5.3 为什么坏的位置主要在上部中间，而不是四周都一样坏

问题不会按画面四周均匀出现，因为训练观测覆盖、场景内容和 3DGS 几何质量都不是均匀的。

从垂直视场看：

```text
PandaSet 原始 front_camera:
  上方视角约 13.8°
  下方视角约 16.5°

PandaSet front_left/front_right 等宽一些的相机:
  垂直总 FOV 约 60°
  单侧向上大约 28° 到 31°

VTD 畸变前 133° pinhole 中间图:
  上方视角约 52.5°
  下方视角约 52.4°

VTD front.dat 后的最终 front_120 中心线:
  上方视角约 33.0°
  下方视角约 32.7°
```

这意味着 133° pinhole 中间图上半部，尤其是画面最上方到上中部，有一大块超过了 PandaSet 原始相机的上视角覆盖。按 133° pinhole 内参粗略估计，画面 `y < 620` 左右已经超过约 `29^\circ` 上视角，基本超出 PandaSet 其他相机的垂直覆盖；`y < 875` 左右已经超出原始中心 `front_camera` 的上视角。VTD 最终 `front_120` 图像会通过 `front.dat` 把上边中心映射到畸变前图像约 `y=544` 的位置，相当于把直接 pinhole 的高仰角区域压缩掉一部分。当前基线已经应用这一步，因此后续判断应看 `aeb_real_front_120_rendered.mp4`，不要再用未畸变 pinhole 中间图代表 VTD 最终效果。

左右方向没有同样严重，主要是因为 PandaSet 的前左、前右、左、右相机给水平侧向区域提供了一部分观测。虽然这些相机不是 `front_120` 的同一光心，但建筑立面、路边车辆和人行道至少在训练图像中出现过。

下半部分也相对更稳定，因为道路、车道线、停靠车辆会在连续帧中反复出现，几何约束比天空和高处结构更强。上部中间则经常对应天空、楼顶、树冠、远处高处边缘等内容，这些区域本身几何弱、纹理少或结构很薄，容易在 3DGS 中形成大 Gaussian、漂浮 Gaussian 或错误深度。`front.dat` 会压缩直接 pinhole 的高仰角区域，但不能凭空补足训练中没有稳定观测到的几何，所以当前畸变后视频里仍能看到一部分黑块、拖影和条纹。

### 5.4 me 数据集前视内参与裁剪

me 数据集中 `CAM_FRONT_120` 原始图像为 `3840 x 2160`。训练预处理逻辑是先按 `downsample=2` 缩放到 `1920 x 1080`，再对 `CAM_FRONT_120` 做裁剪：

```python
crop_up = int((880 - 520) // downsample)  # 180
crop_down = int(520 // downsample)        # 260
crop_left = 480
crop_right = 480
```

因此训练实际使用的前视图像为：

```text
x: 480 -> 1440
y: 180 -> 820
size: 960 x 640
```

内参同步更新为：

```python
intrinsic[0, 0] = fx / downsample
intrinsic[1, 1] = fy / downsample
intrinsic[0, 2] = cx / downsample - crop_left
intrinsic[1, 2] = cy / downsample - crop_up
```

现有 `camera_params.json` 中，me 原始 `CAM_FRONT_120` 和训练裁剪后的内参为：

```text
me 原始 4K CAM_FRONT_120:
  resolution = 3840 x 2160
  fx = 1891.33
  fy = 1874.73
  cx = 1923.24
  cy = 1073.98
  pinhole 等效水平 FOV ≈ 90.85°
  pinhole 等效垂直 FOV ≈ 59.87°

me 训练输入:
  resolution = 960 x 640
  fx = 945.67
  fy = 937.37
  cx = 481.62
  cy = 356.99
  pinhole 等效水平 FOV ≈ 53.77°
  pinhole 等效垂直 FOV ≈ 37.59°
```

这段裁剪没有改变相机光轴方向，内参主点也按裁剪正确平移；但它确实减少了训练输入覆盖的视场。实际调试中，同一帧用 me 原始 4K `CAM_FRONT_120` 内参渲染，比直接用 VTD 133° pinhole `front_120/cam1` 渲染，上部中间问题明显减轻。这说明当前问题不是单纯的 3DGS 模型全局坏掉，而是目标相机视场和投影链路与训练输入不一致。

### 5.5 使用实车大广角图训练的预期

如果后续使用实车 `front_120` 大广角图片训练，并且训练相机内参、外参和投影模型正确，当前上部中间区域应明显改善。原因是训练视场和目标渲染视场一致，`front_120` 高仰角区域不再完全依赖窄前视和侧向相机外推。

训练前把实车大广角图片重采样为高宽各一半是可以接受的工程折中，但必须同步缩放内参：

```text
width, height 乘以 0.5
fx, fy, cx, cy 乘以 0.5
```

这样 FOV 不变，每个像素对应的光线方向仍和原始大广角图像一致。半分辨率训练会降低纹理细节上限，但能保留完整视场覆盖，通常比用窄前视数据外推 `front_120` 更可靠。

这里还必须确认实车大广角图像的投影模型。如果目标输出是 VTD `front_120`，则不能只按 pinhole 内参训练和渲染，还要对齐 `front.dat` 查表畸变。更稳妥的做法有两种：第一，先把真实或 VTD 图像 undistort/rectify 到某个明确的 pinhole 图像并使用对应内参训练；第二，在训练和渲染链路中显式支持真实 ray/LUT 相机模型，并在输出阶段复用 VTD 的 `front.dat`。
