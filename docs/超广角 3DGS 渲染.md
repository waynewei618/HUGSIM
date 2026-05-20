# 超广角 3DGS 渲染

## 1. 当前问题描述

真实车辆前视相机 `front_120` 的 VTD 配置不是单一 pinhole 相机。VTD 先用一个水平 FOV 约 $133.26^\circ$、垂直 FOV 约 $104.94^\circ$、分辨率为 `3840 x 2160` 的内部 pinhole RenderSurface 渲染，再通过 `front.dat` 查表畸变输出最终图像。`front_120` 名称对应的是查表畸变后的最终输出，其中心水平线有效 FOV 约为 $120.13^\circ$。

### 1.1 图片质量

当前画面的主要问题是边缘区域和上部中间区域仍有涂抹、拉伸和局部黑块。图像中心区域还能保留一部分道路、车辆和远处结构，但越接近左右边缘、上边缘和角点，Gaussian splat 越容易被拉成长条状，导致建筑立面、树枝、道路边缘和车辆轮廓变得模糊、变宽、拖尾。

从视觉现象上看，这不是简单的颜色偏差或视频编码问题，而是屏幕空间 splat 本身在超广角边缘被投影成过大的扁长椭圆。多个过大的椭圆相互叠加后，就表现为低频背景被抹开、物体边界被拉长、画面边缘结构不清晰。

早期未加 `front.dat` LUT 畸变时，pandaset/003 场景第 40 帧 pinhole 中间图示例：

![pandaset/003 front_120 pinhole frame 040](assets/pandaset003_real_front120_pinhole_frame040.png)

### 1.2 渲染速度

当前统计来自已经重跑的 VTD 等效 `front_120/cam1` 输出，分辨率均为 `3840 x 2160`。这里把 `total_ms` 作为端到端统计口径，包含 `TiledCameraRenderer.render_camera()` 的 3DGS 渲染耗时和 `front.dat` LUT 畸变后处理耗时。

| 场景 | 帧数 | `total_ms` 平均 | 最小/最大 | `render_camera_ms` 平均 | `postprocess_ms` 平均 | 对应帧率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `me/1` | `97` | `300.547ms` | `249.804/604.931ms` | `261.930ms` | `38.617ms` | `3.33FPS` |
| `pandaset/003` | `80` | `239.828ms` | `219.367/497.182ms` | `200.744ms` | `39.085ms` | `4.17FPS` |
| `pandaset/004` | `80` | `277.669ms` | `228.873/546.196ms` | `237.452ms` | `40.216ms` | `3.60FPS` |

当前单路 VTD 等效 `front_120` 渲染约为 `3.3` 到 `4.2 FPS`，已经低于 10 FPS。相比早期只输出 133° pinhole 中间图的结果，现在多了约 `39ms/frame` 的 LUT 畸变后处理；如果扩展到多路相机渲染，实时性压力会更明显。

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

## 4. 解决方案：分 tile 局部相机渲染，再重投影合成实车前视图

可行的解决方向是：最终输出仍然是一张满足实车前视相机内参的图像，但渲染过程不再直接使用一个完整的超广角 pinhole 相机，而是把目标图像按 tile 切分。每个 tile 使用一个局部小 FOV pinhole 相机单独渲染，再按像素光线关系重采样回目标实车图像。

这样做的目的不是改变最终相机模型，而是让每次 3DGS rasterization 只处理靠近局部光轴的一小块视场。局部 tile 相机的视场角明显小于完整 `front_120` 相机，tile 内 $|x|$、$|y|$ 更小，屏幕空间 2D 协方差不会在整幅超广角边缘被过度拉长。最终合成时再恢复到实车相机的像素坐标，因此输出图像仍然按实车前视内参定义。

### 4.1 不能只做子视口裁剪

如果只是把完整相机的图像切成多个子视口，并给每个 tile 使用：

$$
K_{\mathrm{tile}} =
\begin{bmatrix}
f_x & 0 & c_x - x_0 \\
0 & f_y & c_y - y_0 \\
0 & 0 & 1
\end{bmatrix}
$$

其中 $(x_0, y_0)$ 是 tile 左上角在完整图像中的坐标，那么每个 tile 渲染出的像素可以直接贴回原图。这种方式在几何上等价于完整实车相机的子视口，拼接后内参关系严格正确。

但这种方法不能解决边缘涂抹。因为 tile 边缘区域对应的仍然是完整超广角相机中的大角度光线，Gaussian 在该位置的归一化像平面坐标 $x = X_c / Z_c$、$y = Y_c / Z_c$ 仍然很大，2D 协方差仍会被拉成长椭圆。它只改变了渲染画布大小，没有改变局部投影条件。

### 4.2 局部 tile 相机的构造

真正有用的 tile 渲染应当让每个 tile 拥有自己的局部光轴。设最终实车相机为：

$$
K_{\mathrm{real}},\quad W,\quad H,\quad T_{\mathrm{real}}^{\mathrm{world}}
$$

其中 $K_{\mathrm{real}}$ 是实车前视内参，$W,H$ 是最终图像宽高，$T_{\mathrm{real}}^{\mathrm{world}}$ 是实车相机到世界坐标系的外参。

把目标图像划分为若干 tile，每个 tile 覆盖目标图像中的区域：

$$
[x_0, x_1) \times [y_0, y_1)
$$

先取 tile 中心像素：

$$
p_c =
\begin{bmatrix}
(x_0 + x_1) / 2 \\
(y_0 + y_1) / 2 \\
1
\end{bmatrix}
$$

用实车相机内参反投影得到该 tile 中心光线：

$$
d_c =
\frac{K_{\mathrm{real}}^{-1} p_c}
{\left\lVert K_{\mathrm{real}}^{-1} p_c \right\rVert}
$$

然后构造一个局部 tile 相机，使它的光轴对准 $d_c$。tile 相机和实车相机共享同一个相机中心，只改变朝向：

$$
T_{\mathrm{tile}}^{\mathrm{world}}
= T_{\mathrm{real}}^{\mathrm{world}} R_{\mathrm{tile}}
$$

其中 $R_{\mathrm{tile}}$ 表示从局部 tile 相机坐标系到实车相机坐标系的旋转。它的第三列或前向轴应与 $d_c$ 对齐，具体轴向要与当前 HUGSIM/gsplat 相机坐标约定保持一致。

tile 相机内参 $K_{\mathrm{tile}}$ 使用局部小 FOV。可以按 tile 四角相对 $d_c$ 的最大夹角来计算，也可以先采用固定重叠视场。原则是 tile 相机视场必须完整覆盖目标 tile 加 guard band 对应的实车相机光线。

### 4.3 合成方式：按实车像素反查 tile 图像

局部 tile 渲染不能直接贴回目标图像，因为 tile 相机的光轴已经改变。合成时应以最终实车图像为准，对每个目标像素 $p_{\mathrm{real}} = [u, v, 1]^{\mathsf{T}}$，先得到它在实车相机坐标系下的光线：

$$
d_{\mathrm{real}} = K_{\mathrm{real}}^{-1} p_{\mathrm{real}}
$$

再变换到对应 tile 相机坐标系：

$$
d_{\mathrm{tile}} = R_{\mathrm{tile}}^{-1} d_{\mathrm{real}}
$$

最后投影到 tile 渲染图：

$$
q_{\mathrm{tile}}
\sim
K_{\mathrm{tile}} d_{\mathrm{tile}}
=
K_{\mathrm{tile}} R_{\mathrm{tile}}^{-1} K_{\mathrm{real}}^{-1} p_{\mathrm{real}}
$$

也就是：

$$
q_{\mathrm{tile}}
\sim
H_{\mathrm{real}\rightarrow\mathrm{tile}} p_{\mathrm{real}},
\quad
H_{\mathrm{real}\rightarrow\mathrm{tile}}
=
K_{\mathrm{tile}} R_{\mathrm{tile}}^{-1} K_{\mathrm{real}}^{-1}
$$

由于 tile 相机和实车相机中心相同，二者之间只有纯旋转，不涉及深度，因此这一步是一个 2D homography。实际合成时遍历目标 tile 的像素区域，用 $H_{\mathrm{real}\rightarrow\mathrm{tile}}$ 反查 tile 渲染图并双线性采样。这样最终图像的每个像素仍然对应实车相机 $K_{\mathrm{real}}$ 定义的那条光线。

### 4.4 边界和重叠

tile 之间必须有 guard band，不能只渲染刚好覆盖目标区域的最小视场。原因有两个：

- Gaussian splat 具有屏幕空间半径，靠近 tile 边界的 Gaussian 可能对边界外像素有贡献。
- 重投影采样需要插值，边缘没有冗余像素时容易出现空洞、锯齿或接缝。

建议每个 tile 在目标图像坐标中向外扩展 `32` 到 `128` 像素作为 guard band，再根据扩展后的四角光线确定局部 tile 相机视场。当前实现使用 `64` 像素 guard band，并让 guard 区域也参与最终图像合成：tile 核心区域权重为 `1`，进入 guard band 后按距离平滑衰减到 `0`，多个 tile 的重叠区域做加权平均。这就是当前实现中的 feather blending。它的目的不是改变投影模型，而是避免 tile 边界处因为单 tile 截断、插值和 splat 半径不足产生硬接缝。

### 4.5 并行渲染与速度收益

分 tile 后，每个 tile 的渲染任务相互独立，同一帧内可以并行执行。当前实现采用单 GPU 内的小批量并行方式：

- 先按 tile 渲染图尺寸分组，相同尺寸的 tile 组成一个 batch。
- 每个 batch 最多放 `4` 个 tile 相机，调用 gsplat 的 batched cameras rasterization，一次完成多个局部相机渲染。
- tile 渲染只取最终需要的 RGB，不再额外 rasterize depth 和 3D feature 通道。
- 每个 tile 的实车像素到 tile 图像的 `grid_sample` 网格和 feather 权重缓存在 GPU 上。
- 合成阶段也在 GPU 上完成：对 tile 渲染图做 `grid_sample` 双线性重采样，再按 feather 权重累加，最后只把整张结果图同步回 CPU。

这样避免了旧实现中每个 tile 单独渲染、单独同步到 CPU、再用 NumPy 做大面积双线性采样的开销。pandaset/003 的 profiling 显示，分块路径中 CPU 重投影和 feather 合成曾是主要耗时；迁移到 GPU 合成后，分块渲染速度已经接近整图 pinhole 渲染。

### 4.6 预期效果和限制

该方案能缓解的问题是：由于超广角 pinhole 边缘远离光轴，导致 3DGS 屏幕空间 splat 被拉成长条后产生的涂抹、拖影和结构模糊。

该方案不能解决的问题是：

- 训练数据本身没有覆盖到的视角或遮挡区域。
- Gaussian 模型本身几何错误、漂浮物或动态物体轨迹错误。
- 训练阶段没有显式使用真实 ray/LUT 相机模型时，目标 `front_120` 输出仍会依赖窄前视和侧向相机的外推能力。

验证时应同时检查两类结果：第一是边缘区域的建筑、车辆、道路边界是否明显少拖影；第二是最终合成图像是否仍满足目标相机模型。对纯 pinhole 输出，同一个目标像素 $(u, v)$ 对应的光线由 $K_{\mathrm{real}}^{-1}[u, v, 1]^{\mathsf{T}}$ 定义；对 VTD `front_120` 最终输出，像素到光线的关系还必须经过 `front.dat` 查表映射。

## 5. VTD `front_120` 投影链路校核

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

### 5.1 Display XML 是畸变前的内部 pinhole 视场

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

### 5.2 `front.dat` 查表畸变后，最终水平 FOV 约为 120°

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

### 5.3 3DGS 要对齐 VTD，必须补同样的畸变步骤

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
aeb_hil_vil_render/vtd_front_120/front_120_parameters.json
```

如果要观察畸变前中间结果，可在调用时传 `--disable-real-camera-distortion`。

## 6. 当前基线方案

当前工程基线是 `2 x 2` 分块局部相机渲染，加 `64` 像素 guard band、feather blending、单 GPU tile batch 并行渲染和 GPU 合成。对默认实车相机 `front_120/cam1`，渲染出的 `3840 x 2160`、133° pinhole 中间图还会继续经过 VTD `front.dat` LUT 后处理，输出与 VTD 最终 `front_120` 一致的畸变图像。

超广角分块入口是 `TiledCameraRenderer.render_camera()`。默认参数为：

```text
tile_rows = 1
tile_cols = 1
```

默认值表示不分块，此时内部转调 `GaussianSceneRenderer.render_camera()` 的原始整图渲染路径。需要启用当前 `2 x 2` 基线时，在 `compose_compare_video.py` 命令行中传：

```bash
--render-tile-rows 2 \
--render-tile-cols 2
```

实车前视畸变由调用方提供参数。默认 `front_120/cam1` 使用：

```text
aeb_hil_vil_render/vtd_front_120/front_120_parameters.json
```

该参数文件再指向本地保存的 `front.dat` 符号链接。需要观察畸变前 pinhole 中间图时，传：

```bash
--disable-real-camera-distortion
```

当前基线命令形态为：

```bash
pixi run python aeb_hil_vil_render/compose_compare_video.py \
  <scene_export> \
  <aeb_front_original.mp4> \
  <aeb_trajectory.json> \
  <aeb_camera.json> \
  <aeb_front_compare.mp4> \
  --real-camera-output <aeb_real_front_120_rendered.mp4> \
  --render-tile-rows 2 \
  --render-tile-cols 2
```

其中 `pandaset/003` 和 `pandaset/004` 的训练输出只有 `ckpts/chkpnt30000.pth` 和 `dynamic_*_chkpnt30000.pth`，已在各自场景下整理了硬链接形式的 `export/` 目录供渲染器读取：

```text
outputs/pandaset/003/export/
outputs/pandaset/004/export/
```

当前已重跑并作为基线观察的文件为：

```text
outputs/aeb_hil_vil_render/me/1/aeb_front_compare.mp4
outputs/aeb_hil_vil_render/me/1/aeb_real_front_120_rendered.mp4
outputs/aeb_hil_vil_render/me/1/aeb_real_front_120_rendered.timing.csv

outputs/aeb_hil_vil_render/pandaset/003/aeb_front_compare.mp4
outputs/aeb_hil_vil_render/pandaset/003/aeb_real_front_120_rendered.mp4
outputs/aeb_hil_vil_render/pandaset/003/aeb_real_front_120_rendered.timing.csv

outputs/aeb_hil_vil_render/pandaset/004/aeb_front_compare.mp4
outputs/aeb_hil_vil_render/pandaset/004/aeb_real_front_120_rendered.mp4
outputs/aeb_hil_vil_render/pandaset/004/aeb_real_front_120_rendered.timing.csv
```

速度统计如下：

| 方案 | 分辨率 | 帧数 | 平均耗时 | 最小/最大耗时 | 对应帧率 | 说明 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 早期整图 pinhole，无 LUT | `3840 x 2160` | `80` | `176.120ms` | `159.655/212.675ms` | `5.68FPS` | 旧对照，只输出 VTD `OriginalScene` 类似中间图 |
| 早期 `2 x 2` + feather，无 LUT | `3840 x 2160` | `80` | `238.908ms` | `188.640/724.076ms` | `4.19FPS` | 旧对照，仍未输出最终 `front_120` |
| 当前 `2 x 2` + feather + `front.dat` LUT，`pandaset/003` | `3840 x 2160` | `80` | `239.828ms` | `219.367/497.182ms` | `4.17FPS` | 当前工程基线 |
| 当前 `2 x 2` + feather + `front.dat` LUT，`pandaset/004` | `3840 x 2160` | `80` | `277.669ms` | `228.873/546.196ms` | `3.60FPS` | 当前工程基线 |
| 当前 `2 x 2` + feather + `front.dat` LUT，`me/1` | `3840 x 2160` | `97` | `300.547ms` | `249.804/604.931ms` | `3.33FPS` | 当前工程基线 |

当前版本的首帧包含模型加载后的首次 CUDA kernel 编译、tile 合成网格和 feather 权重缓存构建，因此最大耗时通常出现在首帧。LUT 后处理本身约为 `39ms/frame` 到 `40ms/frame`。当前基线比只输出 pinhole 中间图更慢，但它才是与 VTD 最终 `front_120` 图像对齐的工程基线。

## 7. 当前渲染效果讨论

### 7.1 问题缓解了，但没有完全解决

分块渲染 + feather 后，硬接缝和一部分由超广角边缘 splat 过大导致的拖影得到缓解；补上 `front.dat` LUT 后，输出像素也已经与 VTD 最终 `front_120` 的采样关系对齐。但视频中仍能看到上部中间区域存在明显黑块、拉伸和不稳定纹理。这说明当前残留问题不只是 rasterization 的局部投影问题，还和训练数据覆盖范围、场景几何质量有关。

### 7.2 pandaset/003 原始前视不是大广角

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

### 7.3 为什么坏的位置主要在上部中间，而不是四周都一样坏

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

### 7.4 me 数据集前视内参与裁剪

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

### 7.5 使用实车大广角图训练的预期

如果后续使用实车 `front_120` 大广角图片训练，并且训练相机内参、外参和投影模型正确，当前上部中间区域应明显改善。原因是训练视场和目标渲染视场一致，`front_120` 高仰角区域不再完全依赖窄前视和侧向相机外推。

训练前把实车大广角图片重采样为高宽各一半是可以接受的工程折中，但必须同步缩放内参：

```text
width, height 乘以 0.5
fx, fy, cx, cy 乘以 0.5
```

这样 FOV 不变，每个像素对应的光线方向仍和原始大广角图像一致。半分辨率训练会降低纹理细节上限，但能保留完整视场覆盖，通常比用窄前视数据外推 `front_120` 更可靠。

这里还必须确认实车大广角图像的投影模型。如果目标输出是 VTD `front_120`，则不能只按 pinhole 内参训练和渲染，还要对齐 `front.dat` 查表畸变。更稳妥的做法有两种：第一，先把真实或 VTD 图像 undistort/rectify 到某个明确的 pinhole 图像并使用对应内参训练；第二，在训练和渲染链路中显式支持真实 ray/LUT 相机模型，并在输出阶段复用 VTD 的 `front.dat`。
