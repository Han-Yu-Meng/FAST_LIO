# ZUPT（零速更新）原理与 FAST-LIO 实现说明

> 本文档是 `mid360_localization / src/FAST_LIO` 中 ZUPT 模块的原理调研与实现说明。
> 代码位置：`src/zupt.hpp`（检测器 + 更新）、`src/laserMapping.cpp`（集成）、
> `config/*.yaml`（参数）、`scripts/zupt_check.py`（独立验证）。

---

## 1. 背景：ZUPT 是什么，为什么对地面机器人有效

零速更新（Zero-Velocity Update, ZUPT）源自足绑式行人惯导（Foxlin, IEEE VR 2005，
"Pedestrian Tracking with Shoe-Mounted Inertial Sensors"）。核心观察：

1. **运动具有可检测的"静止相"**。行人步态的站立相、车辆的停车、机器人悬停/待机，
   期间 IMU 测得的比力恒为重力、角速度恒为零偏（加噪声）。
2. **静止 = 免费的、强的伪量测**。此时速度真值为 0，把它作为量测送入 EKF，
   可以周期性地"截断"惯导误差漂移。Foxlin 的分析表明：纯惯导位置误差随时间
   立方增长，周期性 ZUPT 后误差随"步数/静止段数"线性增长。
3. **协方差交叉项带来"回溯修正"**。把 ZUPT 作为 EKF 量测更新（而不是直接把速度
   置零），滤波器会利用速度-位置-姿态误差之间的协方差非对角项，把静止段积累的
   位置/姿态漂移一并拉回来。

对**四自由度地面机器人**（x, y, z, yaw；roll/pitch 由重力确定），ZUPT 效果显著，
原因有三层：

| 约束 | 观测到什么 | 对 4DOF 机器人的意义 |
|---|---|---|
| velocity ZUPT | 速度 v=0 | 直接杀掉速度漂移，经交叉项抑制位置/姿态漂移 |
| angular-rate ZUPT | 陀螺零偏 bg | 陀螺零偏是 yaw 漂移的根源，停车期间精确标定 |
| 重力约束 | roll/pitch + 加计零偏 ba | 把水平姿态钉在重力方向上，等价于 4DOF 中被消掉的两个自由度 |

静止时 LiDAR 里程计本身也可能退化（开阔地、长走廊、动态障碍遮挡），
ZUPT 提供了与场景无关的观测来源——这正是它在低成本 IMU 行人定位
（抬脚/落脚检测 → 选更新策略：不更新 / 局部更新 / 降低协方差）中广泛使用的原因。

---

## 2. 零速检测（zero-velocity detection）

### 2.1 问题形式化

假设检验（Wahlström & Skog, "Fifteen Years of Progress at Zero Velocity:
A Review", arXiv:2008.09208）：

- H0：运动；H1：静止。
- 窗口测量 `z_k = {u_n}, n ∈ [k-w_b, k+w_f]`，窗口长 `W = w_b + w_f + 1`。
- 判据：似然比 `p(z_k|H1) / p(z_k|H0) > γ`。

### 2.2 四种经典检测器（Skog et al., IEEE TBME 2010）

本模块实现了 OpenShoe 参考实现（`GLRT.m / MV.m / MAG.m / ARE.m`，见 §7）中的
四种检测器，通过 `detector_type` 选择：

**GLRT / SHOE（默认，推荐）**——广义似然比检验的闭式解：

```
T = (1/W) Σ_n [ ‖ω_n‖² / σ_g² + ‖a_n − g·ā/‖ā‖‖² / σ_a² ]
```

其中 `ā` 为窗口内加速度均值，`g = 9.81`，`σ_a / σ_g` 为加计/陀螺白噪声标准差
（兼作两项的权重）。静止时两项都服从卡方分布，`T ≈ 3 + 3 = 6` 量级；
运动时迅速变大。**判据：`T ≤ γ` 判为静止**。

**MV（加速度滑动方差）**：`T = (1/W) Σ ‖a_n − ā‖²`
**MAG（加速度模值）**：`T = (1/W) Σ (‖a_n‖ − g)²`
**ARE（角速度能量）**：`T = (1/W) Σ ‖ω_n‖²`

> 经验结论（Skog 2010 评测）：GLRT 综合加计与陀螺，鲁棒性最好；ARE 次之；
> MV 在慢速运动（摆动相减速段）容易误检；MAG 受振动影响大。
> 陀螺信息对零速检测最有价值，加计与陀螺互补。

### 2.3 硬门限与迟滞

为防 σ 参数失配导致统计量漂移，另加两道**硬门限**（0 表示禁用）：

- `max_gyro_mean`：窗口平均角速度模值超过阈值 → 强制判运动；
- `max_acc_dev`：`|‖ā‖ − g|` 超过阈值 → 强制判运动（静止时比力模长恒为 g）。

决策经**施密特触发器**输出：连续 `enter_cnt` 次命中才进入静止，
连续 `exit_cnt` 次未命中才退出静止。默认进入慢（防抖）、退出快（防误触发）。

### 2.4 γ 的标定（重要）

γ 无量纲，与 σ_a/σ_g 的取值耦合，**必须按实测数据标定**：

1. 录一段"真静止"数据（机器人停稳）和一段"真运动"数据；
2. 用 `scripts/zupt_check.py` 里的 Detector 代码（与 zupt.hpp 逐行对应）离线
   回放，统计两段数据的 T 分布；
3. 取 γ 在静止段 T 的 P95 与运动段 T 的 P05 之间；
4. 在线调试可订阅 `/zupt_status`（见 §5），rqt_plot 观察 `linear.y`（统计量）
   与 `linear.x`（静止标志）验证。

综述（arXiv:2008.09208）指出：γ 最优值随运动速度变化（跑步时更低、步行时更高），
固定阈值在混合工况下位置 RMSE 比自适应阈值差 50% 以上。地面机器人"停车/行驶"
两类工况差异大，若行驶中（颠簸、低速蠕行）出现误触发，优先收紧硬门限，
其次上调 γ。

---

## 3. 量测模型（ESKF 中的 ZUPT）

FAST-LIO2 用 IKFoM 流形误差状态 EKF。状态与切空间索引（共 23 维）：

```
[0-2] pos   [3-5] rot   [6-8] offset_R_L_I   [9-11] offset_T_L_I
[12-14] vel   [15-17] bg   [18-20] ba   [21-22] grav (S2)
```

右扰动模型：`R ⊞ δθ = R·Exp(δθ)`（与 `df_dx` 中 `-R·hat(acc)` 的写法一致）。

### 3.1 velocity ZUPT

静止 ⇒ 世界系速度为零：

```
z = 0,   h(x) = vel,   r = z − h = −vel
H[:, 12:15] = I₃,   R = σ_v²·I₃
```

H 只在速度块非零，位置/姿态的修正是经 P 的交叉项传导的——这正是"回溯修正"。

### 3.2 angular-rate ZUPT（观测陀螺零偏）

静止 ⇒ 陀螺测量即零偏：`ω_m = bg + n`：

```
z = mean(ω_window),   h(x) = bg,   r = ω̄ − bg
H[:, 15:18] = I₃,   R = σ_bg²·I₃
```

长期停车时持续收紧 bg，直接抑制 yaw 漂移（地面机器人最关心的自由度）。

### 3.3 重力约束（观测 roll/pitch 与加计零偏）

静止 ⇒ 加计测量：`ā = ba − Rᵀ·grav`（由 FAST-LIO 传播方程
`v̇ = R(a−ba) + grav = 0` 反推；`grav` 指向地心，模长 9.809）：

```
z = mean(a_window),   h(x) = ba − Rᵀ grav,   r = ā − ba + Rᵀ grav
H[:, 3:6]   = skew(Rᵀ grav)     （对 δθ 的雅可比）
H[:, 18:21] = −I₃                （对 δba）
R = σ_g²·I₃
```

该残差对 δθ 的雅可比秩为 2（重力方向只约束横滚/俯仰，绕重力方向的 yaw
不可观）——这与"重力把 6DOF 位姿降为 4DOF（x,y,z,yaw）"是同一件事，
数值验证见 `scripts/zupt_check.py` test 2（解析 vs 数值雅可比误差 < 1e-6）。

### 3.4 更新策略（对应行人定位中的"不更新/局部更新"）

`update_mode`：

- `0 = MODE_NONE`：只检测不更新（用于观察检测器）；
- `1 = MODE_COV_ONLY`：**只收缩协方差、不注入状态修正**——即"局部更新（降低
  协方差）"策略。适用于不完全信任零速先验、但希望压低滤波器不确定度的场合；
- `2 = MODE_FULL`：完整卡尔曼更新（默认）。

按组开关：`vel_zupt / gyro_zupt / gravity_constraint` 可独立启用。
安全护栏：残差/雅可比非有限、修正量 `|δθ| > 5°` 或 `|δv| > 0.5 m/s` 时跳过
更新并告警（防检测器误触发引发状态跳变）。协方差更新用 Joseph 形式保证对称正定。

---

## 4. 与 FAST-LIO 的集成设计

### 4.1 为什么是"独立串行更新"而不是塞进 h_share_model

本仓库的 `esekfom::update_iterated_dyn_share_modified(double R)` 是魔改版本，
量测雅可比**硬编码 12 列**（pos/rot/外参），靠 `(P/R)⁻¹ + HᵀH` 的信息形式更新，
vel/bg/ba 只通过协方差交叉项被间接修正。ZUPT 行需要直接观测 vel(12-14) 与
bg(15-17)，无法塞进该 12 列结构。

方案：**传播后、LiDAR 迭代更新前，用 esekf 的公开接口
（get_x/get_P/change_x/change_P）执行一次独立的 EKF 量测更新**。
对 EKF 而言，串行更新与联合更新数学等价（同一线性化点上信息矩阵可加），
且带来工程好处：

- 点云退化/无效帧时 ZUPT 依然生效（更新点在"点数不足 return"之前）；
- "不更新/只降协方差/全量更新"策略天然可插拔；
- 不改动 IKFoM 工具箱代码，升级风险小。

### 4.2 执行流程

```
imu_cbk (200 Hz)  →  归一化加计  →  zupt::Detector::push()   [滑窗统计 + 迟滞]
loop_once (每帧扫描):
    p_imu->Process()            [IMU 前向传播到 scan 末时刻]
    if (静止)  zupt::apply_update(kf)   [串行 EKF 更新: 3~9 行量测]
    ...ikd-Tree / LiDAR 迭代更新 / 建图 ...
    publish_zupt_status()       [/zupt_status 调试输出]
```

检测器在 IMU 频率上运行（响应快、可全时段统计），ZUPT 更新在扫描频率上执行
（10 Hz，已足够压制漂移；LiDAR 更新随后进行，起点即 ZUPT 修正后的状态）。
`use_imu_odometry` 模式下 `kf_copy` 在 LiDAR 更新后同步，IMU 频率里程计输出
自动获得 ZUPT 修正。

### 4.3 调试输出 `/zupt_status`（geometry_msgs/Twist）

| 字段 | 含义 |
|---|---|
| `linear.x` | 静止标志（1=静止） |
| `linear.y` | 检测统计量 T |
| `linear.z` | `|‖ā‖ − g|` |
| `angular.x` | `‖陀螺均值‖` |
| `angular.y` | `‖ā‖` |
| `angular.z` | 窗口样本数 |

---

## 5. 参数总表（FastLIO.zupt）

| 参数 | 默认 | 含义 |
|---|---|---|
| `enable` | true | 总开关 |
| `window_sec` | 0.15 | 检测窗口（秒） |
| `detector_type` | 0 | 0=GLRT/SHOE, 1=MV, 2=MAG, 3=ARE |
| `sigma_acc` / `sigma_gyr` | 0.15 / 0.02 | GLRT 噪声权重 [m/s², rad/s] |
| `gamma` | 10.0 | 检测阈值（**需标定**，见 §2.4） |
| `max_gyro_mean` | 0.06 | 硬门限：平均角速度 [rad/s]，0=禁用 |
| `max_acc_dev` | 0.5 | 硬门限：\||a|−g| [m/s²]，0=禁用 |
| `enter_cnt` / `exit_cnt` | 2 / 1 | 进入/退出静止的连续命中数 |
| `vel_zupt` / `gyro_zupt` / `gravity_constraint` | true | 三组量测开关 |
| `sigma_vel` | 0.03 | 速度量测噪声 [m/s]（越小拉得越狠） |
| `sigma_bg` | 0.02 | 零偏量测噪声 [rad/s] |
| `sigma_grav` | 0.10 | 重力量测噪声 [m/s²] |
| `update_mode` | 2 | 0=只检测 1=只降协方差 2=完整更新 |

调参建议：

- **振动大的平台**（电机怠速、履带）：静止时统计量会被抬高 → 适当放大
  `sigma_acc` 或上调 `gamma`，或改用 ARE 检测器（对加计振动不敏感）；
- **低速蠕行不算静止**：收紧 `max_gyro_mean`（如 0.03）并上调 `gamma`；
- **停车时间长**：`gyro_zupt` 收益最大（bg 收敛后 yaw 漂移显著下降）；
- **只想托底不想动状态**：`update_mode: 1`（cov_only）；
- 与已有 `planar_constraint`（输出端 roll/pitch/z 清零）不冲突：ZUPT 在滤波器
  内部修正，planar_constraint 只影响输出，两者可同时开。

---

## 6. 与"重力约束 4DOF"相关工作的关系

ZUPT/重力约束谱系（本模块的理论坐标）：

- **VINS-Mono**（Qin & Shen, T-RO 2018）：回环/位姿图 4DOF 精修（x,y,z,yaw），
  roll/pitch 由重力固定——"rotation-constrained yaw 优化"的代表；
- **LIO-Mapping**（Ye, Chen & Liu, ICRA 2019, arXiv:1904.06993）：紧耦合
  LiDAR-IMU 里程计 + rotation-constrained refinement 与全局地图对齐，
  即用户提到的"rotation-constrained yaw 优化"；
- **Gravity-constrained 4DOF ICP**（Babin et al., arXiv:2203.13799，Norlab）：
  把点到面 ICP 的优化变量降为 τ=[yaw, tx, ty, tz]，roll/pitch 由 IMU 重力方向
  固定。已开源进 libpointmatcher（`force4DOF` 选项）。该文引用 Zhang et al. 指出：
  重力使 2 个旋转自由度直接可观，位姿估计退化为 4DOF；
- **RTLIO**（Sensors 2021）：位姿图同样只优化 4DOF（x,y,z,yaw）。

本模块的 `gravity_constraint` 行是同一思想在**滤波器前端**的落地：
每帧静止时把 roll/pitch 拉向重力方向（yaw 不受约束）。若后续要上
4DOF ICP / 位姿图 4DOF 精修，滤波器侧的 roll/pitch 已被 ZUPT-重力约束钉住，
正好是 4DOF 假设成立的前提。

---

## 7. 参考文献与参考实现

**论文**

1. E. Foxlin, "Pedestrian Tracking with Shoe-Mounted Inertial Sensors",
   IEEE Computer Graphics and Applications, 2005.（ZUPT 开山之作）
2. I. Skog, P. Händel, J.-O. Nilsson, "Zero-Velocity Detection — An Algorithm
   Evaluation", IEEE Trans. Biomedical Engineering, 2010.（四种检测器的评测）
3. I. Skog et al., "Evaluation of Zero-Velocity Detectors for Foot-Mounted
   Inertial Navigation Systems", IPIN 2010.
4. J. Wahlström, I. Skog, "Fifteen Years of Progress at Zero Velocity: A Review",
   arXiv:2008.09208, 2020.（综述：GLRT 统一框架、自适应阈值、更新策略分类）
5. P. D. Groves, *Principles of GNSS, Inertial, and Multisensor Integrated
   Navigation Systems*, 2nd ed.（ZUPT 的教科书式误差状态模型）
6. H. Ye, Y. Chen, M. Liu, "Tightly Coupled 3D Lidar Inertial Odometry and
   Mapping" (LIO-Mapping), ICRA 2019.
7. P. Babin et al., "Gravity-constrained Point Cloud Registration",
   arXiv:2203.13799, 2022.（4DOF ICP）
8. T. Qin, S. Shen, "VINS-Mono", IEEE T-RO 2018.（4DOF 位姿图）

**权威开源实现**

1. OpenShoe MATLAB Implementation（Skog/Nilsson，本文检测器公式的来源）：
   `github.com/bijingxue/ZUPT-aided-INS`（`src/GLRT.m` 等）
2. `github.com/meyiao/ImuFusion`（MATLAB `zupt.m`，足绑式 EKF-ZUPT）
3. `github.com/Fixit-Davide/imu_zupt`（ROS 2 C++ ZUPT，IMU 级零速修正）
4. libpointmatcher（Norlab，`force4DOF` 重力约束 ICP）

---

## 8. 验证

`scripts/zupt_check.py`（Python 3 + numpy，独立于 ROS）包含三项检查，
检测器与更新公式和 `zupt.hpp` 一一对应：

1. 合成 IMU（静止 3 s → 运动 3 s → 静止 3 s，200 Hz）上的检测器分类：
   运动段零误触发，静止段零漏检，进入静止延迟 ≈ 0.16 s、退出延迟 ≈ 5 ms；
2. 重力残差 δθ 雅可比的解析 vs 中心差分数值验证（误差 < 1e-6）；
3. Joseph 形式更新的收敛性与协方差正定性（速度误差 → 4.5 mm/s，
   零偏误差 → 0.004 rad/s）。

运行：`python scripts/zupt_check.py`。

**真机验收建议**：静态放置 10 分钟看位置/速度漂移对比（开/关 ZUPT），
再跑"行驶-停车-行驶"序列，用 `/zupt_status` 核对停车段检测标志，
确认行驶段无误触发；最后在长走廊/开阔地（LiDAR 退化场景）验证 ZUPT 托底效果。

---

## 9. 后续可扩展方向

- **IMU 频率 ZUPT**：当前按帧（10 Hz）更新；对行人式高频动静切换可在 IMU
  回调内直接更新（接口已按此设计，把 `apply_update` 挪到 `imu_cbk` 即可）；
- **非零速更新（non-zero velocity update）**：轮速计/角速度杠杆臂估计真实速度
  作为量测，代替强制零速（综述 §VIII 的方向）；
- **自适应阈值**：γ 随运动状态/速度自适应（综述指出混合工况下收益 >50%）；
- **4DOF 深度集成**：配合 `planar_constraint` 或后端 4DOF 位姿图
  （LIO-Mapping 风格 rotation-constrained refinement）；
- **检测器学习化**：综述提到 SVM/LSTM/对比学习检测器，适合步态极端复杂的场景。
