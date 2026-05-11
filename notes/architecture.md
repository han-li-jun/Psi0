# Psi0 项目架构文档

> 整理时间：2026-05-11  
> 基于代码版本：commit e100324

---

## 一、整体系统架构（三层）

```
┌─────────────────────────────────────────────────────────────────┐
│                        Ψ₀ System                               │
│                                                                 │
│   System-2             System-1              System-0           │
│  ┌──────────┐        ┌──────────────┐      ┌──────────────┐    │
│  │ Qwen3VL  │──────▶│  扩散变换器  │────▶│  RL 轨迹控   │    │
│  │  (VLM)   │  特征  │   (~500M)   │ 动作  │  制器 (WBC) │    │
│  │  2B参数  │        │ Flow Match   │ 块   │  IK+力矩    │    │
│  └──────────┘        └──────────────┘      └──────────────┘    │
│  语义理解               行为预测               物理执行           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、代码目录结构与职责

```
Psi0-main/
│
├── src/psi/                          ── 核心库
│   ├── models/psi0.py                   Psi0Model（完整VLA模型）
│   ├── trainers/
│   │   ├── finetune.py                  FinetuneTrainer（微调）
│   │   ├── pretrain.py                  PretrainTrainer（预训练）
│   │   └── posttrain.py                 PosttrainTrainer（后训练）
│   ├── config/
│   │   ├── model_psi0.py                模型超参数配置
│   │   ├── transform.py                 数据变换管道
│   │   └── train/                       各阶段训练配置
│   │       ├── finetune_real_psi0_config.py
│   │       └── pretrain_egodex_psi0_config.py
│   ├── data/
│   │   ├── dataset.py                   Dataset / MixtureDataset
│   │   └── lerobot/compat.py            LeRobot数据格式适配
│   └── deploy/
│       └── psi0_serve_simple.py         FastAPI推理服务
│
├── scripts/                          ── 入口脚本
│   ├── train.py                         ← 训练主入口
│   ├── train/psi0/
│   │   ├── pretrain-egodex-psi0-fast.sh    预训练（200K步）
│   │   └── finetune-real-psi0.sh           微调（真实数据）
│   ├── data/
│   │   ├── raw_to_lerobot.py            原始数据→LeRobot格式
│   │   └── calc_modality_stats.py       计算归一化统计量
│   └── deploy/serve_psi0-rtc.sh         启动推理服务
│
├── real/                             ── 真实世界接口
│   ├── teleop/                          遥操作（数据采集）
│   │   ├── main.py                      遥操作主程序
│   │   ├── vr.py / vr_pico.py           Apple VisionPro/PICO支持
│   │   └── robot_control/               IK求解+力矩计算
│   └── deploy/
│       ├── psi-inference_rtc.py         机器人端推理客户端
│       └── psi-inference.py             标准推理客户端
│
├── baselines/                        ── 对比基线
│   ├── act/                             ACT
│   ├── dp/                              Diffusion Policy
│   ├── gr00t-n1.6/                      GR00T N1.6
│   └── pi05/                            OpenPI π₀.5
│
└── third_party/SIMPLE/               ── 仿真环境
    └── src/simple/
        ├── envs/                        Gym仿真环境
        ├── tasks/                       任务定义（抓取/放置/移动）
        ├── engines/                     MuJoCo / Isaac Sim后端
        ├── datagen/                     CuRobo生成仿真数据
        └── cli/eval.py                  评估入口
```

---

## 三、核心模型架构（Psi0Model）

```
输入
 ├─ 图像 (RGB)
 ├─ 文本指令 ("hold the lunch box")
 ├─ 当前状态 (32维关节角)
 └─ 噪声动作 [B, 6, 7]

         │
         ▼
┌─────────────────────┐
│      Qwen3VL        │  Vision Encoder + Language Model
│   (冻结 or 微调)    │  输出: [B, S, 1920] 视觉-语言特征
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  ObservationProj    │  1920 → 1536 维投影
│  + 状态编码 (15维)  │  视图融合策略: concat / cross / perceiver
└─────────────────────┘
         │ 观测特征 [B, S, 1536]
         │
         ▼
┌─────────────────────┐       时间步 t
│  ActionProjectionIn │◀─── (扩散时间嵌入 256维→1536维)
│  噪声动作编码        │
└─────────────────────┘
         │ 动作特征 [B, 6, 1536]
         │
         ▼
┌─────────────────────────────────────────┐
│         6 × VLATransformerBlock         │
│  ┌─────────────────────────────────┐    │
│  │  AdaLayerNorm (t 条件化)         │    │
│  │  Self-Attention (动作内部)       │    │
│  │  Cross-Attention (动作←观测)     │    │
│  │  FeedForward (GELU, dim=2048)   │    │
│  └─────────────────────────────────┘    │
│  heads=24, head_dim=64, hidden=1536     │
└─────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│  ActionProjectionOut│  1536 → 7 维
│  AdaLN + Linear     │
└─────────────────────┘
         │
         ▼
输出: 预测噪声 [B, 6, 7]
  action = (x, y, z, rx, ry, rz, gripper)
```

**关键超参数**

| 参数 | 值 |
|------|----|
| hidden_dim | 1536 |
| num_attention_heads | 24 (head_dim=64) |
| num_blocks | 6 |
| dim_feedforward | 2048 |
| action_dim | 7 (x,y,z,rx,ry,rz,gripper) |
| action_chunk_size | 6 |
| diffusion_steps | 1000（训练），10（推理） |

---

## 四、完整数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│  1. 数据采集                                                          │
│                                                                      │
│  VR手柄/PICO ──▶ teleop/main.py ──▶ IK求解 ──▶ 宇树G1机器人执行      │
│                        │                                            │
│                        └──▶ 录制: color/*.jpg + data.json           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  2. 数据预处理                                                        │
│                                                                      │
│  raw_to_lerobot.py                                                   │
│  ├─ 手部关键点 → IK → 关节角度                                       │
│  ├─ 动作块化 (action_chunk_size=6)                                   │
│  └─ 输出 LeRobot 格式:                                               │
│     ├─ observation.images.egocentric  [T, H, W, 3]                  │
│     ├─ observation.states             [T, 32]                        │
│     └─ action                         [T, 7]                        │
│                                                                      │
│  calc_modality_stats.py → stats.json (mean/std 归一化参数)           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  3. 训练                                                             │
│                                                                      │
│  scripts/train.py                                                    │
│  ├─ Transform 管道:                                                  │
│  │   RepackTransform → FieldTransform → ModelTransform              │
│  │   (动作块重组)       (归一化)         (VLM预处理)                 │
│  │                                                                  │
│  ├─ Accelerator (DDP / DeepSpeed Zero-3)                            │
│  ├─ AdamW, cosine lr, warmup=5%                                     │
│  ├─ 损失 = MSE(预测噪声, 真实噪声)                                   │
│  │         加权: xyz×0.1, rpy×0.2, gripper×0.1                     │
│  └─ 保存: .runs/finetune/{run_name}/checkpoints/ckpt_XXXXX/        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  4. 部署推理                                                          │
│                                                                      │
│  [服务端]  psi0_serve_simple.py                                      │
│            FastAPI (port=22085)                                      │
│            迭代去噪 10步 → 输出动作块 [6, 7]                          │
│                  ▲                                                   │
│                  │ HTTP                                              │
│  [机器人端] psi-inference_rtc.py                                     │
│            ├─ ZMQ 接收 RealSense 图像                                │
│            ├─ 发送图像+状态 → 服务端                                  │
│            ├─ 接收动作块 [6, 7]                                      │
│            ├─ compute_tau() IK反演 → 关节力矩                        │
│            └─ Unitree DDS → 机器人执行  (60Hz)                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ 可选
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│  5. SIMPLE 仿真评估                                                   │
│                                                                      │
│  third_party/SIMPLE/                                                 │
│  ├─ datagen/ → CuRobo 运动规划生成仿真数据                            │
│  ├─ engines/ → MuJoCo / Isaac Sim 渲染                               │
│  ├─ eval.py  → 开环评估 (MP任务)                                     │
│  └─ eval_decoupled_wbc.py → 闭环评估 (WBC追踪)                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 五、训练流程详解

```python
# scripts/train.py 执行流程
train(config: LaunchConfig)
├─ 1. 动态实例化 Trainer
│     finetune_real_psi0 → FinetuneTrainer
│
├─ 2. init_models()
│     ├─ 加载 Qwen3VL 预训练权重 (flash_attention_2, bfloat16)
│     ├─ 初始化 ActionHeader
│     └─ 可选加载预训练 ActionHeader (action_header.safetensors)
│
├─ 3. 初始化 Accelerator
│     支持: DDP / FSDP / DeepSpeed Zero-3
│     混合精度: bf16
│
├─ 4. 数据加载
│     LeRobotDataConfig → Dataset → DataLoader
│     workers=12, PaddedCollator 对齐序列
│
├─ 5. 优化器
│     AdamW, betas=(0.9, 0.95)
│     cosine lr + warmup(5%) + min_lr=5e-7
│
├─ 6. 训练循环
│     每步: forward → MSE loss → backward → grad_clip → step
│     每 100 步: 日志
│     每  50 步: 验证（推理+L1误差）
│     每5000 步: 保存 checkpoint
│
└─ 7. 推理（验证时）
      x_T (纯噪声) → 10步去噪 → x_0 (预测动作)
      FlowMatchEulerDiscreteScheduler
```

**FinetuneTrainer 损失权重**

```
loss = MSE(pred, target) * [0.1, 0.1, 0.1,   # xyz
                             0.2, 0.2, 0.2,   # rpy
                             0.1]             # gripper
```

---

## 六、第三方依赖说明

| 依赖 | 位置 | 作用 |
|------|------|------|
| `unitree_sdk2_python` | third_party/SIMPLE/third_party | 宇树G1机器人DDS通信 |
| `decoupled_wbc` | third_party/SIMPLE/third_party | Whole Body Controller（全身控制） |
| `curobo` | third_party/SIMPLE/third_party | GPU加速运动规划+IK求解 |
| `gsnet` | third_party/SIMPLE/third_party | 抓取点检测网络 |
| `openpi-client` | third_party/SIMPLE/third_party | π₀.5基线的通信客户端 |
| `lerobot` | lerobot_src | 机器人数据集标准格式 |
| `AMO` | third_party/SIMPLE/third_party | 手部动作适配模型 |
| `gear_sonic` | third_party/SIMPLE/third_party | 音频/语音相关模块 |

**Python 核心依赖**

| 包 | 版本 | 用途 |
|----|------|------|
| torch | 2.7.0 | 深度学习框架 |
| transformers | 4.57.0 | Qwen3VL |
| diffusers | - | 扩散模型调度器 |
| flash_attn | 2.7.4 | 高效注意力 |
| accelerate | 1.7.0 | 多GPU训练 |
| deepspeed | 0.17.1 | Zero-3 显存优化 |
| lerobot | git@09929d8 | 数据格式 |
| wandb | ≥0.20.0 | 训练监控 |

---

## 七、基线模型对比

| 模型 | 目录 | 特点 |
|------|------|------|
| **ACT** | baselines/act | Action Chunking Transformers |
| **Diffusion Policy** | baselines/dp | UNet扩散策略 |
| **GR00T N1.6** | baselines/gr00t-n1.6 | NVIDIA大规模预训练 |
| **OpenPI π₀.5** | baselines/pi05 | JAX/PyTorch双后端 |
| **H-RDT** | baselines/h-rdt | 人类-机器人扩散变换器 |
| **EgoVLA** | baselines/egovla | 第一视角VLA |
| **InternVLA-M1** | baselines/internvla-m1 | 多模态拦截器 |

---

## 八、读代码建议路线

**第一条线（理解训练）**
```
scripts/train.py
  → src/psi/trainers/finetune.py
    → src/psi/models/psi0.py
      → src/psi/config/model_psi0.py
```

**第二条线（理解数据）**
```
scripts/data/raw_to_lerobot.py
  → src/psi/config/transform.py
    → src/psi/data/dataset.py
      → src/psi/data/lerobot/compat.py
```

**第三条线（理解部署）**
```
scripts/deploy/serve_psi0-rtc.sh
  → src/psi/deploy/psi0_serve_simple.py
    → real/deploy/psi-inference_rtc.py
      → real/teleop/robot_control/
```
