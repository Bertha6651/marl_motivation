**文档概述：Phase 1 实验的完整可执行规格说明书。面向实际代码实现和实验执行，覆盖训练流程、配置、参数、监控、评估和分析的全部细节。

**基于**：Flow-GRPO 论文 + AgentFlow 代码库（`train/config.yaml`, `train/rollout.py`, `agentflow/solver.py`）+ verl 框架文档 + Agent RL 最佳实践（2025）

---

## 一、实验概述

### 1.1 实验目标

验证：将 Flow-GRPO 从单角色（仅 Planner）全参数 RL 扩展为**三角色（Planner + Executor + Verifier）LoRA RL**，在仅使用结果奖励（无过程奖励）的条件下，训练是否稳定、是否有效、以及各组件的贡献如何。

### 1.2 实验条件

| 条件                           | 代码            | 训练配置                                    |
| ---------------------------- | ------------- | --------------------------------------- |
| B0: Frozen Baseline          | `B0_frozen`   | 全部组件冻结，Qwen2.5-7B-Instruct，无 RL         |
| B1: Flow-GRPO (original)     | `B1_flowgrpo` | 仅 Planner 全参数 GRPO（引用论文值）               |
| **Ours: Multi-Role LoRA RL** | `MR_LoRA_RL`  | Planner + Executor + Verifier LoRA GRPO |

### 1.3 核心指标

- **训练稳定性**：无 NaN loss、无 OOM、KL divergence < 0.5，针对稳定性后文有详细探讨
- **端到端性能**：目标 benchmark 准确率（尤其 GAIA、搜索类）
- **组件贡献**：消融实验量化每组件增量

---

## 二、训练数据

### 2.1 推荐数据集

基于训练集调研（文档 23），推荐使用：

**方案 A（极简）**：~3,000 samples
- LIMR-1.4K（1,389 条数学）+ NQ subset（1,500 条搜索）


**方案 B（推荐）**：~10,000 samples
- DAPO-Math-17k subset（5,000 条）+ NQ + HotpotQA subset（5,000 条）

**数据格式要求**：`{question: str, result/answer: str, source: str}`--与flow-grpo对其

<font color="#ff0000">【todo】</font>

### 2.2 验证与评估数据

| 用途    | 数据集          | 数量         |
| ----- | ------------ | ---------- |
| 训练中验证 | AIME24       | 30 题       |
| 最终评估  | 4 benchmarks | motivation |
|       |              |            |
<font color="#ff0000">【todo】</font>

---

## 三、参数架构

### 3.1 总体架构

```
Qwen2.5-7B-Instruct (Base Model, BF16, Frozen)
  │  ← FSDP2 分片到 2 GPU，每 GPU ~7GB
  │
  ├── LoRA_Planner: rank=64, alpha=128, all-linear
  │    lr=1e-6, β=0.001, temp=0.5
  │    trainable ≈ 0.4B → ~40MB storage
  │
  ├── LoRA_Executor: rank=64, alpha=128, all-linear
  │    lr=1e-6, β=0.001, temp=0.5
  │    trainable ≈ 0.4B → ~40MB storage
  │
  └── LoRA_Verifier: rank=64, alpha=128, all-linear
       lr=1e-6, β=0.001, temp=0.5
       trainable ≈ 0.4B → ~40MB storage

Generator: Frozen ( Qwen2.5-7B-Instruct) 
总可训练参数: ~1.2B / 7B 
```

### 3.2 初始化

- LoRA A 矩阵：kaiming uniform 随机初始化
- LoRA B 矩阵：**零矩阵** → ΔW = 0 → 初始行为 = Base Model（冷启动）
- 无 SFT warmup（Flow-GRPO 论文已证明 SFT 导致 -19.0% 崩溃）



---

## 四、训练配置

### 4.1 基础环境

| 配置项                 | 值                                                                         | 说明                                             |
| ------------------- | ------------------------------------------------------------------------- | ---------------------------------------------- |
| Base Model          | `Qwen/Qwen2.5-7B-Instruct`                                                | HuggingFace model ID                           |
| 训练框架                | verl (≥0.4.1)                                                             | `python -m agentflow.verl`                     |
| Rollout 引擎          | vLLM (≥0.5.0)                                                             | `actor_rollout_ref.rollout.name: vllm`         |
| 分布式策略               | FSDP2                                                                     | `actor_rollout_ref.actor.strategy: fsdp2`      |
| 硬件                  | 4 × A100 80GB                                                             | GPU 0-1: FSDP2 training; GPU 2-3: vLLM rollout |
| 工具集                 | Base Generator, Python Coder, Google Search, Wikipedia Search, Web Search | 5 tools                                        |
| LLM Judge (Outcome) | GPT-4o                                                                    | 判定最终答案正确性（0/1）                                 |

### 4.2 核心超参数

#### 4.2.1 训练循环参数

| 参数                  | verl 配置路径                     | 值                                                              | 说明                                    |
| ------------------- | ----------------------------- | -------------------------------------------------------------- | ------------------------------------- |
| train_batch_size    | `data.train_batch_size`       | 32                                                             | 每 step 采样 32 个 query                  |
| rollout.n           | `actor_rollout_ref.rollout.n` | 8<font color="#ff0000">【todo】</font>如果减小一点可能5也比较合适，或者还是就8就ok了？ | 每 query 生成 8 条 trajectory（GRPO group） |
| total_epochs        | `trainer.total_epochs`        | 2<font color="#ff0000">【todo】</font> 或者可以先训一个epoch观察一下         | Motivation 实验使用 2 epoch               |
| max_turns           | `TOOL_STEPS`                  | 10                                                             | 每 trajectory 最大轮数                     |
| max_response_length | `data.max_response_length`    | 2048                                                           | Planner action 最大 token 数             |
| max_prompt_length   | `data.max_prompt_length`      | 18432                                                          | 输入 context 最大 token 数                 |

#### 4.2.2 更新参数

| 参数                           | verl 配置路径                            | 值              | 说明                                 |
| ---------------------------- | ------------------------------------ | -------------- | ---------------------------------- |
| ppo_mini_batch_size          | `actor.ppo_mini_batch_size`          | 8              | 每次 optimizer.step() 的 trajectory 数 |
| ppo_micro_batch_size_per_gpu | `actor.ppo_micro_batch_size_per_gpu` | 4              | 纯内存控制，不影响算法                        |
| ppo_epochs                   | —                                    | 2【todo】和上面保持同步 | 每批数据重复更新次数（GRPO 使用 2）              |
| clip_ratio_low               | `actor.clip_ratio_low`               | 0.2            | PPO clip 下界                        |
| clip_ratio_high              | `actor.clip_ratio_high`              | 0.3            | PPO clip 上界                        |
| adv_estimator                | `algorithm.adv_estimator`            | grpo           | GRPO 不使用 critic                    |
| loss_agg_mode                | `actor.loss_agg_mode`                | token-mean     | verl 推荐，long-CoT 稳定                |

#### 4.2.3 Per-Component 参数

| 参数                | Planner    | Executor   | Verifier   | 说明                           |
| ----------------- | ---------- | ---------- | ---------- | ---------------------------- |
| LoRA rank         | 64         | 64         | 64         |                              |
| LoRA alpha        | 128        | 128        | 128        |                              |
| target_modules    | all-linear | all-linear | all-linear |                              |
| learning_rate     | 1e-6       | 1e-6       | 1e-6       |                              |
| kl_loss_coef (β)  | 0.001      | 0.001      | 0.001      | 统一起步，分别监控                    |
| train_temperature | 0.5        | 0.5        | 0.5        | Planner 与 Flow-GRPO 一致（§4.1） |

### 4.3 训练量计算

| 方案    | 训练样本   | Steps/Epoch | 总 Steps |
| ----- | ------ | ----------- | ------- |
| A（极简） | 3,000  | 94          | 94      |
| B（推荐） | 10,000 | 313         | 313     |
epoch 如果改变那么总steps还要乘倍数

```
每 step: 32 query × 8 rollout = 256 trajectory
每 step 更新: 256 / 8 = 32 次 optimizer.step()
总 trajectory (A): 94 × 256 = 24,064 条
总 trajectory (B): 313 × 256 = 80,128 条
```

---

## 五、训练流程

### 5.1 完整 Training Step 流程

```
TRAINING STEP #K
═══════════════════════════════════════════════════════════

Phase 1: ROLLOUT (vLLM GPU 2-3)
  1. 采样 train_batch_size=32 个 query
  2. 对每个 query_i:
     for j in 1..8:  # rollout.n
       使用 LoRA_Planner   → Planner 生成 next_step
       使用 LoRA_Executor  → Executor 生成 tool_command → 工具执行
       使用 LoRA_Verifier  → Verifier 生成 verification
       if STOP or step_count ≥ max_turns: break
     收集 trajectory_i_j (含 token spans)
   → 产出: 256 条 trajectory

Phase 2: REWARD (GPT-4o Judge API)
  3. 对每条 trajectory:
     answer = extract_final_answer(traj.direct_output)
     R = GPT4o_Judge(query, answer, ground_truth) ∈ {0,1}
     traj.reward = R

Phase 3: ADVANTAGE
  4. 对每个 GRPO group (同一 query 的 8 条 trajectory):
     rewards = [traj_1.R, ..., traj_8.R]
     A_j = (R_j - mean(rewards)) / (std(rewards) + 1e-8)
     → 组内所有 tokens 共享 A_j

Phase 4: POLICY UPDATE (FSDP2 GPU 0-1)
  5. 切分 256 trajectory → 32 mini-batch (每组 8 条)
  6. 对每个 mini_batch:
     for component in [Planner, Executor, Verifier]:
       提取 component_tokens + 对应 advantage
       切分 micro_batch (每组 4, 2 micro-batch 梯度累积)
       对每个 micro_batch:
         使用对应 LoRA forward → log_probs_current
         计算 GRPO loss:
           ratio = exp(log_p_current - log_p_old)
           L_GRPO = -mean(min(ratio*A, clip(ratio)*A))
           L_total = L_GRPO - β * KL(LoRA_c || ref_c)
         loss.backward()  # 梯度累积
       累积完成后: optimizer_c.step(); optimizer_c.zero_grad()
  
  7. Weight Sync: FSDP2 → vLLM (更新后的 3 个 LoRA adapter)
```

### 5.2 训练循环伪代码

```python
# verl 训练主循环 (verl/workers/actor/dp_actor.py)
for epoch in range(total_epochs):
    for batch_idx, batch_data in enumerate(train_dataloader):
        # Phase 1-2: Rollout + Reward (vLLM)
        trajectories = rollout_engine.generate(
            queries=batch_data.queries,
            lora_adapters={
                "planner": lora_P_current,
                "executor": lora_E_current,
                "verifier": lora_V_current,
            },
            n=8, max_turns=10
        )
        rewards = judge_engine.evaluate(trajectories)
        
        # Phase 3: Advantage
        advantages = compute_group_normalized_advantages(
            rewards, group_size=8
        )
        
        # Phase 4: Policy Update (FSDP2)
        for component in ["planner", "executor", "verifier"]:
            optimizer = optimizers[component]
            lora = lora_adapters[component]
            
            for mini_batch in split(trajectories, mini_batch_size=8):
                optimizer.zero_grad()
                for micro_batch in split(mini_batch, micro_batch_size=4):
                    with lora.active():
                        log_probs = base_model(micro_batch.input_ids)
                    loss = grpo_loss(
                        log_probs, micro_batch.log_probs_old,
                        micro_batch.advantages, micro_batch.component_mask
                    )
                    loss -= beta[component] * compute_kl(lora, ref_lora)
                    loss.backward()  # accumulate gradients
                optimizer.step()
        
        # Weight sync to vLLM
        for component in ["planner", "executor", "verifier"]:
            sync_lora_to_vllm(lora_adapters[component], component)
        
        # Logging
        if batch_idx % log_freq == 0:
            log_metrics(training_metrics)
```

### 5.3 验证流程

```
每 save_freq 步（或每个 epoch 结束时）:
  1. 加载当前 LoRA checkpoint
  2. 在 AIME24 (30题) 上运行推理（温度=0.0, max_turns=10）
  3. 计算准确率
  4. 记录 validation metrics
```

---

## 六、过程监控指标体系

所有指标均可从 verl 训练日志和 rollout 数据中提取。

### 6.1 训练健康指标（每 training step 记录）

| ID      | 指标                     | 计算方式                                  | 数据来源               | 异常阈值                   | 服务目标               |
| ------- | ---------------------- | ------------------------------------- | ------------------ | ---------------------- | ------------------ |
| **H01** | Avg Outcome Reward     | `mean(R_i)` across 256 trajectories   | verl reward buffer | 连续 50 step 无上升趋势       | 核心训练健康             |
| **H02** | Group Reward Std       | `mean(std(R_group))` across 32 groups | verl reward buffer | 连续 50 step < 0.05      | 训练多样性：过低→无区分度      |
| **H03** | Planner KL Divergence  | `D_KL(π_θP\|π_ref)`                   | verl actor         | > 0.5：策略漂移；< 0.001：无学习 | Per-component 学习速度 |
| **H04** | Executor KL Divergence | 同上 (Executor)                         | verl actor         | 同 H03                  | 同上                 |
| **H05** | Verifier KL Divergence | 同上 (Verifier)                         | verl actor         | 同 H03                  | 同上                 |
| **H06** | Component KL Ratio     | `max(H03,H04,H05) / min(H03,H04,H05)` | derived            | > 10                   | H1 前兆—某组件学习极端分化    |
| **H07** | Planner GRPO Loss      | batch 内 Planner tokens 的 GRPO loss 均值 | verl actor         | NaN/Inf                | Per-component 损失监控 |
| **H08** | Executor GRPO Loss     | 同上 (Executor)                         | verl actor         | NaN/Inf                | 同上                 |
| **H09** | Verifier GRPO Loss     | 同上 (Verifier)                         | verl actor         | NaN/Inf                | 同上                 |

### 6.2 组件质量指标（每 training step 记录）

| ID      | 指标                            | 计算方式                                    | 数据来源                        | 服务目标                    |
| ------- | ----------------------------- | --------------------------------------- | --------------------------- | ----------------------- |
| **Q01** | Tool Call Error Rate          | `P(tool execution returns error)`       | rollout 中的 execution result | Flow-GRPO Figure 6 对应指标 |
| **Q02** | Tool Selection Distribution   | `Cat(5)` 各工具被 Planner 选择的频率             | rollout 中的 tool_name        | Flow-GRPO Figure 5 对应   |
| **Q03** | Avg Trajectory Length         | batch 内 trajectory 的平均 turn 数           | rollout 数据                  | Verifier STOP 行为变化      |
| **Q04** | Verifier STOP Rate            | `P(STOP)` per turn                      | rollout 数据                  | 监控是否过频繁/过少 STOP         |
| **Q05** | Tool Execution Success Rate   | `P(tool returns valid resultl called）)` | rollout 数据                  | Executor 输出质量           |
| **Q06** | Per-Component Avg Token Count | 每组件每 step 平均输出 token 数                  | rollout + token span        | 输出长度趋势（是否崩溃为极简输出）       |

### 6.3 评估指标（每 epoch 或每 save_freq 记录）

| ID          | 指标                       | 计算方式                  | 数据来源                     | 服务目标        |
| ----------- | ------------------------ | --------------------- | ------------------------ | ----------- |
| **E01-E10** | Per-Benchmark Accuracy   | 4 benchmarks 各自准确率    | eval loop + GPT-4o Judge | 核心性能        |
| **E11**     | Weighted Avg Accuracy    | 按 benchmark 样本量加权     | derived                  | 全局性能摘要      |
| **E12**     | Ablation Gain (Planner)  | `Acc(Full) - Acc(-P)` | ablation eval            | Planner 贡献  |
| **E13**     | Ablation Gain (Executor) | `Acc(Full) - Acc(-E)` | ablation eval            | Executor 贡献 |
| **E14**     | Ablation Gain (Verifier) | `Acc(Full) - Acc(-V)` | ablation eval            | Verifier 贡献 |

### 6.4 人工分析指标（训练后抽样）

| ID | 指标 | 方法 | 样本量 | 服务目标 |
|----|------|------|--------|---------|
| **A01** | Failure Root Cause Distribution | 人工标注 30 条失败 trajectory 的根因组件 | 30 | H3 前兆 |
| **A02** | Verifier STOP Timing Accuracy | 人工判定 Verifier STOP 是否在"恰当时机" | 50 | Verifier 训练质量 |
| **A03** | Per-Component Output Quality (Human) | 人工 1-5 Likert 评分 | 50 × 3 | 各组件输出质量基线 |

### 6.5 指标记录频率和存储

| 频率                  | 指标                              | 存储方式                                                               |
| ------------------- | ------------------------------- | ------------------------------------------------------------------ |
| 每 training step     | H01-H09, Q01-Q06                | TensorBoard / WandB（verl 默认 `trainer.logger: ['console','wandb']`） |
| 每 epoch / save_freq | E01-E14                         | 同上 + CSV 本地备份                                                      |
| 训练后                 | A01-A03                         | CSV（人工标注记录）                                                        |
| 每 training step     | 完整 rollout 数据（含 token span）     | Parquet（事后分析）                                                      |
| 每 save_freq         | LoRA checkpoint（3 个 adapter 文件） | safetensors                                                        |

---

## 七、评估协议

### 7.1 自动化评估

- **Benchmark 列表**（从agent-flow中找选择几个代表的作为测试）：
  - 搜索密集型：HotpotQA（总7400->可以随机选一部分）
  - Agentic 推理：GAIA (textual split)（165）
  - 数学推理：AIME2024（30）
  - 科学推理：GPQA（448）->覆盖专业领域知识

| 能力               | GAIA | HotpotQA | AIME2024        | GPQA |
| ---------------- | ---- | -------- | --------------- | ---- |
| 单跳搜索             | ✅    | ✅        | —               | ✅    |
| 多跳推理             | ✅    | ✅        | —               | —    |
| 工具选择             | ✅    | ✅        | ✅（Python Coder） | ✅    |
| Verifier STOP 判断 | ✅    | ✅        | ✅               | ✅    |
| 数学计算             | —    | —        | ✅               | —    |
| 科学知识             | ✅    | —        | —               | ✅    |
<font color="#ff0000">【todo】</font>选择这四个benchmark怎么样，需不需要再减少一点

- **判定方式**：
  - 有 ground-truth answer → exact match 或 GPT-4o Judge（与 Flow-GRPO 一致）
  - 每个 benchmark 3 次独立运行，报告均值 ± 标准差
    - 简单一点，每个benchmark运行一遍就可以


### 7.2 消融评估

基于 vLLM per-request LoRA 选择机制（文档 11 §2.2），评估时选择性不加载某组件 LoRA：

| 评估条件 | Planner | Executor | Verifier | 回答的问题 |
|---------|---------|----------|----------|-----------|
| Full | LoRA | LoRA | LoRA | 全配置性能 |
| -P | **Base** | LoRA | LoRA | Planner LoRA 贡献 |
| -E | LoRA | **Base** | LoRA | Executor LoRA 贡献 |
| -V | LoRA | LoRA | **Base** | Verifier LoRA 贡献 |
| Base | **Base** | **Base** | **Base** | B0 基线（全部冻结） |

**注意**：由于分布偏移（文档 11 §2.4），移除某组件 LoRA 后的性能是下界估计。

### 7.3 人工评估（训练后）

1. **失败轨迹归因**：随机抽样 30 条失败 trajectory，人工标注：
   - 失败根源组件（Planner/Executor/Verifier/外部因素/复合）
   - 该组件的具体失败类型（工具选择错误/参数错误/STOP时机错误等）

2. **组件输出质量**：随机抽样 50 条 trajectory（成功+失败各半），人工对每组件的输出质量进行 1-5 Likert 评分

3. **Verifier STOP 时机评估**：从上述 50 条中，人工判定 Verifier 的 STOP 时机是否恰当

---

## 八、分析方案

### 8.1 训练稳定性分析

#### 1  稳定性评判的三级指标体系

基于文献和工程最佳实践，建立三级评判：

**Level 1**：崩溃信号（任何一项触发 → 立即中断训练）

| 指标                   | 检测方式                          | 阈值         | 来源文献                                                              |
| -------------------- | ----------------------------- | ---------- | ----------------------------------------------------------------- |
| **NaN/Inf Loss**     | `actor_loss` 值检查              | 连续 3 step  | verl logging                                                      |
| **KL 爆炸**            | `kl_divergence` per component | > 1.0      | Post-Training Instrument Cluster (2025): "KL > 1.0 = stop signal" |
| **Gradient Norm 爆炸** | `grad_norm` (verl DEBUG log)  | > 10.0     | Axolotl stability guide                                           |
| **Reward Std = 0**   | `reward_std` across groups    | 连续 10 step | Axolotl: "always 0 → no learning signal"                          |

**Level 2**：预警信号（触发 → 记录警告，考虑干预）

| 指标                   | 检测方式                    | 预警阈值               | 干预措施               | 来源文献                                               |
| -------------------- | ----------------------- | ------------------ | ------------------ | -------------------------------------------------- |
| **KL 偏高**            | per-component KL        | > 0.15             | 提升 β 至 0.002       | verl blog: "> 0.08 warning, > 0.15 red alert"      |
| **KL 为 0**           | per-component KL        | 连续 50 step < 0.001 | 降低 β 至 0.0005      | 自主判定                                               |
| **Entropy 崩溃**       | `entropy` per component | < 0.01             | 提升温度 +0.1          | Axolotl: "< 0.01 → mode collapse"                  |
| **Clip Fraction 过高** | `clip_ratio`            | > 0.3              | 降低 LR 50%          | Axolotl: "> 0.3 → PPO clipping too aggressive"     |
| **零方差 Prompt 过高**    | `frac_reward_zero_std`  | > 0.8              | 增加 rollout.n 或分析数据 | GRESO (2025): effective prompt ratio drops to ~20% |
| **组件 KL 比 极端**       | `max(KL_c) / min(KL_c)` | > 10               | 检查某组件是否退化          | 自主判定                                               |

 **Level 3**：健康趋势信号

| 指标 | 健康范围 | 监控目的 | 来源文献 |
|------|---------|---------|---------|
| **Reward 趋势** | 上升或稳定 | 核心学习信号 | Flow-GRPO Figure 8a |
| **KL 黄金区间** | 0.001-0.08 | 策略在适度探索 | verl blog: "0.03-0.06 golden zone" |
| **Response Length 趋势** | 先上升后稳定 | Flow-GRPO 论文确认的正常模式 | Flow-GRPO §4.5: "after an initial exploratory rise, progressively shortens and stabilizes" |
| **Tool Call Error Rate** | 下降 | Executor 质量改善 | Flow-GRPO Figure 6 |
| **Reward Std** | > 0.1 | Group 内有区分度 | GRPO 机制需要 group variance |
| **Gradient Norm** | 0.001-1.0 | 梯度信号健康 | Axolotl stability guide |

#### 2 在 verl 中的可执行性

verl 框架的 METRIC JSONL 日志（`trainer.logger: ['console','wandb']`）已经记录了以下关键字段：

| verl 日志字段             | 对应指标              | 频率                    |
| --------------------- | ----------------- | --------------------- |
| `critic/rewards/mean` | Level 3 Reward    | per step              |
| `actor/kl_loss`       | Level 1/2 KL      | per step              |
| `actor/pg_loss`       | Level 2 Loss      | per step              |
| `actor/entropy_loss`  | Level 2 Entropy   | per step (if enabled) |
| `actor/grad_norm`     | Level 1 Grad Norm | per step (DEBUG mode) |
| `perf/mfu/actor`      | GPU 利用率           | per step              |

**需要额外启用的监控**：
- `actor_rollout_ref.actor.entropy_coeff` → 设为非零值（如 0.001）以启用 entropy logging
- verl 的 DEBUG 日志级别 → 启用 `grad_norm` 记录
- 自定义 metrics：per-component KL、tool call error rate、zero-variance prompt ratio → 通过 WandB callback 实现


### 8.2 端到端性能分析

- **指标**：E01-E11
- **对比**：B0 (Frozen) vs B1 (Flow-GRPO paper) vs Ours

### 8.3 组件贡献分析

- **指标**：E12-E14 (ablation gains)
- **分析**：各消融条件下的性能衰减量 = 该组件的边际贡献
- **关注**：Executor 贡献是否 > 0 → 证明 Executor 训练有效


---

## 九、代码修改清单-参考

### 9.1 新建文件

| 文件                             | 用途                |
| ------------------------------ | ----------------- |
| `train/lora_multi_config.yaml` | 多 LoRA 配置文件       |
| `train/rollout_multi_lora.py`  | 多 LoRA rollout 支持 |

### 9.2 修改现有文件

| 文件                                       | 修改内容                                                                           | 复杂度         |
| ---------------------------------------- | ------------------------------------------------------------------------------ | ----------- |
| `train/config.yaml`                      | 拆分 per-component LoRA 参数 (lr, rank, alpha, β, temp)                            | 低           |
| `agentflow/agentflow/engine/factory.py`  | `create_llm_engine()` 接受 `lora_config` 参数                                      | 低           |
| `agentflow/agentflow/models/planner.py`  | `__init__` 透传 `lora_config` 到 engine                                           | 低           |
| `agentflow/agentflow/models/executor.py` | 同上                                                                             | 低           |
| `agentflow/agentflow/models/verifier.py` | 同上                                                                             | 低           |
| `agentflow/agentflow/solver.py`          | `construct_solver()` 接受 `lora_configs`；`solve()` 记录 token span                 | 中           |
| `train/rollout.py`                       | `Rollout._maybe_init_agent()` 传递 LoRA 配置；`_solve_and_evaluate()` 记录 token span | 中           |
| verl `Actor.update_policy()`             | 支持多 LoRA 组的独立 loss 计算和更新                                                       | **高**（核心修改） |


---

## 十、风险管理

| 风险                    | 概率  | 缓解                                        |
| --------------------- | --- | ----------------------------------------- |
| verl 多 LoRA 实现复杂度超预期  | 中   | 备选：顺序训练（P→E→V）或先双角色（P+E）                  |
| 共享 advantage 导致信用噪声过大 | 中高  | 观察 A01 归因分析                               |
| 冷启动 LoRA 训练收敛慢        | 低中  | 1 epoch 应处于快速增长期（Predictive Scaling Laws） |
