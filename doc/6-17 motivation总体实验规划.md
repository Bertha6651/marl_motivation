
# 1 背景和问题
大语言模型（LLM）在工具增强推理领域取得了显著进展。以 DeepSeek-R1（Guo et al., 2025）和 SimpleRL（Zeng et al., 2025b）为代表的工作表明，通过强化学习从结果反馈中微调 LLM，可以使其展现出复杂的自纠错和多步推理能力。一个互补的研究方向——工具集成推理（Tool-Integrated Reasoning, TIR）——进一步赋予 LLM 调用外部工具（如网络搜索、代码执行）的能力（Jin et al., 2025; Chen et al., 2025; Feng et al., 2025）。

然而，TIR 方法存在一个架构限制：它们训练的是一个**单一的单体策略（monolithic policy）**，在多轮全上下文推理条件下交织思考与工具调用。Li et al.（2025）指出，这种范式面临两个缩放挑战：(i)随着视野的延长、工具多样性的增长以及环境随着工具反馈的变化,训练变得越来越不稳定(ii) 推理时对于未见过的任务或工具仍然很脆弱

**Agentic 系统范式**提供了一种替代方案。AutoGen、MetaGPT等框架将任务分解到多个专业化模块（planner, coder, critic），通过协作工作流解决复杂问题。然而,在此类系统中实现稳健的协调最终需要培训,因为手工逻辑或静态提示无法可靠地捕获模块何时以及如何协作、 适应不断变化的工具输出或从早期错误中恢复。与此同时,它们引入了新的训练挑战:模块顺序协调、结果反馈通过长推理链传播、状态分布随着工具输出的演变而变化。因此, 大多数系统仍然无需培训,依赖于手工逻辑或提示启发法。部分工作尝试了 SFT 或 preference optimization，但这些离线方法"are decoupled from live dynamics and learn poorly from downstream successes or failures"（Flow-GRPO 论文, §1）。


![[Pasted image 20260617185442.png]]


**AgentFlow 与 Flow-GRPO**在两类范式之间架起了桥梁。AgentFlow 是一个可训练的 in-the-flow agentic 框架，由四个专业化模块（Action Planner P、Tool Executor E、Execution Verifier V、Solution Generator G）通过共享演进记忆 M 和工具集 K 协调工作。其核心训练算法 Flow-GRPO 通过在多轮交互中**将 trajectory-level 的结果奖励广播到每一轮（turn）**，将长程信用分配问题转化为一系列可控的单轮策略更新。
![[Pasted image 20260618084757.png]]
![[Pasted image 20260617185704.png]]

然而，Flow-GRPO 的优化范围存在一个关键局限：**仅 Planner（P）是可训练的**，Executor（E）、Verifier（V）和 Generator（G）全部保持冻结。这意味着：

1. **Executor 能力瓶颈**：即使 Planner 制定了最优计划，Executor 可能因生成错误的工具调用命令或参数不匹配而执行失败
2. **Verifier 判断瓶颈**：Verifier 可能过早或过晚发出 STOP 信号，限制系统在恰当的时机终止
3. **组件间能力不匹配**：训练后的 Planner 可能显著强于 Executor 和 Verifier，形成新的系统瓶颈。同时由于能力不匹配，三个组件之间的沟通和理解也可能存在问题

**MoRAgent**提供了另一个重要视角。该工作将 Agent 任务能力分解为三个角色——Reasoner、Executor 和 Summarizer——并通过 Mixture-of-Roles（MoR）架构实现参数高效微调。MoRAgent 的核心洞察是：角色间的参数隔离有助于专业化学习并且可以通过对多角色进行隔离优化提升真题效果。然而，MoRAgent 使用的是 SFT 范式，其方法不能直接迁移到 RL 场景——事实上，Flow-GRPO 论文的消融实验明确显示，在 AgentFlow 场景下使用 SFT 会导致性能崩溃（-19.0%，Flow-GRPO 论文 Table 3）。

为了解决上述问题，我们提出了一个多角色agentRL框架，希望通过将单角色的flow-grpo扩展到多角色强化学习、以及为每个角色单独增加过程奖励模型全方位提升每个组件的能力，并在强化学习交互中提高各组件的协同能力。

为了验证上述方法的可行性，我们引入下面的motivation实验，我们将方法分为两个前后密切相关的两个层次，希望通过实验和分析实验结果验证上述方法的可行性/有效性，也希望在motivation实验中发现更多有意义的结果来改进目前的方法

具体来说：
1. 第一层次（面向结果奖励的多角色rl）的实验围绕 验证多角色flow-agent的可行性，主要是在flow-grpo上进行调整使其能够支持三个重要组件一起进行强化学习优化（planner,executer,verifier），而为了方便各组件联合训练以及训练的高效性，计划利用lora即插即拔、低参数的特点。具体来说，这一层次的任务主要是
	1. 验证将单组件训练扩展到多组件的可行性，主要围绕训练稳定性/训练之后的效果等方面进行研究
	2. 验证将flow-grpo中的全参数微调，变为高效参数微调，添加lora方式训练，是否会降低模型强化学习效果
	3. 给第二阶段提供保障和理论支持，如果面向结果奖励的方式都训练不稳定或者崩溃那么需要考虑如何进行算法适配了
2. 第二层次（加入prm的多角色rl）的实验 是在第一层次基础上，希望通过对每个组件引入prm过程奖励模型，希望通过增加step级别的细粒度奖励优化每个组件的性能。在这个过程中，每个组件都会根据各自面向任务的特点设计不同维度的评价体系，每一轮次中的每一组件都会得到属于自己的过程奖励，并于第一阶段的结果奖励进行组合，一起进入强化学习优化过程。这一阶段收集完相关数据之后主要是是验证下面几方面的内容
	1. **层次 1（性能验证）**：为每个组件引入独立的过程奖励信号能否进一步提升多角色 RL 的端到端性能？--通过改进信用分配
	2. **层次 2（Motivation 验证——核心）**：独立 PRM 的引入是否会触发以下四个假设性问题？

motivation：每个角色引入单独prm，可能会导致奖励不均匀/不平衡--motivation实验

| 编号  | 假设      | 核心关切                                                       |
| --- | ------- | ---------------------------------------------------------- |
| H1  | 训练组件分割  | 各组件是否因独立 PRM 而只关注局部过程奖励（短期），忽视全局协同（长期）和相邻组件配合？             |
| H2  | 奖励分配不匹配 | 不同组件的 PRM 评分维度/标准本身是否导致不公平的奖励分布（如 A 组件天然容易获高分，B 组件天然难获高分）？ |
| H3  | 信用错误归因  | PRM 能否正确识别每个组件的真实贡献/错误？还是会出现"Planner 的错误被归咎于 Executor"的情况？ |
| H4  | 奖励黑客    | 各组件是否出现"输出讨好 PRM 但无助于任务完成"的行为？                             |
下面会进入具体理论和实验细节环节，对于每个环节都通过两阶段递进的方式进行介绍。


# 2 理论
## 2.1 flow-grpo算法
### 2.1.1 AgentFlow 的 MDP 形式化

AgentFlow 的问题求解过程可描述为多轮 Markov Decision Process（MDP）。给定查询 $q$ 和工具集 $K$，在第 $t$ 轮：

- **状态**：$s^t = (q, K, M^t)$，其中 $M^t$ 是 Memory 在前 $t-1$ 轮累积的内容
- **Planner 动作**：$a^t \sim \pi_\theta(a^t | q, K, M^t)$，包含子目标、选定工具、上下文
- **Executor 执行**：$e^t \sim \mathcal{E}(e^t | a^t, K)$，调用工具并返回观察结果
- **Verifier 判断**：$v^t \sim \mathcal{V}(v^t | q, e^t, M^t)$，产出二元信号 $v^t \in \{0, 1\}$
- **Memory 更新**（若 $v^t = 0$）：$M^{t+1} = f_{\text{mem}}(M^t, a^t, e^t, v^t)$

过程在 $v^t = 1$（终止）或达到最大轮数 $T_{\max}$ 时结束。最终答案 $o$ 由 Generator 基于最终 Memory 生成：$o \sim \mathcal{G}(o | q, M^T)$。

![[Pasted image 20260617190826.png]]

### 2.1.2 Flow-GRPO 的优化目标

Flow-GRPO（Li et al., 2025, §3.2）优化 Planner 策略 $\pi_\theta$ 以最大化期望回报。其目标函数为：

$$\mathcal{J}_{\text{Flow-GRPO}}(\theta) = \mathbb{E}_{(q,y^*)\sim D, \{\tau_i\}_{i=1}^G\sim\pi_{\theta_{\text{old}}}}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}\frac{1}{|a_i^t|}\sum_{j=1}^{|a_i^t|}\min\left\{\rho_{i,j}^t A_i^t, \text{clip}(\rho_{i,j}^t, 1-\epsilon, 1+\epsilon) A_i^t\right\} - \beta D_{KL}(\pi_\theta \| \pi_{\text{ref}})\right]$$

其中：
- $\rho_{i,j}^t = \frac{\pi_\theta(a_{i,j}^t | s_i^t, a_{i,1:j-1}^t)}{\pi_{\theta_{\text{old}}}(a_{i,j}^t | s_i^t, a_{i,1:j-1}^t)}$ 为 token 级重要性比率
- $A_i^t = \frac{\bar{R}(o_i, q, y^*) - \text{mean}(\{\bar{R}(o_k, q, y^*)\}_{k=1}^G)}{\text{std}(\{\bar{R}(o_k, q, y^*)\}_{k=1}^G)}$ 为 group-normalized advantage
- $\bar{R}(o, q, y^*) \in \{0, 1\}$ 为 trajectory-level 结果奖励

ps 结果奖励 $\bar{R}$ 对同一 trajectory 内的所有轮数是恒定的


## 2.2 第一阶段
Flow-GRPO 仅优化 Planner（$\pi_\theta$），Executor $\mathcal{E}$、Verifier $\mathcal{V}$ 和 Generator $\mathcal{G}$ 均保持冻结。这对应优化目标中仅有一个可训练策略。本工作将此扩展为：

$$\mathcal{J}_{\text{Multi-Role}}(\theta_P, \theta_E, \theta_V) = \sum_{c \in \{P, E, V\}} \mathcal{J}_{\text{GRPO}}^{(c)}(\theta_c)$$

其中每个组件 $c$ 拥有独立的策略参数 $\theta_c$（通过独立的 LoRA adapter 实现），共享相同的 trajectory-level 结果奖励 $\bar{R}$ 和 group-normalized advantage $A_i^t$，但独立进行 KL 正则化和参数更新。


**奖励信号**：与 Flow-GRPO 完全一致
- **共享 Advantage：三个组件（Planner、Executor、Verifier）


对于每个组件 $c \in \{P, E, V\}$，GRPO 损失独立计算：

$$\mathcal{L}^{(c)}(\theta_c) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_i}\sum_{t=1}^{T_i}\frac{1}{|a_{i,c}^t|}\sum_{j=1}^{|a_{i,c}^t|}\min\left\{\rho_{i,j}^{t,c} A_i, \text{clip}(\rho_{i,j}^{t,c}, 1-\epsilon, 1+\epsilon) A_i\right\} + \beta_c D_{KL}(\pi_{\theta_c} \| \pi_{\text{ref}})$$

关键差异（相比 Flow-GRPO 单角色版本）：
- $|a_{i,c}^t|$ 仅计算组件 $c$ 的 token 数量
- $\rho_{i,j}^{t,c}$ 仅使用组件 $c$ 的策略比率
- $\beta_c$ 和 $\pi_{\text{ref}}$ 是每个组件独立的（虽然初始值相同）



## 2.3 第二阶段
继承 Phase 1 的 MDP 框架，并扩展加入prm-step级别的奖励信号


Phase 1 中，所有组件的所有 tokens 共享相同的 trajectory-level advantage $A_i^{\text{outcome}}$。Phase 2 将其扩展为 per-component per-step 的形式：

$$A_{i,t}^c = \underbrace{A_i^{\text{outcome}}}_{\text{trajectory-level}} + \underbrace{\alpha \cdot z_{i,t}^c}_{\text{step-level process reward}}$$

其中：
- $A_i^{\text{outcome}} = \frac{\bar{R}(o_i, q, y^*) - \text{mean}(\{\bar{R}(o_k)\}_{k=1}^G)}{\text{std}(\{\bar{R}(o_k)\}_{k=1}^G)}$ — 与 Phase 1 完全相同的 group-normalized outcome advantage（Flow-GRPO 论文 Eq.7）
- $z_{i,t}^c = \frac{s_{i,t}^c - \mu_c}{\sigma_c}$ — 组件 $c$ 的标准化过程奖励
- $s_{i,t}^c \in [0, 1]$ — Critic LLM 对组件 $c$ 在 rollout $i$ 的 turn $t$ 的综合评分（多维度 平均后归一化）
- $\mu_c, \sigma_c$ — 组件 $c$ 的 EMA 评分统计量（$\gamma = 0.99$）
- $\alpha = 0.3$ — 过程奖励贡献权重（主实验配置，可消融）

**关键性质**：
1. **Per-component 独立**：不同组件的过程奖励基于各自的 EMA 标准化，不会因评分尺度的系统性差异而偏斜
2. **Outcome-dominant 层级**：过程奖励作为 outcome advantage 的加法修正项，而非替代项。当 $\alpha = 0$ 时，Phase 2 精确退化为 Phase 1


# 3 两阶段motivation安排

## 3.1 第一阶段
### 3.1.1 参数架构

```
Qwen2.5-7B-Instruct (Base Model, Frozen, BF16 ≈ 14GB)
  │
  ├── LoRA_Planner (独立训练, 约0.4B params)
  │    rank=64, alpha=128, target_modules=all-linear
  │    lr=1e-6, β=0.001, temp_train=0.7
  │
  ├── LoRA_Executor (独立训练, 约0.2B params)
  │    rank=32, alpha=64, target_modules=all-linear
  │    lr=5e-7, β=0.001, temp_train=0.0
  │
  └── LoRA_Verifier (独立训练, 约0.1B params)
       rank=32, alpha=64, target_modules=all-linear
       lr=5e-7, β=0.001, temp_train=0.0

总可训练参数: ~0.7B (Base Model 7B 的 10%)
LoRA 总存储开销: ~70MB (三个 adapter 文件)
LoRA 总优化器开销: ~280MB
Base Model: 14GB (共享，所有组件复用同一份)
```

**参数分配依据**：

| 组件       | LoRA Rank | 分配逻辑                                                  | 参考来源                                 |
| -------- | --------- | ----------------------------------------------------- | ------------------------------------ |
| Planner  | 64        | 任务最复杂（自由文本生成、工具选择、上下文组织），Flow-GRPO 论文显示 Planner 是性能瓶颈 | Flow-GRPO Table 3, verl 文档推荐 rank≥32 |
| Executor | 32        | 任务中等复杂（受约束格式的命令生成），输出空间小于 Planner                     | MoRAgent 的角色容量分配逻辑                   |
| Verifier | 32        | 任务相对简单                                                | verl 文档推荐 rank≥16 对于简单任务             |
<font color="#8064a2">ps：或者全部设置为64？</font>

**初始化策略（冷启动）**：LoRA 的 A 矩阵采用 kaiming uniform 随机初始化，B 矩阵初始化为零矩阵。因此初始时 $\Delta W = BA = 0$，各组件的初始行为与 Base Model（Qwen2.5-7B-Instruct）完全一致

为什么不进行sft？
- Flow-GRPO 论文已证明 SFT 在 AgentFlow 场景下导致灾难性性能崩溃（-19.0%, Table 3），原因是 "token-level imitation objective misaligns with trajectory-level task success"。
![[Pasted image 20260616090649.png]]

### 3.1.2 训练配置

#### 1) 基础环境

| 配置项        | 值                                                                                       | 参考依据                  |
| ---------- | --------------------------------------------------------------------------------------- | --------------------- |
| Base Model | Qwen2.5-7B-Instruct (BF16)                                                              | Flow-GRPO 论文主实验配置     |
| 训练框架       | verl（原生支持 GRPO + LoRA）--需要对多角色lora进行优化                                                  | AgentFlow 底层框架        |
| Rollout 引擎 | vLLM（tensor_parallel_size=1）                                                            | Flow-GRPO 配置          |
| 训练数据       | Search-R1 + DeepMath 混合 (182,190 samples)--<font color="#ff0000">这个训练数据太大了（需要讨论）</font> | Flow-GRPO 论文训练数据      |
| 验证数据       | AIME24 (30 samples)                                                                     | Flow-GRPO 验证配置        |
| 评估集        | 10 benchmarks（搜索4 + Agentic1 + 数学3 + 科学2）                                               | Flow-GRPO 论文评估集       |
| LLM Judge  | GPT-4o                                                                                  | Flow-GRPO 论文 Judge 配置 |
| 工具集        | Base Generator, Python Coder, Google Search, Wikipedia Search, Web Search (5 tools)     | AgentFlow 默认工具集       |
| 硬件         | **4 × A100 80GB**（<font color="#ff0000">2 training FSDP2 + 2 vLLM</font>）               | 基于算力分析                |
| 最大轮数       | **10**（训练与评估统一）/<font color="#ff0000">在文章中为了加速训练，训练过程中的max_turn设置为3</font>              |                       |
|            |                                                                                         |                       |

#### 2) 核心超参数

| 参数                                             | Planner    | Executor   | Verifier   | 说明                      |
| ---------------------------------------------- | ---------- | ---------- | ---------- | ----------------------- |
| LoRA rank                                      | 64         | 32         | 32         | 按任务复杂度分配                |
| LoRA alpha                                     | 128        | 64         | 64         | alpha = rank × 2        |
| <font color="#ff0000">Learning rate</font>     | 1e-6       | 5e-7       | 5e-7       | Planner LR 最高（任务最复杂）    |
| KL β                                           | 0.001      | 0.001      | 0.001      | 统一起步，分别监控               |
| <font color="#ff0000">Train temperature</font> | 0.5        | 0.3        | 0.3        | Executor/Verifier 保持确定性 |
| Target modules                                 | all-linear | all-linear | all-linear | 覆盖注意力+FFN 所有线性层         |
temperature是否需要统一设置，还是planner特别点？
flow-grpo中planner使用0.5--to balance exploration and exploitation.

| 通用参数                      | 值                            |
| ------------------------- | ---------------------------- |
| Rollout n (G)             | 8                            |
| Train batch size          | 32 (32 queries × 8 rollouts) |
| Max response length       | 2048 tokens                  |
| Max prompt length         | 18432 tokens                 |
| Total epochs              | 5                            |
| PPO clip ratio            | low=0.2, high=0.3            |
| Optimizer                 | AdamW                        |
| Save/Validation frequency | 每 2 epochs                   |

### 3.1.3 两阶段推进策略

#### Stage 1: 小规模验证（1-2 GPU, 预计约2小时）

```
数据: 1000 samples (随机采样自训练集)
Epochs: 1
Rollout n: 4 (减少以加速)
Batch size: 8 (8 queries × 4 rollouts)
Max turns: 10

通过标准 (所有项必须通过):
  ☐ 训练不崩溃 (无 NaN loss, 无 OOM)
  ☐ 三个组件的 KL divergence 分别 < 0.5
  ☐ 平均奖励曲线 ≥ 初始值（确认有正向学习）
  ☐ Token span 追踪正确（抽查 5 条 trajectory 的 mask）
  ☐ Checkpoint 保存/加载正常（3 个独立 adapter 文件无损坏）
  ☐ 三个组件的 LoRA 权重均有变化（确认每个组件都在学习）
  ☐ vLLM 多 LoRA 动态切换正常（无 adapter 混淆或加载失败）
```

#### Stage 2: 全量正式实验（4 GPU, 完整训练）

```
前提: Stage 1 全部通过
配置: 完整 182,190 训练集 × 5 epochs
Rollout n=8, batch=32, max_turns=10
3 次随机种子（报告均值和标准差）
每 2 epochs 在 AIME24 上验证
训练完成后在 10 benchmarks 上全量评估
```

### 3.1.4 对比基线

| 编号       | 基线名称                 | 配置                                     | 回答的核心问题                  |
| -------- | -------------------- | -------------------------------------- | ------------------------ |
| **B0**   | Frozen AgentFlow     | Qwen2.5-7B-Instruct，全部组件冻结，无 RL        | 多角色 RL 的绝对提升幅度           |
| **B1**   | Flow-GRPO (original) | 仅 Planner 全参数 GRPO 训练（论文结果复现）          | 多角色 LoRA RL vs 单角色全参数 RL |
| **Ours** | Multi-Role LoRA RL   | Planner+Executor+Verifier LoRA 协同 GRPO | 主要实验条件                   |



### 3.1.5 消融方案

消融分析的目标是精确测量每个组件训练的独立贡献。考虑到资源约束，提供两种实施方案：


训练仅一份全三组件模型，评估时选择性不加载某组件的 LoRA：

| 评估条件 | 操作                                 |
| ---- | ---------------------------------- |
| Full | 全部 LoRA 正常加载                       |
| -E   | Executor 使用 Base Model (LoRA=None) |
| -V   | Verifier 使用 Base Model (LoRA=None) |
| -EV  | Executor 和 Verifier 均使用 Base Model |

局限：<font color="#ff0000">因分布偏移</font>（训练时各组件适应了彼此的 LoRA 行为），移除某组件 LoRA 后的性能是下界估计。需在分析中明确标注此局限。



### 3.1.6 评估协议

**自动化评估**：
- 10 benchmarks 全部使用 exact match / LLM Judge 判定（与 Flow-GRPO 论文 一致）--GAIA方式
- 每个 benchmark 3 次独立运行，报告均值 ± 标准差
- 工具调用错误率统计

**人工分析**（训练完成后）：
- 随机抽样 50 条 trajectory，人工标注各组件的输出质量
- 随机抽样 30 条失败 trajectory，人工归因失败根源组件
- Verifier STOP 时机准确率人工评估（对比理想 STOP 时机）



## 3.2 第二阶段
### 3.2.1 总体架构

Phase 2 完全沿用 Phase 1 的参数架构和训练配置，仅在 reward 计算环节增加 process reward

### 3.2.2 Per-Component PRM 评分维度

每个组件在每个 turn 接收 Critic LLM 的 4维度评分（1-4 Likert），综合分数 $s_{i,t}^c = \frac{1}{4 \times 4} \sum_{k=1}^4 s_{c,k}$（均等权重）。

参考了原则（每个角色从多维度）+agentPRM中关于promise+progress的思想

**Planner PRM（4维度）**：

| 维度           | 评估内容                     | 文献根源                                   |
| ------------ | ------------------------ | -------------------------------------- |
| P1: 工具选择合理性  | 选定的工具是否最适合当前子目标？         | Flow-GRPO Figure 5（工具选择分布优化）           |
| P2: 子目标质量    | 子目标是否具体、可执行、范围恰当？        | AgentFlow Planner Prompt 的 Sub-Goal 约束 |
| P3: 上下文传递完整性 | Context 是否包含工具执行所需的全部信息？ | AgentFlow Planner Prompt 的 Context 约束  |
| P4: Promise  | 该步骤实质推进了问题求解多少？          | AgentPRM                               |


**Executor PRM（4维度）**：

| 维度                         | 评估内容                           | 文献根源                                             |
| -------------------------- | ------------------------------ | ------------------------------------------------ |
| E1: 命令格式正确性                | 命令是否符合 tool 要求的 Python 格式？     | AgentFlow Executor 的输出格式约束                       |
| E2: 参数准确性                  | tool.execute() 参数的名称/类型/值是否正确？ | Flow-GRPO Figure 6（工具调用错误率）                      |
| E3: Context 利用度            | 是否正确使用了 Planner 提供的 Context？   | **本项目创新维度**（跨组件信用归因）                             |
| E4: 执行结果质量                 | 命令实际执行结果的质量（NOT 命令文本质量）        | Agent-RRM 的 outcome evaluation；anti-manipulation |


**Verifier PRM（4维度）**：

| 维度             | 评估内容                     | 文献根源                                             |
| -------------- | ------------------------ | ------------------------------------------------ |
| V1: 完整性评估准确度   | Memory 完整性判断是否准确？        | AgentFlow Verifier 的 Completeness 维度             |
| V2: 不一致性检测     | 是否正确识别了 Memory 中的矛盾/歧义？  | AgentFlow Verifier 的 Inconsistencies/Ambiguities |
| V3: STOP 时机正确性 | STOP/CONTINUE 决策的时机是否恰当？ | **本项目核心维度**                                      |
| V4: 分析质量       | analysis 文本是否逻辑严密、证据具体？  | AgentFlow Verifier 的 analysis 字段                 |


对于各个维度的详细解析和描述见下面文档[[15-Phase2-各组件PRM评分维度设计]]



prompt具体设计如下
- 还有的挑战
- previous_actions_summary 到后期会很长，并且会有很多工具结果的信息，如何将这些信息进行一个很好的提取，让其输出到critic PRM LLM会得到客观的分数

Planner Critic LLM Prompt

```
You are evaluating the quality of a PLANNER's decision in a multi-step agent system.

## Task Context
Query: {query}
Available Tools: {available_tools} (with metadata)
Previous Steps (from Memory): {previous_actions_summary}
Current Step Number: {step_count} / {max_steps}

## Planner Output to Evaluate
Justification: {planner_output.justification}
Sub-Goal: {planner_output.sub_goal}
Selected Tool: {planner_output.tool_name}
Context Provided: {planner_output.context}

## Tool Execution Result (for reference)
Tool: {tool_name}
Execution Result: {execution_result_summary}

## Evaluation Instructions
Rate the Planner's decision on the following 5 dimensions using a 1-5 scale.
For each dimension, provide a brief rationale (1-2 sentences).
IMPORTANT: Base your evaluation on the factual information available (previous steps, tool metadata, execution result), NOT on the rhetorical quality of the output text.

1. tool_selection_adequacy (1-5): Is the selected tool the most appropriate for the current sub-goal?
2. sub_goal_quality (1-5): Is the sub-goal specific, executable, and appropriately scoped?
3. context_completeness (1-5): Does the context contain ALL necessary information for the tool to function?
4. promise (1-5): How well does this step set up future steps for success?

## Output Format (JSON only, no other text)
{
  "tool_selection_adequacy": {"score": int, "rationale": "str"},
  "sub_goal_quality": {"score": int, "rationale": "str"},
  "context_completeness": {"score": int, "rationale": "str"},
  "promise": {"score": int, "rationale": "str"},
  "overall_critique": "str (2-3 sentences identifying the strongest and weakest aspects)"
}
```

 Executor Critic LLM Prompt

```
You are evaluating the quality of an EXECUTOR's tool command in a multi-step agent system.

## Task Context
Query: {query}
Planner's Sub-Goal: {planner_sub_goal}
Planner's Context: {planner_context}
Selected Tool: {tool_name} (metadata: {tool_metadata})
Previous Steps: {previous_actions_summary}

## Executor Output to Evaluate
Analysis: {executor_output.analysis}
Command Generated:{executor_output.command}


## Execution Result (CRITICAL — base your evaluation on this)
Success: {execution_success}
Result: {execution_result_summary}
Error (if any): {error_message}

## Evaluation Instructions
Rate the Executor's output on the following 5 dimensions using a 1-5 scale.
IMPORTANT: Dimension E4 (execution result quality) is based on the ACTUAL execution result, not the command text quality. This prevents "looking good but being useless."

1. command_format_correctness (1-5): Is the command syntactically correct and follows required format?
2. parameter_accuracy (1-5): Are tool.execute() parameters accurate in name, type, and value?
3. context_utilization (1-5): Does the command correctly use context from Planner?
4. execution_result_quality (1-5): What is the ACTUAL quality of the execution outcome? (NOT the command text)


## Output Format (JSON only)
{
  "command_format_correctness": {"score": int, "rationale": "str"},
  "parameter_accuracy": {"score": int, "rationale": "str"},
  "context_utilization": {"score": int, "rationale": "str"},
  "execution_result_quality": {"score": int, "rationale": "str"},
  "overall_critique": "str"
}
```

 Verifier Critic LLM Prompt

```
You are evaluating the quality of a VERIFIER's decision in a multi-step agent system.

## Task Context
Query: {query}
Initial Analysis (from Planner): {query_analysis}
Available Tools: {available_tools_summary}
All Previous Steps (Memory): {memory_actions}

## Verifier Output to Evaluate
Analysis: {verifier_output.analysis}
Decision: {verifier_output.stop_signal} (STOP=ready / CONTINUE=need more)

## Known Information (for reference)
Total Steps Taken: {step_count}
Was Final Answer Correct?: {final_answer_correctness} (UNKNOWN if still in progress)

## Evaluation Instructions
Rate the Verifier's output on the following 5 dimensions. 
For V3 (STOP timing), consider what information was available AT THE TIME of this decision, not with hindsight.
For V5 (progress), if the trajectory is still in progress, estimate based on the current analysis quality.

1. completeness_assessment_accuracy (1-5): How accurately does the Verifier assess Memory completeness?
2. inconsistency_detection (1-5): Does the Verifier detect contradictions/ambiguities in Memory?
3. stop_timing_correctness (1-5): Is the STOP/CONTINUE decision made at the right time?
4. analysis_quality (1-5): Is the analysis logical, specific, and evidence-based?


## Output Format (JSON only)
{
  "completeness_assessment_accuracy": {"score": int, "rationale": "str"},
  "inconsistency_detection": {"score": int, "rationale": "str"},
  "stop_timing_correctness": {"score": int, "rationale": "str"},
  "analysis_quality": {"score": int, "rationale": "str"},
  "overall_critique": "str"
}
```

---
### 3.2.3 实验设计

#### 1） 训练配置

| 参数                   | Phase 1（参考） | Phase 2（本方案）                         | 变更理由       |
| -------------------- | ----------- | ------------------------------------ | ---------- |
| **Process reward α** | **N/A**     | **0.3**                              | Phase 2 新增 |
| **Critic LLM**       | **N/A**     | **GPT-4o-mini** (Stage 2)/待定         |            |
| **EMA decay**        | **N/A**     | **γ = 0.99**                         | Phase 2 新增 |



#### 2） 消融实验

| 消融                       | 配置            | 回答的问题                             |
| ------------------------ | ------------- | --------------------------------- |
| **α=0**                  | 等价于 Phase 1   | Phase 2 与 Phase 1 的精确对比（控制所有其他变量） |
| **α=0.1**                | 弱过程奖励         | 过程奖励的最小有效剂量                       |
| **α=0.5**                | 强过程奖励         | 过程奖励主导时的效果                        |
| **α=1.0**                | 极强过程奖励        | 过程奖励几乎淹没 outcome 时的效果             |
| **No EMA normalization** | 原始 PRM 分数不标准化 | EMA 标准化的必要性                       |


### 3.2.5 Motivation 四个假设的专项验证方案

####  H1：训练组件分割

**核心问题**：如果每个组件拥有独立的 PRM，该 PRM 仅根据该组件的局部输出来评分，那么 GRPO 优化的梯度方向仅朝向"本地 PRM 评分最大化"而非"全局任务成功率最大化"。结果奖励+过程奖励的冲突

**验证方法**：
1. **KL 独立演化分析**：对比 Phase 1（无 PRM）和 Phase 2（有 PRM）的组件间 KL divergence 差异。如果 Phase 2 组件间 KL 差异显著大于 Phase 1 → PRM 推动了组件独立演化（分割）
	1. $$D_{\text{seg}} = \text{Var}(\text{KL}_P, \text{KL}_E, \text{KL}_V)_{\text{Phase 2}} - \text{Var}(\text{KL}_P, \text{KL}_E, \text{KL}_V)_{\text{Phase 1}}$$
2. **相邻组件配合度分析**
	1. 专门分析 Planner→Executor 接口（Context 传递）的错误率变化。
		1. $\rho(\text{err}_P, \text{err}_E)$
		2. 若 $\rho < 0.1$（弱相关），表明两组件的问题是独立的而非协同的
	2. 如果 Planner 的 P3（上下文完整性）评分上升但 Executor 的 E3（Context 利用度）评分下降 → Planner 在"讨好自己的 PRM"而非"帮助 Executor"


####  H2：奖励分配不匹配

**核心问题**：不同组件的 PRM 评分维度本身是否导致不公平的奖励分布？
- 三个组件的 PRM 评分基于**完全不同的评分维度和量规。
- Planner 的评分维度（工具选择、子目标质量）与 Executor 的评分维度（命令格式、参数准确性）在概念上不可比。
- 如果 Critic LLM 天然对某些维度的评分更宽松/严格，那么各组件的 PRM 评分分布会有系统性差异。
	- 即使使用了 EMA 标准化，这种差异可能反映在评分的**区分度**（variance）、**偏度**（skewness）和**时间稳定性**（temporal consistency）上。

**验证方法**：
1. **评分分布分析**：三个组件的原始 PRM 评分 $s^c$ 的均值、标准差、偏度、评分时间稳定性在训练过程中的演变。如果某组件的评分分布系统性偏高或偏低 → 评分维度设计本身可能存在偏差
2. **EMA 标准化前后的对比**：比较 $s^c$（原始）和 $z^c$（标准化后）的分布差异。如果标准化显著改变了某组件的优势信号大小 → 原始评分存在系统性偏差
3. **维度级偏差检测**：检查各维度内部的评分紧缩（如所有样本在某一维度都是 3.5-4.0 分）→ 评分缺乏区分度
4. PRM评分与人工评估一致性评价

#### H3：信用错误归因

**核心问题**：PRM 能否正确识别每个组件的真实贡献/错误？

在 AgentFlow 的多组件流程中，Planner 的输出是 Executor 的输入，Executor 的输出是 Verifier 的输入。如果一个组件产生了一个有缺陷的输出，这个缺陷可能在下游组件中表现为失败，而下游组件 PRM 的评分基于的是"该组件在给定输入下的表现"，这个输入本身就是有问题的。
- Executor 可能因为 Planner 提供的模糊 Context 而生成错误命令 → Executor PRM 给低分 → Executor 被惩罚（但错误根源是 Planner）
- Verifier 可能因为 Executor 产生的混乱 Memory 而误判 → Verifier 被惩罚（但错误根源在更上游）

**验证方法**：
1. **归因一致性分析**：对 30 条失败 trajectory，分别进行 Critic LLM 评分归因、outcome 奖励归因和人工归因。计算三种归因方式的 pairwise 一致率
2. **错误传播链追踪**：选取 Planner 错误→Executor 放大→Verifier 未能纠正的级联失败案例，追踪 PRM 评分在各环节的变化
	1. 识别 "Planner Context 不完整但 Planner PRM 评分正常" 的步骤（$s^P > \text{median} \land$ 人工判定 Context 不完整）
	2. 追踪该步骤的下游：Executor 是否因此失败？Verifier 是否因此误判？

#### H4：奖励黑客

**核心问题**：各组件是否出现"输出讨好 PRM 但无助于任务完成"的行为？

RL 中普遍存在 reward hacking 问题。在 LLM-based agent 的 context 中，Khalifa et al.（2026）证明了 LLM judges 对 CoT manipulation 高度敏感——模型可以通过操纵推理文本（而非改善实际行动）来获取更高评分。在我们的场景中，每个组件都输出 analysis/critique 文本，这些文本可以：
- Planner 写冗长详尽的 Justification 但做出平庸的工具选择
- Executor 写详细整洁的 analysis 但参数使用模糊的默认值
- Verifier 写一套逻辑严密的分析但回避关键的不一致性

**验证方法**：
1. **Discordant Case 检测**：识别 PRM 综合评分高（$z^c > 1.5$）但 trajectory 最终失败的 case。分析这些 case 的占比和特征
2. **文本膨胀检测**：检查各组件的分析文本（Planner justification, Verifier analysis）的长度和"形式上正确"的特征是否随训练增长，但实际执行结果（E4, V3）并未同步改善

多智能体失败原因的探究
- 是各个智能体能力的提升，还是之间协作能力的提升
- 
## 4.额外讨论点
### 4.1 Verifier 输出的可训练性
**Verifier 输出结构**：
- `[Execution Analysis]`：对当前工具执行结果的分析
- `[Memory Analysis]`：对累计 Memory 完整性的 5 维度分析
- `[Verification Status]`：STOP 或 CONTINUE 的二元决策

**代码实现**（`agentflow/models/verifier.py`）：`MemoryVerification` Pydantic model 包含 `analysis: str`（自由文本）和 `stop_signal: bool` 两个字段。

**训练可行性**：Verifier 的完整文本输出（analysis + conclusion）构成 GRPO 计算中的 "Verifier action"。Trajectory 的成功/失败信号自然反映了 Verifier 判断的质量——正确的 STOP 时机会提高最终答案的正确率，错误的 STOP（过早或过晚）会降低成功率。

## 4.2 为什么不优化Solution Generator
generator作为flow-agent的四个重要组件之一，为什么不将generator也纳入多角色优化的体系中呢？针对这个问题，我们从一下几个角度进行解答：

1. **任务本质不同**：Generator 是单步 conditional generation（Memory → Answer），不涉及多轮决策和长时程信用分配
	1. 从 MDP 的角度来看：
		- Planner/Executor/Verifier 是 **policy**（$\pi_\theta$）——选择动作以最大化长期回报
		- Generator 是 **value function 的 output head**（类比）——将累积状态映射为结果。其中generator的输入已经完全包含了 Planner/Executor/Verifier 的所有决策效果
	- 对于generator来说grpo也许不是最优选择，高质量的sft/RLAIF/保持冻结也许才是其优化能力的方法
2. **文献支撑**：DRAFT（2026）实证证明了 extraction 和 reasoner应使用独立的参数组；AgentPRM 和 Agent-RRM 均不将 final answer generation 纳入 PRM 评分
3. **工程简化**：保持 Generator 冻结减少了训练变量，使得 Motivation 实验的归因更清晰

## 4.3 flow-gpro中使用lora+多角色隔离的可行性判断和性能下降
### 4.3.1 lora使用
VERL 框架原生支持 LoRA + GRPO

verl（Volcano Engine Reinforcement Learning，字节跳动 Seed 团队开源）是 AgentFlow 所基于的训练框架，**已原生支持 GRPO + LoRA**。官方文档明确给出了配置方案和最佳实践：

**verl 通过 HuggingFace PEFT 库支持 LoRA，关键配置（来自 verl 文档）：

```yaml
actor_rollout_ref:
  model:
    lora_rank: 32
    lora_alpha: 64
    target_modules: all-linear
  rollout:
    load_format: safetensors  # LoRA 加载必须使用 safetensors
```

**工作流程**：
1. 初始化：`get_peft_model(base_model, lora_config)` 创建 PEFT 模型
2. 训练：`update_policy()` 对 PEFT 模型进行前向+反向传播
3. 同步：每个 training step 后，将更新的 LoRA weights 同步到 vLLM rollout engine
4. 推理：vLLM 使用更新后的 LoRA adapter 进行下一轮 rollout



### 4.3.2 多角色隔离
后续设计里面，我们通过给每个角色独立模型/独立 LoRA，实现了无梯度冲突的多角色隔离模式

- **独立 LoRA 组 + Shared adapter** 是最通用且已验证的架构
- **Group-normalized advantage**（Flow-GRPO 已有）可以自然推广到多组件
- **KL coordination** 可作为角色间协调的额外正则项
- **解耦训练 pipeline** 降低了实现复杂度（可先独立优化，再联合优化）

**当前限制**：verl 的 LoRA 支持假设**一个 Base Model 只有一个 LoRA adapter**。所有训练 tokens 共享同一个 LoRA 的梯度更新。

### 4.3.3 verl多lora代码级别可行性分析
在 AgentFlow 的三组件架构中，我们需要：

1. **三个独立的 LoRA adapter**（Planner rank=64, Executor rank=32, Verifier rank=32）
2. **共享同一个 Base Model**（Qwen2.5-7B-Instruct，BF16 ≈ 14GB）
3. **三个独立的优化器**（不同 learning rate、不同的 KL penalty）
4. **Token 归属追踪**：每个 token 属于哪个组件 → 应该用哪个 LoRA 计算 loss
5. **Rollout 时动态切换**：vLLM 在推理 Planner/Executor/Verifier 时使用不同的 LoRA：

多 LoRA 独立更新的实现路径分析：多 PEFT 模型 + 共享 Based
具体可行性分析如下：

| 子问题                      | 可行性             | 风险  |
| ------------------------ | --------------- | --- |
| 多 PEFT 模型共享 Base         | ✅ 可行（PEFT 原生支持） | 低   |
| 独立优化器                    | ✅ 可行            | 低   |
| vLLM 多 LoRA 动态切换         | ✅ 可行（已修复）       | 低   |
| verl update_policy 修改    | ⚠️ 中等复杂度        | 中   |
| FSDP + 多 LoRA 梯度管理       | ⚠️ 需验证          | 中   |
| LoRA → vLLM 多 adapter 同步 | ⚠️ 需验证          | 低中  |

## 4.4 算力估算以及api调用估算

算力估计：推荐使用 4×A100 80GB，其中 2 GPU 用于 FSDP2 训练，2 GPU 用于 vLLM rollout
- 添加了lora 
- 如果全量微调，根据flow-agaent描述，8 × A100 80GB
训练数据集大小：

| 配置项  | 值                                                                                                                           | 参考依据             |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| 训练数据 | Search-R1 + DeepMath 混合 (182,190 samples)--<font color="#ff0000">这个训练数据太大了（需要讨论）</font>--<font color="#00b0f0">训练集查找</font> | Flow-GRPO 论文训练数据 |
| 验证数据 | AIME24 (30 samples)                                                                                                         | Flow-GRPO 验证配置   |
| 评估集  | 10 benchmarks（搜索4 + Agentic1 + 数学3 + 科学2）                                                                                   | Flow-GRPO 论文评估集  |
论文中使用了a batch size of 32 with 8 rollouts per sample.一个batch有5k samples,这个训练量，叠加下面成本，一个batch就肉眼可见的贵
- prm 评估调用模型：
- critic 结果判断：
	- 每次critic LLM调用-1.5ktoken( 输入+输出：prompt中含query+组件输出+context（大头）+评分指标)
	- 对于每个training step：3组件x平均turn 4 turns（大概）x 8 rollouts->96次调用
	- 全量训练：182190 samples/32 batch x 5epochs=28467 steps
	- 这么算下来成本太贵了，这个逻辑也可以引用到其他两个部分
- 各种工具api调用：网页搜索（gemini api+tavily+openai api）


那么问题来了，如果我们减少训练量/step数，有可能无法得到比较好的训练效果和数据进行分析。
- 如下图所示，在training step前期，reward 提升并不明显，基本在0.5左右徘徊，要知道如果经过上面计算，前半部分的训练本身就很足够了
- 下面的效果还是全量微调的结果，如果加入lora,lora初始化策略是冷启动
	- LoRA 的 A 矩阵采用 kaiming uniform 随机初始化，B 矩阵初始化为零矩阵。因此初始时 $\Delta W = BA = 0$，各组件的初始行为与 Base Model完全一致
	- lora本身性能有可能不如全量微调，前期训练有可能会需要更多的训练样本

![[Pasted image 20260617165612.png]]
ps：这里图训练step只有60多，训练效果呈现也是这样的，不知道图表中的step是不是和我们上面分析的step对应

### 4.4.1 PSDP2与vLLM在grpo rl流程中的角色
GRPO 的每个 training step 包含两个性质截然不同的阶段：
Training Step = Rollout (推理/生成) → Update (训练/反向传播)

verl 字节gpro
梯度/参数

vLLM：在 Rollout 阶段，使用当前策略 $\pi_{\theta_{\text{old}}}$ 生成多条 trajectory
- 推理主导：只做前向传播
- 高吞吐需求：每个step生成8个完成trajectory---rollout可以看着小点
- KV cache密集：长序列的子回归生成需要大量KV cache内存
- 延迟敏感：rollout是step主要时间消耗

FSDP2：在 Update 阶段，使用 rollout 收集的 trajectory 数据计算 GRPO loss 并更新策略参数
- 训练主导：需要前向传播 + 反向传播 + 优化器更新
- 将模型参数、梯度和优化器状态分片（shard）到多个 GPU 上，仅在计算需要时通过 all-gather 收集完整参数

FSDP2 在 GRPO 中的具体工作流
```
对于每个 training step 的 update 阶段:
  1. 从 rollout buffer 加载 trajectory 数据
  2. Mini-batch → Micro-batch 分割
  3. 对每个 micro-batch:
     a. FSDP2 all-gather 收集该层参数
     b. 前向传播（计算 log_probs_current）
     c. 计算 GRPO loss（含 per-component token span 和 advantage）
     d. 反向传播（计算梯度）
     e. FSDP2 reduce-scatter 同步梯度
  4. 优化器 step（更新 LoRA 参数）
  5. FSDP2 同步 LoRA 权重到 vLLM（weight sync）
```

## 4.5 max_turns如何设置
在训练过程中flow-grpo为了加快训练速度设置的为3轮次，推理时设计为10轮次，但是如果我们也设计为3轮次，会导致
1. verifier总是判断为失败，训练到后期容易产生偏差--max-turns
2. 为了让每个组件都得到比较合适的奖励和充分的训练，我认为训练时候也设计10轮次比较合适

![[Pasted image 20260617165255.png]]
![[Pasted image 20260617165338.png]]

## 4.6 一些超参数设置
### 1）β
沿用flow-grpo配置，统一使用 $\beta = 0.001$ 作为三个组件的初始 KL 惩罚系数，但建立了明确的分别监控和调整机制：

| 监控指标                     | 预警阈值             | 调整措施                  |
| ------------------------ | ---------------- | --------------------- |
| 组件 $c$ 的 KL divergence   | > 0.5            | 提升 $\beta_c$ 至 0.002  |
| 组件 $c$ 的 KL divergence   | < 0.001（完全不偏离参考） | 降低 $\beta_c$ 至 0.0005 |
| Planner 和 Executor KL 比值 | > 5× 差异          | 分析是否某组件学习过快/过慢        |

β 控制的是：**策略可以偏离参考策略多远**。
- β 过大 → 策略被强约束在 π_ref 附近 → 无法探索、无法学习
- β 过小 → 策略可能崩溃或退化（如总是输出同一个"安全"的 tool name）
- β=0 → 无 KL 正则化 → 高风险崩溃（Flow-GRPO 的消融没有展示无 KL 的情况，但 PPO 文献普遍认为 KL 是必需的）


### 2） temperature
目前planner使用0.5(flow-grpo算法)
而Executor 和 Verifier 的训练温度为 0.3（适度的探索）

是否需要统一为0.5

### 3） α

## 4.7 EMA 标准化机制的原理
**指数移动平均（Exponential Moving Average, EMA）** 是一种用于追踪非平稳流数据的在线统计算法。在当前方案（16号文档 §4.2）中：

$$\mu_c^{(t)} = \gamma \cdot \mu_c^{(t-1)} + (1-\gamma) \cdot \bar{s}_c^{(t)}$$
$$\sigma_c^{(t)} = \gamma \cdot \sigma_c^{(t-1)} + (1-\gamma) \cdot \text{std}(s_c^{(t)})$$

其中 $\gamma = 0.99$ 是衰减因子，$\bar{s}_c^{(t)}$ 是当前 batch 中组件 $c$ 的评分均值。

**工作原理**：

1. **初始化**：$\mu_c^{(0)} = 0.5$（[0,1] 范围的中点），$\sigma_c^{(0)} = 0.2$（保守估计的标准差）

2. **逐 batch 更新**：每个 training step 后，用当前 batch 的统计量更新 EMA。$\gamma = 0.99$ 意味着新 batch 贡献 1% 的权重——在约 100 个 batch 后，统计量主要由累积历史决定。

3. **标准化**：$z_{i,t}^c = \frac{s_{i,t}^c - \mu_c}{\sigma_c + \epsilon}$，将每个组件的评分转化为零均值、单位方差的 z-score。

4. **自适应**：随着训练进行，各组件的输出质量会变化（希望变好），Critic LLM 的评分分布也会随之漂移。EMA 通过持续追踪使得标准化参数与评分分布保持同步。

**关键行为**：

- Batch 1-10（早期）：统计量主要由初始化值主导（冷启动），标准化不够准确
- Batch 10-100：统计量逐渐由真实数据主导，标准化趋于稳定
- Batch 100+：统计量稳定地追踪评分分布，但 Critic LLM 的评分漂移会被 EMA 平滑吸收

**数值稳定性**：EMA 相比简单的全局均值和方差的优势——在处理 RL 训练的非平稳数据时不会因早期不稳定数据而永久偏离。
