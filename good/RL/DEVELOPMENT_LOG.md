# EEG RL 报警策略 — 开发通讯本

这是接续开发本项目的各个 AI 会话之间的**交接与问题登记本**。每个会话结束前，
把自己做了什么、踩了什么坑、留下了什么决定追加进来，让下一个会话不用重新踩一遍。
深度报告放在各自的专题 md 里（见索引），这里只写结论、坑和决定。

---

## 铁律（任何会话不得违反）

1. **所有修改只落在 `good/RL/` 内。** `ai/` 与 `good/` 其他目录的代码只读；
   允许 import 它们的位置只有 `RL/integrations/`。
2. **`verl-agent-master/` 永不修改、不被核心包 import**（AST 隔离测试强制）。
   只允许逐行数学对照。
3. **chb22-23 已在 R4 被消费**，不再是 untouched 留出集。任何新实验不得再用它们
   调参或做选择；正式确认需要新的患者划分 / chb24 / 外部数据集。
4. 选择规则：**中位 seed**，永远不是最优 seed；目标函数（λ 系数）与守门
   （敏感度 ≥0.80）必须在搜索/训练**之前**冻结并写进结果 payload。
5. **L0 协议：概率导出之后的一切策略实验零 GPU**，只吃冻结的 artifact。
6. 结果一律 content-addressed JSON，落盘在 `artifacts/chbmit/eeg_rl/` 下。

## 文档索引

| 文件 | 内容 |
|---|---|
| `EEG_RL_ALARM_POLICY_PLAN.md` | 总计划 L0→L1-A→L1-B→L2→L3 与 G 系列 |
| `EEG_RL_ALARM_POLICY_RESULTS.md` | R1-R4 完整报告（稳健规则/监督对照/PPO/最终测试） |
| `G0_VERL_ADVANTAGE_PARITY.md` | advantage 数学与 verl-agent 源码逐例对照（11/11，1.2e-7） |
| `G1_GRPO_FINDINGS.md` | record 级 GRPO：死区、信用稀释、5-seed 负结果 |
| `G3_GIGPO_FINDINGS.md` | GiGPO step-level：P9 已修、瓶颈转移为跨被试泛化、5-seed 负结果 |
| `eeg_alarm_policy/` | 核心包（不 import ai/good/verl） |
| `integrations/` | 唯一允许 import ai/good 的只读桥 |

---

## 开发日志（按时间正序）

### 会话 A — 计划与 L0 核心
- 写 `EEG_RL_ALARM_POLICY_PLAN.md`；建核心包骨架：`contracts.py`、
  `artifacts.py`、`evaluator.py`、`features.py`、AST 隔离测试。
- 决定：verl-agent 只做数学参考；chb01/chb21 身份重叠组特殊处理。

### 会话 B — 分割/聚合/规则/CLI + 导出桥
- 新增 `splits.py`（DevelopmentGate、PolicySplit、Tier A 角色）、
  `cohort.py`（SelectionObjective + evaluate_cohort）、`rules.py`
  （TemporalRule/网格/EMA/迟滞）、`search_rules.py` CLI、
  `integrations/export_scheme_c_s1.py` 只读导出适配器。
- chb20 首个真实网格（3220 规则）：最优 0.90/2-of-3/300s（FA/h 3.74→0.254）。
- **迷你留出发现**：该规则在 chb21 上 3/4 检出，guardrail 失败 → 不得晋级。
  法证确认漏报机制是"高阈值+2-of-3 在闪烁型发作上的真漏"，与 300s 不应期无关。
- 教训：单被试（8 个事件）选规则必然过拟合；选择需要 chb01-19 的统计质量。

### 会话 C (sol) — 审查修复、R1-R4、G0
- 审查并修复会话 B 遗留的 4 个真实数据 bug（见问题登记 P1-P4）。
- 真实导出 chb20/21（与 S1 结果 JSON 交叉校验 delta=0）；R1 稳健规则、
  R2 监督对照（LR / MLP 32x32）、R3 PPO（5 seed 未过门槛）、R4 最终测试。
- 关键结果：MLP 10/10 检出、0.837 FA/h，为当前最佳候选；PPO 验证不稳定未晋级。
- **chb22-23 在 R4 被读取** → 从此不再是干净留出（铁律 3）。
- G0：二元版 advantage 与 verl-agent 源码逐例 parity 通过。
- G1（GRPO）开发到一半（`grpo_training.py`/`train_grpo.py` 落盘、聚焦测试通过），
  计划先跑 smoke 时额度耗尽。

### 会话 D — G1 收尾
- 全套 56 测试绿；跑 smoke 发现两个结构性问题（见 P8、P9），修复 P8。
- 正式 5-seed G1：全部 J=0.0（沉默收敛），中位 seed 33，`promoted: false`。
- 写 `G1_GRPO_FINDINGS.md`（含初始化偏置扫描表与信用稀释量化）。
- 运行环境备注：全程 CPU（符合 L0），5 seeds 墙钟约 9 分钟。

### 会话 E — G3 GiGPO（当前会话）
- 确认 verl-agent GiGPO 布局为**一步一行**（`core_gigpo.py` + `ray_trainer.py`：
  step_rewards=折扣 return-to-go，anchor_obs=逐步观测文本，episode 内精确观测聚类）。
- 实现 `gigpo_training.py` / `train_gigpo.py`：episode（=G1 GRPO）+ step 两级优势。
  锚点用 `(record, row)` 而非精确观测哈希——观测含本 rollout 报警史，精确匹配会在
  分歧行碎组；锚点偏差已写进 payload（`anchor_deviation`）。PPO 更新循环从 G1
  抽成 `grpo_update_epochs` 公共函数（行为不变，G1 确定性测试仍绿）。
- 测试 62 绿（新增 6），ruff 干净。正式 5-seed 全部 J=0.0，中位 seed 33，
  `promoted: false`。写 `G3_GIGPO_FINDINGS.md`。
- **关键发现**：P9（信用稀释）被证实修好——训练被试上 ictal/normal logit 分离
  0.15→0.95，step 优势 96-98% 组活跃；但 chb21 分离度 −0.27（反向），瓶颈转移为
  跨被试泛化（chb21 概率尺度比 chb20 低 ~14×，单被试训练学不到尺度不变性，
  见 P12）。这直接支撑 G4（MLP 池化 warm start + KL）的动机。
- 环境备注：conda 环境是 `pytorch`（python 3.10.19 / torch 2.5.1 / numpy 2.0.1，
  与 G1 结果 payload 的 runtime 一致）；ruff 在 `rag` 环境里。CPU 全程，5 seeds 约 12 分钟。

---

## 问题登记簿

| # | 问题 | 根因 | 处置 | 状态 |
|---|---|---|---|---|
| P1 | 导出器用了旧 n_fft=64 的 E2 baseline 缓存 | S1 模型是 128/128/32 协议训练的，baseline 必须同协议重建并校验 SHA256 | sol 修复 | ✅ |
| P2 | natural cache 窗口数属性读错 | 属性名错误 | sol 修复 | ✅ |
| P3 | 规则搜索把 export summary 当 prediction artifact 读 | 两者同目录，`glob("*.json")` 捞到 summary | sol 修复 | ✅ |
| P4 | J 为 None 时排序崩溃 | `-None` TypeError | sol 修复 | ✅ |
| P5 | chb20 单被试选出的 0.90/2/3/300s 规则在 chb21 漏 1 事件 | 8 个事件撑不起 operating point；强被试上选出的自信规则到弱被试失守 | 降级为开发观察；选择改到 chb01-19 池化 | ✅ 已归档 |
| P6 | chb21_19 漏报疑云 | 法证：该发作概率闪烁（14 窗中 3 个 ≥0.90 且不连），2-of-3+高阈值真漏；**非**不应期压制（发作前 300s 无 ≥0.90 窗） | 记录在案；不要再归因给 refractory | ✅ |
| P7 | R3 PPO 5-seed 大幅发散（0/4 检出 到 always-alarm） | 单策略训练患者、事件奖励稀疏、critic 方差 | 已停；教训记入 G 系列动机 | ✅ |
| P8 | **GRPO 密集初始化死区**：随机 init p≈0.5 密集报警 → 300s 不应期饱和 → 同组 rollout 回报逐位相同 → advantage 恒零 → 只有熵梯度，永不学习 | 回报对动作不敏感是结构性的，不是 bug | `GRPOConfig.init_logit_bias=-3`（稀疏起步）；扫描表见 G1 文档（p≤0.12 → 29/29 组激活） | ✅ 已修复+测试 |
| P9 | **record 级信用稀释 ~60:1**：event record ~900 步中仅 ~14 步发作窗，±1 outcome advantage 均摊到全轨迹，报警决策只占 ~1.6% 信号；23 个正常 record 的"沉默省 0.02"信号清晰一致 → 5 seed 全部漂向沉默（ictal/normal logit 分离仅 0.15） | episode 级 group advantage 的结构局限 | G3 用 (record,row) 锚点 + return-to-go step 优势修复：分离 0.15→0.95（chb20），step 优势 96-98% 组活跃 | ✅ G3 已修（但见 P12） |
| P10 | 读出陷阱：稀疏初始化下确定性读出（logit≥0）初始必然 0 报警，**不能**据此判断"策略学没了" | 决策门槛 0 远在初始 logits（≈−3）之上 | 判断策略死活看 ictal/normal logit 分离度与熵轨迹，不看读出报警数 | ✅ 记录 |
| P11 | λ_lat=0.001 的延迟项形同虚设（最优规则延迟罚分 0.0003；chb21 上 69s 延迟尾巴无惩罚） | 预声明系数本就如此 | 保持不变换连续性；若要改必须在 chb01-19 搜索**前**重新声明 | 📌 记录 |
| P12 | **单被试 RL 训练不跨被试泛化**：G3 在 chb20 上分离度 0.95，到 chb21 反向为 −0.27、J=0；chb21 概率尺度（median 0.0072 / q95 0.123）比 chb20（0.1023/0.604）低 ~14×，而 ictal 均值两边都是 0.64；8 维 history 是原始概率，单被试训练学不到尺度不变性 | 观测特征未按被试自身尺度归一 + 训练分布只有一个被试 | 未修；现有 R2 MLP 同样只在 chb20 训练，并不是池化模型。G4 前必须先构建真正的 chb01-19 多被试 MLP/out-of-fold artifact，再测试 warm start + KL | ⏳ 待 G4 |

## 工具链坑（本机 Windows + conda）

- 本项目 conda 环境是 **`pytorch`**（python 3.10.19 / torch 2.5.1 / numpy 2.0.1，
  与 G1 payload 的 runtime 一致）；**ruff 在 `rag` 环境**。`astar-river025` 没有测试。
- `conda run` **不能执行多行 `python -c`** → 一次性诊断写成临时 `.py` 文件跑完即删。
- `conda run` 会吞掉 pytest 的最终汇总行（`N passed`）→ 看进度点行（`-q` 下每行
  `FE.` 点数）或重定向到文件再 tail。
- Glob 工具带绝对 `path` 参数在本机可能匹配失败 → 用相对工作目录的 pattern。
- Bash 工作目录可能在调用间重置 → 命令里显式 `cd /c/ML/astar/good/RL`。
- 策略实验阶段 GPU 就是闲的（L0 协议），不要误报"GPU 没动"是故障。

---

## 当前状态与下一步

- **状态**：L0/R1-R4/G0/G1/G3 完成。最佳候选 = MLP 32x32（10/10，0.837 FA/h，
  J≈0.9876）。RL 三次尝试（PPO、record 级 GRPO、GiGPO）均未过晋级门槛；
  G3 已把训练侧问题（信用稀释）修好，失败原因定位在跨被试泛化（P12）。
- **下一步（按优先级）**：
  1. **G4 前置 + G4：多被试 MLP → GiGPO warm start + KL 约束** —— 先导出
     chb01-19 out-of-fold 概率并训练真正的多被试 actor；现有 R2 是 chb20-only，
     不能冒充池化 warm start。之后训练中加 KL(ref||π) 约束防灾难性遗忘。
     若仍超不过监督 MLP → RL 路线按协议停止。
  2. G2（RLOO，同为 episode 级）信息量低，维持跳过。
  3. MLP 的独立确认（Tier B out-of-fold、bootstrap 置信区间、特征消融）——
     这条线不依赖 RL，可并行。
- **交接清单**（每个会话离开前自查）：测试全绿？ruff 干净？结果落盘为
  content-addressed JSON？本文件追加了自己的会话小节和新问题？
