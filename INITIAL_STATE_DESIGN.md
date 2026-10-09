# 多视频推理框架：State-Init / Focus / Review / Answer

> **历史方案说明（已被实现替代）：** 本文后半部分记录的
> `pair-complete` 方案不再是当前契约。现有实现不会枚举所有视频对，也不会为遗漏
> pair 自动制造 issue；初始状态保持 answer-free，但只表达由问题和现有证据实际触发
> 的关系与缺口。当前行为以 `README.md`、`src/toolcvr/schemas.py` 和
> `src/toolcvr/initial_state.py` 为准。

> 用途：给外部做文献调研的设计说明 + 提问清单。
> 实现位置：`/media/data6/xuejj/ToolCVR`。
> 已实现：Coarse Observation、Initial State Formation（`src/toolcvr/initial_state.py`）、
> Focus 的 Observer 侧（`src/toolcvr/focus.py`）。
> 已有独立实现：Focus Planning、State Update 与双 Agent 内部工具循环协议。
> 尚未完成：把工具循环接入正式多轮入口以及最终 Answer Synthesis。

---

## 0. 任务背景

多视频问答（CrossVid 类）：给 N 个视频 + 一个问题（+选项），要求比较/排序/匹配视频间的关系。
答案格式多样：`sequence`（排序）、`interval`、`choice`、`matching`、`set`、`free_text`。

框架有两个智能体：`ObserverAgent`（看视频，输出结构化事件报告）+ `ReasonerAgent`（只看报告文本，不看帧）。
本文档描述两者之间的状态机。

---

## 1. 整体框架

### 1.1 主线

```
Coarse Observe → Initialize State → Focus → Observe → Review → Update State → Repeat → Answer
```

### 1.2 五阶段流程

```
N 个视频
  ↓
ObserverAgent 粗扫
  ↓
N 份 timestamped event reports
  ↓
Initial State Formation
  ↓
WorkingState_0
  ↓
────────── 迭代循环 ──────────
  ↓
Focus Planning
  ↓
生成 Evidence Request
(video + time window + question)
  ↓
ObserverAgent Focus
  ↓
指定时间窗口稠密抽帧 + VLM 定向理解
  ↓
Focused Evidence
  ↓
Review / State Update
  ↓
WorkingState_{t+1}
  ↓
判断是否还需要继续取证
  ├─ YES → 下一轮 Focus
  └─ NO  → Answer Synthesis
                 ↓
             Final Answer
```

### 1.3 与旧框架的结构区别（这是最重要的一张图）

旧版：

```
粗扫 → FORM：先猜答案 → FOCUS：围绕答案找局部疑问 → VLM 取证 → REVIEW：修答案 → DONE

即   Answer → Evidence → Answer
```

新版：

```
粗扫 → State Initialization → State → Focus Request → Evidence → State Revision
     → Focus Request → ... → Answer

即   Evidence → State → Evidence → State → Answer
```

**旧版里证据是"为答案服务的"**：FOCUS 找的是"围绕已选答案的局部疑问"，
REVIEW 修的是"答案本身"。所以证据再多也只能加固或微调一个已经下注的选择。

**新版里答案不是循环的一部分。** 循环只做一件事：把 WorkingState 上的缺口变成取证请求，
再把证据并回状态。答案在循环外，只在状态不再有值得查询的缺口时才被生成一次。

### 1.4 职责边界（硬约束）

| 模块 | 做什么 | **不做什么** |
|---|---|---|
| Coarse Observer | 生成 timestamped reports | 不推理答案 |
| Initial State | 第一次建立 WorkingState | 不输出答案 |
| Focus Planner | 生成具体视觉 request | **不直接看视频** |
| Focus Observer | 指定窗口稠密观察 | 不决定全局答案 |
| Review | 用新证据更新状态 | **不直接输出 final answer** |
| Answer Synthesizer | 根据最终状态答题 | **不再发起取证** |

### 1.5 阶段 ≠ 智能体（按 `AGENTS.md`）

`AGENTS.md` 规定框架里只有两个具体智能体类：`ObserverAgent` 和 `ReasonerAgent`。
上面六个模块是**阶段**，不是智能体：

- 阶段 1、4 是 `ObserverAgent` 的两种模式：`SCAN`（粗扫）与 `FOCUS`（定向稠密）。
- 阶段 2、3、5 以及 Review 是 `ReasonerAgent` 的不同阶段。
- 跨视频对应、状态校验、覆盖统计都是**工具/模块**，不加 `Agent` 后缀。

这条不是形式主义：它保证了"谁在看视频、谁在看文本"这个边界不被悄悄打破
（`ReasonerAgent` 在正常执行中不接收原始帧）。

### 1.6 一个值得强调的架构性质：`answer_format` 只出现在最后

按 1.2 的数据流，`answer_format` **只在阶段 5 被使用**：

```
阶段 1–4（粗扫 / 建状态 / 取证 / 更新状态）  ← 完全不依赖 answer_format
阶段 5（Answer Synthesis）                   ← question + answer_format + final WorkingState
```

这正好符合"不按 task label 路由"的要求，而且是**结构上**符合，不是靠纪律：
前面的阶段没有拿到 answer_format，想让它们做 task-specific 的事也做不到。
这条应该写进论文——它解释了为什么这套状态构建能跨 `sequence / choice / matching / set / free_text` 复用。

---

## 2. 阶段 1：Coarse Observation（已实现）

`ObserverAgent` 对所有视频先粗扫，得到完整的 timestamped event reports。
每个 event 带 `event_id` / `time_span` / `description` / `uncertainty` / `evidence_frame_ids`。

这部分不变，也是后面一切的地基。**它是唯一一次"无目的"地看视频**，之后的观察都带 request。

---

## 3. 阶段 2：Initial State Formation（已实现）

### 3.1 要取代的旧机制，以及它为什么结构性地坏

旧实现叫 `FORM`：让模型读报告，输出一个 JSON，字段顺序是——

```
best_answer        <- 第一个字段
state_factors      <- 提示词写"只列影响答案的因素"
alternatives       <- 同类中可能性较低的变体
ambiguities.pair   <- 对已选答案内部的分歧点
```

**病灶：后面三个字段全是第一个字段的衍生物。**
- `state_factors` 被提示词限定为"影响答案的因素"，所以答案是 X 时它只能列 X 相关的因素；
- `alternatives` 是"同一答案的较弱变体"，不是别的答案；
- `ambiguities` 描述的是"我选的答案内部哪里有分歧"，不是"我是不是选错了"。

于是模型第一轮就必须下注，之后没有任何**与答案无关**的世界状态可以拿来做对照。
表现为：一轮或两轮直接给最终答案，跑偏之后无法纠正。

配套的加固机制也有同样的毛病：下游有个 review 回合，但它的翻案必须落在 FORM 自己提名的 pair 上
（`_gate_review_flips`），也就是"只能纠正我自己觉得自己错的地方"。
加上 `MAX_EVIDENCE_ROUNDS = 2`，以及解析时 `decision` 默认填 `"DONE"`——漏输出就等于自认完成。

**结论：这不是提示词问题，是 schema 问题。** 只要状态表里存在答案字段，
模型就会去填它，且填了之后整张表都围着它转。

### 3.2 本阶段的定位（一句话，克制版）

> **A pair-complete, answer-free intermediate state that converts omission and uncertainty
> from implicit model behavior into explicit, measurable state variables.**
> （一个 pair-complete、answer-free 的中间状态，把"遗漏"和"不确定"从模型的隐式行为
> 转变成显式的、可度量的状态变量。）

真正有辨识度的**不是"状态表"这个形式**，而是三个性质的组合：
**answer-free**（3.3）+ **pair-complete**（3.5）+ **omission-measurable**（3.5 的计数部分）。
单独任何一个都不新：结构化中间状态早有人做，封闭枚举也早有人做；
是"三者同时成立、且 3.4 条的派生值刻意只覆盖引用而不覆盖语义"这个组合，
把 early commitment 从"要靠 prompt 劝阻的行为"变成了"schema 层面不可能发生的事"。

### 3.3 约束一：状态表结构上不含答案

序列化结果里显式声明：

```
contains_answer: false
contains_completion_decision: false
model_reported_confidence: false
```

没有 `best_answer`、没有 `alternatives`、没有 `DONE`、没有 confidence 字段**可填**。
模型即使想早交卷，也没有地方写。

### 3.4 约束二：派生证据覆盖状态，而不是自报置信度

`working_relations[].status` 由模块从引用结构算出来，**模型输出里的 `status` 字段被直接忽略**：

| 条件 | 派生 status |
|---|---|
| 同一 pair 上存在未解 issue | `unresolved` |
| scope 内每个视频都贡献了被引用事件 | `evidence_complete` |
| 只有部分视频贡献了事件 | `evidence_partial` |

**命名上刻意保守**：这三个值是**关于引用覆盖的事实**，不是关于语义真伪的判断。
"v2 和 v3 都引用了事件"只能推出"这条 relation 有双边的证据覆盖"，
推不出"v2 早于 v3"这个陈述是对的——两个视频都有事件引用，完全可能只是模型读错了。
所以派生的是 `evidence_complete / evidence_partial / unresolved`，而不是 `supported / tentative / unresolved`。
程序不应该替模型判定语义真假，只应该判定"引用是否覆盖了 scope 里的每个视频"。

模型写 `"status": "evidence_complete"` 但只引了单边事件，会被直接改写为 `evidence_partial`
并记一条 `validation_notes`。

**为什么这重要**：模型自报置信度在文献里早被证明不可靠（overconfident / 校准差）。
我们不试图"更好地校准"它，而是**取消这个通道**——把"有多确定"变成对引用结构的可判定查询，
代价是显式承认这个查询只能回答"证据齐不齐"，不能回答"对不对"。

### 3.5 约束三：Pair-complete 关系骨架

设输入 N 个视频，则**每个无序视频对都必须被一条 relation 或一条 issue 覆盖**，共 C(N,2) 条。

- 模型漏掉的 pair 不会被静默丢弃：模块自动补一条 issue（`U-auto-NN`，`filled_by_module: true`），
  在 `validation_notes` 追加一条说明，并在 `coverage` 里计数。
- 于是**模型漏报从"看不见"变成"可计数的指标"**（`coverage.module_filled_pairs`）。
- 同时状态表**永远是完备的**，下游不需要处理"某个 pair 没提"的情况。

**刻画方式（不要说成"任务无关且通用于所有答案格式"）**：
这是一种 **cross-video relational state backbone**，即 pair-complete 的跨视频关系骨架。
它在**构造上是任务无关的**（不管什么题型都能建出来，且永远完备），
但它在**用途上是任务相关的**：

| 题型 | pair 骨架是否承载信息 |
|---|---|
| PSS（排序）、CC（比较）、FSA（找同动作）、matching | 是，pair 本身就是答案的结构 |
| free_text、单视频识别、全局集合性质 | 可能产生大量空转槽位 |

也就是说：骨架**总是可以构造**，但**不是每个槽位都有意义**。
这个边界要在论文里明说，不要包装成普适性主张。

**这个"空转槽位"问题不会在阶段 2 解决，它变成了阶段 3 的核心难点**（见 4.3）。

**"承认不知道"是一个必须显式填写的格位，而不是可以省略的沉默。** 这才是这一条真正的机制价值。

### 3.6 状态表的四+一个部分

| 字段 | 含义 | 硬约束 |
|---|---|---|
| `observed_information` | 报告里**已经看到**的事实 | 每条必须引用存在的 `event_id` |
| `working_relations` | 对某个视频对的**解释性陈述** | scope ≥ 2 个视频，必须引用事件，`status` 由模块派生（只表示引用覆盖，不表示语义真伪） |
| `unresolved_issues` | 某个视频对上**仍未确定**的东西 | issue 必须与答案无关（见 3.7） |
| `candidate_requests` | 待执行的请求池 | 只能指向存在的 issue；阶段 2 **绝不执行** |
| `coverage` | 覆盖统计 | `required / covered / module_filled / unresolved` 四类计数 |

导出指标 **Pair Coverage Recall**：

```
PCR = 1 - module_filled_pairs / C(N,2)
```

即"模型自己意识到多少应该被表示的 pair"，一个中间状态质量指标，而不是最终任务准确率。
它测量的是 state construction recall / 结构性漏报率。

### 3.7 约束四：issue 必须与答案无关

提示词里明确写：

> unresolved_issues 必须描述证据的缺口，绝不能是对某个偏好答案的怀疑。
> 写 "v2 与 v3 的阶段先后关系仍未确定"，**不要**写 "如果答案是 v2 先于 v3，请检查 v2"。

这一条和 3.3 配套：如果 issue 可以从答案推出来，那状态表又变成答案的衍生物了。
所以 issue 必须是**先于答案存在的、关于世界的问句**。

**这个约束会一路传导到阶段 3**：从 issue 生成的 Evidence Request，
其 `question` 字段也必须用可观察状态措辞，不能预设某个答案（见 4.6）。

### 3.8 校验层丢什么、补什么

- **丢**（并记 note）：引用不存在的 event / video / issue；缺 id；重复 id；缺必填文本；
  scope 不足 2 个视频；超出配置上限。
- **补**：漏掉的视频对 → `U-auto-NN` + note + coverage 计数。
- **炸**（直接抛异常，不静默）：跨报告的 event_id 冲突；视频数 < 2；视频 id 重复；问题为空。
- **重试**：LM 路径最多 `maximum_attempts=2` 次；若第一次有未覆盖 pair，
  把**具体是哪些 pair**回喂给模型再来一次。最后一次即使仍不完整，状态表也是结构完备的
  （模块补过），补了几条记录在案。

**一个容易被忽略的设计点**：关系若在校验后不剩任何有效证据引用，则被丢弃，
**它本该覆盖的那个 pair 会被重新补成 unresolved issue**。
"删掉一条坏关系"不会造成覆盖漏洞。

---

## 4. 阶段 3：Focus Planning（未实现，本节是设计提案）

### 4.1 Focus 的定义

> **Focus 的输入不是视频，而是当前 WorkingState。**
> Reasoner 根据 unresolved issue / relation / coverage，生成一个具体的 **Evidence Request**。

Focus 本质是：**Reasoner 把"状态缺口"翻译成"视觉取证请求"。**

```json
{
  "target_video": "v3",
  "time_window": [18, 26],
  "question": "Before stirring begins, has the ingredient already been placed into the pan?",
  "related_issue": "U07"
}
```

### 4.2 与已有 active perception 工作的区别

- **ActiView**（ACL 2025 long, 2025.acl-long.376, arXiv:2410.04659, 清华+复旦）
  是 active perception 的评测基准：限制感知野，模型必须主动 zoom / shift。
  **注意：它是 image-based VQA 基准，不是视频**。~30 个模型，GPT-4o 66.40% vs 人类 84.67%。
  它把 active perception 定义为"基于当前理解设定目标并执行动作"，但**机制由模型自己决定**。
- **Active-O3** 进一步把 where/how to look 直接训练成 active perception policy。

我们的区别是一个清晰的位置差异：

```
ActiView / Active-O3:   current reasoning → model decides where to look
我们的做法:              explicit unresolved-state coverage
                        → rank unresolved pairs/issues
                        → generate targeted request
```

即：**不是 generic active perception，而是 state-coverage-driven active perception**。
感知决策的候选集不是模型现场生成的，而是上游状态表里**已经存在、可枚举、可计数**的缺口。
这把"问什么"从一个生成问题变成了一个**在封闭候选集上的选择问题**。

### 4.3 核心难点：pair-complete ≠ question-complete

这是整件事的题眼，必须先说清楚。

**状态表是 pair-complete 的，但不是 question-complete 的。**
模块给每个未被覆盖的 pair 都补了 issue（3.5），所以 `U` 里同时混着两类东西：

- **真缺口**：和问题有关、只是证据不够的 pair；
- **空转槽位**：问题和这个 pair 没关系，纯粹是因为骨架要求全覆盖才被补出来的。

具体例子：问题是"哪个视频用了不同的工具"，而 v5 和 v7 在这个问题上根本不参与比较。
如果模型没提 (v5,v7)，模块会补一条 `U-auto-XX`——**它和真缺口在数据结构上长得一模一样**。

所以下面这条是**错的**，不能写：

```python
if len(unresolved_issues) == 0: stop    # ✗ 空转槽位会让这个条件永远不成立
```

需要的是 **answer-relevant unresolved coverage**，而不是 raw unresolved count。

### 4.4 设计约束

1. **不使用 task label**，不做 task-specific 路由。
   （由 1.6 保证：Focus Planning 根本拿不到 `answer_format`。）
2. **不让 LLM 自由打分。**
   这一点必须精确理解：不是说"LLM 不能参与"，而是
   **LLM 的输出绝不能被直接当成排序。**

   | 允许 | 禁止 |
   |---|---|
   | LLM 做**二元判断**（这是否被蕴含？） | LLM 输出**基数分数**（重要性 0.8） |
   | LLM 产出**可被验证的引用**（"被 R03 和 R07 蕴含"） | LLM 产出无法复核的排序 |
   | LLM 负责 request 的**语义措辞** | LLM 同时决定候选集、优先级、停止 |

   区分标准是**可复算性**：二元判断 + 引用校验**可以被 harness 审计**
   （引用不存在的 R07 就驳回），而分数不能。
   这和现有代码里 `supporting_events` 必须引用真实 `event_id` 是同一条纪律，不是新发明的规则。

### 4.5 提议的机制

分成五步，**只有第 3 步和第 5 步用到 LLM，且都不产出任何数字**。

#### 第 1 步：机械分类（不需要 LLM）

对每条 unresolved issue，按 scope 内的引用情况分类。全部依据是可校验的 event 引用：

| 类别 | 判据 | 含义 | 对应动作 |
|---|---|---|---|
| `starved` | scope 内至少一个视频在整张状态表里**零引用** | 我们根本没看过这个视频的相关部分 | 需要**新观察** |
| `uninterpreted` | scope 内每个视频都有引用事件，但没形成 relation | 材料两边都有，只是没比过 | **定向对比**（双侧已引用时段） |
| `conflicting` | 存在 scope 与之重叠、且证据分配不一致的 relation | 已有陈述互相矛盾 | **复核**（VERIFY） |
| `isolated` | scope 内所有视频都不出现在任何 relation 里 | 整条支线都没开始 | 低优先 |

`starved` vs `uninterpreted` 这个区分是**纯机械的**（数引用），却直接决定了**该调哪个工具**——
而且它不依赖任何 task label。

#### 第 2 步：机械剪枝与合并

- **蕴含预筛**：如果一条 issue 的 scope 已经被若干 `evidence_complete` 的 relation 完全覆盖，
  它可能已经被蕴含（是否真的蕴含由第 3 步判定）。
- **冲突合并**：scope 重叠的 issue 合并成一个候选，避免对同一条支线重复 Focus。
- **空转降权**：`filled_by_module: true` 且 `isolated` 且两侧引用对称的，直接沉底。

  这里用到一个反直觉但合理的信号：**模型"没提"这个 pair，本身就是数据。**
  它不是"信模型打的分"，而是把模型的**遗漏**当信号——而遗漏本来就已经是这套设计里的
  一等公民（PCR 就是在量它）。现在只是把它从"度量"扩展到"路由"。

#### 第 3 步：LLM 做有归属的蕴含判定（不产出数字）

对每条候选 issue，问一个**二元问题**：

> 这条 issue 是否已经被当前已建立的 relations 蕴含？
> 如果是，**列出蕴含它的 relation ID**。

harness 校验：列出的 relation ID 必须真实存在，且必须**确实覆盖了该 issue 的 pair**。
校验不通过 → 判定作废，issue 保留。

- 判定为"已蕴含" → issue 被**剪掉**（它不值得我们花预算）。
- 判定为"未蕴含" → issue 保留，进入第 4 步。

**关键点**：LLM 在这里做的是**剪枝**，不是**排序**。
剪枝是二元的、有引用的、可驳回的；排序是基数的、不可复核的。
**把 LLM 放在剪枝位而不是排序位，是这套设计能不能站住的分水岭。**

#### 第 4 步：机械排序（在封闭候选集上，全部可复算）

对存活的 issue，优先级是一个**从状态表结构算出来的整数组合**：

```
priority(u) = w1·conflict(u)      # scope 与其他 relation 重叠且证据不一致
            + w2·bridge(u)        # 能否把 relation 图里两个连通分量连起来
            + w3·asymmetry(u)     # 一侧有大量引用、另一侧为零（定向看很便宜且很可能决定性）
            + w4·demand(u)        # 有多少 candidate_requests 指向它（最弱的一项，见下）
            − w5·module_filled(u) # 模型遗漏的折扣
```

每一项都是从 `InitialState` 的字段直接数出来的整数，**不需要任何模型调用**。

> `demand(u)` 这一项要单独标注：`candidate_requests` 是模型生成的，
> 所以这一项本质上是"模型投票"。它是唯一一项**没有结构依据**的。
> 建议默认 `w4 = 0`，只在做消融时打开，用来验证"模型自己指的方向"和"结构算出来的方向"是否一致——
> 如果一致，说明结构信号是对的；如果不一致，那正好是需要解释的现象。

**这一条决定了机制是真的还是假的**：
给定同一个 `initial_state.json`，任何人重跑第 4 步都应该得到**完全相同的顺序**，不需要调模型。
这是"我们没有把 early commitment 搬到排序器里"的可验证证据。

#### 第 5 步：组装 Evidence Request（3/4 个字段是机械的）

这是 Focus 重新定义之后**最重要的一个发现**：`Evidence Request` 的四个字段里，
**只有 `question` 需要 LLM**，其余三个都能从状态表机械继承：

| 字段 | 来源 | 是否需要 LLM |
|---|---|---|
| `target_video` | 从 issue 的 scope 机械得出（`starved` 时是零引用那一侧） | 否 |
| `time_window` | **从被引用事件的 `time_span` 直接继承**（`uninterpreted` 时可取双侧引用的并集并加裕量） | 否 |
| `related_issue` | 直接就是 issue id（第 4 步选出） | 否 |
| `question` | LLM 把这个缺口写成一句**可观察的**问句 | **是** |

也就是说，LLM 在 Focus Planning 里的全部产出就是**一个句子**：
候选集不是它挑的，优先级不是它定的，时间窗不是它猜的，停止也不是它判的。

`time_window` 机械继承这一点尤其关键——**取证窗口来自证据本身，而不是模型对"该看哪一秒"的猜测**。
`starved` 情况是例外：没有引用事件可继承，窗口必须另寻来源
（这是 4.8 里一条没想清楚的问题）。

#### 第 6 步：`question` 的措辞纪律（由 3.7 传导而来）

`question` 必须用**可观察状态**措辞，不能预设某个答案：

| 可以 | 不可以 |
|---|---|
| "Before stirring begins, has the ingredient already been placed into the pan?" | "Is v3 the one that used a different tool?" |
| 问一件**看就能知道**的事 | 问一个**答案判断** |
| 不提及任何候选答案 | 提及 "different"、"correct"、"compared to v1" |

这一条和 3.7 是同一个纪律，只是从 issue 层传导到了 request 层。
如果 request 的问句里已经含着对答案的假设，那重蹈旧 FORM 的覆辙只是延迟了几轮。

### 4.6 Focus Planner 与已有 `FocusTool` 的接口

ToolCVR 里 **Observer 侧的 Focus 已经实现**（`src/toolcvr/focus.py`），接口是：

```python
FocusRequest(
    video_id="v1",
    evidence_need="inspect the action immediately before and after this moment",
    center_timestamp_seconds=42.0,   # 或者 time_span=(38.0, 46.0)，二者必居其一
)

FocusTool.run(request)   # 稠密抽帧 ≤16 帧 → ObserverAgent → 结构化 event
```

和 4.1 的 `Evidence Request` 字段对应关系：

| 1.2 里的 Evidence Request | 已有的 `FocusRequest` |
|---|---|
| `target_video` | `video_id` |
| `time_window` [18, 26] | `time_span=(18.0, 26.0)` |
| `question` | `evidence_need` |
| `related_issue` | **新增字段**（用于把证据并回正确的 issue） |

所以阶段 3 要做的不是重写 Focus 工具，而是**在它前面加一个 Planner**：
输入 WorkingState，输出 `FocusRequest + related_issue`。

### 4.7 停止条件

用 **answer-relevant unresolved coverage**，而不是 raw count：

```
stop  ⟺  没有任何 issue 满足 priority(u) > 0
```

即剩下未解决的 issue 全都落进以下之一：

- 被已建立的 relations 蕴含（第 3 步判定，有引用）；
- `module_filled` 且 `isolated` 且两侧引用对称（结构上没有任何理由去看）；
- 预算耗尽。

`coverage` 里**同时报告两个数**：

- `unresolved_pair_count` —— raw，pair-complete 骨架的完整统计（阶段 2 已有）；
- `prioritized_issue_count` —— 经过蕴含剪枝和排序后**仍有正优先级**的 issue 数。

两者的差就是空转槽位的规模。**这个差本身是个可报告的指标**，
论文里可以拿它来说明 pair-complete 骨架在一个题型上"有效槽位 / 总槽位"的比例。

### 4.7.1 一个必须承认的点：stop 是某一轮的判断，不是证明

因为 5.3 定了状态是**非单调**的，所以 "没有正优先级 issue" 只在**第 t 轮**成立——
后续轮次的证据完全可能把某个已关闭的 issue `REOPEN`。

这不削弱停止条件，反而让它是诚实的：它不假装证明了"已经查清"，
只声明"在当前的证据和状态上，没有结构上值得再查的东西"。

两个后果：

- **`REOPEN` 是防止循环死锁的机制**。如果 Review 发现新证据推翻了一个 `RESOLVE`，
  它必须能重开——否则我们只是把"早期错误锁死状态"从阶段 2 挪到了阶段 4。
- **必须有预算兜底**。既然 stop 不是证明，就不能只靠它退出，`max_rounds` 一类的硬预算必须存在。
  这同时给 4.8 的消融实验提供了可控的预算变量。

### 4.8 可证伪性（这一步决定它是不是真机制）

设计完必须能被打脸。两个可做的实验：

1. **重排序一致性**：同一张状态表，重复跑第 4 步，顺序必须逐位相同；换个实现独立算，也必须相同。
   （不可复算 = 机制不成立。）
2. **优先级消融**：在同样的 Focus 预算下，比较
   (a) 按 `priority` 排序选、(b) 随机选、(c) 按模型自报重要性选。
   如果 (a) 不能显著好于 (b)，那这套结构信号就是装饰。
3. **整轮可回放**：因为 5.3 定了修订历史 append-only，
   `WorkingState_0` + 各轮 `FocusedEvidence` + `StateDelta` 序列就是一次运行的完整记录。
   给定这些，第三方应该能重建每一轮的 `WorkingState_t`，并复核每个 delta 的证据引用。
   （不可回放 = 审计不到中途发生了什么，也就无法诊断 early commitment。）

**第 2 个实验才是论文的主结果。** 没有它，整个阶段 3 只是"设计得像有道理"。
第 3 个则把 2606.22936 强调的 trajectory-level 诊断变成了实际可做的东西。

### 4.9 仍然不确定的（需要调研的）

1. **`bridge` / `asymmetry` 这些量真的是"信息增益"的代理吗？**
   我们是在显式状态表上算，不是在 POMDP 的 belief 上算。形式上接近 VOI，但不是 VOI。
   有没有工作把 VOI 用在"LLM agent 的结构化中间状态"上？
2. **蕴含判定的可靠性**。如果 LLM 频繁误判"已蕴含"而把真缺口剪掉，整个机制就崩了。
   第 3 步的引用校验能挡住一部分（引用不存在 / 不覆盖该 pair），但挡不住"引用真实却推不出结论"。
   **需要测**：误剪率是多少？
3. **`starved` 的 `time_window` 从哪来。**
   `starved` 意味着 scope 内有视频零引用，没有 `time_span` 可继承。
   但这可能是**上游 CLIP 筛选漏了**，不是真的没有素材。
   这时该重新 SCAN 整段、还是该怀疑 CLIP 召回并放宽阈值？这条边界现在没想清楚。
4. **权重 `w1..w5` 是手设的策略，不是学出来的。** 可能被批评为"只是一个启发式"。
   反驳是：它是**透明的、在封闭集上可复算的**，而且 4.8 的消融正是用来检验这个启发式是否抓住了真实结构。
   是否值得学出来（会重新引入不透明性）需要权衡。

---

## 5. 阶段 4：Focused Observation + Review（未实现）

### 5.1 Focused Observation

Observer 收到 request 后才真正看视频：`v3` 的 `18–26s`
→ 稠密抽帧（≤16 帧）→ VLM 定向理解 → **Focused Evidence**。

约束继承自现有 warm start 的不变量：一次 VLM 请求 ≤16 帧；
观察文本由结构化证据确定性渲染，自由文本不进入状态。

### 5.2 Review / State Update

Focused Evidence 送回 Reasoner，**Review 的职责是根据新证据修正 WorkingState，而不是回答原始问题。**

Review 允许的操作分成两层看：**五种推理语义动作**，由**八个类型安全的具体操作**实现。
两层是同一件事的两种粒度，不是两套机制。

#### 5.2.1 推理语义层：五种动作

| 操作 | 作用对象 | 含义 |
|---|---|---|
| `ADD` | information / relation / issue | 新证据产生的新条目 |
| `UPDATE` | relation | 修改陈述或补入证据（`R03`: `evidence_partial` → `evidence_complete`） |
| `RETRACT` | relation / issue | 有证据表明这条陈述是错的 |
| `REOPEN` | 已 resolved 的 issue | 新证据推翻了之前的关闭 |
| `RESOLVE` | issue | 证据齐了，缺口关闭 |

最后两行是关键：**Review 必须能"推翻"和"重开"，而不只是"补充"**。
旧框架的 `_gate_review_flips` 只允许翻自己提名的 pair，所以做不到真正的推翻；
这里 Review 的输入是**完整状态表**，它可以发现任何一对陈述之间的矛盾。

#### 5.2.2 实现操作层：八个具体操作

```
ADD
 ├─ AddInformation        观察到的信息条目（不参与 pair 覆盖）
 ├─ AddRelation           跨视频关系陈述（携带 scope + supporting_events）
 └─ AddIssue              未解决缺口

UPDATE
 └─ UpdateRelation        只改 statement 或 supporting_events

RETRACT
 ├─ RetractRelation
 └─ RetractIssue

REOPEN
 └─ ReopenIssue           只对已 resolved 的 issue 合法

RESOLVE
 └─ ResolveIssue          resolving_relation 必填
```

**为什么要拆。** 五个语义动作描述的是模型想做什么，八个具体操作描述的是系统允许做什么，
后者必须能被构造期校验。合并的代价是具体的：

- `AddRelation` 是唯一携带 `scope` 和 `supporting_events` 的操作。如果 `ADD` 是一个操作，
  它的 payload 就得是 union 类型，`ADD` 一条 information 和 `ADD` 一条 relation
  共用同一个构造函数，非法组合只能在运行时才发现。
- `UpdateRelation` 是唯一能对照"scope 未变"做检查的操作（5.3.1 第 3 条）。
  和 `ADD` 合并后，这条检查就失去了统一的落点。
- `UpdateRelation` / `AddRelation` **在类型上就没有 `status` 字段**。
  模型不是"被禁止"传 status，而是根本没有地方传——这正是 3.4 想要的强度。

一句话：**非法编辑应该不可表达，而不是被运行时检查抓住。**
五个语义动作是给人的，八个操作是给编译器和测试的。

> 实现对应 `src/toolcvr/schemas.py` 的 8 个 dataclass 与 `state_update.py` 的分发逻辑。
> 注意代码里的记录名字是 `AddInformation` 而不是 `AddObservedInformation`。

和阶段 2 一样，Review 的输出也必须是**经过机械校验的结构化更新**：
新增的 relation 必须引用真实 event，`RESOLVE` 必须附带引用证据，
`status` 依然由模块派生而不是 Review 自报（见下方 5.3.1 的第 2 条）。

#### 5.2.3 `RETRACT` 删的是条目，不是这个 pair 的未知性

这是最容易被误读的一条语义，单独写清楚：

> **`RETRACT` 删除的是"当前具体的状态条目"，不是删除"这个 pair 的未知性"。**

pair-complete 不变量（3.5）要求：

```
没有 relation  覆盖该 pair
+ 没有 active issue 覆盖该 pair
        ↓
      必须自动恢复一个 unknown slot
```

所以下面这个序列是**预期语义，不是副作用**：

```
Retract U07
   ↓
U07 消失
   ↓
pair (v1,v3) 失去唯一覆盖者
   ↓
U-auto-XX 出现，filled_by_module = true
```

换句话说，**你没法用 `RETRACT` 把未知性从状态里删掉。** 删掉一条 issue 不等于"这个缺口不存在了"，
只等于"这条具体的记录不再有效"。缺口如果没人接手，模块会重新物化它。

这个性质有两个好处：

1. **状态表永远是可读的。** 任何时刻，一个 video pair 要么被一条 relation 解释，要么被一条 issue 标记为未知，
   不存在"消失在视野里"的第三种情况——而"消失在视野里"正是旧 FORM 里遗漏发生的方式。
2. **`RETRACT` 的语义没有被偷偷放大。** 如果 `RETRACT` issue 能直接消除未知性，
   它就成了一个绕过 `RESOLVE` 校验的后门：模型只要撤回 issue 就能假装缺口没了。
   现在 `RETRACT` 只作用于条目，**关闭缺口仍然只有 `RESOLVE` 一条路，而 `RESOLVE` 必须指名覆盖它的 relation。**

代价是 `U-auto-NN` 会积累，且模块修复会进入 `revision_history`。
这是有意的：历史记录的是"系统实际发生了什么"，不只是"模型提出了什么"，
否则重放无法复现。修复条目在历史里带 `filled_by_module` 标记和自己的 note，可以和模型编辑区分开。

### 5.3 状态的单调性：**非单调，但修订历史 append-only**

**决定（已定）：不做单调增长。**

```
WorkingState_t
   + FocusedEvidence_t
        ↓
     Review
        ↓
    StateDelta_t
        ↓
WorkingState_{t+1}
```

理由很直接：**整个新框架就是为了避免 early commitment。**
如果 WorkingState 只能单调增加，那一旦粗扫阶段把某个 relation 写错，
后面 Focus 拿到反证也只能"再加一条冲突"，却无法真正改变当前状态——
最后还是会被早期错误拖住，等于把 early commitment 从 schema 层搬进了状态更新层。

**定式：`WorkingState is non-monotonic, but revision history is append-only.`**

拆成两半：

| 单调增长（append-only） | 非单调（可修订） |
|---|---|
| `observed_information` | `working_relations` |
| Focused Evidence | `unresolved_issues` |
| `revision_history` 本身 | （及其 `status`） |

一句话：**证据记录单调增长，状态解释允许非单调修订。**

修订条目（不改旧条目，而是往历史里追加）：

```
relation / issue id
  current_status: ...
  version: k
  revision_history: [
    {round_id, operation, previous_value, new_value, evidence_refs}
  ]
```

例如 `R03` 的经历：

```
R03 v0: evidence_partial
R03 v1: evidence_complete
R03 v2: retracted
```

#### 5.3.1 三条由这个决定直接推出的实现约束

这三条不是风格问题，是**不写就会坏**的。

**（1）任何修改都必须让 pair 覆盖不变量在修改后立即成立。**

阶段 2 的不变量是"每个视频对都被一条 relation 或一条 issue 覆盖"，
`InitialState.__post_init__` 直接断言了这一点（3.5）。
下游（Focus Planning 的机械分类、排序，4.5）全部依赖它，所以它在循环里也必须一直成立。

两个会打破它的操作：

| 操作 | 为什么破 | 怎么修 |
|---|---|---|
| `RETRACT` relation | 它可能是某个 pair 的唯一覆盖者 | 模块补一条 `U-auto-NN`（复用阶段 2 机制，见 5.2.3） |
| `RESOLVE` issue | 同上——关闭 issue 就少了一个覆盖者 | **见下面的合法性规则** |

于是 `RESOLVE` 不能是一个"断言完成"的动作，它必须**指明是什么替代了这条 issue**：

> **`RESOLVE` 只在某个 relation 已经覆盖该 issue 的 pair 时才合法。**

这条约束和 `supporting_events` 必须引用真实 `event_id` 是同一条纪律：
Review 不能说"这个缺口解决了"，它必须说"**被 R05 解决了**"，而 R05 得真实存在且真的覆盖那个 pair。
harness 直接校验。这让最常见的那个操作也变成可复核的，而不是一次空洞的断言。

模块自动补出来的覆盖同样要**计数**：**覆盖修复次数** ——
衡量"Review 的编辑有多频繁地打断了骨架"，和 PCR 是一路的指标。

**（2）`status` 的历史是派生的，不是 Review 写的。**

`revision_history` 里会出现 `evidence_partial → evidence_complete` 这种条目，
但**这不是 Review 填的字段**。Review 只提交证据引用的变化，
`status` 由模块按 3.4 的规则重新派生，派生结果作为该轮的快照记入历史。

如果让 Review 直接写 `new_value: "evidence_complete"`，就等于把 3.4 取消掉的
"模型自报状态"从后门放了回来——而且是在一个更难审计的地方。

**（3）`UPDATE` 不得修改 `scope`；改 scope 属于 `RETRACT` + `ADD`。**

`scope` 决定了覆盖关系，也决定了一条 relation 在排序器里的 `bridge` 权重（4.5 第 4 步）。
如果允许 `UPDATE` 偷偷改 scope，那么：
- 覆盖可能在你没察觉的情况下破掉（回到第 1 条）；
- `candidate_requests` 和 Focus Request 里引用过的 relation ID 会指到一个换了含义的对象上。

所以 scope 是 relation 的**身份的一部分**：换 scope 就是换了一条 relation，给新 ID。
这样 ID 在整个运行期间是稳定且有意义的，引用永远可追溯。

---

## 6. 阶段 5：Answer Synthesis（未实现）

只有当前已经没有正优先级的 issue、或预算耗尽，才离开 Focus–Review 循环，
然后**单独调用** Answer Synthesizer：

```
输入：question + answer_format + final WorkingState
输出：最终答案
```

**它不再发起取证。** 这是一条硬约束：
如果 Synthesizer 说"我还需要看 X"，那说明停止条件判错了，应该回到阶段 3，
而不是让 Synthesizer 绕过循环自己去看。

`answer_format` 在这里才第一次出现（见 1.6）。

---

## 7. 实现落点

| 阶段 | 文件 | 状态 |
|---|---|---|
| 1 Coarse Observation | `src/toolcvr/report_generation.py`, `scripts/generate_reports_from_selection.py` | 已实现 |
| 2 Initial State | `src/toolcvr/initial_state.py`, `scripts/build_initial_state.py` | 已实现 |
| 3 Focus Planning | —— | **未实现** |
| 4 Focus Observer | `src/toolcvr/focus.py` | 已实现 |
| 4 Review / Update | `src/toolcvr/state_update.py`, `src/toolcvr/schemas.py` | 已实现 |
| 5 Answer Synthesis | —— | **未实现** |

阶段 2 的 schema 在 `src/toolcvr/schemas.py`：
`VideoEvidenceReport` / `ObservedInformation` / `WorkingRelation` / `UnresolvedIssue` /
`CandidateRequest` / `InitialState`；`InitialState.__post_init__` 直接断言覆盖闭合。

阶段 4 的更新层同样在 `src/toolcvr/schemas.py`：
`WorkingState` / `StateDelta` / `StateRevision` + 8 个操作 dataclass；
唯一入口是 `state_update.apply_state_delta(state, delta) -> WorkingState`，是 `(state, delta)` 的纯函数。

配置项：`maximum_information/relations/issues/requests`（24/24/24/12，超限截断并记 note）、
`maximum_attempts=2`。

已验证：**71 个测试 0 失败**；脚本在合成样本上跑通
（`pairs=3/3 module_filled=0 unresolved_pairs=3`）。
**尚未在真实数据上跑过**——目前 `outputs/` 里只有 CLIP 选择阶段的 `result.json`，
还没有 observer 的 `report.json`。

---

## 8. 与旧 FORM 的对照

| 维度 | 旧 FORM | 新框架 |
|---|---|---|
| 第一个字段 | `best_answer` | `observed_information`（观察，非结论） |
| 状态是否含答案 | 含，且是其他字段的根 | schema 层面不存在 |
| 置信度来源 | 模型自报 | 由证据结构派生 |
| 覆盖完备性 | 靠模型自觉 | C(N,2) 封闭枚举 + 模块补全 + 计数 |
| 漏报后果 | 静默 | `module_filled_pairs` 可计数 |
| 主循环 | Answer → Evidence → Answer | **Evidence → State → Evidence → State → Answer** |
| Focus 选什么 | 围绕已选答案的局部疑问 | 状态表上的缺口（按 4.5 选出） |
| 能否纠错 | review 只能翻自己提名的 pair | Review 可推翻任意矛盾陈述 |
| 完成判定 | 解析默认 `"DONE"` | 由优先级决定，非模型自报 |
| 答案生成时机 | 第一轮 | 循环结束后一次 |

---

## 9. 明确的局限

1. 阶段 3 / 4 的 Review / 5 均未实现，未接入主 pipeline，无端到端评测。
2. 没有 solver，不做约束求解。
3. 不引入 task-specific ontology，**不按 task label 路由不同逻辑**。
4. **没有机制避免空转槽位**——对不需要比较所有 pair 的题型，pair 骨架会产生无意义槽位。
   阶段 3 的蕴含剪枝和降权是**缓解**，不是解决。
5. **派生状态只覆盖"证据齐不齐"，不覆盖"对不对"**（见 3.4）。
6. 阶段 3 的 `w1..w5` 是手设策略。
7. `starved` 情况下的取证窗口来源未定（见 4.9）。
8. **`stop` 不是证明，只是第 t 轮的判断**（见 4.7.1）。非单调状态意味着
   后续轮次可能 `REOPEN` 已关闭的缺口，所以必须有硬预算兜底；预算设多少没有依据。
9. **`REOPEN` 的判定依据未定**。Review 凭什么认定"新证据推翻了一个 `RESOLVE`"？
   如果这个判断也交给 LLM 自由裁量，那 5.3 想避免的问题会在这里复现。
   目前只想到一条：`RESOLVE` 时附带的 `evidence_refs` 若被后续证据直接反证，则强制 `REOPEN`——
   但这条只覆盖"被直接反证"这一种情况。

---

## 10. 术语与文献定位

### 10.1 可用的术语（无需自造）

**premature commitment** —— 已有正式工作，可以直接用。
我们与它们的区别是**干预点**：它们是诊断，我们是结构预防。

- **When Agents Commit Too Soon: Diagnosing Premature Commitment in LLM Agents**
  （Aman Mehta, Snowflake AI Research, arXiv:2606.22936, 2026-06）
  把 **representational commitment** 定义为"多次独立运行在同一推理步的隐状态收敛"
  （last-token hidden state 的平均两两余弦相似度，非零温度采样）。
  主要发现：
  - 第 4 步的隐状态相似度能预测后续行为一致性（r = −0.35，partial r = −0.45，峰值在第 40 层）；
  - **"committed" 不等于 "correct"** —— 承诺-正确和承诺-错误在激活相似度上**不可分**。
    承诺只说明 agent 已经安定下来，不说明它对；
  - 运行期监控器从隐状态检测不一致轨迹，AUROC 最高 0.97；
  - 他们自己承认的局限：final-answer 打分看不到这种失败模式。

  **和我们最直接对应的一篇。** 他们是 **diagnose**（事后探测 agent 是否已经 collapse），
  我们是 **prevent by construction**（answer 字段在 schema 里根本不存在，collapse 无处可写）。
  失败模式假设一致，干预点不同。
  所以我们的贡献应表述为 **architectural prevention of premature commitment**，不是 prompt trick。

- **Answer First, Reason Later: When Commitment Order Costs Accuracy in Diffusion Language Models**
  （Jewon Yeom, Jaewon Sok, Seonghyeon Park, Jeongjae Park, Hwiyeong Lee, Taesup Kim,
  arXiv:2608.05687, 2026-08-06）
  研究 masked diffusion LM 的 **token commitment order**：token 一旦可见就不再被 mask，称为 committed；
  当 final answer 在它前面的 reasoning 还没稳定时就被 commit，即 "answer-first"。
  - LLaDA-8B 在 GSM8K 上无约束解码时，final answer 平均只在轨迹的 **15–24%** 处就 commit 了，
    此时一半 reasoning 区域还被 mask；
  - 病因不是终止信念（EOS pressure 各解码器几乎相同），而是 **reachability**；
  - 2×2 设计显示 CoT **只在有序 commit 下才有用**，交互项 **+34.8 pp（95% CI [26.8, 42.8]）**；
  - 一个很有意思的对照：四选一任务里预留一个 answer token 位置，
    **延迟该位置** 比 **等时长的 reasoning token 延迟** 在三个模型上都更好；
  - 修复是 **frontier-gated commitment**（只允许最左未解析 token 附近 w=16 窗口内 commit），
    准确率 0.528 → 0.852，同时保留最高 4× 并行解码。

  **引用范围必须严格限定**：这是 **diffusion LM 的 decoding / token commitment order**，
  **不是 agent state，也不是多视频推理**。它只能支持一个很窄的论点：
  > 输出结构中让 final answer 过早占据确定位置，可能改变后续推理过程并造成 premature commitment。
  **不能**写成"它证明了 agent 的 FORM schema 会导致锁定"——那个因果跨度太大。

  （一个只能当**类比**、不能当证据的呼应：他们的修复是"只在最左的**未解析 token** 附近允许 commit"，
  我们的骨架是"每个**未解析 pair** 都必须显式存在"。两者都在把"未解析性"变成
  一个显式的、能约束推进过程的对象——但一个是 decoding 层，一个是 agent state 层。）

- **ActiView**（ACL 2025 long, 2025.acl-long.376, arXiv:2410.04659, 清华+复旦）
  active perception 评测基准：限制感知野，模型必须主动 zoom / shift。
  **它是 image-based VQA 基准，不是视频**（引用时注意）。
  ~30 个模型，GPT-4o 66.40% vs 人类 84.67%。机制由模型自己决定，这是与我们最关键的区别。
- **Active-O3** 把 where/how to look 直接训练成 active perception policy。

### 10.2 一条必须守住的红线

整个工作的可信度取决于一句话：**LLM 判真伪，算法定优先级。**

只要排序权还留在模型手里，无论状态表设计得多干净，early commitment 都只是换了个地方发生。
反过来，只要排序是可复算的，4.8 那个消融实验（priority vs 随机 vs 模型自报）就有意义，
整个机制才谈得上是"机制"而不是"提示词工程"。

在 Focus 重新定义之后，这条红线有了更精确的形式：

> **Focus Planning 里 LLM 的全部产出是一个句子。**
> 候选集不是它挑的（第 1、2 步机械），优先级不是它定的（第 4 步机械），
> 时间窗不是它猜的（第 5 步从 `time_span` 继承），停止不是它判的（4.7 机械）。

### 10.3 想请教的问题（给文献调研）

#### A. 定位

- 我们的定位是 **"用一个结构性约束的中间状态取代 answer-first 的 early commitment"**。
  这更接近 structured intermediate representations / belief state / abductive reasoning 的哪一支？
- 有没有现成术语描述"强制对每个变量对表态（关系或未知）"这种**封闭枚举式**的状态构造？

#### B. 早期承诺

- 我们打算直接用 premature commitment（10.1）。有没有工作量化过
  "把 answer 字段挪到输出末尾"或"拆成两阶段"的收益？
- **trajectory-level 诊断指标**：我们把 `coverage.module_filled_pairs`（PCR）当过程指标。
  这类 metric 在文献里叫什么？（2606.22936 的核心论点之一正是 final-answer score
  捕捉不到 trajectory-level failure，这一点可以作为我们的 motivation 引用。）

#### C. Focus 机制（最关键的一节）

> 我们有一张多视频推理的显式状态表，所有视频对都被标注为 relation 或 issue（未解决）。
> 其中一部分 issue 是"真缺口"，另一部分是骨架为了全覆盖而补出来的"空转槽位"，
> 两者在数据结构上不可区分。
>
> 我们想在不使用 task label、也不让 LLM 自由打分的前提下，找出**真正值得 Focus 的那条 unresolved issue**。
> 我们的做法是：机械分类（依据引用分布）→ LLM 做**有引用、可驳回的二元蕴含剪枝**
> → 在封闭候选集上做**可复算的机械排序** → LLM 只写 request 的 question 文本。

**问题：**

1. 在 LLM agent 的主动感知 / 工具调用里，有没有工作用这种**封闭、可枚举的未解决槽位集合**
   驱动"下一步看哪里"，而不是让模型自由生成"我还需要什么信息"？
2. 有没有工作把 LLM 限制在**剪枝位**（二元 + 可验证引用）而把**排序位**留给确定性算法？
   这种"LLM 判真伪、算法定优先级"的分工在文献里怎么称呼？
3. `bridge` / `asymmetry` 这类**图结构量**作为信息增益的代理，有没有先例？
   （关键词猜测：value of information、information gain、submodular 覆盖最大化、
   active perception、question asking / ambiguity resolution、belief state coverage）
4. **状态修订性**：我们已定为 *non-monotonic state, append-only revision history*
   （见 5.3）——理由是单调增长会让早期错误读数锁死状态，
   等于把 early commitment 从 schema 层搬进状态更新层。
   **想找先例或反例**：有没有工作讨论"LLM agent 中间状态的版本化与可追溯推翻"？
   有没有工作论证过中间状态**应该**是单调的（比如为了避免振荡或保证收敛）？
   如果有，我们需要正面回应。

---

## 11. 一句话总结

> 我们把多视频推理从 **"先猜答案再找证据"（Answer → Evidence → Answer）**
> 改成 **"证据建状态、状态驱动取证"（Evidence → State → Evidence → State → Answer）**。
>
> 状态表是 **answer-free** 的（模型没有地方写答案）、**pair-complete** 的（每个视频对都被表态，
> 要么给关系要么显式承认不知道）、**omission-measurable** 的（漏报数可计数）。
> 关系的证据覆盖状态由引用结构派生，只说证据齐不齐，不说关系对不对。
>
> 下一步的 Focus 不是"模型想下一步看什么"，而是**把状态缺口翻译成定向取证请求**：
> 候选集机械枚举、时间窗从已有证据的 `time_span` 继承、优先级机械可复算——
> **LLM 的全部产出只是一个问句的措辞**。
