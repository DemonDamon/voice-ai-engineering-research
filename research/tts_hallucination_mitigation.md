# TTS幻觉问题缓解方案研究

## 论文信息
**标题**: Mitigating Hallucinations in LM-Based TTS Models via Distribution Alignment Using GFlowNets

**作者**: Chenlin Liu, Minghui Fang, Patrick, Wei Zhou, Jie Gao, Jiqing Han

**机构**: 
- Harbin Institute of Technology, China
- Zhejiang University, China
- Tsinghua University, Shenzhen, China

**发布时间**: 2025年8月21日

**arXiv ID**: 2508.15442v1

## 核心问题定义

### LM-based TTS模型的幻觉现象
基于语言模型的文本转语音（TTS）系统经常生成偏离输入文本的幻觉语音，主要表现为：

1. **发音错误（Mispronunciations）**
2. **遗漏（Omissions）**
3. **不自然的重复（Unnatural Repetitions）**

### 现有缓解策略的问题
- 需要大量训练资源
- 引入显著的推理延迟

## 核心解决方案：GOAT框架

### 方法概述
**GFlowNet-guided distribution Alignment for Optimal Trajectory (GOAT)**

这是一个后训练框架，可以在不依赖大量资源或推理成本的情况下缓解幻觉问题。

### 关键创新点

#### 1. 不确定性分析
通过不确定性分析揭示了**幻觉与模型不确定性之间的强正相关性**。

**不确定性计算方法**：

##### Token级别不确定性
```
H_token(P_t) = -Σ p_i * log(p_i)
```
其中 P_t = {p1, ..., p|C|} 是时间步t的token概率分布，|C|是词汇表大小。

##### Word级别不确定性
```
H_word(W_{i:j}) = 1/(j-i) * Σ H_token(P_k)  (k从i到j)
```
提取每个单词W在token序列中的左右边界i和j来计算。

##### Utterance级别不确定性
```
H_utterance(S) = 1/|S| * Σ H_token(P_k)  (k从1到|S|)
```
定义为所有token的平均不确定性，|S|是序列长度。

#### 2. 实证分析结果
在SeedTTS-Eval基准上使用CosyVoice2进行实验：
- 选择200个样本从test-hard子集
- 使用随机多项式采样生成语音
- 使用Paraformer-zh进行ASR并计算WER作为幻觉代理

**关键发现**：
- 低WER的语音通常表现出更低的不确定性，反之亦然
- **Pearson相关系数**: 0.636 (p < 1E-4)
- **Spearman秩相关系数**: 0.649 (p < 1E-4)
- 揭示了统计学上显著的正相关性

#### 3. GOAT优化目标
基于不确定性与幻觉的正相关性，GOAT鼓励模型发现**更确定性和最优的解码路径**。

### 核心技术组件

#### 轨迹流优化
将TTS生成重构为轨迹流优化问题，引入：
- **增强的子轨迹平衡目标（Enhanced Subtrajectory Balance Objective）**
- **锐化的内部奖励（Sharpened Internal Reward）**作为目标分布

#### 奖励温度调整
整合奖励温度调整以优化训练速率，实现稳定性和性能的平衡。

## 实验结果

### 性能提升
在CosyVoice2上的测试结果：
- **字符错误率降低超过50%**（在挑战性测试案例上）
- **不确定性降低高达58%**
- 展示了强大的泛化能力和有效性

### 优势
1. **无需大量训练资源**
2. **不引入推理延迟**
3. **后训练框架**，可应用于已训练模型
4. **强泛化能力**

## 对客户问题的启示

### 针对910B卡推理质量问题
客户反馈的问题：
- 存在乱说话、杂音过多的问题
- 初步排查为生成了大量的幻觉speech token

### 可能的解决方向

#### 1. 不确定性监控
实现token级、word级和utterance级的不确定性监控：
- 在推理过程中计算熵值
- 设置不确定性阈值
- 当不确定性过高时触发重采样或调整解码策略

#### 2. 解码策略优化
- 从随机多项式采样改为更确定性的解码方法
- 使用beam search或top-k/top-p采样
- 引入温度参数控制采样随机性

#### 3. 后训练优化
如果条件允许，可以考虑：
- 应用GFlowNet-based的后训练方法
- 使用910B卡上的小规模数据进行微调
- 针对性优化高不确定性的token生成

#### 4. 硬件适配考虑
910B卡的特殊性：
- CANN框架的算子支持可能影响某些优化技术的实现
- 需要验证熵计算和不确定性分析在NPU上的性能
- 考虑使用ATC转OM模型时保留必要的中间输出用于监控

## 参考文献
- Du et al. (2024b): CosyVoice2
- Gao et al. (2022): Paraformer-zh
- Anastassiou et al. (2024): SeedTTS-Eval benchmark
- Huang et al. (2024a), Ma et al. (2025): 基于不确定性的幻觉分析
