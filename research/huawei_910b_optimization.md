# 华为910B卡推理优化深度研究

## 芯片技术背景

### 910B芯片技术分析

#### 制程与性能
- **制程工艺**：7nm（中芯国际代工）
- **峰值算力**：320 TFLOPS
- **热设计功耗（TDP）**：310W
- **持续算力输出**：256-320 TFLOPS
- **能效比**：接近国际主流AI加速卡水平

#### 架构设计
- **核心架构**：达芬奇（Da Vinci）核心架构
- **支持精度**：
  - FP16半精度浮点运算
  - INT8整数运算
- **设计优势**：混合精度设计，满足大模型训练的计算精度要求，同时优化推理场景的能效比

### 市场定位

#### 国产替代进程
- **2025年预测**：Ascend系列芯片预计占据中国AI加速市场约35%份额
- **下一代产品910C**：性能预计翻倍，有望在高端推理市场形成对NVIDIA L20系列的竞争性替代
- **采购趋势**：互联网企业采购策略显著转变，消费级卡逐渐被Atlas 300i等专业推理卡替代

#### 技术生态挑战
**CUDA生态壁垒**仍然是910B面临的严峻挑战：

1. **软件生态差距**
   - 华为通过构建MindSpore全栈工具链缩小差距
   - 提供PyTorch适配插件

2. **大模型训练支持**
   - 集成Torch-NPU插件
   - 结合DeepSpeed支持千亿参数模型的分布式训练

## 客户问题分析

### 问题1：并发性能瓶颈

#### 现状
- **目标**：单张910B卡实现10路并发
- **实际测试**：
  - 基于Torch NPU推理
  - 显存足够
  - RTF达不到要求（约1.2，不稳定）
  - 目标RTF应接近0.1（10路并发）

#### 技术限制
1. **vLLM昇腾版**
   - 缺少增加自定义模型的功能
   - 华为反馈10月份还不一定能搞定

2. **Triton方案**
   - 昇腾版本成功案例少
   - 算法适配过程遇到很多问题
   - 加速效果可能不如vLLM

### 问题2：推理质量问题

#### 现象
- 910B卡在情感表达、清晰度和流畅度方面弱于英伟达卡
- 存在乱说话、杂音过多的问题
- 初步排查：生成了大量的幻觉speech token

## 解决方案

### 方案A：官方推荐的OM模型转换方案

#### ATC（Ascend Tensor Compiler）转换
参考仓库：
```
https://gitee.com/ascend/ModelZoo-PyTorch/blob/master/ACL_PyTorch/built-in/audio/CosyVoice/CosyVoice2/README.md
```

#### 优势
1. **针对性优化**：OM格式专为昇腾NPU优化
2. **性能提升**：相比Torch NPU直推有显著提升
3. **官方支持**：华为官方维护的转换流程

#### 实施步骤
```bash
# 1. 模型导出为ONNX
python export_onnx.py --model cosyvoice2 --output model.onnx

# 2. 使用ATC转换为OM
atc --model=model.onnx \
    --framework=5 \
    --output=model_optimized \
    --soc_version=Ascend910B \
    --input_shape="text:1,512;speech_prompt:1,16000" \
    --log=error \
    --optypelist_for_implmode="Gelu" \
    --op_select_implmode=high_performance

# 3. 使用ACL进行推理
python acl_inference.py --om_model model_optimized.om
```

### 方案B：Profiling性能优化

#### 使用CANN Profiling工具
参考文档：Profiling性能数据采集-CANN商用版8.0.RC2

#### 优化点识别
1. **空泡（Bubble）检测**
   - 识别模型内部的计算空闲时间
   - 优化算子调度减少空泡

2. **流同步等待优化**
   - 减少不必要的同步点
   - 使用异步执行提升并发

3. **调度性能优化**
   - 优化算子融合
   - 调整batch size和并发策略

#### 实施流程
```python
# 1. 启用Profiling
import torch_npu
torch_npu.npu.set_option({
    'ACL_PROFILING_MODE': '1',
    'ACL_PROFILING_PATH': './profiling_data'
})

# 2. 运行推理收集数据
model.forward(input_data)

# 3. 分析Profiling结果
# 使用CANN提供的可视化工具分析
```

### 方案C：流式输入优化

#### 问题诊断
部分开源版本对流式输入支持不完善，导致：
- LLM输出文字后，需要缓存整句才能输入给TTS模块
- 引入额外等待时间：（单字推理时间 × 平均句字数）

#### 优化方向
1. **流式TTS改造**
   - 实现chunk-based推理
   - 支持增量输入和输出
   - 减少首包延迟

2. **LLM-TTS Pipeline优化**
   ```python
   # 优化前：等待整句
   full_sentence = llm.generate_until_sentence_end()
   audio = tts.synthesize(full_sentence)
   
   # 优化后：流式处理
   for token in llm.generate_stream():
       if is_word_boundary(token):
           audio_chunk = tts.synthesize_chunk(accumulated_tokens)
           play_audio(audio_chunk)
           accumulated_tokens.clear()
   ```

### 方案D：多实例并发优化

#### 当前方案改进
客户现状：6个worker共享一个910B卡，单卡并发6路

#### 优化策略

##### 1. 使用Triton Inference Server
虽然昇腾版Triton成功案例少，但仍是值得尝试的方向：

```yaml
# Triton配置示例
name: "cosyvoice2_ensemble"
platform: "ensemble"
max_batch_size: 6
input [
  {
    name: "text_input"
    data_type: TYPE_STRING
    dims: [ -1 ]
  }
]
output [
  {
    name: "audio_output"
    data_type: TYPE_FP32
    dims: [ -1, 1 ]
  }
]
ensemble_scheduling {
  step [
    {
      model_name: "text_encoder"
      model_version: -1
      input_map {
        key: "INPUT"
        value: "text_input"
      }
      output_map {
        key: "OUTPUT"
        value: "encoded_text"
      }
    },
    {
      model_name: "acoustic_model"
      model_version: -1
      input_map {
        key: "INPUT"
        value: "encoded_text"
      }
      output_map {
        key: "OUTPUT"
        value: "audio_output"
      }
    }
  ]
}
```

##### 2. 动态Batching
```python
# 使用动态batching提升吞吐量
class DynamicBatchInference:
    def __init__(self, model, max_batch_size=6, max_wait_ms=50):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.request_queue = Queue()
    
    async def infer(self, input_data):
        # 将请求加入队列
        future = asyncio.Future()
        self.request_queue.put((input_data, future))
        return await future
    
    async def batch_processor(self):
        while True:
            batch = []
            start_time = time.time()
            
            # 收集batch
            while len(batch) < self.max_batch_size:
                if time.time() - start_time > self.max_wait_ms / 1000:
                    break
                try:
                    item = self.request_queue.get(timeout=0.01)
                    batch.append(item)
                except Empty:
                    if batch:
                        break
            
            if batch:
                # 批量推理
                inputs, futures = zip(*batch)
                outputs = self.model.batch_infer(inputs)
                
                # 返回结果
                for future, output in zip(futures, outputs):
                    future.set_result(output)
```

##### 3. 模型并行 + 数据并行混合
```python
# 将模型拆分到多个NPU核心
# 每个核心处理不同的请求

# 配置1：单卡6路并发
# - 使用6个独立实例
# - 每个实例占用1/6显存

# 配置2：模型并行
# - 将CosyVoice2拆分为多个stage
# - Text Encoder -> Acoustic Model -> Vocoder
# - 每个stage在不同NPU核心上
# - Pipeline并行处理多个请求
```

### 方案E：推理质量优化（解决幻觉token问题）

#### 基于不确定性的质量控制
参考前面的TTS幻觉缓解方案：

```python
class QualityControlledInference:
    def __init__(self, model, uncertainty_threshold=2.5):
        self.model = model
        self.uncertainty_threshold = uncertainty_threshold
    
    def infer_with_quality_control(self, text):
        # 生成speech tokens
        tokens, logits = self.model.generate(text, return_logits=True)
        
        # 计算不确定性
        uncertainties = self.calculate_uncertainty(logits)
        
        # 检测高不确定性区域
        high_uncertainty_regions = self.detect_high_uncertainty(
            uncertainties, 
            threshold=self.uncertainty_threshold
        )
        
        if high_uncertainty_regions:
            # 对高不确定性区域重新生成
            tokens = self.regenerate_uncertain_regions(
                text, 
                tokens, 
                high_uncertainty_regions
            )
        
        # 转换为音频
        audio = self.model.decode(tokens)
        return audio
    
    def calculate_uncertainty(self, logits):
        # Token级别不确定性（熵）
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return entropy
    
    def detect_high_uncertainty(self, uncertainties, threshold):
        # 检测连续的高不确定性区域
        high_uncertainty_mask = uncertainties > threshold
        regions = []
        start = None
        
        for i, is_high in enumerate(high_uncertainty_mask):
            if is_high and start is None:
                start = i
            elif not is_high and start is not None:
                regions.append((start, i))
                start = None
        
        return regions
```

#### 解码策略优化
```python
# 从随机采样改为更确定性的解码
# 方案1：使用beam search
outputs = model.generate(
    input_ids,
    num_beams=4,
    do_sample=False  # 不使用随机采样
)

# 方案2：使用top-k + 温度调整
outputs = model.generate(
    input_ids,
    do_sample=True,
    top_k=10,  # 只从top-10中采样
    temperature=0.7  # 降低温度减少随机性
)

# 方案3：使用nucleus sampling (top-p)
outputs = model.generate(
    input_ids,
    do_sample=True,
    top_p=0.9,  # 累积概率90%
    temperature=0.8
)
```

### 方案F：成本优化方案

#### 问题
当前方案需要450张910B卡，成本过高

#### 优化方向

##### 1. 提升单卡并发能力
- **目标**：从6路提升到10-15路
- **方法**：
  - OM模型优化
  - 动态batching
  - 模型量化（FP16 -> INT8）

##### 2. 模型压缩
```python
# INT8量化
from torch_npu.contrib import transfer_to_npu

# 量化感知训练
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# 转换到NPU
npu_model = transfer_to_npu(quantized_model)
```

##### 3. 模型蒸馏
```python
# 使用CosyVoice 2作为教师模型
# 蒸馏出更小、更快的学生模型

class DistillationTrainer:
    def __init__(self, teacher_model, student_model):
        self.teacher = teacher_model
        self.student = student_model
    
    def distill_loss(self, student_output, teacher_output, labels):
        # KL散度损失
        kl_loss = F.kl_div(
            F.log_softmax(student_output / temperature, dim=-1),
            F.softmax(teacher_output / temperature, dim=-1),
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # 硬标签损失
        hard_loss = F.cross_entropy(student_output, labels)
        
        # 组合损失
        return alpha * kl_loss + (1 - alpha) * hard_loss
```

##### 4. 请求调度优化
```python
class SmartScheduler:
    def __init__(self, num_cards=450):
        self.cards = [CardInfo(i) for i in range(num_cards)]
    
    def schedule_request(self, request):
        # 根据负载选择最优卡
        best_card = min(self.cards, key=lambda c: c.current_load)
        
        # 考虑文本长度和优先级
        if request.priority == "high":
            # 高优先级请求分配到负载较低的卡
            available_cards = [c for c in self.cards if c.current_load < 0.7]
            if available_cards:
                best_card = min(available_cards, key=lambda c: c.current_load)
        
        return best_card.id
```

## 实施路线图

### 阶段1：快速优化（1-2周）
1. 实施OM模型转换方案
2. 使用Profiling工具识别性能瓶颈
3. 优化流式输入pipeline

**预期效果**：RTF从1.2降低到0.8-0.9

### 阶段2：并发优化（1个月）
1. 实现动态batching
2. 优化多实例调度
3. 尝试Triton部署方案

**预期效果**：单卡并发从6路提升到8-10路

### 阶段3：质量优化（1-2个月）
1. 实现基于不确定性的质量控制
2. 优化解码策略
3. 针对910B特性微调模型

**预期效果**：消除幻觉token，质量接近N卡

### 阶段4：成本优化（2-3个月）
1. 模型量化和压缩
2. 模型蒸馏
3. 智能调度系统

**预期效果**：总卡数从450降低到250-300

## 参考资源

### 官方文档
- CANN商用版8.0.RC2文档
- Profiling性能数据采集指南
- ATC工具使用说明

### 开源仓库
- https://gitee.com/ascend/ModelZoo-PyTorch
- CosyVoice2 OM转换示例

### 社区资源
- 华为昇腾论坛
- MindSpore社区
- Torch-NPU GitHub

## 关键联系人
- 华为侧技术支持
- 昇腾社区专家
- 邮箱：woshiqsh1986@126.com（客户联系方式）
