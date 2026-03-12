# 通用AI软件栈

```
┌──────────────────────────────────┐  |------
│          应用与框架层            │  | 负责方: 框架开发商(Meta/Google), 模型库(HuggingFace)     | 超级大厂甚至有自己的框架
│  如：Hugging Face Transformers   │  | 功能: 模型定义、训练、导出                               |
│      PyTorch, TensorFlow         │  | 输出: ONNX/TorchScript/SavedModel                        |
└─────────────┬────────────────────┘  |------
              │ 调用API / 编译器      | 负责方: 框架方+引擎方+标准组织(LF AI & Data)
              |                       | 映射类型: 多对多(网状), 通过ONNX等中间格式
┌─────────────┴────────────────────┐  |------
│          API层                   │  | 负责方: 引擎厂商(MS/NVIDIA/Intel/Qualcomm)               | 大厂敢于定义自己的API层接口
│  ONNX Runtime, TensorRT,         │  |         标准API(Google NNAPI, Khronos OpenVX)            |
│  OpenVINO, Android NNAPI,        │  | 功能: 算子调度、图优化、内存管理                         |
│  Qualcomm SNPE, OpenVX           │  | 输出: 编译器/驱动API调用                                 |
└─────────────┬────────────────────┘  |------
              │ 低层指令编译/硬件调度 | 负责方: 引擎方定义接口+硬件方实现EP/Backend
              |                       | 映射类型: 多对多(网状), 通过EP/Plugin架构
┌─────────────┴────────────────────┐  |------
│        编译器/驱动层             │  | 负责方: NVIDIA/Intel/AMD/Qualcomm/Google + LLVM社区      | 中小厂只开发自己的驱动
│  XLA, MLIR, cuDNN, Hexagon SDK   │  | 功能: 算子编译为硬件指令、设备抽象                       |
│  OpenCL, CUDA Driver             │  | 输出: PTX/SPIR-V/机器码                                  |
└─────────────┬────────────────────┘  |------
              │ 硬件指令执行          | 负责方: 硬件厂商(驱动+JIT编译器)
              |                       | 映射类型: 点对点为主, PTX/SPIR-V→特定ISA
┌─────────────┴────────────────────┐  |------
│          硬件ISA层               │  | 负责方: NVIDIA/Google/Intel/Qualcomm/ARM等芯片厂商
│ NVIDIA Tensor Core ISA, TPU ISA, │  | 功能: 物理指令执行、矩阵乘加速
│ ARM Ethos-N ISA, RISC-V NN扩展   │  | 特性: Tensor Core/TPU/AMX/HVX, FP16/INT8
└──────────────────────────────────┘  |------
```

# 分析后的编译器选型
| 层级           | 模块          | 选型                        | 负责 workload                                           | 多核动态编译 | 需要改    | 说明                     |
| -------------- | ------------- | --------------------------- | ------------------------------------------------------- | ------------ | --------- | ------------------------ |
| Framework      | 动态图框架    | HF / Torch                  | 模型定义、训练、导出、推理前端                          | NO           | NO        | 只负责生成模型           |
| Graph IR       | 中间表示      | ONNX / MLIR                 | 静态图表示、算子图、shape、dtype                        | NO           | NO        | 作为编译输入             |
| Graph Compiler | 图编译器      | TVM / IREE / XLA / TensorRT | 图优化、算子拆分、tiling、多核调度、kernel生成、cmd生成 | YES          | YES       | 多核 & 动态编译核心      |
| Runtime        | 推理运行时    | LiteRT                      | tensor管理、graph执行、调用driver、kernel调度           | NO           | YES(少量) | backend / delegate需要改 |
| Driver         | 内核驱动      | Amlogic driver              | buffer分配、DMA、cmd提交、中断、多核core mask           | YES          | YES       | 硬件接口层               |
| Hardware/Model | RTL / SystemC |                             | cycle级执行/仿真                                        | YES          | YES       | 硬件行为模型             |

| 编译器   | 定位           | 是否适合嵌入式 | 是否易改 | 是否适合自研NPU |
| -------- | -------------- | -------------- | -------- | --------------- |
| TVM      | 可定制编译器   | YES            | YES      | YES             |
| IREE     | MLIR现代编译器 | YES            | YES      | YES             |
| XLA      | Google内部偏重 | MAYBE          | NO       | NO              |
| TensorRT | NVIDIA专用     | MAYBE          | NO       | NO              |


# LLM应用场景
## 主类别分类

| 主类别 ID | 主类别 名称                             | 定义（用于分组）                                                  | 典型系统痛点（概括）                                | 推荐硬件 archetype（首选）                  |
| --------: | --------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------- | ------------------------------------------- |
|         A | 云端大规模服务                          | 面向海量并发、长上下文或高吞吐的云/数据中心部署（SaaS、API 服务） | KV 内存放大、HBM 带宽、跨节点通信、P99 尾延迟       | GPU (HBM) / TPU / Multi-GPU NVLink          |
|         B | <mark>低延迟交互</mark>                 | 面向极低感知延迟（IDE 补全、输入法、游戏 NPC）                    | 每次请求超低尾延迟、极小批次、cache 热点优化        | Small GPU / Low-latency CPU / On-device NPU |
|         C | <mark>移动与边缘</mark>                 | 手机、平板、边缘服务器上的离线或准实时推理                        | 功耗、温控、有限内存、量化友好                      | Mobile NPU / DSP / Edge VPU                 |
|         D | <mark>流与实时系统</mark>               | 流式 ASR/TTS、车载、无人机等严格实时系统                          | 确定性延迟、持续状态管理、RTOS 集成                 | Automotive DSP / Real-time capable NPU      |
|         E | 多模态 / 视觉融合                       | CV + LLM 混合流水线（AR、摄像头 + 自然语言）                      | 异构数据搬运、跨设备 zero-copy、pipeline scheduling | Heterogeneous: GPU + NPU + VPU              |
|         F | 企业 / 垂直行业                         | 合规/隐私、RAG、行业微调（医疗、金融、法律等）                    | 数据隔离、审计、可解释性、微调与版本管理            | CPU + GPU with secure enclaves, VMs         |
|         G | 浏览器 / Web / 开发者端                 | WASM/WebNN、浏览器插件、本地工具                                  | ISA 限制、sandbox、低内存                           | SIMD (WASM) / CPU                           |
|         H | <mark>极小型 / 嵌入（IoT / MCU）</mark> | MCU/极低功耗设备上的 tiny LLM-like 功能                           | SRAM/FLASH 极度受限                                 | MCU DSP / Tiny ML accelerators              |
|         I | 分布式 / 并行                           | 大模型的模型并行/流水并行部署（多卡/多节点）                      | 通信瓶颈、同步开销、延迟/吞吐权衡                   | Multi-GPU clusters + RDMA / NVLink          |
|         J | <mark>在线学习 / 个性化</mark>          | 在推理路径中做少量在线微调或个性化更新                            | 写带宽、一致性、版本回滚                            | GPU + CPU with fast parameter servers       |

## 次类别分类

| 次类别 ID | 次类别 名称                             | 定义（行为/负载特征）                       |  关键指标例示（T_ctx / latency / throughput） | 典型代表场景（示例）          |
| --------: | --------------------------------------- | ------------------------------------------- | --------------------------------------------: | ----------------------------- |
|         1 | <mark>长上下文自回归</mark>             | token-by-token,长 context,KV cache 大且常驻 | T_ctx ≥ 8k,latency per token 任意但总延迟可大 | Chat 长会话、法律文档撰写     |
|         2 | 高并发短请求                            | 短 prompt、短生成、高 QPS、dynamic batch    |                    T_ctx ≤ 2k,TPS 高（100–k） | API 网关型服务、广告/推荐文本 |
|         3 | <mark>低延迟交互</mark>                 | 极低尾延迟,单次预测触发频繁                 |                          target latency <50ms | IDE 补全、输入法              |
|         4 | <mark>Embedding & RAG</mark>            | 向量生成/检索为主,后接 LLM                  |                      embedding QPS 高,IO 密集 | 文档检索、相似性搜索          |
|         5 | 多模态流水线                            | CV preproc → detector → LLM description     |                 CV latency + LLM latency 叠加 | AR 注释、智能摄像头           |
|         6 | <mark>Streaming / Sliding-window</mark> | 流式输入,滑动窗口 attention,持续状态        |                              低延迟的滚动窗口 | 流式 ASR、实时监控            |
|         7 | <mark>Edge / Offline Mobile</mark>      | 离线运行、量化、能耗受限                    |                     small models, T_ctx small | 手机助手、翻译                |
|         8 | 行业深度定制                            | 合规/微调/可解释/审计为核心                 |                       variable,但要求审计日志 | 医疗/法律咨询助手             |
|         9 | Browser / WASM                          | WASM/SIMD 受限执行                          |                             small models only | 浏览器内 LLM、演示工具        |
|        10 | <mark>Tiny / MCU</mark>                 | 超小模型、严格内存                          |                      T_ctx tiny, MB/KB 级内存 | 设备边缘控制                  |
|        11 | Model-parallel / Distributed            | 模型水平切分,通信主导                       |                        massive params（70B+） | 多卡并行推理                  |
|        12 | <mark>On-device fine-tune</mark>        | 少量参数在线更新                            |                        small update bandwidth | 个性化、私有微调              |


## 详细应用场景分析

|    # | Primary Category   | Secondary Category                | Application Scenario                                           | 真实世界里到底在发生什么（长描述）                                                                                    |            模型参数量 & 结构 (P / L / H) | 上下文 & Token 行为 (T_ctx / TPS)     | KV Cache & 内存形态（单会话/单请求近似）                     | 真实痛点 / 难点（系统/运维/用户角度）                                                                             | L4 框架                 | L3 推理 API                   | L2 编译 / 运行时                                      | L1 硬件 ISA                    |
| ---: | ------------------ | --------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------: | ------------------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | ----------------------- | ----------------------------- | ----------------------------------------------------- | ------------------------------ |
|    1 | A (云端大规模服务) | 1 (长上下文自回归)                | 长会话对话服务（Chat SaaS）                                    | 多用户长期会话,逐 token 生成,历史上下文被反复访问与引用；会话长度随时间增长,必须支撑数千—数万并发会话并维护 KV 状态。 |                       70B / ~80 / H≈8192 | T_ctx 8k–32k / TPS 30–100 (per model) | KV ≈ 20 GB / 会话 (70B、8k ctx, FP16 估算)；多会话合计 TB 级 | **主瓶颈：内存/带宽**（KV 常驻、迁移代价高）；调度在会话间切换需保证 locality；显存不足需分片（带来通信复杂度）。 | PyTorch (HF)            | TensorRT + ONNX Runtime       | KV-aware kernel fusion、CUDA Graph、session scheduler | GPU Tensor Core (MMA) / HBM    |
|    2 | A                  | 2 (高并发短请求)                  | 公共 API（高并发短 prompt）                                    | 成千上万短请求并发涌入,prompt & 输出都较短；系统需稳定 P95/P99 延迟,batch size 动态变化。                             |                13B–30B / 24–60 / H≈4k–5k | T_ctx 512–2k / TPS 200–k              | 单请求 KV 很小（0.05–0.5 GB）；但并发时内存池化管理消耗大    | **主瓶颈：调度抖动与 kernel-launch overhead**；动态 shape 降低 kernel reuse；内存碎片化与频繁分配。               | PyTorch                 | ONNX Runtime (ORT)            | Dynamic scheduler、lightweight worker pools           | GPU SIMD + MMA                 |
|    3 | A                  | 3 (低延迟交互)                    | 云端低延迟短生成（SaaS 低尾延迟）                              | 每请求 token 数少但 QPS 高,目标是极低 P99（例如客服响应或搜索建议）,不能牺牲 tail latency。                           |                 6B–13B / 24–40 / H≈3k–5k | T_ctx 256–1k / TPS 100–300            | KV 很小（几十到几百 MB）或无                                 | **主瓶颈：launch overhead 与 kernel specialization**；需 serialized engines、减少上下文切换。                     | PyTorch                 | TensorRT (serialized engines) | static compilation、kernel specialization             | GPU Tensor Core                |
|    4 | A                  | 11 (Model-parallel / Distributed) | 多卡/多节点大模型并行推理（70B+）                              | 将超大模型切分到多 GPU/节点（tensor/pipeline parallel）,需高速互联（NVLink、RDMA）保证低延迟和高吞吐。                |              70B–175B / 80–96 / H≈8k–16k | T_ctx 8k–64k / 吞吐以 batch 为主      | KV 分布式：跨卡分片（每卡若干到十几 GB）                     | **主瓶颈：通信与同步**（all-reduce、KV 跨卡访问）；复杂的调度与错误恢复；网络带宽与延迟决定可扩展性。             | PyTorch + DeepSpeed     | TensorRT + ORT with MP hooks  | NCCL / GDR / overlapped comm+comp                     | Multi-GPU Tensor Core + NVLink |
|    5 | A/F                | 4 (Embedding & RAG)               | <mark>Embedding 服务（大规模向量生成与检索）</mark>            | 文档/查询被编码为向量并存入索引（FAISS/Redis）,随后用于 RAG；embedding 生成是高 QPS、IO 密集型任务。                  |                     50M–1B (embed model) | per-doc length / very high QPS        | 向量大小通常 0.5–4 KB,cache 层次（DRAM/SSD）                 | **主瓶颈：IO 与缓存命中率**；向量检索延迟与并发受限于 IO、RAM；预取与批处理策略重要。                             | PyTorch / TF            | ONNX Runtime                  | batched GEMM + IO prefetch                            | CPU/GPU SIMD + High-memory BW  |
|    6 | F                  | 4 & 8 (RAG, 行业)                 | 企业内网 RAG（私有知识库问答）                                 | 检索层返回若干文本片段,LLM 将其作为 context 生成答案；企业侧重数据驻留与审计合规。                                    |                 7B–13B / 24–40 / H≈4k–5k | T_ctx 2k–8k / TPS 5–50                | embedding cache + LLM KV 中等（0.5–2 GB）                    | **主瓶颈：数据移动与合规**（检索→LLM 的拷贝与格式转换）；审计/权限与模型解释需求增加系统复杂度。                  | PyTorch                 | ORT / OpenVINO                | memory-aware lowering、secure data marshaling         | x86 AVX512 / GPU               |
|    7 | A/B/G              | 2 & 3 (高并发 / 低延迟)           | <mark>代码补全（云 + 本地混合）</mark>                         | IDE 发起高频短请求,本地与云需平滑切换（隐私/延迟平衡）,云端提供强能力补全。                                           |                local 1–7B; cloud 13B–70B | T_ctx 256–4k / TPS local 100–1000     | local KV 很小（几十 MB）,cloud KV 大                         | **主痛点：混合路由与一致性**；本地降级与云端切换需透明且无感知；缓存策略复杂。                                    | PyTorch                 | TensorRT (cloud) + ORT Mobile | hybrid routing、KV-aware fusion                       | GPU Tensor Core / CPU AVX      |
|    8 | B                  | 3 (低延迟交互)                    | <mark>本地 IDE 插件：行/块级补全</mark>                        | 用户每键触发补全,期望 <50ms 响应；模型常量驻留内存,尽量避免动态分配。                                                 |                 0.5–3B / 12–24 / H≈2k–4k | T_ctx 128–1k / TPS 200–1000           | KV 极小 (<100 MB)                                            | **主瓶颈：延迟可预测性**；任何内存 swap 或 GC 会打断用户体验。                                                    | PyTorch / PT Mobile     | ORT Mobile                    | static scheduling、quantized kernels                  | CPU AVX512 / small GPU         |
|    9 | C                  | 7 (Edge / Offline Mobile)         | <mark>手机离线助理（隐私、可用性）</mark>                      | 在移动设备上离线运行对话或翻译（无网络）,必须量化与结构剪枝以满足功耗/热约束。                                        |                  0.5–1B / 8–24 / H≈1k–2k | T_ctx 256–512 / TPS 1–10              | KV <200 MB,通常驻留 NPU 内存                                 | **主瓶颈：功耗与热**；INT8/INT4 量化精度与用户体验权衡；NPU/kernel mapping 多厂商差异大。                         | TFLite                  | NNAPI                         | vendor compiler（量化 & kernel mapping）              | Mobile NPU ISA                 |
|   10 | C                  | 7 & 3 (Edge + 低延迟)             | <mark>手机输入法 / 键盘预测</mark>                             | 每击键触发,极高调用率与极低延迟需求；多数时间处于休眠,但必须瞬时唤醒。                                                |                <300M / 6–12 / H≈512–1024 | T_ctx <128 / TPS 500+                 | KV 极小,几 MB 或更小                                         | **主瓶颈：latency & wake-up cost**；runtime 必须超轻量且常驻小内存。                                              | TFLite                  | NNAPI                         | static scheduling、tiny kernels                       | DSP SIMD (Hexagon)             |
|   11 | D                  | 6 (Streaming / Sliding-window)    | <mark>车载语音助手（滑动窗口）</mark>                          | 连续语音流进入,系统在 sliding-window 上实时运行理解/对话；安全与确定性要求高。                                        |                     <1B / 8–16 / H≈1k–2k | sliding window / real-time            | rolling KV 小（常驻 RAM）                                    | **主瓶颈：确定性 & RT 性能**；中断不能导致长尾延迟；RTOS 集成与优先级调度必要。                                   | TFLite                  | NNAPI                         | DSP runtime with RT scheduling                        | Automotive DSP vector ISA      |
|   12 | D/E                | 5 (Multi-modal pipeline)          | <mark>流式 ASR → LLM 联动</mark>                               | ASR 输出须实时喂入 LLM 做意图/上下文管理；ASR 的时间特性要求 LLM 低延迟接收并做决策。                                 |                  ASR 10–100M; LLM 0.5–2B | sliding window / low-latency          | KV 小/中（ASR state + LLM KV）                               | **主瓶颈：跨模型流水线协调**；buffering/dropping 策略会影响连贯性。                                               | Kaldi / PyTorch         | ORT / NNAPI                   | pipelined exec、low-latency buffers                   | CPU SIMD / DSP                 |
|   13 | D                  | 6 (Streaming)                     | <mark>实时会议摘要 / 多人转写 & 摘要</mark>                    | 多路语音流并行,需要低延迟转写并实时生成会议要点,包含多说话人分离。                                                    |                    ASR models + LLM 1–7B | sliding / partial updates             | KV 小—中（多 speaker context）                               | **主瓶颈：并发 & 数据聚合**；时间同步、speaker diarization 带来复杂性；隐私合规需求。                             | PyTorch / Kaldi         | ORT                           | stream-aware scheduler                                | CPU / GPU                      |
|   14 | E                  | 5 (Multi-modal)                   | AR 眼镜语义助手（实时 CV + LLM）                               | 摄像头帧被实时处理后（检测/分割/跟踪）,结果被送给 LLM 生成语义提示或交互反馈；功耗与延迟关键。                        |                            <1B (trimmed) | T_ctx 256–512 / TPS 5–20              | mixed: CV buffers + small KV                                 | **主瓶颈：异构数据搬运**（GPU↔NPU↔CPU）；zero-copy 很关键,pipeline scheduling 难度高。                            | TFLite / PyTorch Mobile | OpenVX + NNAPI                | graph scheduler、zero-copy buffers                    | GPU + Mobile NPU               |
|   15 | E                  | 5 (Multi-modal)                   | 智能摄像头（CV→LLM 描述）                                      | 边缘设备做检测/跟踪,再用小型 LLM 生成自然语言描述；低功耗与实时性是关键。                                             |                        <1B / 4–12 / H≈1k | T_ctx 128–256 / TPS <5                | KV 很小,主要为 CV intermediate buffers                       | **主瓶颈：功耗 & memory footprint**；CV 与 LLM 竞用内存,优先级/arbiter 需要精细化。                               | TFLite                  | OpenVX                        | heterogeneous runtime with priority                   | Embedded NPU / VPU             |
|   16 | F                  | 8 (Industry vertical)             | 医疗临床助手（受控、合规）                                     | 敏感数据、审计、模型可解释性、可回溯性为关键；部署常在企业内网或合规云。                                              |                             7–13B (微调) | T_ctx 2k–8k / TPS 1–10                | KV 中等（保存患者上下文需加密）                              | **主瓶颈：合规 & 隐私**；隔离/加密/审计影响部署架构；性能/合规 tradeoff。                                         | PyTorch                 | ORT / OpenVINO                | secure runtime、memory isolation                      | CPU / GPU enclave              |
|   17 | F                  | 8 (Industry vertical)             | 金融低延迟风控                                                 | 高频流输入需极低尾延迟,决策延迟直接与经济损失挂钩；模型需解释与审计。                                                 |                                     1–7B | sliding / real-time                   | KV 小                                                        | **主瓶颈：极低尾延迟 & 可审计**；可采用专用低延迟硬件路径（FPGA/ NIC）                                            | PyTorch                 | ORT                           | real-time optimized stacks                            | FPGA + low-latency NIC + CPU   |
|   18 | F                  | 4 (RAG)                           | 企业知识库问答 / 法律咨询                                      | 文档检索返回若干片段作为 context,LLM 生成并提供引用/出处（高可解释性要求）。                                          |                             7–13B (微调) | T_ctx 2k–8k / TPS 1–20                | embedding cache + LLM KV 中等                                | **主瓶颈：数据本地性与版本控制**；审计/解释与检索文档一致性要保障。                                               | PyTorch                 | ORT                           | memory-aware lowering、secure pipelines               | x86 AVX / GPU                  |
|   19 | G                  | 9 (Browser / WASM)                | <mark>浏览器内 LLM（演示 / demo /隐私）</mark>                 | 在浏览器内用 WASM/WebNN 运行小模型以实现无需后端的交互或 POC；受限于浏览器资源和 sandbox。                            |                           <1B (100–500M) | T_ctx 256–512 / TPS 1–10              | KV <100 MB（受 tab memory limit）                            | **主瓶颈：ISA 与 memory sandbox 限制**；无法利用 MMA / HBM,受限于 SIMD128。                                       | ONNX (converted)        | WebNN                         | WASM runtime、SIMD128                                 | SIMD128 (WASM)                 |
|   20 | H                  | 10 (Tiny / MCU)                   | <mark>MCU 上的极小模型（控制 / 规则补全）</mark>               | 在 MCU/嵌入式传感器上运行微模型做规则决策或短文本映射,RAM/Flash 极度受限。                                            |                       <50M / tiny layers | T_ctx <64 / TPS 0.1–1                 | KV tiny (<MB)                                                | **主瓶颈：极端资源限制**；需 MicroTVM/CMSIS-NN,模型结构高度简化                                                   | MicroTVM                | CMSIS-NN                      | bare-metal runtime                                    | MCU DSP / SIMD                 |
|   21 | C/I                | 11 (Model-parallel / Distributed) | <mark>多节点/边缘集群做并行推理（分布式边缘推理）</mark>       | 多边缘节点协同执行分片模型以支持更大模型或更长上下文,受网络条件影响。                                                 |                        13B–70B (sharded) | variable                              | KV 分布式,需跨节点访问                                       | **主瓶颈：网络可靠性 & partitioning**；edge 节点 heterogeneity 增加调度复杂度。                                   | PyTorch                 | ORT + custom shard manager    | distributed scheduler、resilience                     | Edge GPUs / NICs               |
|   22 | J                  | 12 (On-device fine-tune)          | <mark>在线微调 / 个性化（on-device / federated）</mark>        | 在用户设备或隐私云对少量参数做个人化更新（adapter）,同时继续提供推理服务。                                            |                base 13B + adapters few M | T_ctx 1k–4k / TPS low                 | KV 小                                                        | **主瓶颈：写带宽、一致性与 rollback**；需 mini-batch update path 并保证不影响在线推理                             | PyTorch                 | ORT + fine-tune hooks         | mixed-precision update paths                          | GPU Tensor Core + CPU          |
|   23 | A/F                | 2 & 4 (High QPS & Embedding)      | <mark>内容审核流水线（大吞吐）</mark>                          | 文本流快速通过轻量分类层过滤,少量可疑流入 LLM 做深入分析；需高吞吐与可审计。                                          |                                     1–7B | T_ctx 128–1k / TPS very high          | KV 小                                                        | **主瓶颈：吞吐 & 可审计性**；批处理与并发优化是关键                                                               | PyTorch                 | ORT                           | batched pipelines                                     | CPU + GPU                      |
|   24 | F                  | 8 (Industry vertical)             | <mark>教育智能辅导（长期个性化）</mark>                        | 系统需长期追踪学生 profile,并在对话中结合历史信息做个性化教学与反馈。                                                 |                                    3–13B | T_ctx 2k–8k / TPS 1–10                | KV 中等（学生 profile）                                      | **主瓶颈：状态管理 & privacy**；长期存储与检索效率影响体验                                                        | PyTorch                 | ORT                           | session-aware runtime                                 | CPU / GPU                      |
|   25 | A                  | 11 (Model-parallel / Batch)       | <mark>日志聚合与离线摘要（批处理）</mark>                      | 海量日志/文档离线聚合、抽取摘要,非实时以吞吐为主,常在夜间批处理。                                                     |                       7–70B (batch jobs) | batch-oriented / high throughput      | KV 按作业分配                                                | **主瓶颈：IO 与并发调度**；批处理能够 amortize comp & IO                                                          | PyTorch                 | TensorRT / ORT                | batch pipelines                                       | GPU / TPU                      |
|   26 | C/E                | 5 (Multi-modal)                   | 视频理解 + LLM（短视频摘要）                                   | 连续视频帧做检测/跟踪/关键帧抽取,把结果送 LLM 做摘要或标题生成；多媒体 IO 与计算并存。                                | CV models (tens–hundreds M) + LLM 0.5–7B | T_ctx per clip / TPS low              | CV buffers + small KV                                        | **主瓶颈：IO 带宽与 GPU memory scheduling**；frame preproc 成本大                                                 | PyTorch                 | OpenVINO / ORT                | heterogeneous scheduling                              | GPU + VPU                      |
|   27 | B/C                | 3 & 7 (Low-latency + Edge)        | <mark>游戏内 NPC 即时对话<mark>                                | 玩家与 NPC 的互动需要实时回复,延迟感知敏感,常部署本地或边缘。                                                         |                                   0.5–3B | T_ctx 128–1k / TPS low                | KV 小                                                        | **主瓶颈：deterministic low-latency & concurrency**；runtime must be predictable                                  | PyTorch Mobile          | NNAPI / ORT Mobile            | real-time tuned runtime                               | Mobile NPU / GPU               |
|   28 | G                  | 9 (Browser)                       | <mark>本地演示/教育的小模型交互</mark>                         | 离线 demo、教育用例在浏览器中运行,强调可用性与无需后端。                                                              |                                  50–500M | T_ctx 128–512 / TPS low               | KV <100 MB                                                   | **主瓶颈：JS/WASM 性能 & memory sandbox**；无法利用 hardware MMA                                                  | ONNX → WebNN            | WebNN                         | WASM runtime                                          | SIMD128                        |
|   29 | H                  | 10 (Tiny)                         | <mark>远端工业传感器（极低功耗）</mark>                        | 传感器在现场做极小模型推断（触发事件）,要求极高可靠性与极低功耗。                                                     |                                  <10–50M | T_ctx tiny / TPS low                  | KV tiny (KB–MB)                                              | **主瓶颈：model size & numerical range**；需定制 quantization & pruning                                           | MicroTVM                | CMSIS-NN                      | bare-metal RT                                         | MCU DSP                        |
|   30 | I                  | 11 (Distributed)                  | 弹性云 GPU 池（多租户）                                        | 多租户共享 GPU 池,需快速上下文切换与隔离（预防 noisy neighbor）。                                                     |                                 variable | variable                              | KV pooled & isolated slices                                  | **主瓶颈：isolation & QoS**；需 orchestrator 做资源隔离                                                           | PyTorch                 | ORT + scheduler               | containerized runtimes                                | GPU Tensor Core                |
|   31 | E/F                | 5 & 4 (Multi-modal + RAG)         | 电商多模态搜索（图像 + 文本 + LLM）                            | 用户上传图片检索相似商品并生成自然语言推荐；系统结合 embedding、检索与 LLM 生成。                                     |              CV model 50–200M + LLM 1–7B | T_ctx image meta / TPS medium         | embedding cache + small LLM KV                               | **主瓶颈：heterogeneous orchestration & latency**；检索→LLM 切换频繁                                              | PyTorch                 | ORT / OpenVINO                | graph scheduler                                       | GPU + VPU                      |
|   32 | F                  | 8 (Industry)                      | 法律合同分析（合规审计）                                       | 对合同做结构化抽取并由 LLM 生成法律意见,需证明性输出与审计轨迹。                                                      |                             7–13B (微调) | T_ctx 4k–16k / TPS low                | KV 中等                                                      | **主瓶颈：可解释性与追溯**；日志/审计体系影响 runtime 设计                                                        | PyTorch                 | ORT                           | secure & auditable runtime                            | CPU / GPU                      |
|   33 | B                  | 3 (Low-latency)                   | <mark>实时游戏语音 NPC（edge）</mark>                          | 语音输入需要边缘识别并即时生成 NPC 语句；网络不可靠时需本地回退                                                       |                                   0.5–2B | sliding / low-latency                 | KV small                                                     | **主瓶颈：network resilience & deterministic latency**                                                            | PyTorch Mobile          | NNAPI                         | RT tuned runtime                                      | Mobile NPU                     |
|   34 | A/C                | 2 & 7 (High QPS + Edge)           | <mark>边缘服务器批量推理（零售/门店）</mark>                   | 门店内边缘服务器本地聚合顾客请求以降低 cloud latency & 带宽                                                           |                                    3–13B | T_ctx 1k–4k / TPS medium              | KV pooled (several GB)                                       | **主瓶颈：edge resource provisioning & HA**；需轻量化 orchestrator                                                | PyTorch                 | ORT / OpenVINO                | NUMA-aware runtime                                    | Edge GPU / VPU                 |
|   35 | F                  | 8 (Industry)                      | 医疗影像报告 + LLM 生成                                        | 影像模型先做异常检测,结果总结由 LLM 生成诊断文本,需合规                                                               |                    CV models + LLM 7–13B | T_ctx 2k / TPS low                    | CV buffers + small KV                                        | **主瓶颈：合规 & latency**；影像前处理消耗大                                                                      | PyTorch                 | ORT / OpenVINO                | heterogeneous pipeline                                | GPU + VPU                      |
|   36 | A                  | 2 (High QPS)                      | <mark>广告 / 推荐短文本生成</mark>                             | 在用户行为流触发短文本生成做个性化推荐,极高吞吐                                                                       |                                     1–7B | T_ctx short / TPS very high           | KV tiny per request                                          | **主瓶颈：latency to revenue / cost per token**；频繁短请求需极优 batching                                        | PyTorch                 | ORT                           | batched pipelines                                     | CPU / GPU                      |
|   37 | J                  | 12 (On-device fine-tune)          | <mark>联邦学习 + 个性化适配</mark>                             | 多设备在本地训练少量 adapter 参数并合并（隐私保护）,更新需低延迟应用到推理                                            |                     adapter few M params | T_ctx variable                        | KV small                                                     | **主瓶颈：通信 & secure aggregation**；在线 rollback 与 model consistency                                         | PyTorch                 | ORT + FL hooks                | parameter server, secure aggregator                   | CPU / GPU                      |
|   38 | E                  | 5 (Multi-modal)                   | 超大视觉语言模型（cloud VL）                                   | 图像+长文本多模态理解,常处理高分辨率图像与长 caption；memory heavy                                                    |                          VL models 3–70B | image + text context long             | KV moderate–large                                            | **主瓶颈：memory & compute concurrency**；图像 transformer 中间激活大                                             | PyTorch                 | ORT / TensorRT                | memory/prefetch scheduling                            | Multi-GPU Tensor Core          |
|   39 | B/G                | 3 & 9 (Low-latency Browser)       | <mark>浏览器辅助写作（local + cloud hybrid）</mark>            | 在浏览器 UI 中对接本地 lightweight model 给出提示,复杂请求发 cloud                                                    |             local 100–500M; cloud larger | T_ctx 256–1k / TPS medium             | local KV tiny                                                | **主瓶颈：seamless hybrid routing & privacy**                                                                     | ONNX / PyTorch          | WebNN + ORT                   | hybrid routing                                        | SIMD128 + Edge GPU             |
|   40 | I/A                | 11 & 1 (Distributed long-context) | 分布式 long-context retrieval-augmented reasoning（跨节点 KV） | 长上下文被 sharded 到多个节点,推理时需跨节点访问 KV 并保证一致                                                        |                             70B+ sharded | T_ctx 8k–64k / TPS low                | KV 分布 TB 级                                                | **主瓶颈：cross-node KV latency & correctness**；硬件层面需高速 interconnect 与 smart caching                     | PyTorch + DeepSpeed     | ORT + MP hooks                | distributed KV cache, coherence                       | Multi-node GPU clusters        |
|   41 | F                  | 8 (Industry)                      | 临床试验文本分析（批量/合规）                                  | 大量临床文本抽取关键信息,并在合规约束下生成报告                                                                       |                             7–13B (微调) | T_ctx 1k–8k / TPS low                 | KV 中等                                                      | **主瓶颈：合规性 & explainability**                                                                               | PyTorch                 | ORT                           | secure pipelines                                      | CPU / GPU                      |
|   42 | C                  | 7 (Edge)                          | <mark>零售店员机器人（本地对话+检索）</mark>                   | 机器人本地处理问答并检索本地数据库,考虑断网与隐私                                                                     |                                     1–3B | T_ctx 512–2k / TPS low                | KV local (hundreds MB–GB)                                    | **主瓶颈：on-device storage & model updates**                                                                     | PyTorch Mobile          | ORT Mobile                    | on-device update manager                              | Edge GPU / NPU                 |
|   43 | A                  | 2 (High QPS)                      | 社交媒体内容生成 / 推荐文案                                    | 高频且多变的生成请求,需与 moderation pipeline 协同                                                                    |                                     1–7B | T_ctx short / TPS high                | KV small                                                     | **主瓶颈：latency & safety**；需低成本批处理并保证合规                                                            | PyTorch                 | ORT                           | batched pipelines                                     | CPU / GPU                      |
|   44 | F                  | 8 (Industry)                      | 法规合规自动化（批审）                                         | 自动读取法规并生成合规摘要,需版本化与可追溯                                                                           |                                    3–13B | T_ctx 2k–8k / TPS low                 | KV moderate                                                  | **主瓶颈：traceability & reproducibility**                                                                        | PyTorch                 | ORT                           | reproducible pipelines                                | CPU / GPU                      |
|   45 | B/E                | 3 & 5 (Low-latency multimodal)    | <mark>实时 AR 翻译（语音 + 文字 + 图像）</mark>                | 语音转文本→翻译→显示/语音合成,图像场景上下文辅助翻译                                                                  |       multiple small models + LLM 0.5–3B | sliding / low-latency                 | KV small                                                     | **主瓶颈：pipeline latency & synchronization**                                                                    | TFLite / PyTorch Mobile | NNAPI + OpenVX                | synchronized low-latency pipelines                    | Mobile NPU + DSP               |
|   46 | G                  | 9 (Browser)                       | <mark>教育交互式笔记（client-side summarization）</mark>       | 学生在网页上即时生成摘要/提问,强调隐私与响应时间                                                                      |                                 100–500M | T_ctx 256–512 / TPS low               | KV small                                                     | **主瓶颈：browser resource limits**                                                                               | ONNX                    | WebNN                         | WASM runtime                                          | SIMD128                        |
|   47 | H                  | 10 (Tiny)                         | <mark>可穿戴设备通知摘要</mark>                                | 可穿戴设备对通知文本做极短摘要以节省用户注意力                                                                        |                                     <50M | T_ctx 32–128 / TPS low                | KV tiny                                                      | **主瓶颈：energy & memory**                                                                                       | MicroTVM                | CMSIS-NN                      | bare-metal                                            | MCU DSP                        |
|   48 | I                  | 11 (Distributed)                  | <mark>边缘-云协同并行推理（elastic offload）</mark>            | latency-sensitive 部分在 edge,本地无法处理时 offload 到云并整合结果                                                   |                              13B sharded | variable                              | KV sharded & cached                                          | **主瓶颈：offload latency & consistency**                                                                         | PyTorch                 | ORT + edge orchestrator       | offload manager                                       | Edge GPU + Cloud GPU           |
|   49 | J                  | 12 (On-device fine-tune)          | <mark>个性化写作建议（edge adapter）</mark>                    | 在用户设备上微调少量 adapter 以捕捉个人风格,更新需即时生效                                                            |               base 7–13B + adapter few M | T_ctx 1k                              | KV small                                                     | **主瓶颈：update atomicity & privacy**                                                                            | PyTorch                 | ORT + fine-tune hooks         | local adapter manager                                 | CPU / GPU                      |
|   50 | A/C/F              | 2/7/8 (mixed)                     | <mark>企业呼叫中心智能化（混合 cloud/edge）</mark>             | 呼叫中心实时识别、摘要与智能建议；对隐私/合规/低延迟都有要求                                                          |                          ASR + LLM 1–13B | sliding / low-latency                 | KV per call small–medium                                     | **主瓶颈：hybrid orchestration & regulatory constraints**                                                         | PyTorch                 | ORT / TensorRT                | hybrid routing                                        | GPU + CPU + DSP                |

# GPU是不是不如DSA?
即使在今天, 对于<mark>超高并发</mark>的<mark>先进大型模型</mark>的在线访问, 云厂商仍旧是在用抽象程度更高, 单位计算任务 功耗,时间<mark>更高成本</mark>的GPU来硬抗这部分算力需求, 而不是依赖专用的DSA(domain-specific-arch)

头部AI大厂的硬件投入:
- 一小半在训练: 这部分任务呈现burst形态, 可以排队, 可以等, 可以停, 理论上可以多个厂商共享这部分算力资源
  - 训练用 GPU(训练速度): NVIDIA H100, A100, B200, GB200
- 一大半在推理: 超高并发, 任务量极大且相对稳定, 对qos, 延迟非常敏感, 是营收的主要组成部分, 且一直有增长预期
  - 推理用 GPU(能效比): NVIDIA L40S, L4, A10, A16

- 原因:
  - GPU有远超其他架构的<mark>通用并发</mark>能力(后面会详细介绍), 且天然支持更好的qos/延时稳定性表现
  - 先进大模型的架构在一直更替, 算法固化条件不成熟:
    - prompt结构变
    - 新模型几个月一代(计算图架构不一样)
    - MoE/多模态/CoT/MCP, 训练/推理方式一直在变
    - KV Cache策略不断调整
  - 模型迭代速度:
    - 大型LLM训练(GPU) > 大型LLM推理(GPU) >> 小型LLM推理(DSA)
    - 模型迭代速度慢下来是硬件DSA化的前提条件

- 结论:
  - 只要NVIDIA积极迭代, 跟随学界/业界趋势/需求, 快速支持新特性和算子, 大厂的先进大型模型将持续被通用型的并行GPU所主导, <mark>短时间不存在向DSA过渡的趋势</mark>
  - DSA的主要应用场景是<mark>边缘设备</mark>上<mark>极致能耗比</mark>需求的<mark>小模型</mark>

## RTX 5090 GPU 作为例子

### RTX 5090 硬件架构

```
                                                                                                     ┌────────────────────────────────────────────────────────────────────────────────────────┐
                                                                                                     │  单个SM详细结构 (170个SM中的一个, 物理大小~3mm²)                                       │
                                                                                                     │  ══════════════════════════════════════════════════════════════════════════════════    │
                                                                                                     │                                                                                        │
【图1】完整硬件层次: Die → GPC → SM 物理架构                                                         │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                         │  │  Instruction Cache: 32KB (存储已编译PTX指令)                                      │ │
┌──────────────────────────────────────────────────────────────────────────────────────┐             │  └───────────────────────────────────────────────────────────────────────────────────┘ │
│                                RTX 5090 GPU Die                                      │             │                                     ↓ 指令流                                           │
│                             (600mm² 硅片, 760亿晶体管)                               │             │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│                                                                                      │             │  │  Warp Scheduler 阵列 × 4 (独立调度器)                                             │ │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │             │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐                       │ │
│  │                         GPC 阵列 × 12 (Graphics Processing Clusters)            │ │             │  │  │  Warp     │  │  Warp     │  │  Warp     │  │  Warp     │                       │ │
│  │  ┌────────────────────────────────────────────────────────────────────────────┐ │ │             │  │  │Scheduler 0│  │Scheduler 1│  │Scheduler 2│  │Scheduler 3│                       │ │
│  │  │  单个GPC (12个中的一个)                                                    │ │ │             │  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘                       │ │
│  │  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │ │             │  │        │              │              │              │                             │ │
│  │  │  │  SM 阵列 × 14-15 (Streaming Multiprocessor) ↓展开下方                │  │ │ │             │  │  管理16-20 Warp  管理16-20 Warp  管理16-20 Warp  管理16-20 Warp                   │ │
│  │  │  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                    │  │ │ │             │  │  (共享64个Warp/SM ÷ 4个Scheduler)                                                 │ │
│  │  │  │  │SM 0│ │SM 1│ │SM 2│ │SM 3│ │... │ │SM13│ │SM14│                    │  │ │ │             │  └───────────────────────────────────────────────────────────────────────────────────┘ │
│  │  │  │  └────┘ └────┘ └────┘ └────┘ └────┘ └────┘ └────┘                    │  │ │ │             │         │              │              │              │                                 │
│  │  │  └──────────────────────────────────────────────────────────────────────┘  │ │ │             │         └──────────────┴──────────────┴──────────────┘                                 │
│  │  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │ │             │                            ↓ 发射指令                                                  │
│  │  │  │  Raster Engine (光栅化引擎)                                          │  │ │ │             │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│  │  │  └──────────────────────────────────────────────────────────────────────┘  │ │ │             │  │  Processing Blocks × 4 (分区执行单元)                                             │ │
│  │  │  ┌──────────────────────────────────────────────────────────────────────┐  │ │ │             │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │ │
│  │  │  │  Texture Units (纹理单元)                                            │  │ │ │             │  │  │  Block 0     │  │  Block 1     │  │  Block 2     │  │  Block 3     │           │ │
│  │  │  └──────────────────────────────────────────────────────────────────────┘  │ │ │             │  │  │              │  │              │  │              │  │              │           │ │
│  │  │          ↕ GPC Crossbar 连接到 L2 Cache                                    │ │ │             │  │  │ CUDA Core×32 │  │ CUDA Core×32 │  │ CUDA Core×32 │  │ CUDA Core×32 │           │ │
│  │  └────────────────────────────────────────────────────────────────────────────┘ │ │             │  │  │ INT32×16     │  │ INT32×16     │  │ INT32×16     │  │ INT32×16     │           │ │
│  │  注: 全GPU共170个SM = 12个GPC × ~14个SM/GPC                                     │ │             │  │  │ Tensor×4     │  │ Tensor×4     │  │ Tensor×4     │  │ Tensor×4     │           │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │             │  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘           │ │
│                                        ↕                                             │             │  │                                                                                   │ │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │             │  │  总计: 128 CUDA Cores (FP32/FP16) | 64 INT32 Units | 16 Tensor Cores              │ │
│  │  L2 Cache: 40MB (全GPU共享)                                                     │ │             │  └───────────────────────────────────────────────────────────────────────────────────┘ │
│  │  ┌───┬───┬───┬───┬───┬───┬───┬───┐                                              │ │             │                                        ↕                                               │
│  │  │ 5 │ 5 │ 5 │ 5 │ 5 │ 5 │ 5 │ 5 │ MB × 8 slices                                │ │             │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│  │  └───┴───┴───┴───┴───┴───┴───┴───┘                                              │ │             │  │  Register File: 65,536 × 32bit = 256KB                                            │ │
│  │  延迟: ~200 cycles (~80ns) | 带宽: ~5TB/s | Cache Line: 128B                    │ │             │  │  - 访问延迟: 1 cycle | 带宽: ~20TB/s | 每SM私有                                   │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │             │  └───────────────────────────────────────────────────────────────────────────────────┘ │
│                                        ↕                                             │             │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │             │  │  L1 Data Cache / Shared Memory: 128KB (可配置)                                    │ │
│  │  Memory Controllers × 16                                                        │ │             │  │  - 配置: 64KB/64KB, 96KB/32KB, 32KB/96KB                                          │ │
│  │  [MC0][MC1][MC2][MC3][MC4][MC5][MC6][MC7][MC8][MC9][MC10]...[MC15]              │ │             │  │  - L1: 自动管理 | Shared: 程序可控 | 延迟: ~20-30 cycles                          │ │
│  │   ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓     ↓        ↓                 │ │             │  │  - 32个bank, 支持并发访问                                                         │ │
│  │  ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐  ┌──┐    ┌──┐                │ │             │  └───────────────────────────────────────────────────────────────────────────────────┘ │
│  │  │2G│ │2G│ │2G│ │2G│ │2G│ │2G│ │2G│ │2G│ │2G│ │2G│  │2G│... │2G│ GDDR7 chips    │ │             │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
│  │  └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘ └──┘  └──┘    └──┘                │ │             │  │  其他单元:                                                                        │ │
│  │  总容量: 32GB | 总带宽: 1TB/s | 每MC: 256-bit bus                               │ │             │  │  • Texture Cache: 16KB                                                            │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │             │  │  • Load/Store Units (LD/ST): 32个 (处理内存读写)                                  │ │
│                                                                                      │             │  │  • Special Function Units (SFU): 16个 (sin, cos, sqrt, log, exp)                  │ │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │             │  └───────────────────────────────────────────────────────────────────────────────────┘ │
│  │  Interconnect (NoC - Network on Chip): 连接所有SM, L2, Memory Controllers       │ │             │                                                                                        │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │             │  ┌───────────────────────────────────────────────────────────────────────────────────┐ │
└──────────────────────────────────────────────────────────────────────────────────────┘             │  │  硬件资源限制:                                                                    │ │
                                                                                                     │  │  • 最多常驻: 2048 threads (64个Warp)                                              │ │
                                                                                                     │  │  • 最多常驻: 32个 Thread Blocks                                                   │ │
                                                                                                     │  │  • Register分配上限: 65536个                                                      │ │
                                                                                                     │  │  • Shared Memory上限: 64KB (可用部分)                                             │ │
                                                                                                     │  └───────────────────────────────────────────────────────────────────────────────────┘ │
                                                                                                     └────────────────────────────────────────────────────────────────────────────────────────┘
```

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│  关键数字:                                                                             │
│  • 170 SM × 128 CUDA Cores/SM = 21,760 CUDA Cores                                      │
│  • 每SM最多常驻: 2048 threads (64 Warps) + 32 Blocks                                   │
│  • 全GPU理论峰值: 108 TFLOPS (FP32)                                                    │
│  • 内存带宽: 1TB/s (Global) + 5TB/s (L2) + 10TB/s (L1) + 20TB/s (Register)             │
│                                                                                        │
│  层次树:                                                                               │
│  Die (芯片)                                                                            │
│   ├─ GPC × 12                                                                          │
│   │   └─ SM × 14-15                                                                    │
│   │       ├─ Instruction Cache (32KB)                                                  │
│   │       ├─ Warp Scheduler × 4                                                        │
│   │       │   └─ Warp Pool (16 Warps/Scheduler)                                        │
│   │       │       └─ Warp (32 threads, 共享PC)                                         │
│   │       │           └─ Thread (独立寄存器)                                           │
│   │       ├─ Processing Block × 4                                                      │
│   │       │   ├─ CUDA Core × 32 (FMA单元)                                              │
│   │       │   ├─ INT32 Unit × 16                                                       │
│   │       │   └─ Tensor Core × 4                                                       │
│   │       ├─ Register File (256KB)                                                     │
│   │       ├─ Shared Memory / L1 (128KB)                                                │
│   │       ├─ LD/ST Units × 32                                                          │
│   │       └─ SFU × 16                                                                  │
│   ├─ L2 Cache (40MB, 8 slices)                                                         │
│   └─ Memory Controller × 16 → GDDR7 (32GB)                                             │
│                                                                                        │
│  执行流程:                                                                             │
│  1. Kernel Launch → Grid of Blocks (程序启动)                                          │
│  2. Block Scheduler → 分配Block到SM (资源检查: threads, registers, shared mem)         │
│  3. SM → 分割Block为Warp (每32 threads)                                                │
│  4. Warp Scheduler → 从Ready的Warp中选择1个 (Scoreboard + Arbiter)                     │
│  5. 发射指令 → 广播到32个CUDA Core并行执行 (SIMT)                                      │
│  6. Latency Hiding → Warp stall时切换到其他Warp (隐藏内存延迟)                         │
│  7. Block完成 → 释放资源 → 加载新Block (持续执行)                                      │
│                                                                                        │
│  性能优化关键:                                                                         │
│  • 高占用率: 多个Block/Warp常驻 → 更好的Latency Hiding                                 │
│  • 避免分支发散: Warp内thread执行同一路径 → 100%效率                                   │
│  • 内存合并访问: Warp内连续访问 → 最大带宽利用                                         │
│  • 使用Shared Memory: 减少Global Memory访问 (400× faster)                              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### RTX 5090 架构层级对照表

| 层级                | 上一层 → 本层关系 | 全芯片数量 (RTX 5090)    | 下一层 (直接包含)  | 抽象本质       | 主要职责            | 调度/管理对象 | 程序员可见 | 缺失此层后果     |
| ------------------- | ----------------- | ------------------------ | ------------------ | -------------- | ------------------- | ------------- | ---------- | ---------------- |
| Die (芯片)          | —                 | 1                        | GPC ×12            | 物理边界       | 承载整个GPU         | Grid/Kernel   | ❌          | GPU不存在        |
| GPC (图形处理集群)  | 1 Die → 12        | 12                       | SM ×14-15 (不均)   | 前端/分区      | 组织SM、前端与互联  | SM组          | ❌          | 前端与调度失控   |
| SM (流多处理器)     | 1 GPC → ~15       | 170                      | Warp+执行单元 (64) | 核心计算单元   | 资源分配与Warp调度  | Block/Warp    | ⚠️          | 并行计算能力崩溃 |
| Thread Block (CTA)  | Grid → 多Block    | 动态 (≤硬件上限)         | Warp               | 资源与同步容器 | 同步、共享内存隔离  | Warp          | ✅          | 无法线程同步     |
| Warp                | 1 Block → 多Warp  | 10,880 (170×64 常驻上限) | Thread ×32         | SIMT最小执行批 | SIMT指令发射与执行  | 单条指令      | ⚠️          | SIMT语义失效     |
| Thread (线程)       | 1 Warp → 32       | 348,160 (170×2048 常驻)  | —                  | 编程抽象       | 独立控制流(逻辑)    | 标量操作      | ✅          | 退化为SIMD       |
| CUDA Core           | 1 SM → 128        | 21,760 (170×128)         | —                  | FP32/INT32 ALU | 标量算术运算        | Lane执行      | ❌          | 无通用算力       |
| Tensor Core (第5代) | 1 SM → 16         | 2,720 (170×16)           | —                  | 矩阵乘加引擎   | Warp级矩阵运算(MMA) | Warp          | ⚠️          | AI/ML性能崩塌    |

### 示例GPU代码: GEMM (矩阵乘法)

GEMM是LLM的核心算子 (Linear层、Attention、FFN),覆盖90%以上的计算量。

```
数据在内存层次中的位置:
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Memory层级          │ 容量 (per SM)  │ 带宽        │ 延迟       │ 本例数据     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Register File       │ 256KB          │ ~20 TB/s    │ 1 cycle    │ acc, a0-b1   │
│  Shared Mem / L1     │ 128KB (可配)   │ ~10 TB/s    │ ~20 cycle  │ As, Bs       │
│  L2 Cache            │ 40MB (全局共享)│ ~5 TB/s     │ ~200 cycle │ (自动缓存)   │
│  Global Mem (GDDR7)  │ 32GB (全局共享)│ 1 TB/s      │ ~400 cycle │ A, B, C      │
└─────────────────────────────────────────────────────────────────────────────────┘
```
#### CUDA Core版 (Tiled GEMM)

```cpp
// C[M,N] = A[M,K] × B[K,N], 每个Block计算C的32×32子块
// 
// 【编译器视角】Block分配: 256 threads (16×16) → 8个Warp
//  单block最多32warp(1024线程), 而SM最多驻留64 warp(2048线程), 所以一个SM通常会同时驻留多个block
//  warp scheduler(类似cpu多发射的概念)保证不同warp(不管是不是一个block的)的指令可以pipe起来, 让alu(标量/向量/张量/shader)跑满
//  这其实是在鼓励程序员手动对算子的数据结构做出更细粒度的拆分, 让算子的数据和运算单元耦合程度更高, 提高计算效率 (程序员保证并行语义正确, 硬件才能给予极高吞吐)
//  编译器和驱动不会有意识地分配不同算子偏好(标量/向量/张量/shader)的block到同一个sm(尝试提高alu的总体利用率), 仅按照硬件利用率动态分配
//  动态分配后, block和sm是静态绑定, 不会migrate
//
//   - 本kernel: 256 threads + 8KB shared → SM可驻留~6个Block (48 Warp)
//   - 48个Warp足够隐藏Global Memory ~400 cycle延迟
//
// 数据位置:  A,B,C 【DDR】   As,Bs 【L1/Shared】   acc,a0,b0... 【Reg】

__global__ void block_gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    int tid = threadIdx.y * 16 + threadIdx.x;                  // 【8个Warp并发驻留】Scheduler在不同cycle交错发射
                                                               // 每个warp被发射时, 32个thread以SIMT方式同时执行
                                                               // Warp0: tid=[0-31], Warp1: tid=[32-63]...Warp7: tid=[224-255]
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    int tx = threadIdx.x, ty = threadIdx.y;                    // 【Reg】线程私有, 1 cycle
    int row = blockIdx.y * 32 + ty * 2;                        // 每个thread负责C的2×2子块
    int col = blockIdx.x * 32 + tx * 2;                        // Warp0: row[0-3],col[0-31]; Warp7: row[28-31],col[0-31]
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    __shared__ float As[32][32];                               // 【L1】Block级资源: 8个Warp共享8KB Shared Memory
    __shared__ float Bs[32][32];                               // 【L1】SM内128KB共享, ~20 cycle访问
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    float acc[2][2] = {{0}};                                   // 【Reg】私有累加器, 32个thread各自维护acc[2][2]
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    for (int k0 = 0; k0 < K; k0 += 32) {                       // 【8个Warp并发执行】Scheduler每cycle从ready warp选择发射
        // ─────────────────────────────────────────────────────────────────────────────────────────
        As[ty*2][tx*2]     = A[(row)*K + k0+tx*2];             // 【DDR→Reg→L1】Warp0发射LOAD: 32 Core并行读Global
        As[ty*2][tx*2+1]   = A[(row)*K + k0+tx*2+1];           //   Warp0被scoreboard标记pending, 暂时不能再issue
        As[ty*2+1][tx*2]   = A[(row+1)*K + k0+tx*2];           //   Scheduler立刻切换到其他ready warp
        As[ty*2+1][tx*2+1] = A[(row+1)*K + k0+tx*2+1];         //   Warp1-7继续执行, 隐藏~400 cycle延迟
                                                               //   数据到达Reg后, 再STORE到L1 (~20 cycle)
        // ─────────────────────────────────────────────────────────────────────────────────────────
        Bs[ty*2][tx*2]     = B[(k0+ty*2)*N + col];             // 【DDR→Reg→L1】同上, Warp0写Bs[0-1][*]
        Bs[ty*2][tx*2+1]   = B[(k0+ty*2)*N + col+1];           //   32 thread并行写, 访问不同bank, 无conflict
        Bs[ty*2+1][tx*2]   = B[(k0+ty*2+1)*N + col];           //   不同Warp写不同地址, 可交错执行
        Bs[ty*2+1][tx*2+1] = B[(k0+ty*2+1)*N + col+1];         //
        // ─────────────────────────────────────────────────────────────────────────────────────────
        __syncthreads();                                       // 【Block级barrier】硬件检查: 所有8个Warp都到达
                                                               //   Warp0-7全部stall, 等待最慢的warp
                                                               //   硬件计数器: 8/8 warps到达 → barrier释放
        // ─────────────────────────────────────────────────────────────────────────────────────────
        for (int k = 0; k < 32; k++) {                         // 【SIMT锁步】Warp内32 thread共享PC, 同时执行
            float a0 = As[ty*2][k];                            // 【L1→Reg】~20 cycle, t0读As[0][k], t1读As[0][k]...
            float a1 = As[ty*2+1][k];                          //   数据复用: As[row][k]被同行16个thread共享
            float b0 = Bs[k][tx*2];                            // 【L1→Reg】
            float b1 = Bs[k][tx*2+1];                          //
            acc[0][0] += a0 * b0;                              // 【Reg→ALU→Reg】FMA, 1 cycle, 32 Core并行执行
            acc[0][1] += a0 * b1;                              //   操作数从各自Reg读, 结果写回各自Reg
            acc[1][0] += a1 * b0;                              //   无需同步: Warp内天然锁步
            acc[1][1] += a1 * b1;                              //
        }
        // ─────────────────────────────────────────────────────────────────────────────────────────
        __syncthreads();                                       // 防止快的warp覆盖下一轮L1数据
    }
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    C[(row)*N + col]       = acc[0][0];                        // 【Reg→DDR】各Warp发射STORE, 32 thread写相邻地址
    C[(row)*N + col+1]     = acc[0][1];                        //   → Memory Controller合并为128B请求
    C[(row+1)*N + col]     = acc[1][0];                        //   8个Warp交错执行, 隐藏~400 cycle延迟
    C[(row+1)*N + col+1]   = acc[1][1];                        //
    // ═════════════════════════════════════════════════════════════════════════════════════════════
    // 分支发散机制 (实际GEMM不需要)
    // ═════════════════════════════════════════════════════════════════════════════════════════════
    if (tid < 32) {                                            // 【Warp间分支】Predicate Mask:
                                                               //   Warp0: mask=0xFFFFFFFF (全部32 thread进入if)
                                                               //   Warp1-7: mask=0x00000000 (跳过if体)
                                                               //   *** 此后只有Warp0继续执行if体 ***
        // ─────────────────────────────────────────────────────────────────────────────────────────
        float sum = 0.0f;                                      // 【Reg】Warp0: 32个thread各自初始化sum
        for (int i = tid; i < 256; i += 32)                    // Warp0循环: t0处理i=0,32,64...; t1处理i=1,33,65...
            sum += As[i/32][i%32];                             // 【L1→Reg→ALU→Reg】32 thread从L1读不同位置, SIMT锁步
        // ─────────────────────────────────────────────────────────────────────────────────────────
        if (tid == 0)                                          // 【Warp内分支】Predicate Mask = 0x00000001
                                                               //   t0: mask=1, 执行STORE
                                                               //   t1-31: mask=0, 闲置 (31个CUDA Core浪费)
            C[0] = sum;                                        // 【Reg→DDR】只有t0写结果
    }
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    // 【Block完成】8个Warp退出 → 硬件释放L1+Reg → SM加载下一个Block
}
```

#### Tensor Core版 (WMMA API)

```cpp
// Tensor Core: Warp级矩阵乘专用硬件, 16×16×16 MMA约4-8 cycle (vs CUDA Core ~128 cycle)
// 
// 【Tensor Core操作是Warp级的】
//   - 32个thread作为整体发起MMA指令, 不是每个thread独立操作
//   - 16×16矩阵的256个元素分散在32个thread的Register中
//   - fragment: 编译器管理的分布式数据结构, 程序员不需要知道哪个thread持有哪个元素
//
// 数据位置:  A,B,C 【DDR】   As,Bs 【L1/Shared】   a_frag,b_frag,c_frag 【Reg分布式】

#include <mma.h>
using namespace nvcuda;

__global__ void block_gemm_tensor(const half* A, const half* B, float* C, int M, int N, int K) {
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    int tid = threadIdx.x;                                     // 【1个Warp】本kernel每Block只有32 thread
    int warp_row = blockIdx.y;                                 // 【Reg】每个Block处理C的16×16子块
    int warp_col = blockIdx.x;                                 //
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    __shared__ half As[16][16];                                // 【L1】512B staging buffer
    __shared__ half Bs[16][16];                                // 【L1】512B
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> a_frag;    // 【Reg分布式】每thread ~8 half
    wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> b_frag;    // 【Reg分布式】每thread ~8 half
    wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;                 // 【Reg分布式】每thread ~8 float
    wmma::fill_fragment(c_frag, 0.0f);                         // 【Reg】32 thread各自初始化自己持有的c_frag部分
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    for (int k0 = 0; k0 < K; k0 += 16) {                       //
        // ─────────────────────────────────────────────────────────────────────────────────────────
        for (int i = tid; i < 256; i += 32) {                  // 【DDR→Reg→L1】32 thread协作加载16×16=256 half
            int r = i / 16, c = i % 16;                        //   每thread加载8个元素, 交错访问
            As[r][c] = A[(warp_row*16 + r)*K + k0 + c];        //   scoreboard标记pending, 隐藏~400 cycle
            Bs[r][c] = B[(k0 + r)*N + warp_col*16 + c];        //
        }
        // ─────────────────────────────────────────────────────────────────────────────────────────
        __syncthreads();                                       // 【barrier】等待32 thread都完成DDR→L1
        // ─────────────────────────────────────────────────────────────────────────────────────────
        wmma::load_matrix_sync(a_frag, &As[0][0], 16);         // 【L1→Reg分布式】32 thread协作读, 256元素分布到各自Reg
        wmma::load_matrix_sync(b_frag, &Bs[0][0], 16);         // 【L1→Reg分布式】硬件自动分配哪个thread持有哪个元素
        // ─────────────────────────────────────────────────────────────────────────────────────────
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);        // 【Reg→TensorCore→Reg】MMA指令:
                                                               //   Scheduler发射一条HMMA指令
                                                               //   32 thread的Reg数据喂入Tensor Core
                                                               //   4×4×4硬件矩阵乘, 流水线完成16×16×16
                                                               //   4-8 cycle完成, 结果写回32 thread的Reg
        // ─────────────────────────────────────────────────────────────────────────────────────────
        __syncthreads();                                       // 防止下一轮覆盖L1数据
    }
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    wmma::store_matrix_sync(&C[warp_row*16*N + warp_col*16], c_frag, N, wmma::mem_row_major);
                                                               // 【Reg分布式→DDR】32 thread协作写, 256元素合并写出
    // ─────────────────────────────────────────────────────────────────────────────────────────────
    // 【Block完成】Warp退出 → 硬件释放L1+Reg → SM加载下一个Block
}
```
# 移动端API的问题

不同操作系统平台的万千上层应用开发者才是我们最终渗透到的服务对象, 从列举的50种场景可知, 顶层应用绝大部分都是高度客制化的, 开发者要自行完成: 
  - 应用场景理解 -> 
  - pytorch/tensorflow算法/模型开发 -> 
  - 全平台硬件适配做硬件加速 (目前成本过高, <mark>绝大部分移动端团队无法承受</mark>, 极大限制了移动端AI应用的开发)

最后一步:全平台适配, :
  - 每台不同的设备对于NNAPI和openVX的算子支持程度是不同的, 哪怕支持cpu fallback, 模型可以跑起来, <mark>底层的性能仍然是极端不可控的</mark>, 某些平台的实现可能极其低效, 完全不满足场景需求, 应用开发者维护成本极高

目前Google NNAPI的流程:
  - Acceleration Service API (Beta)	在用户设备上跑benchmark,返回推荐配置, 然后应用端根据结果配置 (<mark>治标不治本</mark>)
  - AI Edge Portal (Private Preview) 开发者在100+设备云端测试	只有大厂用得起,小团队没资源 (<mark>同样成本爆炸</mark>)

真正的解决方案:
  - OS的NNAPI, 提高对底层硬件厂商的要求, 强推设备分级制度, <mark>对不同子集的硬件设备提出具体的算子支持和性能需求</mark>, 为上层不同类别的应用提供基础的流畅运行保证, 极大降低开发成本
  - 难点: 推动困难, <mark>合理的分级会暴漏现在大部分硬件或多或少的问题, 得罪大部分硬件厂商</mark> (Google要公开说: “这批设备,不配跑这类 AI 应用”)

结论: 在绝大部分移动端应用开发团队完全用不起来AI加速器的情况下, <mark>现在的AI加速器到底是谁在用</mark>? 商业模式是什么?
  - 板级产品开发商(垂直整机)确定固定的AI应用场景, 确认可行性, 生产规模, 向SoC硬件提供商提出具体的AI加速需求, 硬件提供商做针对性开发, 提供编译器支持
  - 少部分头部软件大厂自己的killer应用, 会去尝试做全平台适配
  - 相反, 绝大部分通用开发者根本不会考虑调用AI加速器

# 移动端典型workload

作为移动端/端侧SoC NPU提供厂商, 毫无疑问走的是DSA路线, 根据上面50个应用场景的具体分析, 以下几个模型将能cover我们大部分应用场景, 应当重点分析:

| 模型类别                    | 代表模型                                         | 参数量             | 核心架构特性                                         | 业务场景 / 关注痛点                 | 主要测试重点                                                   |                      |
| --------------------------- | ------------------------------------------------ | ------------------ | ---------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------- | -------------------- |
| **标准移动版**              | **google/gemma‑2‑2b‑it**                         | 2.6B               | Decoder‑only Transformer, RoPE, RMSNorm, GQA         | 移动端通用交互、长序列生成          | KV Cache 管理（2k‑4k）、4‑bit / INT8 量化适配、RoPE / GQA 支持 | ([Hugging Face][1])  |
| **多模态融合（VLM）**       | **Qwen/Qwen3‑VL‑2B‑Instruct**                    | 2B                 | Vision + Text 多模态（视觉编码 & 语言解码,长上下文） | AR 眼镜、智能摄像头等多模态流式场景 | 异构流水线调度、动态 Shape 支持、视觉 Token 注入效率           | ([Hugging Face][2])  |
| **极小型 / 常驻任务**       | **HuggingFaceTB/SmolLM2‑135M (或 SmolLM2‑360M)** | ~0.1B              | Tiny‑scale Transformer                               | 手机输入法、嵌入式极低功耗常驻推理  | SRAM 驻留、唤醒延迟、冷启动能耗                                | ([SIOS Tech Lab][3]) |
| **深度推理 / 复杂逻辑**     | **microsoft/Phi‑3.5‑mini‑Instruct (3.8B)**       | 3.8B               | Dense Transformer, 更深层结构                        | 离线办公、复杂逻辑推理              | 计算密集度、长 Context Prefill 性能                            | —                    |
| **向量 / 检索 & Embedding** | **TaylorAI/bge‑micro‑v2**                        | ~Embedding 384‑dim | Sentence‑Transformers 嵌入模型                       | 私有知识库、RAG Embedding & 检索    | 高吞吐 Embedding QPS、IO 密集型访问                            | ([Hugging Face][4])  |
| **图像生成 (Diffusion)**    | **stabilityai/sd‑turbo**                         | ~1B                | CLIP + UNet + VAE (Diffusion, 非 Transformer)        | 端侧图像生成、创意工具              | UNet 去噪迭代、VAE 解码、跨模态 (文本→图像)                    | ([Hugging Face][5])  |

## 训练,微调,推理一体化计算图：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Model_LLM(nn.Module):
    """
    统一 DAG(有向无环图) 模型,三种模式共享
    mode: "pretrain" | "align" | "infer"
    """

    def __init__(self, vocab, dim, n_layers):
        super().__init__()

        # -------- 模型结构（不随 mode 变）--------
        self.embed = nn.Embedding(vocab, dim)                                          # 词 → 向量（词嵌入空间）
        self.layers = nn.ModuleList([TransformerBlock(dim) for _ in range(n_layers)])  # transformer 主干 (上下文表征空间)
        self.policy_head = nn.Linear(dim, vocab, bias=False)                           # 向量 → 词表 logits（policy: 选 token）
        self.critic_head = nn.Linear(dim, 1)                                           # 向量 → 偏好（critic: 估计价值）

        # -------- 运行时状态 --------
        self.mode = None     # "pretrain" | "align" | "infer"
        self.cache = False   # KV 缓存（推理加速）
        self.grad = False    # 是否反向传播

        # -------- 优化器 --------
        # 训练模式：更新全参数
        self.optim_full = torch.optim.AdamW(self.parameters(), lr=3e-4)
        # 对齐模式：只更新 主干网 + critic头
        self.optim_align = torch.optim.AdamW(list(self.layers.parameters()) + list(self.critic_head.parameters()), lr=1e-4)
        self.optim = None

    def set_mode(self, mode: str):
    #  模式       | train/eval | KV缓存 | 梯度 | 可训练参数             | 用到的输出
    # ------------|------------|--------|------|------------------------|---------------------------
    #  pretrain   | train()    | 关闭   | 开启 | 全部                   | logits → CE loss
    #  align      | train()    | 关闭   | 开启 | layers + critic_head   | logits + values → DPO + critic loss
    #  infer      | eval()     | 开启   | 关闭 | 无                     | logits + cache

        assert mode in ["pretrain", "align", "infer"]
        self.mode = mode

        if mode == "pretrain":           # 全量训练
            self.train()                 # Dropout 生效 + BatchNorm更新(LayerNorm不影响)
            self.cache = False           # 不缓存（每次算完整序列）
            self.grad = True             # 要反向传播
            self.optim = self.optim_full
            for p in self.parameters():
                p.requires_grad = True   # 所有参数可训练

        elif mode == "align":            # 对齐: (SFT、RLHF、DPO微调), 在固定语言空间里调整行为偏好
            self.train()                 # Dropout 生效 + BatchNorm更新(LayerNorm不影响)
            self.cache = False
            self.grad = True
            self.optim = self.optim_align
            for p in self.embed.parameters():
                p.requires_grad = False  # 冻结 embedding（词嵌入空间）
            for p in self.layers.parameters():
                p.requires_grad = True   # 训练 transformer backbone（上下文表征空间）
            for p in self.policy_head.parameters():
                p.requires_grad = False  # 冻结 policy_head（这句话下一个token应该是什么？）(如果能保证稳定性,也可以酌情考虑放开policy头)
            for p in self.critic_head.parameters():
                p.requires_grad = True   # 训练 critic_head（这句话是不是符合我们的偏好,该不该说？）

        elif mode == "infer":            # 推理
            self.eval()                  # Dropout 关闭 + BatchNorm不更新(LayerNorm不影响)
            self.cache = True            # 缓存 KV（自回归加速）
            self.grad = False            # 不反向
            self.optim = None            # 不更新参数
            for p in self.parameters():
                p.requires_grad = False  # 无需梯度

    # ============================================================
    # 统一 forward
    # ============================================================
    def forward(self, ids, *, labels=None, past=None, chosen=None, rejected=None):
        """
        统一条计算路径
        """
        assert self.mode is not None, "set_mode() first"

        def _forward(x_ids, x_past):
            """共享的forward path"""
            x = self.embed(x_ids)                     # [B, T] → [B, T, D]
            T = x.size(1)
            
            # -------- 不同模式的 mask 策略 --------
            # pretrain: causal mask,完整序列自回归
            # align:    causal mask,完整序列自回归
            # infer:    有 KV cache 时新 token 可看所有历史,无需 mask；否则 causal mask
            if self.mode == "infer" and self.cache and x_past:
                causal_mask = None  # 增量推理：新 token 可以 attend 所有历史 KV
            else:
                causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            
            new_past = []
            for i, layer in enumerate(self.layers):
                kv = x_past[i] if (self.cache and x_past) else None
                x, kv = layer(x, attention_mask=causal_mask, past_key_values=kv, use_cache=self.cache) # 是否开启串行自回归加速(KV缓存)
                if self.cache:
                    new_past.append(kv)

            logits = self.policy_head(x)              # [B, T, D] → [B, T, V]
            values = self.critic_head(x).squeeze(-1)  # [B, T, D] → [B, T]
            return logits, values, new_past

        # -------- pretrain: --------
        if self.mode == "pretrain":
            # 如果是多模态, 骨干网表征的学习更加困难,可能会用对比学习/自监督来提升表征空间的质量 (数据增强 -> 对比loss)

            logits, _, _ = _forward(ids, past)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

            if self.grad:
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            return {"loss": loss.detach(), "logits": logits}

        # -------- align: Actor_Critic baseline --------
        if self.mode == "align":
            # prompt: "怎么入侵别人的电脑？"
            # chosen = "我无法提供入侵他人电脑的方法,这是违法行为。如果你对网络安全感兴趣,建议学习合法的渗透测试课程。"
            # rejected = "首先你需要扫描目标端口,然后使用 nmap..." # 拒绝回答

            # chosen 和 rejected 共用 backbone
            logit_c, value_c, _ = _forward(chosen, None)  # [B, T, V]
            logit_r, value_r, _ = _forward(rejected, None)

            # 1) DPO policy_head 偏好损失
            #    促使生成分布偏向 chosen 答案,远离 rejected 答案
            logp_c = seq_logp(logit_c, chosen)  # [B]
            logp_r = seq_logp(logit_r, rejected)  # [B]
            loss_dpo = -torch.log(torch.sigmoid(logp_c - logp_r)).mean()  # 对 batch 求均值

            # 2) DPO critic_head 偏好损失
            #    训练价值网络提升判别力,使得 chosen 回答得分高于 rejected 回答
            value_diff = value_c.mean() - value_r.mean()
            loss_critic = -torch.log(torch.sigmoid(value_diff))

            # 3) 联合训练
            #    让loss梯度反压回主干网和critic头
            loss = loss_dpo + 0.1 * loss_critic

            # 注意, critic头就是会重新塑造transformer骨干网的表征空间,让好偏好和不好偏好的上下文表征发生聚类(形成分界面)

            if self.grad:
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

            return {
                "loss_dpo": loss_dpo.detach(),                                           # 下降 -> 偏好学习在进行
                "loss_critic": loss_critic.detach(),                                     # 下降 -> Critic学习在进行
                "logp_c - logp_r": logp_c.detach() - logp_r.detach(),                    # 越来越大->模型越来越偏好 chosen
                "value_c - value_r": value_c.mean().detach() - value_r.mean().detach(),  # 越来越大->critic 学会区分好坏
            }

        # -------- infer: 只返回 logits, 更新KV cache --------
        if self.mode == "infer":
            with torch.no_grad():
                logits, _, new_past = _forward(ids, past)
            return {"logits": logits, "past": new_past}

```

