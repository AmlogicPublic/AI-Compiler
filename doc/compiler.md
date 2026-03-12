## CV vs 端侧LLM vs 服务器LLM 特性对比

典型规模: 端侧LLM ~1B–4B, 服务器LLM ~7B+

| 特性            | CV                      | 端侧LLM                     | 服务器LLM                          |
| --------------- | ----------------------- | --------------------------- | ---------------------------------- |
| graph exec      | V                       | V + token loop              | V + token loop + scheduler         |
| token loop      | X                       | V                           | V                                  |
| dynamic sched   | X                       | 轻量                        | 重                                 |
| activation life | layer级                 | token级                     | token级                            |
| schedule        | compile-time            | runtime(轻量)               | runtime(重)                        |
| kernel reuse    | 高                      | 中                          | 低                                 |
| runtime复杂度   | 低                      | 中                          | 极高                               |
| compiler复杂度  | 高                      | 中                          | 中                                 |
| 算子类型        | conv / depthwise / pool | matmul / attention          | matmul / attention                 |
| shape           | static                  | semi-dynamic                | highly dynamic (seq/kv/batch/beam) |
| batch           | 常见                    | 少(通常=1)                  | serving大量                        |
| KV cache        | 无                      | 小(几十MB，可静态分配)      | 大(GB级，必须paged)                |
| paged KV        | 无                      | 可选                        | 必须                               |
| memory pattern  | dense                   | KV cache(静态buffer)        | KV cache + paged memory            |
| allocator       | 静态                    | 半静态                      | 动态                               |
| fusion          | conv-bn-relu            | matmul-softmax-matmul       | matmul-softmax-matmul              |
| latency         | 全链路强要求            | decode延迟                  | decode延迟 + throughput            |
| delegate可行性  | 高                      | V(Apple/QC/MTK)             | X(不用delegate)                    |
| quant           | int8                    | fp16/int8/int4              | fp16/bf16/int8/int4/fp8            |
| NPU适配         | 很成熟                  | 可行(CoreML/QNN/NeuroPilot) | 困难，GPU为主                      |
| 模型大小        | MB ~ 100MB              | 几百MB~几GB                 | 几GB~几十GB                        |

核心结论:

| 结论                | CV              | 端侧LLM            | 服务器LLM        |
| ------------------- | --------------- | ------------------ | ---------------- |
| runtime vs compiler | compiler-driven | compiler + runtime | runtime OS-level |
| delegate可行        | V(主流)         | V(Apple/QC/MTK)    | X(不用delegate)  |
| Mid IR必须          | -               | V(简化)            | V(完整)          |
| runtime必须分开     | V               | V                  | V                |
| paged KV必须        | -               | X                  | V                |
| scheduler必须       | -               | 轻量               | 重(OS级)         |

- CV: compiler驱动系统，delegate主导
- 端侧LLM: compiler+runtime混合系统，delegate+runtime
- 服务器LLM: runtime OS级系统，codegen+runtime (scheduler/allocator/pager/batcher/serving)

## 技术栈层级对比

| 层级     | CV                    | 端侧LLM                 | 服务器LLM              |
| -------- | --------------------- | ----------------------- | ---------------------- |
| Frontend | ONNX / TFLite / TF    | torch.export / HF       | torch.export / HF      |
| Graph IR | TFLite graph / ONNX   | torch.fx                | torch.fx / Relax       |
| High IR  | MLIR / StableHLO      | MLIR / StableHLO        | MLIR / Relax           |
| Mid IR   | target-aware IR       | layout + fusion + quant | +paged + sched + alloc |
| Low IR   | LLVM / NPU            | LLVM / Metal / GPU      | PTX / SPIR-V / Metal   |
| Kernel   | NPU lib / oneDNN      | NPU lib / Metal         | Triton / CUTLASS       |
| Runtime  | LiteRT / ONNX Runtime | 轻量runtime + delegate  | vLLM / TRT-LLM         |
| Backend  | delegate              | delegate + codegen      | codegen                |

Mid IR 对比:

| 功能            | 端侧LLM | 服务器LLM |
| --------------- | ------- | --------- |
| KV layout       | V       | V         |
| quant layout    | V       | V         |
| fused attention | V       | V         |
| paged KV        | X       | V         |
| beam schedule   | X       | V         |
| dynamic batch   | X       | V         |
| serving sched   | X       | V         |

端侧 Mid IR = layout + fusion + quant
服务器 Mid IR = layout + fusion + quant + paged + schedule + allocator

| 公司     | Mid IR              |
| -------- | ------------------- |
| NVIDIA   | NVIR                |
| Apple    | MIL + Metal IR      |
| Qualcomm | QNN IR              |
| Google   | HLO → Linalg → LLVM |
| TVM      | TIR                 |
| IREE     | Flow → HAL          |

## 统一策略 (CV + 端侧LLM 为设计目标)

| 层级        | 统一/分开 | 原因                                          |
| ----------- | --------- | --------------------------------------------- |
| Frontend    | 分开      | 输入格式差异大，强行统一没意义                |
| Graph IR    | 分开      | 各家格式不同                                  |
| High IR     | 统一      | MLIR，最值得统一的层                          |
| Mid IR      | 部分统一  | CV + 端侧LLM 可共享 layout/fusion/quant pass  |
| Low IR      | 统一      | LLVM / GPU IR                                 |
| Kernel接口  | 统一      | 调用约定统一，实现分开                        |
| Runtime     | 部分统一  | CV静态 + 端侧LLM轻量runtime可共享调度框架     |
| Backend接口 | 统一      | delegate/codegen统一抽象，端侧LLM也走delegate |

关键变化: 端侧LLM让delegate重新可行，CV和端侧LLM可以共享更多基础设施
工业界共识: MLIR是唯一可以统一的层

## 架构

```
                              Frontend
                       (ONNX / TF / torch.export / HF)
                                  ↓
                              Graph IR
                        (ONNX / FX / TFLite)
                                  ↓
                              High IR
                       (MLIR / StableHLO / Relax)
                                  ↓
          ┌───────────────────────┼───────────────────────┐
          ↓                       ↓                       ↓
    CV Compiler          端侧LLM Compiler          服务器LLM Compiler
    ┌───────┐              ┌───────────┐            ┌─────────────┐
    │layout │              │KV layout  │            │KV layout    │
    │fusion │              │fusion     │            │paged mem    │
    │static │              │quant      │            │beam sched   │
    └───┬───┘              └─────┬─────┘            │dynamic batch│
        ↓                        ↓                  └──────┬──────┘
    CV Mid IR            端侧LLM Mid IR          服务器LLM Mid IR
    (layout+fusion)      (layout+fusion+quant)    (layout+fusion+quant
        ↓                        ↓                 +paged+sched+alloc)
    CV Low IR            端侧LLM Low IR                   ↓
  (LLVM / NPU)        (LLVM / Metal / GPU)        服务器LLM Low IR
        ↓                        ↓                (PTX / SPIRV / Metal)
   ┌────┴────┐              ┌────┴────┐                    ↓
   ↓         ↓              ↓         ↓              ┌─────┴─────┐
Delegate  Codegen       Delegate   Codegen        Codegen    Custom
(NPU/ANE)  (CPU)     (NPU/ANE/GPU) (CPU/GPU)      (GPU)     Kernel
   ↓         ↓              ↓         ↓              ↓           ↓
   └────┬────┘              └────┬────┘              └────┬──────┘
        ↓                        ↓                        ↓
   CV Runtime            端侧LLM Runtime          服务器LLM Runtime
   (静态调度)            (轻量runtime)           ┌────────┼────────┐
        │                 token loop             │        │        │
        │                 + static KV         Scheduler MemMgr Launcher
        │                 + delegate          (token/  (KV/   (kernel
        │                       │              batch)  paged)  call)
        └───────────┬───────────┘                         │
                    │                    ┌────────────────┘
                    ↓                    ↓
          ┌─────────┼─────────┐    ┌─────┼────┐
          ↓         ↓         ↓    ↓          ↓
         CPU       GPU       NPU  GPU        TPU
       (oneDNN) (Metal)  (QNN/ANE) (CUDA)   (XLA)
```

Runtime 差异:

| 功能          | CV Runtime     | 端侧LLM Runtime     | 服务器LLM Runtime    |
| ------------- | -------------- | ------------------- | -------------------- |
| graph exec    | V              | V                   | V                    |
| token loop    | X              | V                   | V                    |
| KV cache      | X              | V(静态buffer)       | V(paged)             |
| paged KV      | X              | X                   | V                    |
| scheduler     | X              | 轻量                | 重                   |
| async stream  | 少             | 中                  | 多                   |
| multi backend | 中             | 中                  | 高                   |
| serving       | X              | X                   | V                    |
| 调度模型      | 静态graph exec | token loop+delegate | 动态token loop       |
| 内存管理      | 静态分配       | 静态KV buffer       | KV cache+paged+alloc |
| 复杂度        | 低             | 中                  | 极高(OS级)           |

## 分流与同步策略

决策时机: 编译期静态决定

| 粒度     | 分流说明           | 同步点       | 同步机制           | 同步原理              | 开销来源          | 工业界实现                         | 推荐度   |
| -------- | ------------------ | ------------ | ------------------ | --------------------- | ----------------- | ---------------------------------- | -------- |
| workload | 整个模型走不同路径 | 无需同步     | -                  | -                     | 无                | 简单部署场景                       | 简单场景 |
| subgraph | 子图级别切分       | subgraph边界 | 异步copy / 零拷贝  | DMA + event / unified | 边界tensor copy   | QNN delegate / ONNX Runtime EP     | V 主流   |
| op       | 单算子级别         | op边界       | 阻塞copy           | memcpy + wait         | 每op都要拷贝+等待 | 不推荐                             | X 不推荐 |
| backend  | 同一runtime多后端  | 显式barrier  | stream同步 / event | 多stream + dependency | barrier等待       | CUDA stream / Metal command buffer | V 常用   |

同步机制:
- 阻塞copy: memcpy + wait，CPU fallback
- 异步copy: DMA + event signal (cudaMemcpyAsync)
- 零拷贝: unified memory / 共享buffer (CoreML MLMultiArray / Android ION)
- stream同步: 多stream + dependency (CUDA stream / Metal)
- pipelined execution: compute与copy overlap (TRT-LLM / vLLM / DeepSpeed / Megatron)

原则: 粒度越大同步越少 / 优先异步隐藏延迟 / 能零拷贝就不memcpy / 避免CPU ↔ GPU频繁ping-pong

## 工业界参考

端侧 (CV + 端侧LLM):

| 厂商     | 方案                                          |
| -------- | --------------------------------------------- |
| Apple    | CoreML → MIL → Metal → ANE (LLM delegate可行) |
| Qualcomm | MLIR → QNN IR → delegate + 轻量LLM runtime    |
| MediaTek | NeuroPilot → NPU delegate + LLM runtime       |
| Samsung  | NPU LLM → delegate + runtime                  |
| Google   | LiteRT → delegate (CV) + AI Edge (LLM)        |
| NVIDIA   | TensorRT → CUDA → Jetson GPU (端侧LLM可行)    |

服务器LLM:

| 厂商   | 方案                           |
| ------ | ------------------------------ |
| NVIDIA | Torch → NVIR → Triton → CUDA   |
| Google | StableHLO → XLA → GPU/TPU      |
| AMD    | MLIR → ROCm → HIP              |
| Intel  | OpenVINO → oneDNN → GPU        |
| Meta   | Torch → TorchInductor → Triton |

端侧方案: MLIR/HLO → LLM pass → delegate/codegen → 轻量runtime
服务器方案: Torch → mid IR → codegen → heavy runtime
