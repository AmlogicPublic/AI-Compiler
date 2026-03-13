# IREE vs TVM 编译路线对比

## 编译产物与文件结构

| 维度 | IREE | TVM |
|------|------|-----|
| **IR 文件** | `prefill.mlirbc` (111KB) / `decode.mlirbc` (114KB) | `prefill.txt` (327KB, 2792行) / `decode.txt` (334KB, 2837行) |
| **运行时文件** | `prefill.vmfb` (175KB) / `decode.vmfb` (242KB) | `prefill.so` (1.2MB) / `decode.so` (1.1MB) |
| **权重文件** | `smollm2_135m.irpa` (621MB) | `params.npz` (513MB) |
| **IR 格式** | MLIR Bytecode 二进制，不可读 | Relax Script 文本，可读可改 |
| **运行时格式** | FlatBuffer (可作 ZIP 打开)，含 Host bytecode + Device .so + 元数据 | ELF 动态库，原生机器码 + 符号表 |
| **权重格式** | IRPA (支持对齐优化、safetensors、GGUF) | NumPy npz (通用格式) |
| **Polyglot ZIP** | 默认启用，可禁用减小文件 | 无 |

## 运行时架构与执行流程

| 维度 | IREE | TVM |
|------|------|-----|
| **高级 IR** | Flow Dialect (数据流图) | Relax IR (函数式图) |
| **设备抽象** | HAL Dialect (硬件抽象层) | Target 配置 (编译期绑定) |
| **任务编排** | Stream Dialect (编译期确定任务流) | PackedFunc (运行时调度) |
| **执行层** | VM bytecode 解释执行 | 原生代码直接执行 |
| **后端** | HAL Driver: llvm-cpu, cuda, vulkan, metal | Target: llvm, cuda, opencl, rocm |
| **文件加载** | FlatBuffer 反序列化 → VM Module | dlopen → 符号解析 |
| **初始化** | VM + HAL 驱动初始化 | Runtime Context + Device 初始化 |
| **权重加载** | ParameterIndex 异步加载，scope 自动注入 | np.load 全量加载，按顺序传参 |
| **执行方式** | VM 解释 Host Code，HAL 执行 Device Code | 直接调用原生函数 |

## 内存与资源管理

| 维度 | IREE | TVM |
|------|------|-----|
| **代码段大小** | VM bytecode (~200KB 紧凑) + 嵌入 Device Code | 优化后机器码 (~1MB 较大) |
| **权重加载** | 可延迟/异步加载，支持流式 | 需全量预加载 |
| **运行时开销** | VM 调度器 + HAL 驱动常驻 | 仅函数入口 + 缓冲区 |
| **峰值控制** | 异步参数加载可降低峰值 | 全部参数需常驻 |
| **SmolLM2-135M 估计** | ~650MB (权重) + ~100MB (激活) | ~550MB (权重) + ~100MB (激活) |
| **跨设备传输** | HAL Buffer 统一管理，H↔D 和 D↔D DMA | 显式 copy，手动触发 |
| **共享缓冲区** | Buffer View，同一物理内存多逻辑引用 | NDArray 独立，不支持共享 |
| **零拷贝** | ✅ HAL Buffer 跨模块/设备共享 | ❌ 需显式传递拷贝 |
| **生命周期** | 引用计数自动管理 | 手动管理作用域和释放 |

## 异构与同步

| 维度 | IREE | TVM |
|------|------|-----|
| **多设备混合** | ✅ VM 统一编排多设备 | ❌ 单次编译绑定单一目标 |
| **CPU+GPU 协同** | ✅ Stream 自动分配任务 | ❌ 需手动分拆，多次编译 |
| **动态设备选择** | ✅ 运行时选择 HAL 驱动 | ❌ 编译时确定 |
| **同步策略** | Stream wait/signal 编译期插入，运行时自动 | PackedFunc 调用时隐式同步 |
| **异步执行** | ✅ 多 Stream 并行，显式同步点 | ❌ 调用即阻塞 |
| **参数加载同步** | 异步，与计算可并行 | 同步，需等待完成 |
| **细粒度控制** | ✅ Stream 级 barrier/fence | ❌ 仅函数级 |
| **依赖表达** | 编译期静态分析 | 运行时顺序执行隐含 |

## 图模型签名

| 阶段 | 输入 | 输出 |
|------|------|------|
| **Prefill** | `input_ids (1, 32)` + `attention_mask (1, 32)` | `logits (1, 32, 49152)` + 60× KV `(1, 3, 32, 64)` |
| **Decode** | `input_ids (1, 1)` + `attention_mask (1, 101)` + `position_ids (1, 1)` + `cache_position (1,)` + 60× past KV `(1, 3, 100, 64)` | `logits (1, 1, 49152)` + 60× KV `(1, 3, 101, 64)` |

## 设计哲学

| 维度 | IREE | TVM |
|------|------|-----|
| **核心理念** | 编译期确定一切 (异构编排、同步、内存) | 运行时灵活性优先 |
| **执行模型** | VM 解释 + HAL 驱动 (统一性换间接层) | 原生执行 (直接性能) |
| **部署形态** | 单一 .vmfb 自包含 | .so + 权重文件组合 |
| **适用场景** | 多设备异构、边缘部署 | 单设备高性能、研究调试 |
