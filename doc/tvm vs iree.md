# IREE vs TVM 编译路线对比

## 总览对比表

| 维度             | IREE                              | TVM                                 |
| ---------------- | --------------------------------- | ----------------------------------- |
| **IR 文件格式**  | `.mlirbc` (MLIR Bytecode, 二进制) | `.txt` (TVM Relax Script, 文本可读) |
| **IR 文件大小**  | prefill 111KB / decode 114KB      | prefill 327KB / decode 334KB        |
| **编译产物格式** | `.vmfb` (FlatBuffer)              | `.so` (动态链接库)                  |
| **编译产物大小** | prefill 175KB / decode 242KB      | prefill 1.2MB / decode 1.1MB        |
| **权重文件格式** | `.irpa` (IREE Parameter Archive)  | `.npz` (NumPy Compressed)           |
| **权重文件大小** | 621MB                             | 513MB                               |
| **IR 可读性**    | X 二进制不可读                    | V 文本可读可调试                    |
| **参数传递方式** | 自动绑定到 VM module              | 显式传入函数参数                    |
| **支持后端**     | llvm-cpu, cuda, vulkan, metal     | llvm, cuda, opencl, rocm            |
| **调度优化**     | IREE 内置优化                     | 需配合 meta_schedule                |

## 编译流程对比

| 阶段         | IREE                          | TVM                   |
| ------------ | ----------------------------- | --------------------- |
| **前端入口** | `torch.export`                | `torch.export`        |
| **导出工具** | `iree.turbine.aot.export()`   | `tvm.relax.frontend`  |
| **中间表示** | MLIR (`.mlirbc`)              | TVM Relax IR (`.txt`) |
| **编译器**   | `iree.compiler.compile_str()` | `tvm.build()`         |
| **最终产物** | `.vmfb` + `.irpa`             | `.so` + `.npz`        |

## 权重处理对比

| 操作         | IREE                                                    | TVM                                              |
| ------------ | ------------------------------------------------------- | ------------------------------------------------ |
| **导出 API** | `aot.externalize_module_parameters(model, scope)`       | `np.savez()` 或 ONNX initializer                 |
| **保存 API** | `aot.save_module_parameters(path, model)`               | `np.savez_compressed(path, **params)`            |
| **加载 API** | `ireert.ParameterIndex().load(path)`                    | `np.load(path)` / `onnx.numpy_helper.to_array()` |
| **绑定方式** | `param_index.create_provider(scope="model")` → 自动注入 | `tvm.nd.array(arr, device)` → 显式传参           |

## Runtime 接口对比

| 操作           | IREE                                                    | TVM                                              |
| -------------- | ------------------------------------------------------- | ------------------------------------------------ |
| **初始化**     | `ireert.Config("local-task")`                           | `tvm.cpu()` / `tvm.cuda()`                       |
| **加载模块**   | `ireert.VmModule.copy_buffer(ctx.instance, vmfb_bytes)` | `tvm.runtime.load_module(lib_path)`              |
| **创建执行器** | `ctx.modules.module["main"]`                            | `relax.VirtualMachine(ex, device)`               |
| **调用函数**   | `main_fn(input_ids, attention_mask)`                    | `vm["main"](input_ids, attention_mask, *params)` |
| **输出转换**   | `outputs[0].to_host()`                                  | `outputs[0].numpy()`                             |

## 图模型签名

| 阶段        | 输入                                                                                                                           | 输出                                                      |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- |
| **Prefill** | `input_ids (1, 32)` + `attention_mask (1, 32)`                                                                                 | `logits (1, 32, 49152)` + 60× KV cache `(1, 3, 32, 64)`   |
| **Decode**  | `input_ids (1, 1)` + `attention_mask (1, 101)` + `position_ids (1, 1)` + `cache_position (1,)` + 60× past KV `(1, 3, 100, 64)` | `logits (1, 1, 49152)` + 60× updated KV `(1, 3, 101, 64)` |

## 代码示例对比

| 场景         | IREE                                                                                                                                                                                                                     | TVM                                                                                                                                                            |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **加载权重** | `param_index = ireert.ParameterIndex()`<br>`param_index.load(params_path)`<br>`provider = param_index.create_provider(scope="model")`<br>`ctx.add_vm_module(ireert.create_io_parameters_module(ctx.instance, provider))` | `params = []`<br>`for init in onnx_model.graph.initializer:`<br>`    arr = onnx.numpy_helper.to_array(init)`<br>`    params.append(tvm.nd.array(arr, device))` |
| **执行推理** | `outputs = main_fn(input_ids, attention_mask)`                                                                                                                                                                           | `outputs = vm["main"](input_ids, attention_mask, *params)`                                                                                                     |
| **获取结果** | `logits = outputs[0].to_host()`                                                                                                                                                                                          | `logits = outputs[0].numpy()`                                                                                                                                  |

核心计算逻辑相同 (LLaMA-style attention + RoPE + SwiGLU MLP)，差异在于 IR 表示、编译工具链和 runtime API。
