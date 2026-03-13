from pathlib import Path

from private import MODEL_ROOT

COMPILED_DIR = MODEL_ROOT / "compiler_iree" / "compiled"


def load_iree_module(vmfb_path: Path, params_path: Path | None = None):
    import iree.runtime as ireert

    assert vmfb_path.exists(), f"VMFB not found: {vmfb_path}\nRun 'python shared/stage0_export/export.py iree' first"

    config = ireert.Config("local-task")
    with open(vmfb_path, "rb") as f:
        vmfb_bytes = f.read()

    ctx = ireert.SystemContext(config=config)
    if params_path is not None and params_path.exists():
        param_index = ireert.ParameterIndex()
        param_index.load(str(params_path))
        param_provider = param_index.create_provider(scope="model")
        param_module = ireert.create_io_parameters_module(ctx.instance, param_provider)
        ctx.add_vm_module(param_module)

    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb_bytes)
    ctx.add_vm_module(vm_module)
    return ctx
