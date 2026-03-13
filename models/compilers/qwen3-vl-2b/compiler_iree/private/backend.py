from pathlib import Path


MODEL_ROOT = Path(__file__).resolve().parents[2]
COMPILED_DIR = MODEL_ROOT / "compiler_iree" / "compiled"


def load_iree_module(vmfb_path: Path):
    import iree.runtime as ireert

    assert vmfb_path.exists(), f"VMFB not found: {vmfb_path}\nRun 'python stage0_export/export.py iree' first"

    config = ireert.Config("local-task")
    with open(vmfb_path, "rb") as f:
        vmfb_bytes = f.read()

    ctx = ireert.SystemContext(config=config)
    vm_module = ireert.VmModule.copy_buffer(ctx.instance, vmfb_bytes)
    ctx.add_vm_module(vm_module)
    return ctx
