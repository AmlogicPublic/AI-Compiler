import importlib.util
import sys
from pathlib import Path

STAGE0_DIR = Path(__file__).resolve().parent
SHARED_EXPORT = STAGE0_DIR / "export.py"


def run_export_backend(backend: str) -> None:
    sys.path.insert(0, str(STAGE0_DIR))
    spec = importlib.util.spec_from_file_location(f"qwen3_stage0_export_{backend}", SHARED_EXPORT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.argv = [str(SHARED_EXPORT), backend]
    module.main()
