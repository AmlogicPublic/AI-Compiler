import importlib.util
import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
SHARED_EXPORT = MODEL_ROOT / "stage0_export" / "export.py"


def main():
    sys.path.insert(0, str(SHARED_EXPORT.parent))
    spec = importlib.util.spec_from_file_location("smollm2_shared_export", SHARED_EXPORT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sys.argv = [str(SHARED_EXPORT), "tvm"]
    module.main()


if __name__ == "__main__":
    main()
