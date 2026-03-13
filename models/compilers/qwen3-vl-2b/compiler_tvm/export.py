import sys
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODEL_ROOT))
from shared.stage0_export.compiler_entry import run_export_backend


def main():
    run_export_backend("tvm")


if __name__ == "__main__":
    main()
