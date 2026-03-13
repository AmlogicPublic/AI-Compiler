# Compiler Module Principles

- **shared**: Code that works for all backends (IREE, TVM, ...)
- **private**: Code that only works for one specific backend
- Backend-specific code uses lazy imports to avoid polluting shared modules
- Keep all models' directory structure symmetric

## Directory Layout
```
models/compilers/<model>/
├── shared/                    # IREE & TVM shared code
│   └── stage0_export/
│       ├── export.py          # Entry point
│       ├── stage_prefill.py   # Prefill wrapper + export
│       ├── stage_decode.py    # Decode wrapper + export
│       └── stage_common.py    # Utilities
├── compiler_iree/
│   ├── private/               # IREE-specific runtime
│   ├── export.py              # Calls shared export
│   └── run.py                 # Runs inference with IREE
└── compiler_tvm/
    ├── private/               # TVM-specific runtime
    ├── export.py              # Calls shared export
    └── run.py                 # Runs inference with TVM
```
