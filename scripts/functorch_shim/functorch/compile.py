# Compatibility exports used by torch._dynamo/vLLM with the ROCm torch 2.9 build.
from torch._functorch.partitioners import *  # noqa: F401,F403

try:
    from torch._functorch.aot_autograd import make_boxed_func  # noqa: F401
except ImportError:
    from torch._functorch._aot_autograd.utils import make_boxed_func  # noqa: F401

try:
    from torch._functorch.aot_autograd import make_boxed_compiler  # noqa: F401
except ImportError:
    pass

try:
    from torch._functorch.compilers import nop  # noqa: F401
except ImportError:
    pass

import importlib as _importlib

for _module_name in (
    "torch._functorch._aot_autograd",
    "torch._functorch.aot_autograd",
    "torch._functorch.compilers",
    "torch._functorch",
):
    try:
        _module = _importlib.import_module(_module_name)
    except Exception:
        continue
    for _name in ("draw_graph", "get_aot_graph_name", "get_graph_being_compiled"):
        if _name not in globals() and hasattr(_module, _name):
            globals()[_name] = getattr(_module, _name)
