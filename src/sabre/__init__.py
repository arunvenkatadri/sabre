from .agent import explain_and_suggest
from .core import explain, explain_diff
from .packs import Pack, current_pack, list_packs, register_pack, use
from .session import disable_memory, enable_memory, recap, reset_memory

__all__ = [
    "explain",
    "explain_diff",
    "explain_and_suggest",
    "enable_memory",
    "disable_memory",
    "recap",
    "reset_memory",
    "use",
    "register_pack",
    "current_pack",
    "list_packs",
    "Pack",
    "load_ipython_extension",
]


def load_ipython_extension(ipython):
    from .magic import ExplainMagics
    ipython.register_magics(ExplainMagics)
