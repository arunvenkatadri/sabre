from .core import explain

__all__ = ["explain", "load_ipython_extension"]


def load_ipython_extension(ipython):
    from .magic import ExplainMagics
    ipython.register_magics(ExplainMagics)
