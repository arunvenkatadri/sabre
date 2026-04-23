from IPython.core.magic import Magics, line_magic, magics_class

from .core import explain


@magics_class
class ExplainMagics(Magics):
    @line_magic
    def explain(self, line: str):
        """%explain [expr] — explain the last cell output, or the given expression."""
        expr = line.strip()
        if expr:
            obj = self.shell.ev(expr)
        else:
            obj = None
        return explain(obj)
