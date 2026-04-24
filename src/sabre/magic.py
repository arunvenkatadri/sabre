from IPython.core.error import UsageError
from IPython.core.magic import Magics, line_magic, magics_class

from .agent import explain_and_suggest
from .core import explain, explain_diff
from .packs import current_pack, list_packs, use
from .session import recap


@magics_class
class ExplainMagics(Magics):
    @line_magic
    def explain(self, line: str):
        """%explain [expr] — explain the last cell output, or the given expression."""
        expr = line.strip()
        obj = self.shell.ev(expr) if expr else None
        return explain(obj)

    @line_magic
    def explain_diff(self, line: str):
        """%explain_diff a b — compare two expressions in the current namespace."""
        parts = line.strip().split()
        if len(parts) != 2:
            raise UsageError("Usage: %explain_diff <expr_a> <expr_b>")
        a = self.shell.ev(parts[0])
        b = self.shell.ev(parts[1])
        return explain_diff(a, b)

    @line_magic
    def suggest(self, line: str):
        """%suggest [expr] — explain and propose next-step cells with buttons."""
        expr = line.strip()
        obj = self.shell.ev(expr) if expr else None
        return explain_and_suggest(obj)

    @line_magic
    def recap(self, line: str):
        """%recap — show sabre's rolling notebook memory."""
        from IPython.display import Markdown, display
        display(Markdown(f"```\n{recap()}\n```"))

    @line_magic
    def use(self, line: str):
        """%use [pack-name] — activate a domain pack, or show the active one.

        Use `%use none` (or `%use -`) to clear the active pack.
        """
        from IPython.display import Markdown, display

        name = line.strip()
        if not name:
            active = current_pack()
            if active is None:
                display(Markdown("_No pack active._ Run `%packs` to list available packs."))
            else:
                display(Markdown(f"**Active pack:** `{active.name}` — {active.description}"))
            return
        if name in ("none", "-", "off", "clear"):
            use(None)
            display(Markdown("_Pack cleared._"))
            return
        pack = use(name)
        display(Markdown(f"**Activated:** `{pack.name}` — {pack.description}"))

    @line_magic
    def packs(self, line: str):
        """%packs — list available domain packs."""
        from IPython.display import Markdown, display
        active = current_pack()
        active_name = active.name if active else None
        lines = ["**Available packs:**", ""]
        for name, desc in list_packs():
            marker = " ←" if name == active_name else ""
            lines.append(f"- `{name}` — {desc}{marker}")
        display(Markdown("\n".join(lines)))
