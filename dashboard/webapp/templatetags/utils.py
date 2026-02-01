import math
from django import template

register = template.Library()


@register.filter
def get_item(d, key):
    try:
        return d.get(key, None)
    except Exception:
        return None


@register.filter
def display_cell(v):
    """
    Always show a visible placeholder instead of blank/NaN/None.
    """
    if v is None:
        return "—"

    try:
        if isinstance(v, float) and math.isnan(v):
            return "—"
    except Exception:
        pass

    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return "—"
    return s


@register.filter
def inr(v):
    """
    Format price as INR with commas, or show — if missing.
    """
    if v is None:
        return "—"

    try:
        if isinstance(v, float) and math.isnan(v):
            return "—"
    except Exception:
        pass

    try:
        n = float(v)
        return "₹ {:,.0f}".format(n)
    except Exception:
        return "—"
