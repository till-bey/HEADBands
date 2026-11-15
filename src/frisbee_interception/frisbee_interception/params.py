"""Declare a dataclass config as ROS 2 parameters '<section>.<field>' and read them back."""

from __future__ import annotations

from dataclasses import fields


def declare_section(node, section: str, defaults) -> dict:
    out = {}
    for f in fields(defaults):
        val = getattr(defaults, f.name)
        if isinstance(val, tuple):
            val = list(val)
        if isinstance(val, dict):
            continue
        out[f.name] = node.declare_parameter(f"{section}.{f.name}", val).value
    return out
