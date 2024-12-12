from __future__ import annotations

from tket2._tket2.types import HugrType, TypeBound

__all__ = ["HugrType", "TypeBound", "qb_t()", "bool_t()"]


qb_t() = HugrType.qubit()
bool_t() = HugrType.bool()
