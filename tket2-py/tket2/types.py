from __future__ import annotations

from tket2._tket2.types import HugrType, TypeBound

__all__ = ["HugrType", "TypeBound", "QB_T", "BOOL_T"]


QB_T = HugrType.qubit()
BOOL_T = HugrType.bool()
