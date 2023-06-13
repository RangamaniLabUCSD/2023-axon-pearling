"""Pint registry"""
__all__ = ["unit"]

import pint

unit = pint.UnitRegistry()


T = 310 * unit.degK
KT = (unit.boltzmann_constant * T).to(unit.nanonewton * unit.micrometer)
Kb = 60*KT
R = 0.05 * unit.micron


# for kb_scale in range(10,110, 10):
#     Kb = kb_scale*KT
#     print(f"max tension for {kb_scale} KT: {(1/(4*R*R)*Kb).to(unit.millinewton/unit.meter)}")
