import driver
from pathlib import Path

# import unit
# Tension value from Bar-Ziv paper
# t = 4e-2* unit.erg/unit.cm**2
# print(t.to(unit.nanonewton/unit.micrometer))

driver.run_simulation(
    {
        "osmolarity": 0.1,
        "tension": 0.001,
        "kb_scale": 70,
        "target_volume_scale": 3,
        "reservoir_volume": 700,
        "output_dir": Path("."),
    }
)
