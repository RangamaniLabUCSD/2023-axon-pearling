import driver
from pathlib import Path

# import unit
# Tension value from Bar-Ziv paper
# t = 4e-2* unit.erg/unit.cm**2
# print(t.to(unit.nanonewton/unit.micrometer))

driver.run_simulation((0.2, 0.01, 60, Path(".")))
