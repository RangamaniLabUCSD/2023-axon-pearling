import driver
from pathlib import Path

# import unit
# Tension value from Bar-Ziv paper
# t = 4e-2* unit.erg/unit.cm**2
# print(t.to(unit.nanonewton/unit.micrometer))

driver.run_simulation((0.1, 0.001, 20, 2.5, 500, Path(".")))
